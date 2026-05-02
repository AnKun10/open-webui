"""Qwen3-VL Image-Aware Context Compressor — Open WebUI Filter Function.

Paste this entire file into Admin Panel → Functions → New Function.
Source of truth lives in the repo at vast-templates/qwen3-vl-8b/functions/.
"""

import asyncio
import base64
import copy
import hashlib
import json as _json
import logging
import os
import time
from typing import Iterator, Optional

import aiosqlite
import httpx
from pydantic import BaseModel, Field

VERSION = "0.1.0"

log = logging.getLogger("qwenvl_image_compress")

CAPTION_SYSTEM_PROMPT = """\
Bạn là image captioner. Mô tả ảnh trong 1-2 câu khách quan, không quá 60 từ.
Cần nêu:
  - Chủ thể chính (người/vật/cảnh).
  - Văn bản nhìn thấy trong ảnh, copy nguyên văn nếu ngắn.
  - Bố cục/màu sắc nổi bật nếu liên quan.
KHÔNG suy diễn cảm xúc, KHÔNG khen chê, KHÔNG bịa chi tiết.
Trả về DUY NHẤT phần caption, không prefix \"Caption:\" hay markdown."""

CAPTION_USER_TEXT = "Mô tả ảnh này."

ROUTER_SYSTEM_PROMPT = """\
Bạn là router cho 1 hệ thống chat đa phương thức.
Cho 1 câu hỏi text-only của user và mô tả các ảnh user đã upload trước đó,
quyết định xem có cần gửi PIXEL của các ảnh đó cho LLM trả lời không.

Trả LLM cần nhìn pixel khi:
  - Câu hỏi tham chiếu trực tiếp ảnh: \"ảnh đó\", \"cái này\", \"hình thứ N\", \"trên màn hình\".
  - Câu hỏi đòi visual detail: màu, vị trí, đếm, OCR chính xác, so sánh ảnh.
  - Câu hỏi tiếp tục chủ đề liên quan đến nội dung ảnh.

Trả LLM KHÔNG cần pixel khi:
  - Câu hỏi đổi sang chủ đề mới không liên quan ảnh.
  - Câu hỏi tổng quát không có đại từ chỉ ảnh và caption đã đủ context.

Output DUY NHẤT 1 JSON object, không markdown:
  {\"need_images\": true|false, \"reason\": \"<1 câu ngắn tiếng Việt>\"}"""

ROUTER_USER_TEMPLATE = (
    "Ảnh đã upload trước đó (theo thứ tự):\n"
    "{captions_block}\n\n"
    "Câu hỏi mới của user:\n"
    "\"\"\"\n{user_text}\n\"\"\"\n"
)


class CaptionCache:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS captions (
        img_hash    TEXT PRIMARY KEY,
        caption     TEXT NOT NULL,
        model       TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        bytes_size  INTEGER,
        user_id     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_created ON captions(created_at);
    """

    def __init__(self, path: str):
        self.path = path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            async with aiosqlite.connect(self.path) as db:
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA synchronous = NORMAL")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.executescript(self.SCHEMA)
                await db.commit()
            self._initialized = True

    async def get(self, h: str) -> Optional[str]:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT caption FROM captions WHERE img_hash = ?", (h,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def put(
        self,
        h: str,
        caption: str,
        model: str,
        bytes_size: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (h, caption, model, now_ms, bytes_size, user_id),
            )
            await db.commit()

    async def get_many(self, hashes: list[str]) -> dict[str, str]:
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        async with aiosqlite.connect(self.path) as db:
            sql = f"SELECT img_hash, caption FROM captions WHERE img_hash IN ({placeholders})"
            async with db.execute(sql, hashes) as cur:
                return {h: c async for h, c in cur}

    async def put_many(self, items: list[tuple[str, str, str, Optional[int], Optional[str]]]) -> None:
        """items: (hash, caption, model, bytes_size, user_id)."""
        if not items:
            return
        now_ms = int(time.time() * 1000)
        rows = [(h, c, m, now_ms, sz, uid) for h, c, m, sz, uid in items]
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()


def has_images(msg: dict) -> bool:
    """Check if a message contains any image_url parts with non-empty URLs."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        p.get("type") == "image_url" and p.get("image_url", {}).get("url")
        for p in content
    )


def iter_image_parts(msgs: list[dict]) -> Iterator[tuple[int, int, str]]:
    """Yield (msg_idx, content_idx, url) for every image_url part."""
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    yield i, j, url


def find_latest_image_turn(msgs: list[dict]) -> Optional[int]:
    """Return the index of the latest user turn with images, or None."""
    latest: Optional[int] = None
    for i, msg in enumerate(msgs):
        if msg.get("role") == "user" and has_images(msg):
            latest = i
    return latest


def text_of(msg: dict) -> str:
    """Extract text content from a message (string or multimodal)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text") or ""
    return ""


async def hash_image_url(url: str, fetch_base: str,
                         fetch_timeout_s: int = 10) -> tuple[str, bytes]:
    """Return (sha256_hex, raw_bytes) for an image_url part.

    Supports data: URLs, absolute http(s) URLs, and relative paths
    (resolved against fetch_base, e.g. http://127.0.0.1:3000)."""
    if url.startswith("data:"):
        if "," not in url:
            raise ValueError("malformed data URL")
        raw = base64.b64decode(url.split(",", 1)[1])
        if not raw:
            raise ValueError("data URL payload is empty")
    elif url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=fetch_timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.content
    elif url.startswith("/"):
        full = f"{fetch_base.rstrip('/')}{url}"
        async with httpx.AsyncClient(timeout=fetch_timeout_s) as client:
            r = await client.get(full)
            r.raise_for_status()
            raw = r.content
    else:
        raise ValueError(f"Unsupported image URL scheme: {url[:32]!r}")
    return hashlib.sha256(raw).hexdigest(), raw


def rewrite_messages(msgs: list[dict],
                     keep_idx: Optional[int],
                     captions_by_url: dict[str, str]) -> list[dict]:
    """Return a deep-copied messages list with images stripped at all turns
    except `keep_idx`. Stripped images become `[Past image #N: <caption>]`
    text parts appended at the end of the message's content."""
    out = copy.deepcopy(msgs)
    for i, msg in enumerate(out):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if i == keep_idx:
            continue
        new_parts: list[dict] = []
        stripped_captions: list[str] = []
        img_n = 0
        for part in content:
            if part.get("type") == "image_url":
                img_n += 1
                url = part.get("image_url", {}).get("url", "")
                cap = captions_by_url.get(url) or "(no caption)"
                stripped_captions.append(f"[Past image #{img_n}: {cap}]")
            else:
                new_parts.append(part)
        if stripped_captions:
            extra_text = "\n".join(stripped_captions)
            if new_parts and new_parts[-1].get("type") == "text":
                new_parts[-1]["text"] = new_parts[-1]["text"] + "\n" + extra_text
            else:
                new_parts.append({"type": "text", "text": extra_text})
        msg["content"] = new_parts
    return out


async def caption_one(data_url: str,
                      base_url: str,
                      api_key: str,
                      model: str,
                      max_tokens: int,
                      timeout_s: int) -> str:
    """Call vLLM to caption a single image. Return trimmed caption text.

    Args:
        data_url: Image URL (data:, http://, https://, or relative path).
        base_url: vLLM base URL (e.g., "http://vllm/v1").
        api_key: Authorization bearer token.
        model: Model name (e.g., "qwen3-vl-8b").
        max_tokens: Max response tokens.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Trimmed caption text.

    Raises:
        httpx.HTTPStatusError: If vLLM returns a non-2xx status.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": CAPTION_USER_TEXT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload, headers=headers,
        )
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()


async def route(user_text: str,
                captions: list[str],
                base_url: str,
                api_key: str,
                model: str,
                max_tokens: int,
                timeout_s: int,
                failopen_keep: bool) -> tuple[bool, str]:
    """Call vLLM to decide whether images are needed for the user's question.

    Args:
        user_text: User's text-only message.
        captions: List of image captions (in order).
        base_url: vLLM base URL (e.g., "http://vllm/v1").
        api_key: Authorization bearer token.
        model: Model name (e.g., "qwen3-vl-8b").
        max_tokens: Max response tokens.
        timeout_s: HTTP request timeout in seconds.
        failopen_keep: If True, return (True, reason) on error; else (False, reason).

    Returns:
        (decision, reason) where decision is True if images are needed.

    Raises:
        Nothing — errors are caught and handled per failopen_keep.
    """
    captions_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
    user_content = ROUTER_USER_TEMPLATE.format(
        captions_block=captions_block, user_text=user_text,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        parsed = _json.loads(raw)
        decision = bool(parsed.get("need_images"))
        reason = str(parsed.get("reason", ""))[:200]
        return decision, reason
    except (httpx.HTTPError, _json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("router call failed: %s; failopen_keep=%s", e, failopen_keep)
        return failopen_keep, f"router failure: {type(e).__name__}"


async def ensure_captions(scanned: list[tuple[int, int, str, str, bytes]],
                          cache: CaptionCache,
                          base_url: str,
                          api_key: str,
                          model: str,
                          max_tokens: int,
                          timeout_s: int,
                          user_id: Optional[str]) -> dict[str, str]:
    """Ensure all scanned images have captions, either from cache or fresh.

    Args:
        scanned: List of (msg_idx, content_idx, url, hash, raw_bytes).
        cache: CaptionCache instance (already initialized).
        base_url: vLLM base URL (e.g., "http://vllm/v1").
        api_key: Authorization bearer token.
        model: Model name (e.g., "qwen3-vl-8b").
        max_tokens: Max response tokens for caption calls.
        timeout_s: HTTP request timeout in seconds.
        user_id: Optional user ID to record with new captions.

    Returns:
        {url: caption} for every successfully captioned image.
        On caption failure, the url is OMITTED from the result.
    """
    if not scanned:
        return {}

    # Gather all hashes and check cache
    hashes = [h for _, _, _, h, _ in scanned]
    hits = await cache.get_many(hashes)

    out: dict[str, str] = {}
    misses: list[tuple[str, str, bytes]] = []

    for _, _, url, h, raw in scanned:
        if h in hits:
            out[url] = hits[h]
        else:
            misses.append((url, h, raw))

    # If all cache hits, return early
    if not misses:
        return out

    # Caption all misses in parallel
    async def _one(url: str) -> tuple[str, Optional[str]]:
        try:
            cap = await caption_one(url, base_url, api_key, model, max_tokens, timeout_s)
            return url, cap or None
        except Exception as e:
            log.warning("caption failed for url=%s err=%s", url[:60], e)
            return url, None

    results = await asyncio.gather(*(_one(url) for url, _, _ in misses))

    # Collect successful captions and prepare batch write
    new_rows: list[tuple[str, str, str, Optional[int], Optional[str]]] = []
    for (url, h, raw), (url2, cap) in zip(misses, results):
        if cap:
            out[url] = cap
            new_rows.append((h, cap, model, len(raw), user_id))

    # Batch write new captions to cache
    if new_rows:
        await cache.put_many(new_rows)

    return out


class Filter:
    class Valves(BaseModel):
        vllm_base_url: str = Field(default="http://127.0.0.1:8000/v1")
        vllm_api_key: str = Field(default="sk-dummy")
        caption_model: str = Field(default="qwen3-vl-8b")
        router_model: str = Field(default="qwen3-vl-8b")
        cache_db_path: str = Field(default="/workspace/openwebui-data/img_captions.db")
        webui_internal_base: str = Field(default="http://127.0.0.1:3000")
        caption_max_tokens: int = Field(default=80)
        router_max_tokens: int = Field(default=60)
        caption_timeout_s: int = Field(default=30)
        router_timeout_s: int = Field(default=15)
        router_failopen_keep: bool = Field(default=True)
        priority: int = Field(default=5)

    class UserValves(BaseModel):
        enabled: bool = Field(default=True)
        force_keep_all_images: bool = Field(default=False)
        show_thinking_log: bool = Field(default=True)
        show_live_status: bool = Field(default=True)

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self._cache: Optional[CaptionCache] = None

    async def _ensure_cache(self) -> CaptionCache:
        if self._cache is None:
            self._cache = CaptionCache(self.valves.cache_db_path)
            await self._cache.init()
        return self._cache

    async def _emit_status(self, emit, description: str, *, done: bool = False,
                           hidden: bool = False, allowed: bool = True) -> None:
        if not (emit and allowed):
            return
        try:
            await emit({"type": "status", "data": {
                "description": description, "done": done, "hidden": hidden,
            }})
        except Exception as e:
            log.debug("status emit failed: %s", e)

    def _build_thinking_log(self, *,
                             n_images: int, n_misses: int, decision_label: str,
                             captions_used: list[tuple[str, str]],
                             user_text: Optional[str],
                             route_reason: Optional[str],
                             tokens_saved: int) -> str:
        lines = ["<details>", f"<summary>🧠 Image compressor reasoning ({n_images} ảnh, {n_misses} caption mới, {decision_label})</summary>", ""]
        lines.append("**Step 1 — Image scan**")
        lines.append(f"- Tổng {n_images} ảnh; cache miss: {n_misses}, hit: {n_images - n_misses}")
        lines.append("")
        if captions_used:
            lines.append("**Step 2 — Captions in use**")
            for h_short, cap in captions_used:
                lines.append(f"- `{h_short}` → \"{cap[:120]}\"")
            lines.append("")
        if user_text is not None:
            lines.append("**Step 3 — Router**")
            lines.append(f"- User: \"{user_text[:200]}\"")
            lines.append(f"- {decision_label}")
            if route_reason:
                lines.append(f"- Reason: *{route_reason}*")
            lines.append("")
        lines.append("**Step 4 — Rewrite**")
        if tokens_saved > 0:
            lines.append(f"- Token estimate saved: ~{tokens_saved}")
        else:
            lines.append("- Images preserved; no tokens saved")
        lines.append("</details>")
        lines.append("")
        return "\n".join(lines)

    def _estimate_image_tokens(self, raw: bytes) -> int:
        return max(800, len(raw) // 800)

    async def _emit_thinking_log(self, emit, user_valves, scanned, misses,
                                  decision_label, captions_by_url,
                                  user_text, route_reason, tokens_saved) -> None:
        if not (emit and user_valves.show_thinking_log):
            return
        captions_used = [
            (h[:8], captions_by_url.get(url, "(no caption)"))
            for (_, _, url, h, _) in scanned
        ]
        content = self._build_thinking_log(
            n_images=len(scanned), n_misses=len(misses),
            decision_label=decision_label,
            captions_used=captions_used,
            user_text=user_text, route_reason=route_reason,
            tokens_saved=tokens_saved,
        )
        try:
            await emit({"type": "message", "data": {"content": content}})
        except Exception as e:
            log.debug("thinking-log emit failed: %s", e)

    async def inlet(self, body: dict, __user__: Optional[dict] = None,
                    __metadata__: Optional[dict] = None,
                    __event_emitter__=None) -> dict:
        try:
            return await self._inlet_impl(body, __user__, __metadata__, __event_emitter__)
        except Exception as e:
            log.exception("ImageCompressor passthrough due to error: %s", e)
            return body

    async def _inlet_impl(self, body, __user__, __metadata__, __event_emitter__) -> dict:
        start = time.monotonic()
        msgs = body.get("messages") or []
        if not msgs:
            return body

        user_valves_raw = (__user__ or {}).get("valves")
        user_valves = (
            user_valves_raw if isinstance(user_valves_raw, self.UserValves)
            else self.UserValves(**(user_valves_raw or {}))
        )
        if not user_valves.enabled:
            return body

        last = msgs[-1] if msgs[-1].get("role") == "user" else None
        if last is None:
            return body

        url_list = list(iter_image_parts(msgs))
        if not url_list:
            return body

        cache = await self._ensure_cache()

        scanned: list[tuple[int, int, str, str, bytes]] = []
        for msg_idx, c_idx, url in url_list:
            try:
                h, raw = await hash_image_url(url, self.valves.webui_internal_base)
                scanned.append((msg_idx, c_idx, url, h, raw))
            except Exception as e:
                log.warning("hash skipped url=%s err=%s", url[:60], e)

        existing = await cache.get_many([h for *_, h, _ in scanned])
        misses = [s for s in scanned if s[3] not in existing]
        if misses:
            await self._emit_status(
                __event_emitter__,
                f"🖼️ Captioning {len(misses)} new image(s)...",
                allowed=user_valves.show_live_status,
            )

        captions_by_url = await ensure_captions(
            scanned=scanned, cache=cache,
            base_url=self.valves.vllm_base_url, api_key=self.valves.vllm_api_key,
            model=self.valves.caption_model,
            max_tokens=self.valves.caption_max_tokens,
            timeout_s=self.valves.caption_timeout_s,
            user_id=(__user__ or {}).get("id"),
        )

        decision_label: str
        route_reason: Optional[str] = None
        user_text_for_log: Optional[str] = None
        tokens_saved = 0

        if user_valves.force_keep_all_images:
            await self._emit_status(
                __event_emitter__, "✅ Compressor done (force_keep_all_images)",
                done=True, allowed=user_valves.show_live_status,
            )
            decision_label = "force_keep_all_images"
            keep_idx: Optional[int] = None  # body untouched, keep_idx unused
            await self._emit_thinking_log(
                __event_emitter__, user_valves, scanned, misses,
                decision_label, captions_by_url, user_text_for_log,
                route_reason, tokens_saved,
            )
            log.info(_json.dumps({
                "event": "image_compress_inlet",
                "chat_id": (__metadata__ or {}).get("chat_id"),
                "n_images": len(scanned),
                "cache_hits": len(scanned) - len(misses),
                "cache_misses": len(misses),
                "decision": "force_keep",
                "tokens_saved": tokens_saved,
                "latency_ms": int((time.monotonic() - start) * 1000),
            }))
            return body

        if has_images(last):
            keep_idx = len(msgs) - 1
            decision_label = "kept new upload"
        else:
            latest_idx = find_latest_image_turn(msgs)
            if latest_idx is None:
                return body
            await self._emit_status(
                __event_emitter__,
                "🧭 Routing: do we need pixels for this question?",
                allowed=user_valves.show_live_status,
            )
            captions_for_router = [
                captions_by_url.get(url, "(no caption)")
                for (mi, _, url, _, _) in scanned if mi == latest_idx
            ]
            user_text_for_log = text_of(last)
            decision, route_reason = await route(
                user_text=user_text_for_log, captions=captions_for_router,
                base_url=self.valves.vllm_base_url, api_key=self.valves.vllm_api_key,
                model=self.valves.router_model,
                max_tokens=self.valves.router_max_tokens,
                timeout_s=self.valves.router_timeout_s,
                failopen_keep=self.valves.router_failopen_keep,
            )
            if decision:
                keep_idx = latest_idx
                decision_label = "🎯 Router: keep images"
            else:
                keep_idx = None
                decision_label = "🎯 Router: drop images"
            await self._emit_status(
                __event_emitter__, decision_label,
                allowed=user_valves.show_live_status,
            )

        if keep_idx is None:
            tokens_saved = sum(self._estimate_image_tokens(raw) for *_, raw in scanned)
        else:
            tokens_saved = sum(
                self._estimate_image_tokens(raw)
                for (mi, _, _, _, raw) in scanned if mi != keep_idx
            )

        body["messages"] = rewrite_messages(msgs, keep_idx, captions_by_url)
        await self._emit_status(
            __event_emitter__, "✅ Compressor done",
            done=True, allowed=user_valves.show_live_status,
        )
        await self._emit_thinking_log(
            __event_emitter__, user_valves, scanned, misses,
            decision_label, captions_by_url, user_text_for_log,
            route_reason, tokens_saved,
        )
        log.info(_json.dumps({
            "event": "image_compress_inlet",
            "chat_id": (__metadata__ or {}).get("chat_id"),
            "n_images": len(scanned),
            "cache_hits": len(scanned) - len(misses),
            "cache_misses": len(misses),
            "decision": "keep" if keep_idx is not None else "drop",
            "tokens_saved": tokens_saved,
            "latency_ms": int((time.monotonic() - start) * 1000),
        }))
        return body
