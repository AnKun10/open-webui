"""Qwen3-VL Image-Aware Context Compressor — Open WebUI Filter Function.

Paste this entire file into Admin Panel → Functions → New Function.
Source of truth lives in the repo at vast-templates/qwen3-vl-8b/functions/.
"""

import asyncio
import base64
import copy
import hashlib
import os
import time
from typing import Iterator, Optional

import aiosqlite
import httpx

VERSION = "0.1.0-dev"

CAPTION_SYSTEM_PROMPT = """\
Bạn là image captioner. Mô tả ảnh trong 1-2 câu khách quan, không quá 60 từ.
Cần nêu:
  - Chủ thể chính (người/vật/cảnh).
  - Văn bản nhìn thấy trong ảnh, copy nguyên văn nếu ngắn.
  - Bố cục/màu sắc nổi bật nếu liên quan.
KHÔNG suy diễn cảm xúc, KHÔNG khen chê, KHÔNG bịa chi tiết.
Trả về DUY NHẤT phần caption, không prefix \"Caption:\" hay markdown."""

CAPTION_USER_TEXT = "Mô tả ảnh này."


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
