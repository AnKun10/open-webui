# Qwen3-VL Image-Aware Context Compressor (Filter) — Design

**Date:** 2026-05-01
**Status:** Approved (design phase)
**Related:** Vast template `vast-templates/qwen3-vl-8b/` (Open WebUI + vLLM serving Qwen3-VL-8B-Instruct)

## 1. Goal

Reduce image-token blowup in long multimodal chats with Qwen3-VL-8B by
selectively stripping pixel data from older user turns before the request
reaches vLLM, while preserving semantic context via cached captions and a
text-only follow-up router.

The compressor lives as an **Open WebUI Filter Function**, attached to model
`qwen3-vl-8b` only. No upstream code is forked. The filter is toggleable
per-chat in the UI and per-user via UserValves.

## 2. Scope & non-goals

### In scope
- Inlet-side rewrite of `body['messages']` for the Qwen3-VL model.
- Cached image captioning via Qwen3-VL itself (one HTTP call per unseen image).
- LLM-based router that decides, on text-only follow-ups, whether older images
  need to be re-included (with pixels) to answer correctly.
- Persistent SQLite cache of `sha256(image_bytes) → caption`.
- Live status events and a collapsible thinking-log block in the chat UI.
- Failure-safe degradation: any error path returns the body unchanged.

### Out of scope (v1)
- Text-history compression. Qwen3-VL-8B has 32k context; observed chats stay
  well under the limit on text alone.
- Open WebUI long-term `Memory` feature (the per-user `Memory` table) — separate concern.
- Cross-chat retrieval / RAG over chat history.
- Image downsampling, re-encoding, or `min_pixels`/`max_pixels` tuning.
- Auto re-caption when the caption model changes (manual cache delete only).
- Web UI for cache inspection (use `sqlite3` CLI).
- Streaming caption output.
- Multi-tenant cache isolation (captions are content-addressable; sharing across
  users is a feature, not a bug).

## 3. Constraints & decisions

| Decision | Choice |
|----------|--------|
| Where logic lives | **Open WebUI Filter Function** (no fork, no extra container) |
| Hook used | `inlet` only (no `outlet`, no `stream`) |
| Caption model | `qwen3-vl-8b` via local vLLM (`http://127.0.0.1:8000/v1`) |
| Router model | `qwen3-vl-8b` (text-only call) — same instance |
| Image retention rule | Keep pixels only for the **most recent image-bearing user turn** |
| Override | Router decides, on text-only follow-up turns, whether to revive that turn's pixels |
| Cache backend | **SQLite WAL** at `/workspace/openwebui-data/img_captions.db` |
| Cache key | `sha256(image_bytes)` (content-addressable, cross-user) |
| Caption sharing | Cross-user (same bytes ⇒ same caption) |
| Failure mode | **Fail-open** everywhere (passthrough on any exception) |
| Router fail policy | `router_failopen_keep=True` by default (preserve accuracy) |

## 4. Architecture

```
┌─────────────────────────── Open WebUI process (port 3000) ─────────────────┐
│                                                                            │
│  POST /api/chat/completions                                                │
│         │                                                                  │
│         ▼                                                                  │
│  utils/middleware.py:process_chat_payload                                  │
│         │ (after pipeline filters, before memory/RAG/tool handlers)        │
│         ▼                                                                  │
│  ╔════ Filter.inlet ═════════════════════════════════════════════════╗    │
│  ║                                                                    ║    │
│  ║   A. Scan messages[] for image_url parts → list[(idx, hash, url)] ║    │
│  ║   B. Cache lookup; for misses, parallel caption calls to vLLM ────╫────┐
│  ║   C. Classify last user turn:                                      ║    │
│  ║         has_new_images? → keep_idx = len(msgs)-1                  ║    │
│  ║         text_only?      → router call → keep_idx = latest_img_idx ╫────┤
│  ║                                              or None              ║    │
│  ║   D. Rewrite msgs: at all turns ≠ keep_idx, replace each          ║    │
│  ║      image_url part with a text part:                              ║    │
│  ║         "[Past image #N: <caption>]"                               ║    │
│  ║                                                                    ║    │
│  ╚════════════════════════════════════════════════════════════════════╝    │
│         │                                                                  ▼
│         ▼                                          ┌──────────────────────────────┐
│   modified body                                    │ vLLM (port 8000)             │
│         │                                          │  qwen3-vl-8b                 │
│         ▼                                          │  - main chat                 │
│  POST http://127.0.0.1:8000/v1/chat/completions ──▶│  - caption (B)               │
│                                                    │  - router  (C, text-only)    │
│                                                    └──────────────────────────────┘
│                                                                                  
│   /workspace/openwebui-data/img_captions.db (SQLite WAL, persisted volume)       │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Pipeline position
Inside `utils/middleware.py`, `process_filter_functions(filter_type='inlet')`
is invoked at line ~2311, after the pipeline inlet filter and **before** the
memory/RAG/file-context handlers. The filter therefore sees the raw user
payload and runs ahead of `chat_memory_handler`, `process_chat_files`, and
function-calling assembly.

### 4.2 Data flow scenarios

| # | User turn | Filter actions | Outcome |
|---|-----------|---------------|---------|
| 1 | Upload 3 images + text | Hash, cache miss → 3 parallel caption calls; `has_new=True`, `keep_idx=last` | All 3 images kept; cache populated |
| 2 | Text-only "ảnh thứ 2 màu gì?" | Cache hit; router call returns `{need_images:true,…}`; `keep_idx=msg #1` | Images at msg #1 kept |
| 3 | Text-only "explain Python decorators" | Cache hit; router returns `{need_images:false,…}`; `keep_idx=None` | All images stripped → captions inline |
| 4 | Upload 1 NEW image | Cache miss for new image only; `has_new=True`, `keep_idx=last` | Old images stripped, new image kept |

### 4.3 Design invariants
1. **Idempotent**: the filter is deterministic given cached captions and
   `temperature=0` router output — running `inlet` twice on the same body
   yields the same result.
2. **Stateless w.r.t. body**: filter mutates only the transient request body;
   does not write to `chat.chat`. Reload of the chat shows the original images.
3. **Read-only against Open WebUI DB**: filter does not depend on Open WebUI
   internal models or Alembic migrations.
4. **Fail-open**: a top-level `try/except` in `inlet` returns `body` unchanged
   on any uncaught exception.

## 5. Components

### 5.1 File layout
Single Python file (~350 lines + tests):

```
qwenvl_image_compress.py
├── module-level constants (CAPTION_SYSTEM, ROUTER_SYSTEM, ROUTER_USER_TEMPLATE)
├── class CaptionCache               (~60 lines, aiosqlite wrapper)
└── class Filter                     (~250 lines)
    ├── class Valves(BaseModel)
    ├── class UserValves(BaseModel)
    ├── __init__
    ├── async def inlet              (top-level guard + dispatch)
    ├── async def _inlet_impl
    ├── async def _scan_images
    ├── async def _hash_image
    ├── async def _ensure_captions
    ├── async def _caption_one
    ├── async def _route
    ├──      def _has_images
    ├──      def _find_latest_image_turn
    ├──      def _text_of
    ├──      def _rewrite_messages
    └──      def _estimate_image_tokens
```

### 5.2 Valves (admin)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `vllm_base_url` | str | `http://127.0.0.1:8000/v1` | OpenAI-compatible endpoint |
| `vllm_api_key` | str | `sk-dummy` | Auth header |
| `caption_model` | str | `qwen3-vl-8b` | Vision-capable |
| `router_model` | str | `qwen3-vl-8b` | Text-capable; can later point to a smaller model |
| `cache_db_path` | str | `/workspace/openwebui-data/img_captions.db` | Persistent volume |
| `caption_max_tokens` | int | 80 | Hard cap |
| `router_max_tokens` | int | 60 | Hard cap |
| `caption_timeout_s` | int | 30 | httpx timeout |
| `router_timeout_s` | int | 15 | httpx timeout |
| `router_failopen_keep` | bool | `True` | On router failure, keep images (preserves accuracy) |
| `priority` | int | 5 | Inter-filter ordering (lower runs earlier) |
| `webui_internal_base` | str | `http://127.0.0.1:3000` | Used only when an image part references a relative URL (`/api/...`); rarely hit because Open WebUI normally inlines images as data URLs |

### 5.3 UserValves (per-user)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `enabled` | bool | `True` | Per-user kill-switch |
| `force_keep_all_images` | bool | `False` | Bypass router, always keep |
| `show_thinking_log` | bool | `True` | Render collapsible `<details>` block |
| `show_live_status` | bool | `True` | Emit live status bubbles during inlet |

### 5.4 Filter API surface

```python
async def inlet(
    self,
    body: dict,
    __user__: dict,
    __metadata__: dict,
    __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> dict:
    """Main entry. Wraps _inlet_impl in a top-level try/except for fail-open."""
```

`__id__`, `__model__` are not required but may be added later for filter-aware
routing or model-id checks.

### 5.5 Filter binding
- Toggle: `self.toggle = True` → per-chat enable button in the UI.
- The filter inlet block as a whole already runs ahead of `chat_memory_handler`
  and `process_chat_files` (see `utils/middleware.py:2311` vs `:2336`+); the
  `priority` valve only orders this filter relative to **other filters** in
  the inlet block. Default 5 keeps it well-positioned among any future
  co-installed filters.
- Bound to model via **Admin Panel → Models → qwen3-vl-8b → Filters → tick**.
  Stored in `model.info.meta.filterIds` (read by `utils/filter.py:35`).

## 6. Prompts

### 6.1 Caption call
**System:**
```
Bạn là image captioner. Mô tả ảnh trong 1-2 câu khách quan, không quá 60 từ.
Cần nêu:
  - Chủ thể chính (người/vật/cảnh).
  - Văn bản nhìn thấy trong ảnh, copy nguyên văn nếu ngắn.
  - Bố cục/màu sắc nổi bật nếu liên quan.
KHÔNG suy diễn cảm xúc, KHÔNG khen chê, KHÔNG bịa chi tiết.
Trả về DUY NHẤT phần caption, không prefix "Caption:" hay markdown.
```

**User:** `Mô tả ảnh này.` + `image_url` part.

**Settings:** `max_tokens=80`, `temperature=0.2`, `stream=False`.

### 6.2 Router call
**System:**
```
Bạn là router cho 1 hệ thống chat đa phương thức.
Cho 1 câu hỏi text-only của user và mô tả các ảnh user đã upload trước đó,
quyết định xem có cần gửi PIXEL của các ảnh đó cho LLM trả lời không.

Trả LLM cần nhìn pixel khi:
  - Câu hỏi tham chiếu trực tiếp ảnh: "ảnh đó", "cái này", "hình thứ N", "trên màn hình".
  - Câu hỏi đòi visual detail: màu, vị trí, đếm, OCR chính xác, so sánh ảnh.
  - Câu hỏi tiếp tục chủ đề liên quan đến nội dung ảnh.

Trả LLM KHÔNG cần pixel khi:
  - Câu hỏi đổi sang chủ đề mới không liên quan ảnh.
  - Câu hỏi tổng quát không có đại từ chỉ ảnh và caption đã đủ context.

Output DUY NHẤT 1 JSON object, không markdown:
  {"need_images": true|false, "reason": "<1 câu ngắn tiếng Việt>"}
```

**User template:**
```
Ảnh đã upload trước đó (theo thứ tự):
{captions_block}

Câu hỏi mới của user:
"""
{user_text}
"""
```

**Settings:** `max_tokens=60`, `temperature=0.0`, `response_format={"type":"json_object"}`,
`stream=False`.

## 7. Thinking log UX

Two layers, both gated by independent UserValves.

### 7.1 Live status bubbles (`type: "status"`)
Persisted to chat via `socket/main.py:802-808` (`Chats.add_message_status_to_chat_by_id_and_message_id`),
so they survive page reloads.

| When | description | done | hidden |
|------|-------------|------|--------|
| Inlet start, cache miss > 0 | `🖼️ Captioning {N} new image(s)...` | False | False |
| Each caption complete | `✏️ Captioned image {i}/{N}` | False | True |
| Before router | `🧭 Routing: do we need pixels for this question?` | False | False |
| Router done | `🎯 Router: keep images` / `🎯 Router: drop images, saved ~{T} tokens` | False | False |
| All done | `✅ Compressor done` | True | False |

### 7.2 Collapsible reasoning block (`type: "message"`)
After all status events, emit a single `type: "message"` event whose `data.content`
is the `<details>` block below. Open WebUI persists the content into the assistant
message record (see `socket/main.py:810+`); exact placement (prepend vs append)
must be verified during implementation — the user-visible outcome required is
that the block is rendered alongside the assistant response and survives reload.

```markdown
<details>
<summary>🧠 Image compressor reasoning ({n_imgs} ảnh, {n_new} caption mới, router: {decision})</summary>

**Step 1 — Image scan**
- Tổng {n_imgs} ảnh
- Cache hit: {hits}, miss: {misses}

**Step 2 — Captioning** *(if any miss)*
- `{hash_short}` → "{caption}"  *(latency: {ms}ms)*
- ...

**Step 3 — Router** *(if text-only)*
- User: "{user_text}"
- Decision: `need_images={bool}`
- Reason: *{reason}*  *(latency: {ms}ms)*

**Step 4 — Rewrite**
- Keep images at msg #{idx} OR strip all
- Token estimate saved: ~{tokens}
</details>
```

### 7.3 Token estimation
Heuristic for the "saved ~N tokens" display:
```python
def _estimate_image_tokens(data_url: str) -> int:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw_bytes = len(b64) * 3 // 4
    return max(800, raw_bytes // 800)  # rough, adequate for UX display
```
Not exact — purely indicative.

## 8. Caching layer

### 8.1 Schema
```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS captions (
    img_hash    TEXT PRIMARY KEY,    -- sha256 hex (64 chars) of raw image bytes
    caption     TEXT NOT NULL,
    model       TEXT NOT NULL,       -- 'qwen3-vl-8b'
    created_at  INTEGER NOT NULL,    -- epoch ms
    bytes_size  INTEGER,
    user_id     TEXT                 -- nullable, audit only
);

CREATE INDEX IF NOT EXISTS idx_created ON captions(created_at);
```

### 8.2 Hash strategy
Hash the **raw decoded image bytes**, not the URL. Open WebUI may store
either `data:image/...;base64,...` URLs or HTTP(S)/relative URLs; the hashing
function handles both:

```python
async def _hash_image(self, content_part: dict) -> tuple[str, bytes]:
    url = content_part["image_url"]["url"]
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1])
    elif url.startswith(("http://", "https://", "/")):
        full_url = url if url.startswith("http") else f"{self.valves.webui_internal_base}{url}"
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(full_url)
            r.raise_for_status()
            raw = r.content
    else:
        raise ValueError(f"Unsupported image URL scheme: {url[:32]}")
    return hashlib.sha256(raw).hexdigest(), raw
```

### 8.3 Concurrency
- **Within an inlet call**: `asyncio.gather` for parallel caption calls;
  `INSERT OR IGNORE` on writes to handle race with concurrent inlets that
  caption the same image.
- **Cross-process**: Open WebUI runs single-worker by default
  (`UVICORN_WORKERS=1`); WAL mode allows safe concurrent reads with the
  occasional write. If `UVICORN_WORKERS>1` is later set, the same WAL +
  `INSERT OR IGNORE` strategy still applies.

### 8.4 Eviction
Not implemented in v1. Each row is ~200 bytes; 100k images ~20 MB. Will revisit
only if disk usage becomes a concern.

### 8.5 Migration
If `Valves.caption_model` changes, existing rows remain valid (lookup still
hits, just with the older model's caption). To force re-caption: delete the
DB file (`rm /workspace/openwebui-data/img_captions.db`); the filter recreates
it on next inlet.

## 9. Error handling

Top-level guard wraps the entire `_inlet_impl`:

```python
async def inlet(self, body, __user__=None, __metadata__=None,
                __event_emitter__=None) -> dict:
    try:
        return await self._inlet_impl(body, __user__, __metadata__, __event_emitter__)
    except Exception as e:
        log.exception("ImageCompressor passthrough due to error: %s", e)
        if __event_emitter__:
            with contextlib.suppress(Exception):
                await __event_emitter__({"type": "status", "data": {
                    "description": f"⚠️ Image compressor error, passthrough: {type(e).__name__}",
                    "done": True,
                }})
        return body
```

| Failure | Detection | Fallback | UI signal |
|---------|-----------|----------|-----------|
| vLLM unreachable (caption) | `httpx.ConnectError`/timeout | Skip caption; image stays in prompt | `⚠️ Caption skipped (vLLM unreachable)` |
| Caption response empty/garbage | empty string check | Same as above | `⚠️ Caption empty, image kept as-is` |
| Router call fails / bad JSON | exception / `json.JSONDecodeError` | Apply `router_failopen_keep` (default keep) | `⚠️ Router failed, keeping images` |
| Cache DB init fails | `OperationalError` | No-cache mode: caption every call, no persist | `⚠️ Cache disabled` |
| Cache lookup mid-flight error | `OperationalError` | Treat as miss → re-caption | hidden |
| Image bytes corrupt | `binascii.Error` / fetch error | Skip image; keep as-is | `⚠️ Image #N skipped (bad data)` |
| `__user__ is None` (background) | None check | Skip UserValves; use Valves only | hidden |
| Empty/malformed `body['messages']` | length check | Return body unchanged | hidden |
| Any other exception | top-level except | Return body unchanged + log full traceback | `⚠️ Compressor error, passthrough` |

## 10. Testing

### 10.1 Unit tests (`test_qwenvl_image_compress.py`)
- `TestScanImages`: text-only, single image, 3 images, data URL vs HTTP URL.
- `TestRewriteMessages`: keep_idx targeting; strip-all; caption substitution;
  text part preserved; role unchanged.
- `TestCaptionCache`: init creates schema; get miss/hit; put-then-get;
  `get_many`; concurrent writes don't corrupt; init idempotent;
  disk-full does not crash.
- `TestPolicyDecision` (with `respx` mocking vLLM): fresh upload strips old;
  text-only router-yes keeps latest turn; text-only router-no strips all;
  no prior images passthrough; `force_keep_all_images` UserValve;
  `enabled=False` UserValve.
- `TestErrorHandling`: vLLM unreachable; router invalid JSON;
  caption timeout; top-level exception passthrough.

### 10.2 Integration test (manual on Vast pod)
`scripts/integration_test_image_compressor.sh`:
1. Scenario 1: upload 3 images + ask. Verify cache row count = 3.
2. Scenario 2: text follow-up referencing image. Verify status events
   include "Router: keep images".
3. Scenario 3: text follow-up changing topic. Verify "Router: drop images,
   saved ~XXXX tokens" status and that `body['messages']` sent to vLLM
   contains zero `image_url` parts (verified via vLLM access log).

### 10.3 Manual UI checklist
- [ ] Toggle filter on chat → see thinking-log block on first response.
- [ ] Upload 1 image + ask → block shows captioning details.
- [ ] Reload page → block persists.
- [ ] Follow-up about image → router decides "keep".
- [ ] Follow-up change topic → router decides "drop", token saved > 0.
- [ ] Disable `UserValves.enabled` → no filter activity.
- [ ] Enable `force_keep_all_images` → router not called.
- [ ] Disable `show_thinking_log` → no `<details>` block, status bubbles still appear.
- [ ] Disable `show_live_status` → no status bubbles, `<details>` block still appears.
- [ ] Stop vLLM → chat does not break (passthrough mode), status indicates failure.

## 11. Deployment

### 11.1 Filter installation (one-time)
1. Author `qwenvl_image_compress.py` locally; run unit tests.
2. **Open WebUI Admin Panel → Functions → `+`** → paste code → Save with
   name "Qwen3-VL Image Compressor".
3. **Admin Panel → Models → qwen3-vl-8b → Filters** → tick the new filter
   → Save.

### 11.2 `onstart.sh` updates required for memory + filter to work durably

These are independent of the filter code itself but are prerequisites for
persistent behavior on a Vast pod:

```bash
mkdir -p /workspace/openwebui-data
export DATA_DIR=/workspace/openwebui-data

# Stable secret across pod restarts so JWTs/sessions survive
export WEBUI_SECRET_KEY=$(cat /workspace/.webui-secret 2>/dev/null \
    || (openssl rand -hex 32 | tee /workspace/.webui-secret))

# Use plural form; remove legacy single-URL form
unset OPENAI_API_BASE_URL OPENAI_API_KEY
export OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1
export OPENAI_API_KEYS=sk-dummy
```

`DATA_DIR=/workspace/openwebui-data` ensures both Open WebUI's main SQLite
database and the filter's `img_captions.db` survive pod recreation.

### 11.3 Rollback
1. Disable filter binding from model: **Admin → Models → qwen3-vl-8b → Filters**
   → uncheck. Effect immediate, no restart.
2. Disable filter globally: **Admin → Functions → toggle off** the filter.
3. If cache DB is corrupt: `rm /workspace/openwebui-data/img_captions.db`.
   Filter recreates on next inlet; first re-encounter of any image incurs a
   caption call.

## 12. Observability

### 12.1 Structured log line (per inlet)
```python
log.info(json.dumps({
    "event": "image_compress_inlet",
    "chat_id": __metadata__.get("chat_id"),
    "n_images": len(images),
    "cache_hits": len(existing),
    "cache_misses": len(missing),
    "decision": "keep" if keep_idx is not None else "drop",
    "tokens_saved": tokens_saved,
    "latency_ms": int((time.monotonic() - start) * 1000),
}))
```

### 12.2 Log levels
- `INFO` — one line per inlet (above).
- `WARNING` — caption fail, router fail, cache fail.
- `ERROR` — top-level exception with traceback.
- `DEBUG` — full caption text, full router output JSON.

### 12.3 Inspection helper
`scripts/dump_captions.py`:
```python
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
for row in con.execute(
    "SELECT img_hash, substr(caption,1,80), model, created_at "
    "FROM captions ORDER BY created_at DESC LIMIT 50"
):
    print(row)
```

Run via SSH: `python scripts/dump_captions.py /workspace/openwebui-data/img_captions.db`.

## 13. Open questions for v2 (not in this spec)

- Make caption shareable across multiple Open WebUI instances by switching
  to a shared Postgres table.
- Add a tiny dedicated text-only router model (Qwen2.5-1.5B-Instruct) on a
  second vLLM port to cut router latency from ~150ms to ~30ms.
- Per-image age-based downsampling tier (mid-age images at 224px) as an
  alternative to outright stripping.
- Image-aware semantic retrieval: revive only the *specific* image referenced
  by the user message, not the whole turn.
