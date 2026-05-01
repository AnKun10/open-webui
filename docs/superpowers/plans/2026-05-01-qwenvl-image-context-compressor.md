# Qwen3-VL Image-Aware Context Compressor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Open WebUI Filter Function that strips image pixels from older user turns and revives them only when an LLM router judges a text-only follow-up needs visual context. Captions cached in a SQLite file keyed by `sha256(image_bytes)`.

**Architecture:** Single-file Python filter module dropped into Open WebUI's Functions store, attached to model `qwen3-vl-8b`. The `inlet` hook rewrites `body['messages']` before the request reaches vLLM. Two HTTP calls to Qwen3-VL itself per worst-case inlet: one for captioning new images, one for routing text-only follow-ups. State persisted in `/workspace/openwebui-data/img_captions.db` (SQLite WAL).

**Tech Stack:** Python 3.11+, `httpx`, `aiosqlite`, `pydantic` v2, `pytest`, `pytest-asyncio`, `respx` (httpx mocking).

**Spec:** `docs/superpowers/specs/2026-05-01-qwenvl-image-context-compressor-design.md`

---

## File structure

```
vast-templates/qwen3-vl-8b/
├── functions/
│   ├── qwenvl_image_compress.py         # Filter source (paste into Admin UI)
│   ├── conftest.py                      # pytest fixtures
│   ├── pytest.ini                       # asyncio_mode=auto
│   ├── requirements-dev.txt             # pytest, respx, etc.
│   └── test_qwenvl_image_compress.py    # unit tests (~600 lines)
├── scripts/
│   ├── dump_captions.py                 # CLI cache inspector
│   └── integration_test_image_compressor.sh
└── onstart.sh                           # update with DATA_DIR + WEBUI_SECRET_KEY + plural OPENAI_API_BASE_URLS
```

The filter is a single self-contained `.py` file. After local TDD, the engineer pastes its contents into **Open WebUI Admin → Functions → New Function** and ticks it under **Models → qwen3-vl-8b → Filters**. No backend fork.

Tests run locally with `pytest` against the file directly — no Open WebUI runtime required.

---

## Task 1: Project skeleton + test infrastructure

**Files:**
- Create: `vast-templates/qwen3-vl-8b/functions/requirements-dev.txt`
- Create: `vast-templates/qwen3-vl-8b/functions/pytest.ini`
- Create: `vast-templates/qwen3-vl-8b/functions/conftest.py`
- Create: `vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py` (stub)
- Create: `vast-templates/qwen3-vl-8b/functions/test_qwenvl_image_compress.py` (smoke test only)

- [ ] **Step 1: Create `requirements-dev.txt`**

```
pytest>=8.0
pytest-asyncio>=0.23
respx>=0.21
httpx>=0.27
aiosqlite>=0.20
pydantic>=2.5
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = .
python_files = test_*.py
addopts = -v --tb=short
```

- [ ] **Step 3: Create `conftest.py`**

```python
"""Shared pytest fixtures for filter tests."""
import base64
import hashlib
import pytest
from pathlib import Path


@pytest.fixture
def cache_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_captions.db")


def _make_data_url(payload: bytes) -> str:
    b64 = base64.b64encode(payload).decode()
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def make_image():
    """Factory: synthetic image data URL + its sha256 hex hash."""
    def _factory(payload: bytes = b"\x89PNG\r\n\x1a\nfakedata"):
        url = _make_data_url(payload)
        h = hashlib.sha256(payload).hexdigest()
        return url, h
    return _factory


@pytest.fixture
def text_msg():
    return lambda text, role="user": {"role": role, "content": text}


@pytest.fixture
def multimodal_msg():
    """Builder: build a user message with text + N image data URLs."""
    def _factory(text: str, image_urls: list[str], role: str = "user"):
        content = [{"type": "text", "text": text}]
        content.extend({"type": "image_url", "image_url": {"url": u}} for u in image_urls)
        return {"role": role, "content": content}
    return _factory
```

- [ ] **Step 4: Create empty filter stub `qwenvl_image_compress.py`**

```python
"""Qwen3-VL Image-Aware Context Compressor — Open WebUI Filter Function.

Paste this entire file into Admin Panel → Functions → New Function.
Source of truth lives in the repo at vast-templates/qwen3-vl-8b/functions/.
"""

VERSION = "0.1.0-dev"
```

- [ ] **Step 5: Create smoke test `test_qwenvl_image_compress.py`**

```python
import qwenvl_image_compress as mod


def test_module_imports():
    assert mod.VERSION.startswith("0.1.0")
```

- [ ] **Step 6: Install dev deps + run smoke test**

```bash
cd vast-templates/qwen3-vl-8b/functions
python -m venv .venv && . .venv/Scripts/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest test_qwenvl_image_compress.py -v
```
Expected: `test_module_imports PASSED`.

- [ ] **Step 7: Commit**

```bash
git add vast-templates/qwen3-vl-8b/functions/
git commit -m "feat(qwen3-vl-filter): scaffold filter project + pytest harness"
```

---

## Task 2: CaptionCache — schema, init, get/put roundtrip

**Files:**
- Modify: `vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py` (add `CaptionCache`)
- Modify: `vast-templates/qwen3-vl-8b/functions/test_qwenvl_image_compress.py` (add `TestCaptionCache`)

- [ ] **Step 1: Write failing tests**

Append to `test_qwenvl_image_compress.py`:

```python
import pytest
from qwenvl_image_compress import CaptionCache


class TestCaptionCacheInit:
    async def test_init_creates_db_file(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        from pathlib import Path
        assert Path(cache_db_path).exists()

    async def test_init_idempotent(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        await c.init()  # second call must not raise

    async def test_get_returns_none_for_miss(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        assert await c.get("deadbeef" * 8) is None

    async def test_put_then_get_roundtrip(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        await c.put("a" * 64, "a cat", "qwen3-vl-8b", bytes_size=12345, user_id="u1")
        assert await c.get("a" * 64) == "a cat"
```

- [ ] **Step 2: Run, confirm failure**

```bash
pytest test_qwenvl_image_compress.py::TestCaptionCacheInit -v
```
Expected: ImportError or AttributeError on `CaptionCache`.

- [ ] **Step 3: Implement `CaptionCache` in `qwenvl_image_compress.py`**

Append:

```python
import asyncio
import os
import time
from typing import Optional
import aiosqlite


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

    async def put(self, h: str, caption: str, model: str,
                  bytes_size: Optional[int] = None,
                  user_id: Optional[str] = None) -> None:
        now_ms = int(time.time() * 1000)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (h, caption, model, now_ms, bytes_size, user_id),
            )
            await db.commit()
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py::TestCaptionCacheInit -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py \
        vast-templates/qwen3-vl-8b/functions/test_qwenvl_image_compress.py
git commit -m "feat(qwen3-vl-filter): CaptionCache init + get/put"
```

---

## Task 3: CaptionCache — `get_many`, `put_many`, race tolerance

- [ ] **Step 1: Write failing tests** — append to `test_qwenvl_image_compress.py`:

```python
class TestCaptionCacheBatch:
    async def test_get_many_empty_returns_empty(self, cache_db_path):
        c = CaptionCache(cache_db_path); await c.init()
        assert await c.get_many([]) == {}

    async def test_get_many_returns_only_present(self, cache_db_path):
        c = CaptionCache(cache_db_path); await c.init()
        await c.put("h1" + "0" * 62, "cap1", "qwen3-vl-8b")
        result = await c.get_many(["h1" + "0" * 62, "missing" + "0" * 57])
        assert result == {"h1" + "0" * 62: "cap1"}

    async def test_put_many_inserts_all(self, cache_db_path):
        c = CaptionCache(cache_db_path); await c.init()
        items = [
            ("a" * 64, "ca", "qwen3-vl-8b", 100, None),
            ("b" * 64, "cb", "qwen3-vl-8b", 200, "u1"),
        ]
        await c.put_many(items)
        assert await c.get("a" * 64) == "ca"
        assert await c.get("b" * 64) == "cb"

    async def test_put_many_race_uses_first_writer(self, cache_db_path):
        """Two concurrent put_many for same hash: second is silently dropped."""
        c = CaptionCache(cache_db_path); await c.init()
        h = "c" * 64
        await c.put_many([(h, "first", "qwen3-vl-8b", None, None)])
        await c.put_many([(h, "second", "qwen3-vl-8b", None, None)])
        assert await c.get(h) == "first"  # INSERT OR IGNORE preserves first
```

- [ ] **Step 2: Run, confirm failure** (`AttributeError: get_many`).

- [ ] **Step 3: Implement** — add to `CaptionCache`:

```python
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
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py::TestCaptionCacheBatch -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): CaptionCache batch ops + race tolerance"
```

---

## Task 4: Image scanning helpers

Pure functions: extract image parts, find latest image-bearing user turn, get text of a message.

- [ ] **Step 1: Write failing tests**

```python
from qwenvl_image_compress import (
    iter_image_parts, find_latest_image_turn, has_images, text_of,
)


class TestImageScan:
    def test_text_only_message_has_no_images(self, text_msg):
        assert not has_images(text_msg("hello"))

    def test_multimodal_message_with_images(self, multimodal_msg, make_image):
        url, _ = make_image()
        m = multimodal_msg("what?", [url])
        assert has_images(m)

    def test_multimodal_message_text_only_part(self, multimodal_msg):
        m = multimodal_msg("hello", [])
        assert not has_images(m)

    def test_iter_image_parts_yields_indices(self, multimodal_msg, make_image):
        u1, _ = make_image(b"a")
        u2, _ = make_image(b"b")
        msgs = [
            {"role": "system", "content": "you are helpful"},
            multimodal_msg("look", [u1, u2]),
        ]
        parts = list(iter_image_parts(msgs))
        # Each: (msg_idx, content_idx, url)
        assert len(parts) == 2
        assert parts[0] == (1, 1, u1)
        assert parts[1] == (1, 2, u2)

    def test_find_latest_image_turn_none_when_no_images(self, text_msg):
        msgs = [text_msg("hi"), text_msg("yo", role="assistant"), text_msg("ok")]
        assert find_latest_image_turn(msgs) is None

    def test_find_latest_image_turn_returns_index(self, text_msg, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [
            multimodal_msg("first", [u]),     # idx 0
            text_msg("answer", role="assistant"),
            text_msg("follow"),                # idx 2 — no images
        ]
        assert find_latest_image_turn(msgs) == 0

    def test_find_latest_picks_most_recent(self, multimodal_msg, text_msg, make_image):
        u1, _ = make_image(b"a")
        u2, _ = make_image(b"b")
        msgs = [
            multimodal_msg("a", [u1]),                 # idx 0
            text_msg("ok", role="assistant"),
            multimodal_msg("b", [u2]),                 # idx 2
        ]
        assert find_latest_image_turn(msgs) == 2

    def test_text_of_string_content(self, text_msg):
        assert text_of(text_msg("hello world")) == "hello world"

    def test_text_of_multimodal(self, multimodal_msg, make_image):
        u, _ = make_image()
        m = multimodal_msg("describe this", [u])
        assert text_of(m) == "describe this"

    def test_text_of_no_text_part(self, make_image):
        u, _ = make_image()
        m = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}]}
        assert text_of(m) == ""
```

- [ ] **Step 2: Run, confirm failure** (ImportError).

- [ ] **Step 3: Implement** — append to `qwenvl_image_compress.py`:

```python
from typing import Iterator


def has_images(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(p.get("type") == "image_url" for p in content)


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
    latest: Optional[int] = None
    for i, msg in enumerate(msgs):
        if msg.get("role") == "user" and has_images(msg):
            latest = i
    return latest


def text_of(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
    return ""
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py::TestImageScan -v
```
Expected: 10 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): image scanning + text extraction helpers"
```

---

## Task 5: Image hashing (data URL + relative URL)

- [ ] **Step 1: Write failing tests**

```python
import hashlib
import respx
import httpx
from qwenvl_image_compress import hash_image_url


class TestHashImage:
    async def test_data_url_hashes_raw_bytes(self):
        payload = b"\x89PNG\r\nfake"
        b64 = base64.b64encode(payload).decode()
        url = f"data:image/png;base64,{b64}"
        h, raw = await hash_image_url(url, fetch_base="http://localhost:0")
        assert h == hashlib.sha256(payload).hexdigest()
        assert raw == payload

    async def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError):
            await hash_image_url("file:///etc/passwd", fetch_base="http://localhost:0")

    @respx.mock
    async def test_relative_url_fetches_from_base(self):
        payload = b"\xff\xd8\xff\xe0fake_jpeg"
        respx.get("http://127.0.0.1:3000/api/v1/files/abc/content").respond(
            200, content=payload
        )
        h, raw = await hash_image_url(
            "/api/v1/files/abc/content",
            fetch_base="http://127.0.0.1:3000",
        )
        assert h == hashlib.sha256(payload).hexdigest()
        assert raw == payload

    @respx.mock
    async def test_absolute_http_url_fetched_directly(self):
        payload = b"data"
        respx.get("https://cdn.example.com/img.png").respond(200, content=payload)
        h, raw = await hash_image_url(
            "https://cdn.example.com/img.png",
            fetch_base="http://ignored",
        )
        assert h == hashlib.sha256(payload).hexdigest()

    @respx.mock
    async def test_http_404_raises(self):
        respx.get("http://127.0.0.1:3000/missing").respond(404)
        with pytest.raises(httpx.HTTPStatusError):
            await hash_image_url("/missing", fetch_base="http://127.0.0.1:3000")
```

Add `import base64` near the top of the test file if not already imported.

- [ ] **Step 2: Run, confirm failure** (ImportError on `hash_image_url`).

- [ ] **Step 3: Implement** — append to `qwenvl_image_compress.py`:

```python
import base64
import hashlib
import httpx


async def hash_image_url(url: str, fetch_base: str,
                         fetch_timeout_s: int = 10) -> tuple[str, bytes]:
    """Return (sha256_hex, raw_bytes) for an image_url part.

    Supports data: URLs, absolute http(s) URLs, and relative paths
    (resolved against fetch_base, e.g. http://127.0.0.1:3000)."""
    if url.startswith("data:"):
        if "," not in url:
            raise ValueError("malformed data URL")
        raw = base64.b64decode(url.split(",", 1)[1])
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
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py::TestHashImage -v
```
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): hash_image_url for data + http URLs"
```

---

## Task 6: Message rewriting (`rewrite_messages`)

- [ ] **Step 1: Write failing tests**

```python
from qwenvl_image_compress import rewrite_messages


class TestRewriteMessages:
    def _captions_for(self, urls):
        return {url: f"cap[{i}]" for i, url in enumerate(urls)}

    def test_keep_idx_preserves_target_turn(self, multimodal_msg, text_msg, make_image):
        u, _ = make_image()
        msgs = [
            multimodal_msg("a", [u]),       # 0
            text_msg("b", role="assistant"),
            multimodal_msg("c", [u]),       # 2 — keep
        ]
        out = rewrite_messages(msgs, keep_idx=2, captions_by_url=self._captions_for([u]))
        # turn 2 unchanged
        assert out[2] == msgs[2]
        # turn 0 image stripped — check no image_url part remains
        assert all(p.get("type") != "image_url" for p in out[0]["content"])
        # caption present in text
        text = next(p["text"] for p in out[0]["content"] if p["type"] == "text")
        assert "cap[0]" in text

    def test_keep_idx_none_strips_all(self, multimodal_msg, make_image):
        u1, _ = make_image(b"a")
        u2, _ = make_image(b"b")
        msgs = [multimodal_msg("look", [u1, u2])]
        out = rewrite_messages(msgs, keep_idx=None, captions_by_url={u1: "A", u2: "B"})
        text_parts = [p["text"] for p in out[0]["content"] if p["type"] == "text"]
        assert all(p.get("type") != "image_url" for p in out[0]["content"])
        assert any("A" in t for t in text_parts)
        assert any("B" in t for t in text_parts)

    def test_text_only_message_passthrough(self, text_msg):
        msgs = [text_msg("hello")]
        out = rewrite_messages(msgs, keep_idx=None, captions_by_url={})
        assert out == msgs

    def test_role_unchanged(self, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [multimodal_msg("x", [u], role="user")]
        out = rewrite_messages(msgs, keep_idx=None, captions_by_url={u: "C"})
        assert out[0]["role"] == "user"

    def test_missing_caption_falls_back_to_placeholder(self, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [multimodal_msg("x", [u])]
        out = rewrite_messages(msgs, keep_idx=None, captions_by_url={})  # no caption
        text_parts = [p["text"] for p in out[0]["content"] if p["type"] == "text"]
        assert any("(no caption)" in t for t in text_parts)
        assert all(p.get("type") != "image_url" for p in out[0]["content"])

    def test_keep_idx_preserves_pixels_at_target(self, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [multimodal_msg("x", [u])]
        out = rewrite_messages(msgs, keep_idx=0, captions_by_url={u: "C"})
        # image_url part must still exist at turn 0
        assert any(p.get("type") == "image_url" for p in out[0]["content"])
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — append:

```python
import copy


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
            new_parts.append({"type": "text", "text": extra_text})
        msg["content"] = new_parts
    return out
```

- [ ] **Step 4: Run, confirm pass**.

```bash
pytest test_qwenvl_image_compress.py::TestRewriteMessages -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): rewrite_messages strips images + injects captions"
```

---

## Task 7: Caption call to vLLM (`caption_one`)

- [ ] **Step 1: Write failing tests**

```python
from qwenvl_image_compress import caption_one, CAPTION_SYSTEM_PROMPT


class TestCaptionOne:
    @respx.mock
    async def test_caption_returns_content(self, make_image):
        url, _ = make_image()
        respx.post("http://vllm/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"content": "  a cat sleeping  "}}]},
        )
        cap = await caption_one(
            data_url=url,
            base_url="http://vllm/v1",
            api_key="sk-x",
            model="qwen3-vl-8b",
            max_tokens=80,
            timeout_s=10,
        )
        assert cap == "a cat sleeping"

    @respx.mock
    async def test_caption_request_shape(self, make_image):
        url, _ = make_image()
        route = respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "x"}}]}
        )
        await caption_one(url, "http://vllm/v1", "sk-x", "qwen3-vl-8b", 80, 10)

        call = route.calls[0]
        body = call.request.read()
        import json as J
        payload = J.loads(body)
        assert payload["model"] == "qwen3-vl-8b"
        assert payload["max_tokens"] == 80
        assert payload["temperature"] == 0.2
        assert payload["stream"] is False
        # Two messages: system + user (text + image)
        assert payload["messages"][0]["role"] == "system"
        assert CAPTION_SYSTEM_PROMPT.strip() in payload["messages"][0]["content"]
        user_content = payload["messages"][1]["content"]
        assert any(p.get("type") == "image_url" for p in user_content)

    @respx.mock
    async def test_caption_empty_response_returns_empty(self, make_image):
        url, _ = make_image()
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": ""}}]}
        )
        cap = await caption_one(url, "http://vllm/v1", "sk-x", "qwen3-vl-8b", 80, 10)
        assert cap == ""

    @respx.mock
    async def test_caption_http_error_raises(self, make_image):
        url, _ = make_image()
        respx.post("http://vllm/v1/chat/completions").respond(500)
        with pytest.raises(httpx.HTTPStatusError):
            await caption_one(url, "http://vllm/v1", "sk-x", "qwen3-vl-8b", 80, 10)
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — append:

```python
CAPTION_SYSTEM_PROMPT = """\
Bạn là image captioner. Mô tả ảnh trong 1-2 câu khách quan, không quá 60 từ.
Cần nêu:
  - Chủ thể chính (người/vật/cảnh).
  - Văn bản nhìn thấy trong ảnh, copy nguyên văn nếu ngắn.
  - Bố cục/màu sắc nổi bật nếu liên quan.
KHÔNG suy diễn cảm xúc, KHÔNG khen chê, KHÔNG bịa chi tiết.
Trả về DUY NHẤT phần caption, không prefix \"Caption:\" hay markdown."""

CAPTION_USER_TEXT = "Mô tả ảnh này."


async def caption_one(data_url: str,
                      base_url: str,
                      api_key: str,
                      model: str,
                      max_tokens: int,
                      timeout_s: int) -> str:
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
```

- [ ] **Step 4: Run, confirm pass**.

```bash
pytest test_qwenvl_image_compress.py::TestCaptionOne -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): caption_one HTTP call to vLLM"
```

---

## Task 8: Router call to vLLM (`route`)

- [ ] **Step 1: Write failing tests**

```python
from qwenvl_image_compress import route, ROUTER_SYSTEM_PROMPT


class TestRoute:
    @respx.mock
    async def test_route_yes(self):
        respx.post("http://vllm/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {
                "content": '{"need_images": true, "reason": "user nói \\"ảnh thứ 2\\""}'
            }}]},
        )
        decision, reason = await route(
            user_text="ảnh thứ 2 màu gì?",
            captions=["a cat", "a dog"],
            base_url="http://vllm/v1",
            api_key="sk-x",
            model="qwen3-vl-8b",
            max_tokens=60,
            timeout_s=5,
            failopen_keep=True,
        )
        assert decision is True
        assert "ảnh" in reason

    @respx.mock
    async def test_route_no(self):
        respx.post("http://vllm/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {
                "content": '{"need_images": false, "reason": "đổi chủ đề"}'
            }}]},
        )
        decision, reason = await route(
            "explain decorators", ["a cat"],
            "http://vllm/v1", "sk-x", "qwen3-vl-8b", 60, 5, failopen_keep=True,
        )
        assert decision is False
        assert reason == "đổi chủ đề"

    @respx.mock
    async def test_route_invalid_json_failopen_keep(self):
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "not json at all"}}]},
        )
        decision, reason = await route(
            "x", ["c"], "http://vllm/v1", "sk-x", "qwen3-vl-8b",
            60, 5, failopen_keep=True,
        )
        assert decision is True
        assert "fail" in reason.lower() or "invalid" in reason.lower()

    @respx.mock
    async def test_route_invalid_json_failopen_drop(self):
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "not json"}}]},
        )
        decision, reason = await route(
            "x", ["c"], "http://vllm/v1", "sk-x", "qwen3-vl-8b",
            60, 5, failopen_keep=False,
        )
        assert decision is False

    @respx.mock
    async def test_route_http_error_failopen_keep(self):
        respx.post("http://vllm/v1/chat/completions").respond(500)
        decision, reason = await route(
            "x", ["c"], "http://vllm/v1", "sk-x", "qwen3-vl-8b",
            60, 5, failopen_keep=True,
        )
        assert decision is True

    @respx.mock
    async def test_route_request_shape(self):
        route_mock = respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {
                "content": '{"need_images": true, "reason": "x"}'
            }}]},
        )
        await route(
            "u msg", ["c1", "c2"], "http://vllm/v1", "sk-x", "qwen3-vl-8b",
            60, 5, failopen_keep=True,
        )
        import json as J
        payload = J.loads(route_mock.calls[0].request.read())
        assert payload["temperature"] == 0.0
        assert payload["response_format"] == {"type": "json_object"}
        assert ROUTER_SYSTEM_PROMPT.strip() in payload["messages"][0]["content"]
        # captions block + user message must be in user content
        user_content = payload["messages"][1]["content"]
        assert "1. c1" in user_content
        assert "2. c2" in user_content
        assert "u msg" in user_content
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — append:

```python
import json as _json
import logging

log = logging.getLogger("qwenvl_image_compress")


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


async def route(user_text: str,
                captions: list[str],
                base_url: str,
                api_key: str,
                model: str,
                max_tokens: int,
                timeout_s: int,
                failopen_keep: bool) -> tuple[bool, str]:
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
```

- [ ] **Step 4: Run, confirm pass**.

```bash
pytest test_qwenvl_image_compress.py::TestRoute -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): route classifier with JSON mode + failopen"
```

---

## Task 9: `ensure_captions` — cache lookup + parallel caption

- [ ] **Step 1: Write failing tests**

```python
from qwenvl_image_compress import ensure_captions


class TestEnsureCaptions:
    @respx.mock
    async def test_all_cache_hits_no_http(self, cache_db_path, make_image):
        c = CaptionCache(cache_db_path); await c.init()
        u1, h1 = make_image(b"a"); u2, h2 = make_image(b"b")
        await c.put(h1, "cap A", "qwen3-vl-8b")
        await c.put(h2, "cap B", "qwen3-vl-8b")

        # respx with no routes; any HTTP call would raise.
        result = await ensure_captions(
            scanned=[(0, 1, u1, h1, b"a"), (0, 2, u2, h2, b"b")],
            cache=c,
            base_url="http://vllm/v1",
            api_key="sk",
            model="qwen3-vl-8b",
            max_tokens=80,
            timeout_s=10,
            user_id=None,
        )
        assert result == {u1: "cap A", u2: "cap B"}

    @respx.mock
    async def test_misses_trigger_caption_calls(self, cache_db_path, make_image):
        c = CaptionCache(cache_db_path); await c.init()
        u, h = make_image(b"new")
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "fresh caption"}}]},
        )
        result = await ensure_captions(
            scanned=[(0, 1, u, h, b"new")],
            cache=c,
            base_url="http://vllm/v1", api_key="sk", model="qwen3-vl-8b",
            max_tokens=80, timeout_s=10, user_id="u1",
        )
        assert result == {u: "fresh caption"}
        # cache populated
        assert await c.get(h) == "fresh caption"

    @respx.mock
    async def test_caption_failure_skips_url(self, cache_db_path, make_image):
        c = CaptionCache(cache_db_path); await c.init()
        u_ok, h_ok = make_image(b"ok"); u_bad, h_bad = make_image(b"bad")
        await c.put(h_ok, "ok caption", "qwen3-vl-8b")
        respx.post("http://vllm/v1/chat/completions").respond(503)
        result = await ensure_captions(
            scanned=[(0, 1, u_ok, h_ok, b"ok"), (0, 2, u_bad, h_bad, b"bad")],
            cache=c,
            base_url="http://vllm/v1", api_key="sk", model="qwen3-vl-8b",
            max_tokens=80, timeout_s=10, user_id=None,
        )
        assert result.get(u_ok) == "ok caption"
        assert u_bad not in result   # failure → url omitted
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — append:

```python
import asyncio


async def ensure_captions(scanned: list[tuple[int, int, str, str, bytes]],
                          cache: CaptionCache,
                          base_url: str,
                          api_key: str,
                          model: str,
                          max_tokens: int,
                          timeout_s: int,
                          user_id: Optional[str]) -> dict[str, str]:
    """scanned: (msg_idx, content_idx, url, hash, raw_bytes).
    Returns {url: caption} for every successfully captioned image."""
    if not scanned:
        return {}
    hashes = [h for _, _, _, h, _ in scanned]
    hits = await cache.get_many(hashes)
    out: dict[str, str] = {}
    misses: list[tuple[str, str, bytes]] = []
    for _, _, url, h, raw in scanned:
        if h in hits:
            out[url] = hits[h]
        else:
            misses.append((url, h, raw))

    if not misses:
        return out

    async def _one(url: str) -> tuple[str, Optional[str]]:
        try:
            cap = await caption_one(url, base_url, api_key, model, max_tokens, timeout_s)
            return url, cap or None
        except Exception as e:
            log.warning("caption failed for url=%s err=%s", url[:60], e)
            return url, None

    results = await asyncio.gather(*(_one(url) for url, _, _ in misses))
    new_rows: list[tuple[str, str, str, Optional[int], Optional[str]]] = []
    for (url, h, raw), (url2, cap) in zip(misses, results):
        if cap:
            out[url] = cap
            new_rows.append((h, cap, model, len(raw), user_id))
    if new_rows:
        await cache.put_many(new_rows)
    return out
```

- [ ] **Step 4: Run, confirm pass**.

```bash
pytest test_qwenvl_image_compress.py::TestEnsureCaptions -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): ensure_captions parallel cache+caption"
```

---

## Task 10: Filter class skeleton + Valves + UserValves

- [ ] **Step 1: Write failing tests**

```python
from qwenvl_image_compress import Filter


class TestFilterValves:
    def test_valves_defaults(self):
        f = Filter()
        assert f.valves.vllm_base_url == "http://127.0.0.1:8000/v1"
        assert f.valves.caption_model == "qwen3-vl-8b"
        assert f.valves.router_failopen_keep is True
        assert f.valves.priority == 5
        assert f.toggle is True

    def test_user_valves_defaults(self):
        f = Filter()
        uv = f.UserValves()
        assert uv.enabled is True
        assert uv.force_keep_all_images is False
        assert uv.show_thinking_log is True
        assert uv.show_live_status is True

    def test_valves_overridable(self):
        f = Filter()
        f.valves = Filter.Valves(caption_max_tokens=120, priority=1)
        assert f.valves.caption_max_tokens == 120
        assert f.valves.priority == 1
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — append:

```python
from pydantic import BaseModel, Field


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
```

- [ ] **Step 4: Run, confirm pass**.

```bash
pytest test_qwenvl_image_compress.py::TestFilterValves -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): Filter class skeleton + Valves + UserValves"
```

---

## Task 11: `Filter.inlet` — fresh upload path (Scenarios 1 + 4)

- [ ] **Step 1: Write failing tests**

```python
class TestInletFreshUpload:
    @respx.mock
    async def test_first_upload_three_images_no_strip(
        self, cache_db_path, multimodal_msg, make_image
    ):
        u1, _ = make_image(b"a"); u2, _ = make_image(b"b"); u3, _ = make_image(b"c")
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "cap"}}]},
        )
        f = Filter()
        f.valves = Filter.Valves(
            vllm_base_url="http://vllm/v1",
            cache_db_path=cache_db_path,
        )
        body = {"messages": [multimodal_msg("look", [u1, u2, u3])]}
        out = await f.inlet(body=body, __user__={"id": "u1"}, __metadata__={"chat_id": "c1"})
        # all 3 images preserved at last (and only) turn
        n_images = sum(
            1 for p in out["messages"][0]["content"] if p.get("type") == "image_url"
        )
        assert n_images == 3

    @respx.mock
    async def test_new_upload_strips_prior_turn(
        self, cache_db_path, text_msg, multimodal_msg, make_image
    ):
        u_old, _ = make_image(b"old"); u_new, _ = make_image(b"new")
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "x cap"}}]},
        )
        f = Filter()
        f.valves = Filter.Valves(
            vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path,
        )
        body = {"messages": [
            multimodal_msg("first", [u_old]),
            text_msg("ok", role="assistant"),
            multimodal_msg("now this", [u_new]),
        ]}
        out = await f.inlet(body=body, __user__={"id": "u1"}, __metadata__={"chat_id": "c1"})
        # first user turn: image stripped
        first_imgs = [p for p in out["messages"][0]["content"] if p.get("type") == "image_url"]
        assert first_imgs == []
        # last user turn: image preserved
        last_imgs = [p for p in out["messages"][2]["content"] if p.get("type") == "image_url"]
        assert len(last_imgs) == 1
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement `_inlet_impl` and `inlet`** — append to `Filter`:

```python
    async def inlet(self, body: dict, __user__: Optional[dict] = None,
                    __metadata__: Optional[dict] = None,
                    __event_emitter__=None) -> dict:
        try:
            return await self._inlet_impl(body, __user__, __metadata__, __event_emitter__)
        except Exception as e:
            log.exception("ImageCompressor passthrough due to error: %s", e)
            return body

    async def _inlet_impl(self, body, __user__, __metadata__, __event_emitter__) -> dict:
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

        # Step A: scan
        url_list = list(iter_image_parts(msgs))
        if not url_list:
            return body

        cache = await self._ensure_cache()

        # Step B: hash + ensure captions
        scanned: list[tuple[int, int, str, str, bytes]] = []
        for msg_idx, c_idx, url in url_list:
            try:
                h, raw = await hash_image_url(
                    url, self.valves.webui_internal_base,
                )
                scanned.append((msg_idx, c_idx, url, h, raw))
            except Exception as e:
                log.warning("hash skipped url=%s err=%s", url[:60], e)

        captions_by_url = await ensure_captions(
            scanned=scanned,
            cache=cache,
            base_url=self.valves.vllm_base_url,
            api_key=self.valves.vllm_api_key,
            model=self.valves.caption_model,
            max_tokens=self.valves.caption_max_tokens,
            timeout_s=self.valves.caption_timeout_s,
            user_id=(__user__ or {}).get("id"),
        )

        # Step C: classify
        if user_valves.force_keep_all_images:
            return body

        if has_images(last):
            keep_idx = len(msgs) - 1
        else:
            latest_idx = find_latest_image_turn(msgs)
            if latest_idx is None:
                return body
            captions_for_router = [
                captions_by_url.get(url, "(no caption)")
                for (mi, _, url, _, _) in scanned if mi == latest_idx
            ]
            decision, _reason = await route(
                user_text=text_of(last),
                captions=captions_for_router,
                base_url=self.valves.vllm_base_url,
                api_key=self.valves.vllm_api_key,
                model=self.valves.router_model,
                max_tokens=self.valves.router_max_tokens,
                timeout_s=self.valves.router_timeout_s,
                failopen_keep=self.valves.router_failopen_keep,
            )
            keep_idx = latest_idx if decision else None

        # Step D: rewrite
        body["messages"] = rewrite_messages(msgs, keep_idx, captions_by_url)
        return body
```

- [ ] **Step 4: Run, confirm pass**.

```bash
pytest test_qwenvl_image_compress.py::TestInletFreshUpload -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): inlet fresh-upload path"
```

---

## Task 12: `Filter.inlet` — text-only follow-up with router (Scenarios 2 + 3)

- [ ] **Step 1: Write failing tests**

```python
class TestInletTextFollowup:
    @respx.mock
    async def test_text_followup_router_yes_keeps_images(
        self, cache_db_path, text_msg, multimodal_msg, make_image
    ):
        u, h = make_image(b"img"); urls = [u]
        # Pre-seed cache so caption call is not needed
        c = CaptionCache(cache_db_path); await c.init()
        await c.put(h, "a screenshot", "qwen3-vl-8b")
        # Only one route registered: the router call
        respx.post("http://vllm/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {
                "content": '{"need_images": true, "reason": "user references image"}'
            }}]},
        )
        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [
            multimodal_msg("first", urls),                      # idx 0 — image
            text_msg("ok", role="assistant"),
            text_msg("ảnh này màu gì?"),                        # idx 2 — text only
        ]}
        out = await f.inlet(body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"})
        # Image preserved at idx 0
        first = out["messages"][0]
        assert any(p.get("type") == "image_url" for p in first["content"])

    @respx.mock
    async def test_text_followup_router_no_strips_all(
        self, cache_db_path, text_msg, multimodal_msg, make_image
    ):
        u, h = make_image(b"img")
        c = CaptionCache(cache_db_path); await c.init()
        await c.put(h, "a screenshot", "qwen3-vl-8b")
        respx.post("http://vllm/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {
                "content": '{"need_images": false, "reason": "topic shift"}'
            }}]},
        )
        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [
            multimodal_msg("first", [u]),
            text_msg("ok", role="assistant"),
            text_msg("explain Python decorators"),
        ]}
        out = await f.inlet(body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"})
        first = out["messages"][0]
        assert all(p.get("type") != "image_url" for p in first["content"])
        # Caption substituted into text
        text = next(p["text"] for p in first["content"] if p["type"] == "text")
        assert "a screenshot" in text
```

- [ ] **Step 2: Run, confirm failure** (router branch not reached because Task 11 already added this code path → tests should actually pass without further code if Task 11 was complete. If a test fails, it indicates a real bug in Task 11 — fix there).

- [ ] **Step 3: If tests pass already, skip implementation step** and move to Step 4. If they fail, fix `_inlet_impl` based on the failure.

- [ ] **Step 4: Run all inlet tests**

```bash
pytest test_qwenvl_image_compress.py::TestInletFreshUpload \
       test_qwenvl_image_compress.py::TestInletTextFollowup -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "test(qwen3-vl-filter): cover router yes/no branches in inlet"
```

---

## Task 13: UserValves enforcement

- [ ] **Step 1: Write failing tests**

```python
class TestInletUserValves:
    async def test_disabled_user_returns_body_unchanged(
        self, cache_db_path, multimodal_msg, make_image
    ):
        u, _ = make_image()
        f = Filter()
        f.valves = Filter.Valves(cache_db_path=cache_db_path)
        body = {"messages": [
            multimodal_msg("a", [u]),
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]}
        before = copy.deepcopy(body)
        uv = Filter.UserValves(enabled=False)
        out = await f.inlet(
            body=body, __user__={"id": "u", "valves": uv}, __metadata__={},
        )
        assert out == before  # no mutation

    @respx.mock
    async def test_force_keep_all_images_skips_router(
        self, cache_db_path, multimodal_msg, text_msg, make_image
    ):
        u, h = make_image(b"x")
        c = CaptionCache(cache_db_path); await c.init()
        await c.put(h, "cap", "qwen3-vl-8b")
        # No respx routes — any HTTP call would explode.
        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [
            multimodal_msg("a", [u]),
            text_msg("y", role="assistant"),
            text_msg("explain decorators"),
        ]}
        uv = Filter.UserValves(force_keep_all_images=True)
        out = await f.inlet(
            body=body, __user__={"id": "u", "valves": uv}, __metadata__={},
        )
        # Image still present at turn 0 (no strip)
        first = out["messages"][0]
        assert any(p.get("type") == "image_url" for p in first["content"])
```

Add `import copy` at the top of the test file if not already.

- [ ] **Step 2: Run, confirm pass** — these should already pass given Task 11's implementation. If the second test fails because hashing or `ensure_captions` ran when `force_keep_all_images=True`, that's still fine since hashing is offline (data URL) and the cache is pre-seeded → no HTTP call. Tests should be GREEN.

- [ ] **Step 3: Run full suite to confirm no regression**

```bash
pytest test_qwenvl_image_compress.py -v
```
Expected: all PASSED.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "test(qwen3-vl-filter): UserValves enabled + force_keep_all_images"
```

---

## Task 14: Top-level error guard

- [ ] **Step 1: Write failing tests**

```python
class TestInletErrorGuard:
    async def test_uncaught_exception_returns_body_unchanged(self, monkeypatch):
        f = Filter()

        async def boom(*a, **kw):
            raise RuntimeError("simulated")

        monkeypatch.setattr(f, "_inlet_impl", boom)
        body = {"messages": [{"role": "user", "content": "hi"}]}
        before = copy.deepcopy(body)
        out = await f.inlet(body=body, __user__={"id": "u"}, __metadata__={})
        assert out == before

    @respx.mock
    async def test_vllm_unreachable_passes_through(
        self, cache_db_path, multimodal_msg, make_image
    ):
        u, _ = make_image()
        # No respx route → ConnectError on caption call.
        f = Filter()
        f.valves = Filter.Valves(
            vllm_base_url="http://nonexistent.invalid/v1",
            cache_db_path=cache_db_path,
            caption_timeout_s=1,
        )
        body = {"messages": [multimodal_msg("a", [u])]}
        out = await f.inlet(
            body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"},
        )
        # Caption failed → image not stripped (no caption to substitute,
        # and last turn anyway). Body should still have the image.
        assert any(
            p.get("type") == "image_url" for p in out["messages"][0]["content"]
        )
```

- [ ] **Step 2: Run, confirm pass** — Task 11's `inlet` already wraps `_inlet_impl` in try/except. Tests should be GREEN.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test(qwen3-vl-filter): top-level guard + vllm-unreachable passthrough"
```

---

## Task 15: Live status events (`type: "status"`)

- [ ] **Step 1: Write failing tests**

```python
class TestStatusEvents:
    @respx.mock
    async def test_emits_caption_start_and_done(
        self, cache_db_path, multimodal_msg, make_image
    ):
        u, _ = make_image()
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "c"}}]},
        )
        events: list[dict] = []
        async def emit(ev): events.append(ev)

        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [multimodal_msg("a", [u])]}
        await f.inlet(
            body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"},
            __event_emitter__=emit,
        )
        descriptions = [e["data"]["description"] for e in events if e.get("type") == "status"]
        assert any("Captioning" in d for d in descriptions)
        assert any("done" in d.lower() for d in descriptions)

    @respx.mock
    async def test_emits_router_decision(
        self, cache_db_path, multimodal_msg, text_msg, make_image
    ):
        u, h = make_image(); c = CaptionCache(cache_db_path)
        await c.init(); await c.put(h, "cap", "qwen3-vl-8b")
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {
                "content": '{"need_images": false, "reason": "topic"}'
            }}]},
        )
        events = []
        async def emit(ev): events.append(ev)
        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [
            multimodal_msg("a", [u]),
            text_msg("y", role="assistant"),
            text_msg("decorators?"),
        ]}
        await f.inlet(
            body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"},
            __event_emitter__=emit,
        )
        descriptions = [e["data"]["description"] for e in events]
        assert any("Routing" in d for d in descriptions)
        assert any("drop" in d for d in descriptions)

    async def test_show_live_status_false_emits_nothing(
        self, cache_db_path, multimodal_msg, make_image
    ):
        u, _ = make_image()
        events = []
        async def emit(ev): events.append(ev)
        f = Filter()
        f.valves = Filter.Valves(cache_db_path=cache_db_path)
        body = {"messages": [multimodal_msg("a", [u])]}
        # No respx mock → caption call would fail; but show_live_status disabled so
        # status events should be absent regardless of HTTP outcome.
        uv = Filter.UserValves(show_live_status=False, show_thinking_log=False)
        await f.inlet(
            body=body, __user__={"id": "u", "valves": uv},
            __metadata__={"chat_id": "c"},
            __event_emitter__=emit,
        )
        assert events == []
```

- [ ] **Step 2: Run, confirm failure** (no status currently emitted).

- [ ] **Step 3: Add status emission helper to Filter** — append:

```python
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
```

Now thread emit + UserValves into `_inlet_impl`. Replace the body of `_inlet_impl` from Task 11 (full replacement, given branching logic is now richer):

```python
    async def _inlet_impl(self, body, __user__, __metadata__, __event_emitter__) -> dict:
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

        # Decide what captions are missing (for status messaging)
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
            base_url=self.valves.vllm_base_url,
            api_key=self.valves.vllm_api_key,
            model=self.valves.caption_model,
            max_tokens=self.valves.caption_max_tokens,
            timeout_s=self.valves.caption_timeout_s,
            user_id=(__user__ or {}).get("id"),
        )

        if user_valves.force_keep_all_images:
            await self._emit_status(
                __event_emitter__,
                "✅ Compressor done (force_keep_all_images)",
                done=True, allowed=user_valves.show_live_status,
            )
            return body

        if has_images(last):
            keep_idx: Optional[int] = len(msgs) - 1
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
            decision, _reason = await route(
                user_text=text_of(last), captions=captions_for_router,
                base_url=self.valves.vllm_base_url,
                api_key=self.valves.vllm_api_key,
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

        body["messages"] = rewrite_messages(msgs, keep_idx, captions_by_url)
        await self._emit_status(
            __event_emitter__,
            "✅ Compressor done",
            done=True, allowed=user_valves.show_live_status,
        )
        return body
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py::TestStatusEvents test_qwenvl_image_compress.py::TestInletFreshUpload test_qwenvl_image_compress.py::TestInletTextFollowup -v
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): live status events during inlet"
```

---

## Task 16: Collapsible thinking log (`type: "message"`)

- [ ] **Step 1: Write failing tests**

```python
class TestThinkingLog:
    @respx.mock
    async def test_thinking_log_emitted_after_inlet(
        self, cache_db_path, text_msg, multimodal_msg, make_image
    ):
        u, h = make_image()
        c = CaptionCache(cache_db_path); await c.init()
        await c.put(h, "a screenshot", "qwen3-vl-8b")
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {
                "content": '{"need_images": false, "reason": "topic shift"}'
            }}]},
        )
        events = []
        async def emit(ev): events.append(ev)
        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [
            multimodal_msg("a", [u]),
            text_msg("y", role="assistant"),
            text_msg("explain decorators"),
        ]}
        await f.inlet(
            body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"},
            __event_emitter__=emit,
        )
        msg_events = [e for e in events if e.get("type") == "message"]
        assert len(msg_events) == 1
        content = msg_events[0]["data"]["content"]
        assert "<details>" in content
        assert "Image compressor reasoning" in content
        # mentions decision and the caption text
        assert "drop" in content.lower() or "router" in content.lower()
        assert "a screenshot" in content

    async def test_show_thinking_log_false_skips_block(
        self, cache_db_path, text_msg, multimodal_msg, make_image
    ):
        u, h = make_image()
        c = CaptionCache(cache_db_path); await c.init()
        await c.put(h, "x", "qwen3-vl-8b")
        events = []
        async def emit(ev): events.append(ev)
        f = Filter()
        f.valves = Filter.Valves(cache_db_path=cache_db_path)
        body = {"messages": [multimodal_msg("a", [u])]}
        uv = Filter.UserValves(show_thinking_log=False)
        await f.inlet(
            body=body, __user__={"id": "u", "valves": uv},
            __metadata__={"chat_id": "c"}, __event_emitter__=emit,
        )
        msg_events = [e for e in events if e.get("type") == "message"]
        assert msg_events == []
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — add helpers to `Filter`:

```python
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
```

Update `_inlet_impl` (replace it again — full version below) to collect data for the log and emit a final `type: "message"` event:

```python
    async def _inlet_impl(self, body, __user__, __metadata__, __event_emitter__) -> dict:
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
        return body

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
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py -v
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): collapsible thinking-log block"
```

---

## Task 17: Structured logging

- [ ] **Step 1: Write failing tests**

```python
import json as _json_test


class TestStructuredLog:
    @respx.mock
    async def test_logs_inlet_summary(
        self, caplog, cache_db_path, multimodal_msg, make_image
    ):
        import logging
        u, _ = make_image()
        respx.post("http://vllm/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "c"}}]},
        )
        f = Filter()
        f.valves = Filter.Valves(vllm_base_url="http://vllm/v1", cache_db_path=cache_db_path)
        body = {"messages": [multimodal_msg("a", [u])]}
        with caplog.at_level(logging.INFO, logger="qwenvl_image_compress"):
            await f.inlet(
                body=body, __user__={"id": "u"},
                __metadata__={"chat_id": "chat-1"},
            )
        summaries = [
            r.message for r in caplog.records
            if r.name == "qwenvl_image_compress" and r.message.startswith("{")
        ]
        assert summaries
        payload = _json_test.loads(summaries[-1])
        assert payload["event"] == "image_compress_inlet"
        assert payload["chat_id"] == "chat-1"
        assert payload["n_images"] >= 1
        assert "decision" in payload
        assert "latency_ms" in payload
```

- [ ] **Step 2: Run, confirm failure**.

- [ ] **Step 3: Implement** — at the very top of `_inlet_impl` add `import time` (already there) and `start = time.monotonic()`. At the end (before returning), add the log line:

```python
        # near top of _inlet_impl
        start = time.monotonic()

        # ... existing body ...

        # before final `return body`
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
```

(Add the `log.info(...)` immediately before `return body` in the main path, AND in the `force_keep_all_images` early-return path with `decision="force_keep"`.)

- [ ] **Step 4: Run, confirm pass**

```bash
pytest test_qwenvl_image_compress.py::TestStructuredLog -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(qwen3-vl-filter): structured per-inlet log line"
```

---

## Task 18: Helper scripts

**Files:**
- Create: `vast-templates/qwen3-vl-8b/scripts/dump_captions.py`
- Create: `vast-templates/qwen3-vl-8b/scripts/integration_test_image_compressor.sh`

- [ ] **Step 1: Create `scripts/dump_captions.py`**

```python
#!/usr/bin/env python3
"""Dump recent captions from the filter cache. Usage:

    python dump_captions.py /workspace/openwebui-data/img_captions.db
"""
import sqlite3
import sys


def main(path: str, limit: int = 50) -> None:
    con = sqlite3.connect(path)
    rows = con.execute(
        "SELECT img_hash, substr(caption,1,80), model, created_at "
        "FROM captions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    for h, cap, model, ts in rows:
        print(f"{h[:12]}  [{model}]  {ts}  {cap}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
```

- [ ] **Step 2: Create `scripts/integration_test_image_compressor.sh`**

```bash
#!/usr/bin/env bash
# Run on a Vast pod that has Open WebUI + vLLM up and the filter installed.
# Requires: API token from Open WebUI Settings → Account → API Keys.
set -eu

BASE="${OPENWEBUI_BASE:-http://127.0.0.1:3000}"
TOKEN="${OPENWEBUI_TOKEN:?Set OPENWEBUI_TOKEN to a valid API key}"
DB="${CAPTION_DB:-/workspace/openwebui-data/img_captions.db}"

count_captions() {
    sqlite3 "$DB" "SELECT count(*) FROM captions;" 2>/dev/null || echo 0
}

before=$(count_captions)
echo "Captions in cache before: $before"

# Scenario 1: upload 1 small synthetic PNG + ask
PNG_B64=$(printf '\x89PNG\r\n\x1a\nfake' | base64 | tr -d '\n')
read -r -d '' PAYLOAD <<EOF || true
{
  "model": "qwen3-vl-8b",
  "stream": false,
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "this is a smoke test image"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${PNG_B64}"}}
    ]}
  ]
}
EOF
curl -fsS -X POST "$BASE/api/chat/completions" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" >/dev/null
sleep 2

after=$(count_captions)
echo "Captions in cache after: $after"
if [ "$after" -gt "$before" ]; then
    echo "PASS: caption row added"
else
    echo "FAIL: no new caption row"
    exit 1
fi
```

- [ ] **Step 3: chmod and smoke-check that the dump script runs**

```bash
chmod +x vast-templates/qwen3-vl-8b/scripts/integration_test_image_compressor.sh
python vast-templates/qwen3-vl-8b/scripts/dump_captions.py 2>&1 | head -3
```
Expected: prints docstring (since no path arg) and exits with code 2.

- [ ] **Step 4: Commit**

```bash
git add vast-templates/qwen3-vl-8b/scripts/
git commit -m "feat(qwen3-vl-filter): cache dump + integration smoke scripts"
```

---

## Task 19: Update `onstart.sh` for persistence

**Files:**
- Create or modify: `vast-templates/qwen3-vl-8b/onstart.sh`

The user's existing onstart (from prior conversation) is the source. Add the four prerequisites the spec calls out: `DATA_DIR`, `WEBUI_SECRET_KEY`, plural `OPENAI_API_BASE_URLS`, drop legacy single-URL form.

- [ ] **Step 1: Read the current onstart.sh** if it exists, else create from the user's pasted version.

```bash
ls vast-templates/qwen3-vl-8b/onstart.sh 2>/dev/null && cat vast-templates/qwen3-vl-8b/onstart.sh || echo "MISSING"
```

- [ ] **Step 2: Write the updated onstart.sh** at `vast-templates/qwen3-vl-8b/onstart.sh`. Full content (preserves the user's existing structure; the four new lines are marked):

```bash
#!/bin/bash
set -e
exec > >(tee -a /workspace/logs/onstart.log) 2>&1

echo "=== Open WebUI + vLLM bootstrap: $(date) ==="

mkdir -p /workspace/logs /workspace/.hf_cache /workspace/.venvs /workspace/openwebui-data
export HF_HOME=/workspace/.hf_cache
export PIP_ROOT_USER_ACTION=ignore
SYS_PY=/usr/bin/python3

# NEW (filter persistence): persistent data dir for Open WebUI DB + caption cache
export DATA_DIR=/workspace/openwebui-data

# NEW (filter persistence): stable JWT secret across pod restarts
export WEBUI_SECRET_KEY=$(cat /workspace/.webui-secret 2>/dev/null \
    || (openssl rand -hex 32 | tee /workspace/.webui-secret))

# ---- 1. vLLM in system Python (idempotent) ----
if ! $SYS_PY -c "import vllm" 2>/dev/null; then
  echo "[1/4] Installing vllm..."
  $SYS_PY -m pip install --no-cache-dir \
    vllm==0.19.1 \
    "transformers!=5.3.*" \
    "accelerate==1.12.0" \
    "aiohttp>=3.13.3"
else
  echo "[1/4] vllm already installed, skip."
fi

# ---- 2. Open WebUI in ISOLATED venv ----
WEBUI_VENV=/workspace/.venvs/webui
if [ ! -x "$WEBUI_VENV/bin/open-webui" ]; then
  echo "[2/4] Creating isolated venv for open-webui..."
  apt-get install -y python3-venv >/dev/null 2>&1 || true
  $SYS_PY -m venv "$WEBUI_VENV"
  "$WEBUI_VENV/bin/pip" install --no-cache-dir --upgrade pip
  "$WEBUI_VENV/bin/pip" install --no-cache-dir open-webui aiosqlite httpx
else
  echo "[2/4] open-webui venv exists, skip."
fi

# ---- 3. Start vLLM (port 8000) ----
echo "[3/4] Starting vLLM..."
tmux kill-session -t vllm 2>/dev/null || true
tmux new -d -s vllm "\
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  HF_HOME=/workspace/.hf_cache \
  /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --served-model-name qwen3-vl-8b \
    --host 127.0.0.1 --port 8000 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --mm-encoder-attn-backend TORCH_SDPA \
    2>&1 | tee /workspace/logs/vllm.log"

# ---- 4. Start Open WebUI (port 3000) ----
echo "[4/4] Waiting for vLLM readiness, then starting Open WebUI..."
tmux kill-session -t webui 2>/dev/null || true
tmux new -d -s webui "\
  until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done; \
  echo 'vLLM ready, launching Open WebUI...'; \
  ENABLE_BASE_MODELS_CACHE=false \
  DATA_DIR=$DATA_DIR \
  WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY \
  OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1 \
  OPENAI_API_KEYS=sk-dummy \
  PORT=3000 HOST=127.0.0.1 \
  WEBUI_AUTH=True \
  $WEBUI_VENV/bin/open-webui serve --host 127.0.0.1 --port 3000 \
  2>&1 | tee /workspace/logs/webui.log"

echo "=== Bootstrap done. Tail logs: tmux attach -t vllm | webui ==="
```

Changes vs original (call out for reviewer):
- `DATA_DIR=/workspace/openwebui-data` exported at the top **and** passed into the webui tmux session.
- `WEBUI_SECRET_KEY` derived from a stored file so JWTs survive restart.
- `OPENAI_API_BASE_URLS` (plural) replaces deprecated `OPENAI_API_BASE_URL`.
- `OPENAI_API_KEYS` (plural) replaces `OPENAI_API_KEY`.
- vLLM readiness probe switched from `/v1/models` to `/health` (former returns 200 before weights finish loading).
- Removed `vllm-omni` install (not used by Qwen3-VL via standard vLLM entrypoint).
- Removed duplicate `--gpu-memory-utilization` flag and redundant `--attention-backend TRITON_ATTN` flag (kept the env var form).
- Added `aiosqlite httpx` to the webui venv install so the filter's deps are available when Open WebUI imports the function module.

- [ ] **Step 3: Verify `bash -n` syntax**

```bash
bash -n vast-templates/qwen3-vl-8b/onstart.sh
```
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add vast-templates/qwen3-vl-8b/onstart.sh
git commit -m "feat(qwen3-vl-filter): onstart.sh persistence + filter venv deps"
```

---

## Task 20: Final local pre-deploy verification

- [ ] **Step 1: Run the full test suite**

```bash
cd vast-templates/qwen3-vl-8b/functions
. .venv/Scripts/activate    # or .venv/bin/activate on Linux
pytest -v
```
Expected: all tests PASSED, no warnings of consequence.

- [ ] **Step 2: Compute final filter file size + count public exports**

```bash
wc -l qwenvl_image_compress.py
python -c "import qwenvl_image_compress as m; print([n for n in dir(m) if not n.startswith('_')])"
```
Expected: 400-550 lines; exports include `Filter`, `CaptionCache`, `caption_one`, `route`, `ensure_captions`, `rewrite_messages`, helpers.

- [ ] **Step 3: Manual deployment dry-run notes (no commit)**

Document for the operator: copy `qwenvl_image_compress.py` contents → paste into Open WebUI **Admin Panel → Functions → New Function**; save with name "Qwen3-VL Image Compressor"; bind via **Models → qwen3-vl-8b → Filters**. Then run `scripts/integration_test_image_compressor.sh` on the pod with `OPENWEBUI_TOKEN` set.

- [ ] **Step 4: Final commit (release marker)**

Bump `VERSION = "0.1.0"` in `qwenvl_image_compress.py` (drop the `-dev` suffix).

```bash
git add vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py
git commit -m "chore(qwen3-vl-filter): release 0.1.0"
```

---

## Self-review

**Spec coverage:**
| Spec § | Implementation task |
|--------|---------------------|
| §4 architecture, scenarios | Tasks 11, 12 |
| §5.2 Valves | Task 10 |
| §5.3 UserValves | Task 10, 13, 15, 16 |
| §5.4 inlet API | Task 11, 14 |
| §5.5 binding | Manual deploy (Task 20 step 3) |
| §6.1 caption prompt | Task 7 |
| §6.2 router prompt | Task 8 |
| §7.1 status events | Task 15 |
| §7.2 thinking-log block | Task 16 |
| §7.3 token estimation | Task 16 |
| §8 caching | Tasks 2, 3 |
| §9 error handling | Tasks 8 (router failopen), 9 (caption fail), 14 (top-level guard) |
| §10 testing | Tasks 1-17 unit; Task 18 integration script |
| §11 deployment + onstart | Tasks 18, 19, 20 |
| §12 observability | Tasks 17, 18 |

**Placeholder scan:** none. All steps contain runnable code or exact commands.

**Type consistency:**
- `CaptionCache.put(...)` signature uses keyword args `bytes_size`/`user_id`; matches usage in Task 9.
- `caption_one(data_url, base_url, api_key, model, max_tokens, timeout_s)` — same positional order in Tasks 7, 9, 11.
- `route(user_text, captions, base_url, api_key, model, max_tokens, timeout_s, failopen_keep)` — same order in Tasks 8, 11.
- `rewrite_messages(msgs, keep_idx, captions_by_url)` — consistent.
- `ensure_captions` returns `dict[url -> caption]` — consumed via `captions_by_url` everywhere.
- `Filter._inlet_impl` rewritten in Tasks 11, 15, 16, 17 — final version is the one in Task 16 augmented by Task 17's logging additions.

**Note on Task 16's full rewrite:** Task 16 replaces `_inlet_impl` wholesale (the body grows enough that a diff would be harder to follow than a full restatement). Task 17 adds two `log.info(...)` lines and `start = time.monotonic()` at the top of that final body. Both are noted in their respective Step 3.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-01-qwenvl-image-context-compressor.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
