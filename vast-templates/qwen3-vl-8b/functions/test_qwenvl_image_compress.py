import base64
import hashlib
import pytest
import respx
import httpx
import qwenvl_image_compress as mod
from qwenvl_image_compress import (
    CaptionCache,
    iter_image_parts,
    find_latest_image_turn,
    has_images,
    text_of,
    hash_image_url,
)


def test_module_imports():
    assert mod.VERSION.startswith("0.1.0")


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


class TestCaptionCacheBatch:
    async def test_get_many_empty_returns_empty(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        assert await c.get_many([]) == {}

    async def test_get_many_returns_only_present(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        await c.put("h1" + "0" * 62, "cap1", "qwen3-vl-8b")
        result = await c.get_many(["h1" + "0" * 62, "missing" + "0" * 57])
        assert result == {"h1" + "0" * 62: "cap1"}

    async def test_put_many_inserts_all(self, cache_db_path):
        c = CaptionCache(cache_db_path)
        await c.init()
        items = [
            ("a" * 64, "ca", "qwen3-vl-8b", 100, None),
            ("b" * 64, "cb", "qwen3-vl-8b", 200, "u1"),
        ]
        await c.put_many(items)
        assert await c.get("a" * 64) == "ca"
        assert await c.get("b" * 64) == "cb"

    async def test_put_many_race_uses_first_writer(self, cache_db_path):
        """Two concurrent put_many for same hash: second is silently dropped."""
        c = CaptionCache(cache_db_path)
        await c.init()
        h = "c" * 64
        await c.put_many([(h, "first", "qwen3-vl-8b", None, None)])
        await c.put_many([(h, "second", "qwen3-vl-8b", None, None)])
        assert await c.get(h) == "first"  # INSERT OR IGNORE preserves first


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

    def test_text_of_returns_empty_string_when_text_is_none(self):
        m = {"role": "user", "content": [{"type": "text", "text": None}]}
        assert text_of(m) == ""

    def test_has_images_false_for_empty_url(self):
        m = {"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": ""}},
        ]}
        assert not has_images(m)

    def test_has_images_false_for_missing_url_key(self):
        m = {"role": "user", "content": [
            {"type": "image_url", "image_url": {}},
        ]}
        assert not has_images(m)


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
