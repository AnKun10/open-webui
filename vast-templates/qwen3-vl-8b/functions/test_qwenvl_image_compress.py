import pytest
import qwenvl_image_compress as mod
from qwenvl_image_compress import CaptionCache


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
