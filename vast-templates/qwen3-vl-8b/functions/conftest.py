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
