import asyncio
import base64
import copy
import hashlib
import json as _json_test
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
    caption_one,
    route,
    ensure_captions,
    Filter,
    CAPTION_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
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

    async def test_data_url_empty_payload_raises(self):
        with pytest.raises(ValueError, match="empty"):
            await hash_image_url("data:image/png;base64,", fetch_base="http://x")


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
        out = mod.rewrite_messages(msgs, keep_idx=2, captions_by_url=self._captions_for([u]))
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
        out = mod.rewrite_messages(msgs, keep_idx=None, captions_by_url={u1: "A", u2: "B"})
        text_parts = [p["text"] for p in out[0]["content"] if p["type"] == "text"]
        assert all(p.get("type") != "image_url" for p in out[0]["content"])
        assert any("A" in t for t in text_parts)
        assert any("B" in t for t in text_parts)

    def test_text_only_message_passthrough(self, text_msg):
        msgs = [text_msg("hello")]
        out = mod.rewrite_messages(msgs, keep_idx=None, captions_by_url={})
        assert out == msgs

    def test_role_unchanged(self, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [multimodal_msg("x", [u], role="user")]
        out = mod.rewrite_messages(msgs, keep_idx=None, captions_by_url={u: "C"})
        assert out[0]["role"] == "user"

    def test_missing_caption_falls_back_to_placeholder(self, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [multimodal_msg("x", [u])]
        out = mod.rewrite_messages(msgs, keep_idx=None, captions_by_url={})  # no caption
        text_parts = [p["text"] for p in out[0]["content"] if p["type"] == "text"]
        assert any("(no caption)" in t for t in text_parts)
        assert all(p.get("type") != "image_url" for p in out[0]["content"])

    def test_keep_idx_preserves_pixels_at_target(self, multimodal_msg, make_image):
        u, _ = make_image()
        msgs = [multimodal_msg("x", [u])]
        out = mod.rewrite_messages(msgs, keep_idx=0, captions_by_url={u: "C"})
        # image_url part must still exist at turn 0
        assert any(p.get("type") == "image_url" for p in out[0]["content"])


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


class TestFilterValves:
    def test_valves_defaults(self):
        f = Filter()
        assert f.valves.vllm_base_url == "http://127.0.0.1:8000/v1"
        assert f.valves.caption_model == "qwen3-vl-8b"
        assert f.valves.router_failopen_keep is True
        assert f.valves.priority == 5
        # Deliberately not setting self.toggle — see comment in Filter.__init__.
        # Open WebUI 0.9.2 gates filters with `toggle=True` behind a per-chat
        # enable button which conflicts with `is_global=True` auto-apply.
        assert not hasattr(f, "toggle")

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
        descriptions = [e["data"]["description"] for e in events if e.get("type") == "status"]
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


class TestEnsureCacheConcurrency:
    async def test_ensure_cache_concurrent_first_calls_share_instance(self, cache_db_path):
        f = Filter()
        f.valves = Filter.Valves(cache_db_path=cache_db_path)
        c1, c2 = await asyncio.gather(f._ensure_cache(), f._ensure_cache())
        assert c1 is c2


class TestInletEmptyScanned:
    async def test_all_hash_fail_returns_body_unchanged(
        self, cache_db_path, multimodal_msg, text_msg
    ):
        # Use an unsupported URL scheme so hash_image_url raises ValueError for every image.
        bad_url_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "ftp://nope/x.png"}},
            ],
        }
        f = Filter()
        f.valves = Filter.Valves(cache_db_path=cache_db_path)
        body = {"messages": [bad_url_msg, text_msg("ok", role="assistant"), text_msg("more")]}
        before = copy.deepcopy(body)
        out = await f.inlet(
            body=body, __user__={"id": "u"}, __metadata__={"chat_id": "c"},
        )
        assert out == before  # unchanged because no images could be hashed
