"""Coverage-focused behaviour tests for ``llm.client`` (Ollama mocked at httpx)."""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import pytest

from active_inference.llm.client import LLMConfig, OllamaClient


class FakeOllama:
    """In-memory Ollama HTTP API served through ``httpx.MockTransport``."""

    def __init__(self, models: Optional[List[str]] = None) -> None:
        self.models: List[str] = list(models or [])
        self.requests: List[httpx.Request] = []
        self.fail: Dict[str, int] = {}
        self.pull_adds = True
        self.delete_removes = True
        self.stream_lines: List[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.fail:
            return httpx.Response(self.fail[path], json={"error": "boom"})
        body = json.loads(request.content) if request.content else {}
        if path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": m, "size": len(m)} for m in self.models]},
            )
        if path == "/api/pull":
            if self.pull_adds:
                self.models.append(body["name"])
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/delete":
            if self.delete_removes and body["name"] in self.models:
                self.models.remove(body["name"])
            return httpx.Response(200)
        if path == "/api/generate" and body.get("stream"):
            return httpx.Response(200, text="\n".join(self.stream_lines))
        if path == "/api/generate":
            return httpx.Response(200, json={"response": f"echo:{body['prompt']}"})
        if path == "/api/chat":
            last = body["messages"][-1]["content"]
            return httpx.Response(200, json={"message": {"content": f"chat:{last}"}})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.9.1"})
        return httpx.Response(404)

    def last_json(self, path: str) -> Dict[str, Any]:
        for req in reversed(self.requests):
            if req.url.path == path:
                return json.loads(req.content)
        raise AssertionError(f"no request to {path}")


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    fake = FakeOllama(models=["gemma3:2b"])
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return fake


@pytest.fixture
async def client(ollama: FakeOllama) -> Any:
    c = OllamaClient(LLMConfig())
    assert await c.initialize() is True
    yield c
    await c.close()


def test_config_to_dict_round_trip() -> None:
    cfg = LLMConfig(default_model="m", temperature=0.1, debug=True)
    data = cfg.to_dict()
    assert data["default_model"] == "m"
    assert data["temperature"] == 0.1
    assert LLMConfig(**data) == cfg


def test_logger_level_follows_debug_flag() -> None:
    assert OllamaClient(LLMConfig(debug=True)).logger.level == logging.DEBUG
    assert OllamaClient(LLMConfig(debug=False)).logger.level == logging.INFO


def test_sync_context_manager_returns_self() -> None:
    c = OllamaClient(LLMConfig())
    with c as entered:
        assert entered is c


class TestInitialize:
    async def test_success_records_models(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        assert client._is_initialized is True
        assert client._available_models == ["gemma3:2b"]
        assert client._model_info["gemma3:2b"]["size"] == len("gemma3:2b")

    async def test_refresh_failure_returns_false(self, ollama: FakeOllama) -> None:
        ollama.fail["/api/tags"] = 500
        c = OllamaClient(LLMConfig())
        assert await c.initialize() is False
        assert c._is_initialized is False
        assert c._available_models == []
        await c.close()

    async def test_no_models_pulls_default(self, ollama: FakeOllama) -> None:
        ollama.models = []
        c = OllamaClient(LLMConfig(default_model="tiny"))
        assert await c.initialize() is True
        assert ollama.last_json("/api/pull") == {"name": "tiny"}
        assert c._available_models == ["tiny"]
        await c.close()

    async def test_no_models_and_pull_fails(self, ollama: FakeOllama) -> None:
        ollama.models = []
        ollama.pull_adds = False
        c = OllamaClient(LLMConfig())
        assert await c.initialize() is False
        assert c._is_initialized is False
        await c.close()

    async def test_unexpected_exception_closes_client(
        self, ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = OllamaClient(LLMConfig())

        async def explode() -> bool:
            raise RuntimeError("kaput")

        monkeypatch.setattr(c, "_refresh_models", explode)
        assert await c.initialize() is False
        assert c.client is None
        assert c._is_initialized is False

    async def test_exception_before_client_created(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*_: Any, **__: Any) -> None:
            raise OSError("no sockets")

        monkeypatch.setattr(httpx, "AsyncClient", broken)
        c = OllamaClient(LLMConfig())
        assert await c.initialize() is False
        assert c.client is None


class TestPull:
    async def test_pull_without_existing_client_uses_temporary_one(
        self, ollama: FakeOllama
    ) -> None:
        c = OllamaClient(LLMConfig())
        assert c.client is None
        # Refresh after the pull uses self.client (None) and fails, so the
        # pulled model is not observed as available.
        assert await c.pull_model("new") is False
        assert ollama.last_json("/api/pull") == {"name": "new"}
        assert c.client is None

    async def test_pull_http_error(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.fail["/api/pull"] = 500
        assert await client.pull_model("x") is False

    async def test_pull_success(self, client: OllamaClient, ollama: FakeOllama) -> None:
        assert await client.pull_model("llama") is True
        assert "llama" in client._available_models


class TestGenerateAndChat:
    async def test_requires_initialization(self) -> None:
        c = OllamaClient(LLMConfig())
        with pytest.raises(RuntimeError, match="not initialized"):
            await c.generate("hi")
        with pytest.raises(RuntimeError, match="not initialized"):
            await c.chat([{"role": "user", "content": "hi"}])
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in c.generate_stream("hi"):
                pass

    async def test_generate_payload_and_response(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        out = await client.generate(
            "why?", system_prompt="sys", temperature=0.2, max_tokens=7
        )
        assert out == "echo:why?"
        payload = ollama.last_json("/api/generate")
        assert payload["model"] == "gemma3:2b"
        assert payload["system"] == "sys"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.2
        assert payload["options"]["num_predict"] == 7
        assert payload["options"]["top_k"] == 40

    async def test_generate_pulls_missing_model(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        assert await client.generate("q", model="other") == "echo:q"
        assert ollama.last_json("/api/pull") == {"name": "other"}

    async def test_generate_unpullable_model(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.pull_adds = False
        with pytest.raises(ValueError, match="Model nope not available"):
            await client.generate("q", model="nope")
        with pytest.raises(ValueError, match="Model nope not available"):
            await client.chat([{"role": "user", "content": "q"}], model="nope")

    async def test_generate_http_error_propagates(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.fail["/api/generate"] = 503
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("q")

    async def test_chat(self, client: OllamaClient, ollama: FakeOllama) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello"},
        ]
        assert await client.chat(msgs, top_p=0.5) == "chat:hello"
        payload = ollama.last_json("/api/chat")
        assert payload["messages"] == msgs
        assert payload["options"]["top_p"] == 0.5

    async def test_chat_pulls_missing_model(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        msgs = [{"role": "user", "content": "x"}]
        assert await client.chat(msgs, model="m2") == "chat:x"
        assert "m2" in client._available_models

    async def test_chat_http_error_propagates(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.fail["/api/chat"] = 500
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat([{"role": "user", "content": "x"}])


class TestStreaming:
    async def test_stream_yields_until_done(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.stream_lines = [
            json.dumps({"response": "Hel"}),
            "",
            "not json",
            json.dumps({"other": 1}),
            json.dumps({"response": "lo", "done": True}),
            json.dumps({"response": "ignored"}),
        ]
        chunks = [c async for c in client.generate_stream("p", repeat_penalty=1.3)]
        assert chunks == ["Hel", "lo"]
        payload = ollama.last_json("/api/generate")
        assert payload["stream"] is True
        assert payload["options"]["repeat_penalty"] == 1.3

    async def test_stream_http_error(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.fail["/api/generate"] = 500
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in client.generate_stream("p"):
                pass


class TestModelManagement:
    async def test_info_and_list(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.models.append("extra")
        assert await client.list_models() == ["gemma3:2b", "extra"]
        assert await client.get_model_info("extra") == {"name": "extra", "size": 5}
        assert await client.get_model_info("missing") is None

    async def test_delete(self, client: OllamaClient, ollama: FakeOllama) -> None:
        assert await client.delete_model("gemma3:2b") is True
        assert client._available_models == []
        assert ollama.last_json("/api/delete") == {"name": "gemma3:2b"}

    async def test_delete_not_removed(
        self, client: OllamaClient, ollama: FakeOllama
    ) -> None:
        ollama.delete_removes = False
        assert await client.delete_model("gemma3:2b") is False

    async def test_delete_error(self, client: OllamaClient, ollama: FakeOllama) -> None:
        ollama.fail["/api/delete"] = 404
        assert await client.delete_model("gemma3:2b") is False


class TestHealthAndLifecycle:
    async def test_healthy(self, client: OllamaClient) -> None:
        assert await client.health_check() == {
            "status": "healthy",
            "version": "0.9.1",
            "available_models": ["gemma3:2b"],
            "default_model": "gemma3:2b",
            "initialized": True,
        }

    async def test_unhealthy(self, client: OllamaClient, ollama: FakeOllama) -> None:
        ollama.fail["/api/version"] = 500
        result = await client.health_check()
        assert result["status"] == "unhealthy"
        assert "500" in result["error"]
        assert result["available_models"] == []
        assert result["initialized"] is False

    async def test_async_context_manager(self, ollama: FakeOllama) -> None:
        async with OllamaClient(LLMConfig()) as c:
            assert c._is_initialized is True
            assert c.client is not None
        assert c.client is None
        assert c._is_initialized is False

    async def test_close_without_client(self) -> None:
        c = OllamaClient(LLMConfig())
        c._is_initialized = True
        await c.close()
        assert c._is_initialized is False
