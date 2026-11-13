"""Wave 2 tests: PromptResponseCache, async non-blocking execution, admission telemetry, API key guard, and Anthropic /v1/messages."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from bie.core.engine import BIEEngine
from bie.core.prompt_cache import PromptResponseCache
from bie.serve.openai_compat import app, configure_engine


class _CountingMockLlama:
    def __init__(self, reply: str = "<think>reasoning</think>Final local answer"):
        self.calls = 0
        self.reply = reply

    def __call__(self, prompt: str, **kwargs):
        self.calls += 1
        return {"choices": [{"text": self.reply}]}


def test_prompt_response_cache_lru_and_ttl():
    cache = PromptResponseCache(max_entries=2, ttl_seconds=60.0)
    k1 = PromptResponseCache.build_key("chat", {"q": 1})
    k2 = PromptResponseCache.build_key("chat", {"q": 2})
    k3 = PromptResponseCache.build_key("chat", {"q": 3})

    cache.put(k1, {"ans": 1}, tokens_saved=100)
    cache.put(k2, {"ans": 2}, tokens_saved=100)
    assert cache.get(k1) == {"ans": 1}  # k1 refreshed
    cache.put(k3, {"ans": 3}, tokens_saved=100)  # evicts k2
    assert cache.get(k2) is None
    assert cache.get(k1) == {"ans": 1}
    assert cache.stats()["hits"] == 2
    assert cache.stats()["tokens_saved"] == 200

    short_cache = PromptResponseCache(max_entries=5, ttl_seconds=1.0)
    short_cache.put(k1, {"ans": 1}, tokens_saved=10)
    short_cache._store[k1].created_at = time.time() - 5.0
    assert short_cache.get(k1) is None


def test_openai_compat_cache_hit_and_telemetry_reset():
    mock = _CountingMockLlama(reply="Deterministic output")
    engine = BIEEngine("cyber-tiel-coder-35b")
    engine.model = mock
    configure_engine(engine)
    client = TestClient(app)
    try:
        payload = {
            "messages": [{"role": "user", "content": "Return deterministic code"}],
            "temperature": 0.0,
        }
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200
        assert mock.calls == 1

        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200
        assert r2.json().get("cached") is True
        assert mock.calls == 1  # Zero extra model calls

        tel = client.get("/v1/bie/telemetry").json()
        assert tel["prompt_cache"]["hits"] >= 1
        assert tel["admission"]["completed_requests"] >= 1

        clr = client.delete("/v1/bie/cache").json()
        assert clr["cleared_entries"] >= 1
    finally:
        configure_engine(None)


def test_optional_api_key_guard(monkeypatch):
    monkeypatch.setenv("BIE_API_KEY", "secret-local-key")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"x-api-key": "secret-local-key"}).status_code == 200
    assert (
        client.get("/v1/models", headers={"Authorization": "Bearer secret-local-key"}).status_code
        == 200
    )


def test_anthropic_messages_thinking_tools_and_streaming():
    mock = _CountingMockLlama(
        reply='<think>Need weather</think>Checking.\n<tool_call>{"name": "get_weather", "arguments": {"city": "SP"}}</tool_call>'
    )
    engine = BIEEngine("cyber-tiel-coder-35b")
    engine.model = mock
    configure_engine(engine)
    client = TestClient(app)
    try:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "cyber-tiel-coder-35b",
                "system": "You are local Claude replacement.",
                "messages": [{"role": "user", "content": "Weather in SP?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Fetch weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 512,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["stop_reason"] == "tool_use"
        block_types = [b["type"] for b in body["content"]]
        assert "thinking" in block_types
        assert "tool_use" in block_types
        tool_blk = next(b for b in body["content"] if b["type"] == "tool_use")
        assert tool_blk["name"] == "get_weather"
        assert tool_blk["input"] == {"city": "SP"}

        stream_resp = client.post(
            "/v1/messages",
            json={
                "model": "cyber-tiel-coder-35b",
                "messages": [{"role": "user", "content": "Stream hello"}],
                "stream": True,
            },
        )
        assert stream_resp.status_code == 200
        assert "event: message_start" in stream_resp.text
        assert "event: message_stop" in stream_resp.text
    finally:
        configure_engine(None)
