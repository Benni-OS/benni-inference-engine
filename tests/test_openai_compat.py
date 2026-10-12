"""Tests for the OpenAI-compatible serve layer without a real GPU."""

import pytest
from fastapi.testclient import TestClient

from bie.serve.openai_compat import app, configure_engine


class FakeEngine:
    def __init__(self, model_name="Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf", chunks=None, text="Hello BIE"):
        self.model_name = model_name
        self.model = object()
        self.chunks = chunks or ["Hello ", "BIE"]
        self.text = text
        self.last = None

    def complete(self, prompt, **kwargs):
        self.last = (prompt, kwargs)
        return {"choices": [{"text": self.text}]}

    def stream(self, prompt, **kwargs):
        self.last = (prompt, kwargs)
        return iter(self.chunks)


@pytest.fixture(autouse=True)
def _reset_engine():
    configure_engine(None)
    yield
    configure_engine(None)


@pytest.fixture()
def client():
    return TestClient(app)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["engine"] == "benni-inference-engine"


def test_ready_503_when_not_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()["detail"]["error"]
    assert body["type"] == "engine_not_ready"


def test_ready_503_when_model_not_loaded(client):
    engine = FakeEngine()
    engine.model = None
    configure_engine(engine)
    assert client.get("/ready").status_code == 503


def test_ready_200_when_loaded(client):
    configure_engine(FakeEngine())
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_models_default(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    assert body["data"][0]["owned_by"] == "benni-os"


def test_models_uses_configured_engine(client):
    configure_engine(FakeEngine(model_name="custom-gguf"))
    response = client.get("/v1/models")
    assert response.json()["data"][0]["id"] == "custom-gguf"


def test_completions_503_when_not_loaded(client):
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["type"] == "engine_not_ready"


def test_completions_requires_prompt(client):
    response = client.post("/v1/completions", json={})
    assert response.status_code == 422


def test_completions_validates_temperature(client):
    response = client.post("/v1/completions", json={"prompt": "hi", "temperature": 5})
    assert response.status_code == 422


def test_completions_returns_text_completion(client):
    engine = FakeEngine(text="resposta")
    configure_engine(engine)
    response = client.post(
        "/v1/completions",
        json={"prompt": "pergunta", "temperature": 0.0, "max_tokens": 16},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    assert body["id"].startswith("cmpl-")
    assert body["choices"][0]["text"] == "resposta"
    assert body["choices"][0]["finish_reason"] == "stop"
    prompt, options = engine.last
    assert prompt == "pergunta"
    assert options == {"temperature": 0.0, "max_tokens": 16}


def test_completions_stream_sse(client):
    engine = FakeEngine(chunks=["a", "b"])
    configure_engine(engine)
    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert text.startswith("data: ")
    assert "data: [DONE]\n\n" in text
    assert '"delta": {"content": "a"}' in text or '"content": "a"' in text
    assert engine.last[1] == {"temperature": 0.7, "max_tokens": 2048}


def test_chat_completions_503_when_not_loaded(client):
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "oi"}]}
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["type"] == "engine_not_ready"


def test_chat_completions_requires_messages(client):
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422


def test_chat_completions_returns_chat_completion(client):
    engine = FakeEngine(text="resposta")
    configure_engine(engine)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "seja breve"},
                {"role": "user", "content": "oi"},
            ],
            "temperature": 0.2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "resposta"
    prompt, options = engine.last
    assert prompt == "system: seja breve\nuser: oi"
    assert options["temperature"] == 0.2


def test_chat_completions_stream_sse(client):
    engine = FakeEngine(chunks=["olá", " mundo"])
    configure_engine(engine)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "oi"}], "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]\n\n" in response.text



def test_chat_completions_multimodal_vision(client):
    engine = FakeEngine(text="Vejo um código e um diagrama")
    configure_engine(engine)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "O que você vê?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                            },
                        },
                    ],
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
    assert body["choices"][0]["message"]["content"] == "Vejo um código e um diagrama"
    prompt, _ = engine.last
    assert "[IMAGE_1]" in prompt


def test_bie_arbiter_endpoint(client):
    response = client.get("/v1/bie/arbiter?mode=cooperative_split")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "cooperative_split"
    assert "primary_ngl" in body
    assert "vision_ngl" in body
    assert body["primary_ngl"] > 0
    assert body["vision_ngl"] > 0
