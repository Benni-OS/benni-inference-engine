"""Tests for BIEEngine chat method, token usage calculation, and guard integration."""

import pytest
from bie.core.engine import BIEEngine


class MockLlama:
    def __init__(self):
        self.last_prompt = None

    def __call__(self, prompt, **kwargs):
        self.last_prompt = prompt
        return {"choices": [{"message": {"role": "assistant", "content": "Engine response [VERIFIED]"}, "text": "Engine response [VERIFIED]"}]}


def test_engine_chat_method():
    engine = BIEEngine("qwen3-8b")
    engine.model = MockLlama()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is BIE?"},
    ]

    res = engine.chat(messages, task_type="BRAVO", apply_skill81=True, guard=True)

    assert res["object"] == "chat.completion"
    assert res["model"] == "qwen3-8b"
    assert res["choices"][0]["message"]["content"] == "Engine response [VERIFIED]"
    assert res["usage"]["prompt_tokens"] > 0
    assert res["usage"]["completion_tokens"] > 0
    assert "guard" in res
    assert res["guard"]["passed"] is True
    assert "timings" in res
    assert res["timings"]["tokens_per_second"] >= 0


def test_engine_chat_empty_messages_raises():
    engine = BIEEngine("qwen3-8b")
    engine.model = MockLlama()
    with pytest.raises(ValueError):
        engine.chat([])
