"""Tests for BIE Engine lifecycle with a mocked llama.cpp backend."""
import sys
import types
from typing import ClassVar

import pytest

from bie.core.engine import BIEEngine
from bie.core.moe_scheduler import MoEScheduler

FIXED_CONFIG = {"ngl": 24, "model_profile": "known", "warnings": []}


class FakeLlama:
    instances: ClassVar[list["FakeLlama"]] = []
    last_call: ClassVar[tuple | None] = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLlama.instances.append(self)

    def __call__(self, prompt, **kwargs):
        FakeLlama.last_call = (prompt, kwargs)
        return {"choices": [{"text": "mocked"}]}


class StreamingLlama:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return iter(self.chunks)


def _install_fake_llama(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    FakeLlama.instances = []
    FakeLlama.last_call = None


def test_load_creates_llama_with_scheduler_ngl(tmp_path, monkeypatch):
    model_file = tmp_path / "qwen.gguf"
    model_file.write_bytes(b"fake-gguf")
    _install_fake_llama(monkeypatch)
    monkeypatch.setattr(MoEScheduler, "compute", lambda self, **kwargs: dict(FIXED_CONFIG))
    engine = BIEEngine("qwen.gguf", model_path=model_file, n_ctx=8192)
    config = engine.load()
    assert engine.model is FakeLlama.instances[-1]
    assert config["model_path"] == str(model_file)
    assert config["ngl"] == 24
    llama = engine.model
    assert llama.kwargs["n_gpu_layers"] == 24
    assert llama.kwargs["n_ctx"] == 8192
    assert llama.kwargs["model_path"] == str(model_file)


def test_load_without_llama_cpp_raises(tmp_path, monkeypatch):
    model_file = tmp_path / "qwen.gguf"
    model_file.write_bytes(b"fake-gguf")
    module = types.ModuleType("llama_cpp")
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    engine = BIEEngine("qwen.gguf", model_path=model_file)
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        engine.load()


def test_complete_requires_load():
    engine = BIEEngine("qwen3-30b-a3b")
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.complete("hello")


def test_complete_rejects_invalid_prompt():
    engine = BIEEngine("qwen3-30b-a3b")
    engine.model = FakeLlama()
    with pytest.raises(ValueError):
        engine.complete("   ")
    with pytest.raises(ValueError):
        engine.complete(123)


def test_complete_forwards_prompt_and_kwargs():
    engine = BIEEngine("qwen3-30b-a3b")
    engine.model = FakeLlama()
    result = engine.complete("hello", temperature=0.0)
    assert result["choices"][0]["text"] == "mocked"
    assert FakeLlama.last_call == ("hello", {"temperature": 0.0})


def test_stream_yields_text_chunks():
    engine = BIEEngine("qwen3-30b-a3b")
    chunks = [{"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}]
    fake = StreamingLlama(chunks)
    engine.model = fake
    assert list(engine.stream("hello")) == ["a", "b"]
    assert fake.calls[0][1]["stream"] is True


def test_stream_skips_empty_text():
    engine = BIEEngine("qwen3-30b-a3b")
    chunks = [{"choices": [{"text": ""}]}, {"choices": [{"text": "x"}]}]
    engine.model = StreamingLlama(chunks)
    assert list(engine.stream("hi")) == ["x"]


def test_unload_resets_state():
    engine = BIEEngine("qwen3-30b-a3b")
    engine.model = FakeLlama()
    engine.runtime_config = {"ngl": 1}
    engine.unload()
    assert engine.model is None
    assert engine.runtime_config is None


def test_resolve_model_path_explicit(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")
    engine = BIEEngine("qwen.gguf", model_path=model_file)
    assert engine.resolve_model_path() == model_file


def test_resolve_model_path_from_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "qwen.gguf").write_bytes(b"data")
    monkeypatch.setenv("BIE_MODEL_DIR", str(model_dir))
    engine = BIEEngine("qwen")
    assert engine.resolve_model_path() == model_dir / "qwen.gguf"


def test_resolve_model_path_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("BIE_MODEL_DIR", str(tmp_path))
    engine = BIEEngine("missing")
    with pytest.raises(FileNotFoundError):
        engine.resolve_model_path()
