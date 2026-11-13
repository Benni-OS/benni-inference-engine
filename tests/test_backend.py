"""Tests for the version-safe llama.cpp backend adapter."""
import importlib.metadata
import sys
import types
from pathlib import Path

import pytest

from bie.core.backend import BackendConfig, InferenceBackend, LlamaCppBackend


class OpenLlama:
    """Accepts any constructor argument, like **kwargs shims."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": "mocked"}]}


class StrictLlama:
    """Exposes only the stable, known constructor contract."""

    def __init__(self, model_path, n_ctx, n_threads, n_batch, n_gpu_layers, verbose):
        self.kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_batch": n_batch,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": "mocked"}]}


def _config(**overrides) -> BackendConfig:
    base = {
        "model_path": Path("models/qwen.gguf"),
        "n_ctx": 8192,
        "n_threads": 4,
        "n_batch": 512,
        "ngl": 24,
        "expert_cache_slots": 128,
        "moe_offload_pattern": r"\.ffn_.*_exps\.=CPU",
        "verbose": False,
    }
    base.update(overrides)
    return BackendConfig(**base)


def _install_llama(monkeypatch, cls):
    module = types.ModuleType("llama_cpp")
    module.Llama = cls
    monkeypatch.setitem(sys.modules, "llama_cpp", module)


def test_backend_satisfies_protocol():
    assert isinstance(LlamaCppBackend(), InferenceBackend)


def test_load_forwards_supported_kwargs(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.10")
    backend = LlamaCppBackend()
    report = backend.load(_config())
    assert report["backend"] == "llama-cpp-python"
    kwargs = report["backend_init_kwargs"]
    assert kwargs["model_path"] == str(_config().model_path)
    assert kwargs["n_gpu_layers"] == 24
    assert kwargs["n_ctx"] == 8192
    assert kwargs["n_experts"] == 128
    assert "offload_kqv" not in kwargs
    assert "moe_offload_pattern" in report["skipped_args"]


def test_offload_kqv_version_gated(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.30")
    backend = LlamaCppBackend()
    kwargs = backend.load(_config())["backend_init_kwargs"]
    assert kwargs["offload_kqv"] is True


def test_offload_kqv_skipped_on_old_version(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.10")
    backend = LlamaCppBackend()
    report = backend.load(_config())
    assert "offload_kqv" not in report["backend_init_kwargs"]
    assert "offload_kqv" in report["skipped_args"]


def test_offload_kqv_absent_when_ngl_zero(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.30")
    backend = LlamaCppBackend()
    report = backend.load(_config(ngl=0))
    assert "offload_kqv" not in report["backend_init_kwargs"]
    assert "offload_kqv" not in report["skipped_args"]


def test_strict_backend_drops_unsupported_args(monkeypatch):
    _install_llama(monkeypatch, StrictLlama)
    backend = LlamaCppBackend()
    report = backend.load(_config())
    kwargs = report["backend_init_kwargs"]
    assert kwargs["n_gpu_layers"] == 24
    assert kwargs["verbose"] is False
    assert "n_experts" not in kwargs
    assert "n_experts" in report["skipped_args"]


def test_missing_backend_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    backend = LlamaCppBackend()
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        backend.load(_config())


def test_complete_requires_load():
    backend = LlamaCppBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.complete("hello")


def test_complete_forwards_to_model(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    backend = LlamaCppBackend()
    backend.load(_config())
    result = backend.complete("hello", temperature=0.0)
    assert result["choices"][0]["text"] == "mocked"


class _StreamingModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return iter(self.chunks)


def test_stream_yields_text_chunks(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    backend = LlamaCppBackend()
    backend.model = _StreamingModel(
        [{"choices": [{"text": "a"}]}, {"choices": [{"text": ""}]}, {"choices": [{"text": "b"}]}]
    )
    assert list(backend.stream("hello")) == ["a", "b"]
    assert backend.model.calls[0][1]["stream"] is True


def test_unload_resets_model(monkeypatch):
    _install_llama(monkeypatch, OpenLlama)
    backend = LlamaCppBackend()
    backend.load(_config())
    assert backend.model is not None
    backend.unload()
    assert backend.model is None
