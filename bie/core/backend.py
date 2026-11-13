"""Version-safe adapter for the llama.cpp Python backend.

pyproject declares ``llama-cpp-python`` without a minimum version, so the
constructor contract must be probed at runtime instead of assumed.
"""
from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .kv_quant import GGML_TYPE_MAP, KVQuantType

# Parameter names stable across llama-cpp-python releases.
_STABLE_KWARGS = frozenset(
    {"model_path", "n_ctx", "n_threads", "n_batch", "n_gpu_layers", "verbose"}
)
# Minimum llama-cpp-python release for version-gated offload flags.
_VERSION_GATED_KWARGS = {"offload_kqv": (0, 2, 15)}


@dataclass(frozen=True)
class BackendConfig:
    """Backend-agnostic request to load a model, derived from BIE policy."""

    model_path: Path
    n_ctx: int
    n_threads: int
    n_batch: int
    ngl: int
    expert_cache_slots: int
    moe_offload_pattern: str
    verbose: bool = False
    kv_quant: KVQuantType = "q4_0"
    flash_attn: bool = True
    type_k: int | None = None
    type_v: int | None = None


@runtime_checkable
class InferenceBackend(Protocol):
    """Contract for swapping the underlying inference runtime."""

    def load(self, config: BackendConfig) -> dict[str, Any]: ...
    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]: ...
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]: ...
    def unload(self) -> None: ...


class LlamaCppBackend:
    """Translate BIE policy into llama.cpp constructor arguments.

    Only arguments accepted by the installed release are forwarded; unknown or
    version-gated flags are skipped and reported instead of raising.
    """

    def __init__(self) -> None:
        self.model: Any | None = None

    @staticmethod
    def _llama() -> type[Any]:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required; install the CPU or CUDA extra."
            ) from exc
        return Llama

    @staticmethod
    def _accepted_params(cls_: type[Any]) -> set[str] | None:
        """Return accepted init params, or None when the class accepts **kwargs."""
        try:
            signature = inspect.signature(cls_.__init__)
        except (TypeError, ValueError):
            return set(_STABLE_KWARGS)
        accepted: set[str] = set()
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return None
            accepted.add(name)
        return accepted

    @staticmethod
    def _version_at_least(minimum: tuple[int, int, int]) -> bool:
        try:
            raw = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            return False
        parts: list[int] = []
        for chunk in raw.split(".")[:3]:
            digits = ""
            for char in chunk:
                if char.isdigit():
                    digits += char
                else:
                    break
            parts.append(int(digits) if digits else 0)
        return tuple(parts) >= minimum

    def _build_init_kwargs(
        self, cls: type[Any], config: BackendConfig
    ) -> tuple[dict[str, Any], dict[str, str]]:
        accepted = self._accepted_params(cls)
        kwargs: dict[str, Any] = {}
        notes: dict[str, str] = {}

        def forward(name: str, value: Any) -> None:
            if accepted is None or name in accepted:
                kwargs[name] = value
            else:
                notes[name] = "not accepted by installed llama-cpp-python"

        forward("model_path", str(config.model_path))
        forward("n_ctx", config.n_ctx)
        forward("n_threads", config.n_threads)
        forward("n_batch", config.n_batch)
        forward("n_gpu_layers", config.ngl)
        forward("verbose", config.verbose)
        if self._version_at_least((0, 3, 0)):
            forward("n_threads_batch", config.n_threads)
            if config.ngl > 0 and config.flash_attn:
                forward("flash_attn", True)
            resolved_type_k = (
                config.type_k
                if config.type_k is not None
                else GGML_TYPE_MAP.get(config.kv_quant, 1)
            )
            resolved_type_v = (
                config.type_v
                if config.type_v is not None
                else GGML_TYPE_MAP.get(config.kv_quant, 1)
            )
            if config.kv_quant != "f16" or config.type_k is not None:
                forward("type_k", resolved_type_k)
            if config.kv_quant != "f16" or config.type_v is not None:
                forward("type_v", resolved_type_v)
        if config.ngl > 0:
            if self._version_at_least(_VERSION_GATED_KWARGS["offload_kqv"]):
                forward("offload_kqv", True)
            else:
                notes["offload_kqv"] = (
                    "requires llama-cpp-python>=" + ".".join(map(str, _VERSION_GATED_KWARGS["offload_kqv"]))
                )
        if config.expert_cache_slots:
            if accepted is not None and "n_experts" in accepted:
                kwargs["n_experts"] = config.expert_cache_slots
            elif getattr(cls, "__name__", "") == "OpenLlama":
                kwargs["n_experts"] = config.expert_cache_slots
            else:
                notes["n_experts"] = "not accepted by installed llama-cpp-python (managed dynamically via hot_pin)"
        if config.moe_offload_pattern:
            notes["moe_offload_pattern"] = "no stable llama.cpp constructor parameter"
        return kwargs, notes

    def load(self, config: BackendConfig) -> dict[str, Any]:
        """Instantiate the backend and report exactly which args were applied."""
        cls = self._llama()
        kwargs, notes = self._build_init_kwargs(cls, config)
        self.model = cls(**kwargs)
        return {
            "backend": "llama-cpp-python",
            "backend_init_kwargs": kwargs,
            "skipped_args": notes,
        }

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("backend is not loaded; call load() first")
        return self.model(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        if self.model is None:
            raise RuntimeError("backend is not loaded; call load() first")
        options = {**kwargs, "stream": True}
        for chunk in self.model(prompt, **options):
            choices = chunk.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                if text:
                    yield text

    def unload(self) -> None:
        self.model = None
