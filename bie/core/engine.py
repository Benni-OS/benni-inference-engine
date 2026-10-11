"""BIE Core Engine — llama.cpp backend with explicit lifecycle, streaming, and Skill 81 integration."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .backend import BackendConfig, LlamaCppBackend
from .hot_pin import HotPinManager
from .kv_manager import KVManager
from .moe_scheduler import MoEScheduler
from ..skill81.ceiling_extractor import CeilingExtractor
from ..skill81.hallucination_guard import HallucinationGuard


class BIEEngine:
    """Inference facade for the local BIE runtime integrating Skill 81, KV compression, and expert hot-pinning."""

    def __init__(
        self,
        model_name: str,
        port: int = 8080,
        *,
        model_path: str | Path | None = None,
        n_ctx: int = 32768,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.port = port
        self.model_path = Path(model_path) if model_path else None
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.n_batch = n_batch
        self.verbose = verbose
        self.model: Any | None = None
        self.backend: LlamaCppBackend | None = None
        self.runtime_config: dict[str, Any] | None = None

        # Integrated Skill 81 & memory management components
        self.ceiling_extractor = CeilingExtractor()
        self.guard = HallucinationGuard()
        self.kv_manager = KVManager(ctx_size=self.n_ctx)
        self.hot_pin = HotPinManager(n_layers=48, n_experts=128)

    def resolve_model_path(self) -> Path:
        """Resolve a GGUF path from an explicit path or BIE_MODEL_DIR."""
        candidate = self.model_path or Path(self.model_name)
        if candidate.is_file():
            return candidate
        model_dir = Path(os.getenv("BIE_MODEL_DIR", "models"))
        candidates = [model_dir / self.model_name]
        if not self.model_name.endswith(".gguf"):
            candidates.append(model_dir / f"{self.model_name}.gguf")
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Model not found: {self.model_name}. Set model_path or BIE_MODEL_DIR."
        )

    def load(self) -> dict[str, Any]:
        """Load the model using the hardware recommendation from MoEScheduler."""
        model_path = self.resolve_model_path()
        config = MoEScheduler(self.model_name).compute()
        backend = LlamaCppBackend()
        backend_config = BackendConfig(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            ngl=int(config.get("ngl", 0)),
            expert_cache_slots=int(config.get("expert_cache_slots", 0)),
            moe_offload_pattern=config.get("moe_offload_pattern", ""),
            verbose=self.verbose,
        )
        report = backend.load(backend_config)
        self.backend = backend
        self.model = backend.model
        self.runtime_config = {**config, "model_path": str(model_path), **report}
        return self.runtime_config

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("BIEEngine is not loaded. Call load() first.")
        return self.model

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a non-streaming completion."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        model = self._require_model()
        return model(prompt, **kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        task_type: str = "general",
        apply_skill81: bool = True,
        guard: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute chat completion with Skill 81 steering, KV compression, and optional guard check."""
        if not messages:
            raise ValueError("messages must be a non-empty list")
        model = self._require_model()

        # Apply Skill 81 steering if enabled
        effective_messages = messages
        if apply_skill81:
            effective_messages = self.ceiling_extractor.extract(
                messages,
                model_profile={"model_id": self.model_name},
                task_type=task_type,
            )

        # Apply KV context management & pruning
        effective_messages = self.kv_manager.compress(effective_messages)

        # Render prompt string (or format for backend)
        prompt_str = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in effective_messages)

        t0 = time.time()
        result = model(prompt_str, **kwargs)
        elapsed = max(1e-6, time.time() - t0)

        # Extract generated content
        if isinstance(result, dict) and "choices" in result:
            choice = result["choices"][0]
            text = choice.get("text") or choice.get("message", {}).get("content", "")
        else:
            text = str(result)

        # Compute usage tokens
        prompt_tokens = self.kv_manager.estimate_tokens(prompt_str)
        completion_tokens = self.kv_manager.estimate_tokens(text)
        total_tokens = prompt_tokens + completion_tokens

        response_payload: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "timings": {
                "latency_s": round(elapsed, 3),
                "tokens_per_second": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0.0,
            },
        }

        if guard:
            last_prompt = messages[-1].get("content", "") if messages else prompt_str
            response_payload["guard"] = self.guard.check(text, last_prompt)

        return response_payload

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield generated text chunks from llama.cpp."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        model = self._require_model()
        options = {**kwargs, "stream": True}
        for chunk in model(prompt, **options):
            choices = chunk.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                if text:
                    yield text

    def unload(self) -> None:
        """Release the model reference and runtime configuration."""
        if self.backend is not None:
            self.backend.unload()
        self.backend = None
        self.model = None
        self.runtime_config = None
