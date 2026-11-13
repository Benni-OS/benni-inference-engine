"""BIE Core Engine — llama.cpp backend with explicit lifecycle, streaming, and Skill 81 integration."""

from __future__ import annotations

import inspect
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..serve.telemetry import TelemetryEngine
from ..serve.tool_caller import ToolCallingEngine
from ..skill81.ceiling_extractor import CeilingExtractor
from ..skill81.hallucination_guard import HallucinationGuard
from ..skill81.injection_firewall import PromptInjectionFirewall
from .admission import AdmissionController
from .backend import BackendConfig, LlamaCppBackend
from .dual_model import DualModelOrchestrator
from .embeddings import EmbeddingsEngine
from .grammar_guard import GrammarGuard
from .hot_pin import HotPinManager
from .kv_manager import KVManager
from .moe_scheduler import MoEScheduler
from .prompt_templates import PromptTemplateEngine
from .router import TaskRouter
from .vision import VisionEngine
from .vram_arbiter import VRAMArbiter


class BIEEngine:
    """Inference facade for the local BIE runtime integrating Skill 81, KV compression,

    admission control, speculative decoding, tool calling, and dual-model vision.
    """

    DEFAULT_MODEL: str = "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    DEFAULT_VISION_MODEL: str = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
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
        logical_cpus = os.cpu_count() or 4
        physical_default = max(4, logical_cpus // 2) if logical_cpus >= 8 else logical_cpus
        self.n_threads = n_threads or physical_default
        self.n_batch = n_batch
        self.verbose = verbose
        self.model: Any | None = None
        self.backend: LlamaCppBackend | None = None
        self.runtime_config: dict[str, Any] | None = None

        # Integrated components
        self.ceiling_extractor = CeilingExtractor()
        self.guard = HallucinationGuard()
        self.firewall = PromptInjectionFirewall()
        self.kv_manager = KVManager(ctx_size=self.n_ctx)
        self.hot_pin = HotPinManager(n_layers=48, n_experts=128)
        self.embeddings = EmbeddingsEngine()
        self.admission = AdmissionController()
        self.router = TaskRouter(local_context_limit=self.n_ctx)
        self.telemetry = TelemetryEngine()
        self.vision_engine = VisionEngine()
        self.arbiter = VRAMArbiter()
        self.dual_orchestrator = DualModelOrchestrator(
            primary_model=self.model_name,
            vision_model=self.DEFAULT_VISION_MODEL,
        )

    def resolve_model_path(self) -> Path:
        """Resolve a GGUF path from an explicit path, BIE_MODEL_DIR, or local models dirs."""
        candidate = self.model_path or Path(self.model_name)
        if candidate.is_file():
            return candidate
        model_dir = Path(os.getenv("BIE_MODEL_DIR", "models"))
        search_dirs = [
            model_dir,
            Path("bie/models"),
            Path(__file__).parents[1] / "models",
        ]
        candidates: list[Path] = []
        for s_dir in search_dirs:
            candidates.append(s_dir / self.model_name)
            if not self.model_name.endswith(".gguf"):
                candidates.append(s_dir / f"{self.model_name}.gguf")
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Model not found: {self.model_name}. Set model_path or BIE_MODEL_DIR."
        )

    def load(self) -> dict[str, Any]:
        """Load the model using the hardware recommendation from MoEScheduler."""
        model_path = self.resolve_model_path()
        config = MoEScheduler(self.model_name, n_ctx=self.n_ctx).compute()
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
            kv_quant=config.get("kv_quant", "q4_0"),
            flash_attn=bool(config.get("flash_attn", True)),
            type_k=config.get("type_k"),
            type_v=config.get("type_v"),
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

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Generate normalized vector embeddings."""
        return self.embeddings.embed(texts, model=self.model)

    def _sanitize_options(self, prompt: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Defensively clamp max_tokens so prompt + max_tokens never exceeds n_ctx."""
        options = dict(kwargs)
        model = self.model
        n_ctx = getattr(self, "n_ctx", None)
        if n_ctx is None and model is not None and hasattr(model, "n_ctx"):
            n_ctx = model.n_ctx() if callable(model.n_ctx) else model.n_ctx
        if not n_ctx:
            n_ctx = 8192

        prompt_tokens = 0
        if model is not None and hasattr(model, "tokenize") and callable(model.tokenize):
            try:
                prompt_tokens = len(model.tokenize(prompt.encode("utf-8", errors="ignore")))
            except Exception:
                prompt_tokens = self.kv_manager.estimate_tokens(prompt)
        else:
            prompt_tokens = self.kv_manager.estimate_tokens(prompt)

        available = n_ctx - prompt_tokens
        if available <= 0:
            raise ValueError(f"Requested prompt ({prompt_tokens} tokens) exceeds context window of {n_ctx}")

        if "max_tokens" in options:
            requested_max = options["max_tokens"]
            if requested_max is not None and isinstance(requested_max, int):
                if requested_max > available:
                    options["max_tokens"] = max(1, available)
                elif requested_max <= 0:
                    options["max_tokens"] = max(1, min(4096, available))

        return options

    @staticmethod
    def _supports_grammar(fn: Any) -> bool:
        """Return True when the callable explicitly accepts ``grammar`` or belongs to ``llama_cpp``."""
        module_name = getattr(type(fn), "__module__", "") or getattr(fn, "__module__", "") or ""
        if module_name.startswith("llama_cpp"):
            return True
        try:
            sig = inspect.signature(fn)
            return "grammar" in sig.parameters
        except (TypeError, ValueError):
            return False

    def _scan_messages_with_firewall(
        self,
        messages: list[dict[str, Any]],
        *,
        block_on_malicious: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run Cyber Sentinel PromptInjectionFirewall on user/tool turns."""
        sanitized_messages: list[dict[str, Any]] = []
        matched_rules: list[str] = []
        highest_threat = "clean"

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "tool") and isinstance(content, str) and content:
                inspection = self.firewall.inspect(content)
                if not inspection.safe:
                    matched_rules.extend(inspection.matched_rules)
                    if inspection.threat_level == "malicious":
                        highest_threat = "malicious"
                        if block_on_malicious:
                            raise PermissionError(
                                f"Blocked by Cyber Sentinel Firewall: {', '.join(inspection.matched_rules)}"
                            )
                    elif highest_threat == "clean":
                        highest_threat = "suspicious"
                    sanitized_messages.append({**msg, "content": inspection.sanitized_prompt})
                    continue
            sanitized_messages.append(msg)

        return sanitized_messages, {
            "threat_level": highest_threat,
            "matched_rules": sorted(set(matched_rules)),
            "sanitized": bool(matched_rules),
        }

    def _prepare_chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        task_type: str = "general",
        apply_skill81: bool = True,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        security_scan: bool = True,
        block_on_malicious: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Normalize multimodal payloads, enforce firewall, inject tools/Skill 81/schema, and prune KV."""
        normalized_messages: list[dict[str, Any]] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                text, images = self.vision_engine.extract_image_payloads(content)
                if images:
                    text = self.vision_engine.format_vision_prompt(text, len(images))
                normalized_messages.append({**m, "content": text})
            else:
                normalized_messages.append(dict(m))

        security_report = {"threat_level": "clean", "matched_rules": [], "sanitized": False}
        if security_scan:
            normalized_messages, security_report = self._scan_messages_with_firewall(
                normalized_messages, block_on_malicious=block_on_malicious
            )

        effective_messages = normalized_messages
        if tools:
            effective_messages = ToolCallingEngine.inject_tools(effective_messages, tools)

        if apply_skill81:
            effective_messages = self.ceiling_extractor.extract(
                effective_messages,
                model_profile={"model_id": self.model_name},
                task_type=task_type,
            )

        if schema:
            last_msg = dict(effective_messages[-1])
            last_msg["content"] = GrammarGuard.format_prompt_for_schema(
                str(last_msg.get("content", "")), schema
            )
            effective_messages = effective_messages[:-1] + [last_msg]

        effective_messages = self.kv_manager.compress(effective_messages)
        return effective_messages, security_report

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a non-streaming completion with optional GBNF schema enforcement."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        model = self._require_model()

        effective_prompt = prompt
        if schema:
            effective_prompt = GrammarGuard.format_prompt_for_schema(prompt, schema)

        sanitized_kwargs = self._sanitize_options(effective_prompt, kwargs)
        if schema and "grammar" not in sanitized_kwargs and self._supports_grammar(model):
            grammar = GrammarGuard.to_llama_grammar(schema)
            if grammar is not None:
                sanitized_kwargs["grammar"] = grammar

        t0 = time.time()
        result = model(effective_prompt, **sanitized_kwargs)
        elapsed = max(1e-6, time.time() - t0)

        # Record telemetry
        text = result.get("choices", [{}])[0].get("text", "") if isinstance(result, dict) else str(result)
        tokens_est = self.kv_manager.estimate_tokens(text)
        self.telemetry.record_request(latency_s=elapsed, tokens_count=tokens_est)

        if schema:
            val_report = GrammarGuard.validate(text, schema=schema)
            if isinstance(result, dict):
                result["schema_validation"] = val_report

        return result

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        task_type: str = "general",
        apply_skill81: bool = True,
        guard: bool = False,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        security_scan: bool = True,
        block_on_malicious: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute chat completion with firewall, Skill 81, native templates, GBNF grammar, and reasoning parser."""
        if not messages:
            raise ValueError("messages must be a non-empty list")
        model = self._require_model()

        effective_messages, security_report = self._prepare_chat_messages(
            messages,
            task_type=task_type,
            apply_skill81=apply_skill81,
            tools=tools,
            schema=schema,
            security_scan=security_scan,
            block_on_malicious=block_on_malicious,
        )

        template_style = PromptTemplateEngine.detect_style(self.model_name)
        rendered_prompt = PromptTemplateEngine.render(
            effective_messages, style=template_style, add_generation_prompt=True
        )
        sanitized_kwargs = self._sanitize_options(rendered_prompt, kwargs)

        use_native_chat = (
            hasattr(model, "create_chat_completion")
            and callable(model.create_chat_completion)
        )

        t0 = time.time()
        if use_native_chat:
            native_kwargs = dict(sanitized_kwargs)
            if schema and "grammar" not in native_kwargs and self._supports_grammar(model.create_chat_completion):
                grammar = GrammarGuard.to_llama_grammar(schema)
                if grammar is not None:
                    native_kwargs["grammar"] = grammar
            result = model.create_chat_completion(messages=effective_messages, **native_kwargs)
        else:
            if schema and "grammar" not in sanitized_kwargs and self._supports_grammar(model):
                grammar = GrammarGuard.to_llama_grammar(schema)
                if grammar is not None:
                    sanitized_kwargs["grammar"] = grammar
            result = model(rendered_prompt, **sanitized_kwargs)
        elapsed = max(1e-6, time.time() - t0)

        # Extract generated content
        native_tool_calls: list[dict[str, Any]] = []
        if isinstance(result, dict) and "choices" in result:
            choice = result["choices"][0]
            msg_obj = choice.get("message", {}) or {}
            raw_text = choice.get("text") or msg_obj.get("content", "") or ""
            native_tool_calls = list(msg_obj.get("tool_calls") or [])
        else:
            raw_text = str(result)

        clean_text, reasoning_content = PromptTemplateEngine.parse_reasoning(raw_text)

        # Compute usage tokens
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or self.kv_manager.estimate_tokens(rendered_prompt))
        completion_tokens = int(usage.get("completion_tokens") or self.kv_manager.estimate_tokens(raw_text))
        total_tokens = prompt_tokens + completion_tokens

        # Record telemetry
        self.telemetry.record_request(
            latency_s=elapsed,
            decode_tok_s=round(completion_tokens / elapsed, 1) if elapsed > 0 else 0.0,
            tokens_count=completion_tokens,
        )

        # Parse tool calls if tools were provided or detected
        tool_calls = native_tool_calls or (
            ToolCallingEngine.parse_tool_calls(clean_text)
            if (tools or "<tool_call>" in raw_text)
            else []
        )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": clean_text,
        }
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        response_payload: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
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
            "security": security_report,
        }

        if schema:
            response_payload["schema_validation"] = GrammarGuard.validate(clean_text, schema=schema)

        if guard:
            last_prompt = messages[-1].get("content", "") if messages else rendered_prompt
            response_payload["guard"] = self.guard.check(clean_text, str(last_prompt))

        return response_payload

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        task_type: str = "general",
        apply_skill81: bool = True,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        security_scan: bool = True,
        block_on_malicious: bool = True,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream OpenAI delta dicts from native chat completion or rendered template fallback."""
        if not messages:
            raise ValueError("messages must be a non-empty list")
        model = self._require_model()

        effective_messages, _ = self._prepare_chat_messages(
            messages,
            task_type=task_type,
            apply_skill81=apply_skill81,
            tools=tools,
            schema=schema,
            security_scan=security_scan,
            block_on_malicious=block_on_malicious,
        )
        template_style = PromptTemplateEngine.detect_style(self.model_name)
        rendered_prompt = PromptTemplateEngine.render(
            effective_messages, style=template_style, add_generation_prompt=True
        )
        sanitized_kwargs = self._sanitize_options(rendered_prompt, kwargs)

        use_native_chat = (
            hasattr(model, "create_chat_completion")
            and callable(model.create_chat_completion)
        )
        if use_native_chat:
            stream_kwargs = {**sanitized_kwargs, "stream": True}
            if schema and "grammar" not in stream_kwargs and self._supports_grammar(model.create_chat_completion):
                grammar = GrammarGuard.to_llama_grammar(schema)
                if grammar is not None:
                    stream_kwargs["grammar"] = grammar
            for chunk in model.create_chat_completion(messages=effective_messages, **stream_kwargs):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta:
                        yield delta
            return

        stream_kwargs = {**sanitized_kwargs, "stream": True}
        if schema and "grammar" not in stream_kwargs and self._supports_grammar(model):
            grammar = GrammarGuard.to_llama_grammar(schema)
            if grammar is not None:
                stream_kwargs["grammar"] = grammar
        for chunk in model(rendered_prompt, **stream_kwargs):
            choices = chunk.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                if text:
                    yield {"content": text}

    def vision(
        self,
        image_input: str | Path | bytes,
        prompt: str = "Analyze this image in detail.",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convenience method for multimodal vision inference."""
        if isinstance(image_input, bytes):
            import base64

            b64_str = base64.b64encode(image_input).decode("utf-8")
            img_ref = f"data:image/jpeg;base64,{b64_str}"
        else:
            img_ref = str(image_input)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_ref}},
                ],
            }
        ]
        return self.chat(messages, task_type="vision", **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield generated text chunks from llama.cpp."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        model = self._require_model()
        sanitized_kwargs = self._sanitize_options(prompt, kwargs)
        options = {**sanitized_kwargs, "stream": True}
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
