"""OpenAI-compatible serving layer for the BIE runtime."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from bie.core.admission import AdmissionController
from bie.core.embeddings import EmbeddingsEngine
from bie.core.engine import BIEEngine
from bie.core.grammar_guard import GrammarGuard
from bie.core.prompt_cache import PromptResponseCache
from bie.core.prompt_templates import PromptTemplateEngine
from bie.core.video_engine import VideoEngine
from bie.core.video_generator import VideoGenerator, VideoGenParams
from bie.core.video_upscaler import UpscaleConfig, VideoUpscaler
from bie.core.vision import VisionEngine
from bie.core.vram_arbiter import VRAMArbiter
from bie.serve.tool_caller import ToolCallingEngine
from bie.skill81.hallucination_guard import HallucinationGuard
from bie.skill81.injection_firewall import PromptInjectionFirewall

VERSION = "0.7.0"
DEFAULT_PRIMARY_MODEL = "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
DEFAULT_VISION_MODEL = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
DEFAULT_VIDEO_UNDERSTANDING_MODEL = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"

app = FastAPI(title="benni-inference-engine", version=VERSION)
_engine: BIEEngine | None = None
_guard = HallucinationGuard()
_firewall = PromptInjectionFirewall()
_cache = PromptResponseCache()
_admission = AdmissionController(
    max_concurrent=int(os.getenv("BIE_MAX_CONCURRENT", "2")),
    max_queue_depth=int(os.getenv("BIE_MAX_QUEUE", "32")),
)
_inference_lock = threading.Lock()
_fallback_embeddings = EmbeddingsEngine()
_vision_engine = VisionEngine()
_video_engine = VideoEngine()
_video_upscaler = VideoUpscaler()
_video_generator = VideoGenerator(upscaler=_video_upscaler)
_arbiter = VRAMArbiter()


@app.middleware("http")
async def _optional_api_key_guard(request: Request, call_next: Callable[..., Any]) -> Any:
    expected_key = os.getenv("BIE_API_KEY", "").strip()
    if expected_key and request.url.path not in ("/health", "/ready"):
        auth_header = request.headers.get("authorization", "")
        x_api_key = request.headers.get("x-api-key", "")
        bearer = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
        if bearer != expected_key and x_api_key.strip() != expected_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing BIE API key.",
                        "type": "authentication_error",
                    }
                },
            )
    return await call_next(request)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]] | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class CompletionRequest(BaseModel):
    model: str = DEFAULT_PRIMARY_MODEL
    prompt: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False
    schema_constraint: dict[str, Any] | None = None
    use_cache: bool = True


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_PRIMARY_MODEL
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False
    task_type: str = "general"
    apply_skill81: bool = False
    guard: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    use_cache: bool = True


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = DEFAULT_PRIMARY_MODEL
    system: str | list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    apply_skill81: bool = False


class EmbeddingRequest(BaseModel):
    model: str = DEFAULT_PRIMARY_MODEL
    input: str | list[str]


class GuardCheckRequest(BaseModel):
    prompt: str
    output: str


class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = "distorted, blurry, bad anatomy, low quality, artifacts, watermark"
    model: str = "wan2.1-t2v-14b"
    image_url: str | None = None
    width: int = 832
    height: int = 480
    frames: int = 49
    fps: int = 16
    steps: int = 30
    guidance_scale: float = 5.0
    enable_teacache: bool = True
    teacache_threshold: float = 0.15
    seed: int | None = None
    upscale_after: bool = True
    upscale_factor: int = 2
    target_fps: int = 60
    upscale_mode: str = "fast_cinema"


class VideoUpscaleRequest(BaseModel):
    video_path: str
    scale: int = Field(default=2, ge=2, le=4)
    target_fps: int = Field(default=60, ge=24, le=120)
    mode: str = "fast_cinema"
    codec: str = "h264_nvenc"
    tile_size: int = 256
    crf: int = 18


def configure_engine(engine: BIEEngine | None) -> None:
    """Inject or clear the runtime engine; intended for startup and tests."""
    global _engine
    _engine = engine
    _cache.clear()


async def _run_inference_sync(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute blocking llama.cpp / engine inference off the event loop under lock."""
    def _locked() -> Any:
        with _inference_lock:
            return fn(*args, **kwargs)

    return await anyio.to_thread.run_sync(_locked)


def _require_engine() -> BIEEngine:
    if _engine is None or _engine.model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": {"type": "engine_not_ready", "message": "BIE engine is not loaded"}},
        )
    return _engine


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _sse(chunks: Iterator[str], request_id: str, model: str) -> Iterator[str]:
    for text in chunks:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


def _sse_deltas(deltas: Iterator[dict[str, Any]], request_id: str, model: str) -> Iterator[str]:
    for delta in deltas:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    final_payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


from bie.serve.health_monitor import HealthMonitor

_health_monitor = HealthMonitor()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "engine": "benni-inference-engine", "version": VERSION}


@app.get("/v1/bie/health/detailed")
async def bie_health_detailed() -> dict[str, Any]:
    """Expose detailed resource health, VRAM, and thermal diagnostics."""
    return _health_monitor.check_health()


@app.get("/ready")
async def ready() -> dict[str, Any]:
    _require_engine()
    return {"status": "ready", "engine": "benni-inference-engine", "version": VERSION}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    model = _engine.model_name if _engine else DEFAULT_PRIMARY_MODEL
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "benni-os"},
            {"id": DEFAULT_VISION_MODEL, "object": "model", "owned_by": "benni-os"},
            {"id": "cyber-tiel-coder-35b", "object": "model", "owned_by": "benni-os"},
            {"id": "gemma-4-26b-vision", "object": "model", "owned_by": "benni-os"},
            {"id": "qwen2.5-vl-7b-video", "object": "model", "owned_by": "benni-os"},
            {"id": "wan2.1-t2v-14b", "object": "model", "owned_by": "benni-os"},
            {"id": "wan2.1-i2v-14b", "object": "model", "owned_by": "benni-os"},
            {"id": "ltx-video-0.9.5", "object": "model", "owned_by": "benni-os"},
            {"id": "ltx-video-2.3", "object": "model", "owned_by": "benni-os"},
            {"id": "bie-video-upscaler-pipeline", "object": "model", "owned_by": "benni-os"},
        ],
    }


@app.get("/v1/videos/models")
async def video_models() -> dict[str, Any]:
    """Expose supported video generation and understanding models with hardware placement plans."""
    return {
        "video_generation": VideoGenerator.SUPPORTED_MODELS,
        "video_understanding": {
            "default_model": DEFAULT_VIDEO_UNDERSTANDING_MODEL,
            "supported_models": ["qwen2.5-vl-7b-video", "qwen2.5-vl-3b-video", "minicpm-v-2.6"],
        },
        "video_upscaler": {
            "default_pipeline": "bie-video-upscaler-pipeline",
            "nvenc_accelerated": _video_upscaler.nvenc_available,
            "supported_modes": ["fast_cinema", "ultra_4k", "pure_spatial", "pure_temporal"],
        },
    }


@app.post("/v1/videos/generations")
async def generate_video(request: VideoGenerationRequest) -> dict[str, Any]:
    """Generate high-fidelity video with Wan2.1 or LTX-Video and automatic upscale."""
    params = VideoGenParams(**request.model_dump())
    plan = _video_generator.compute_generation_plan(params)
    task = _video_generator.create_task(params)
    _video_generator.execute_task(task.task_id)

    updated_task = _video_generator.get_task(task.task_id)
    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to retrieve task")

    return {
        "task_id": updated_task.task_id,
        "status": updated_task.status,
        "progress": updated_task.progress,
        "output_path": updated_task.output_path,
        "metrics": updated_task.metrics,
        "execution_plan": plan,
    }


@app.post("/v1/videos/upscale")
async def upscale_video(request: VideoUpscaleRequest) -> dict[str, Any]:
    """Run spatial super-resolution and temporal frame interpolation on existing video."""
    cfg = UpscaleConfig(
        scale=request.scale,
        target_fps=request.target_fps,
        mode=request.mode,
        codec=request.codec,
        tile_size=request.tile_size,
        crf=request.crf,
    )
    in_p = Path(request.video_path)
    if not in_p.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")

    plan = _video_upscaler.compute_pipeline_plan(in_p, cfg)
    out_path = _video_upscaler.upscale(in_p, config=cfg)

    return {
        "status": "completed",
        "output_path": str(out_path),
        "plan": plan,
    }


@app.get("/v1/videos/{task_id}")
async def get_video_task(task_id: str) -> dict[str, Any]:
    """Retrieve video generation task status and output."""
    task = _video_generator.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Video task {task_id} not found")
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "output_path": task.output_path,
        "error_message": task.error_message,
        "metrics": task.metrics,
    }


@app.get("/v1/bie/arbiter")
async def bie_arbiter(mode: str = "primary_exclusive") -> dict[str, Any]:
    """Expose GPU VRAM allocation plan between Cyber-Tiel-Coder-35B and Gemma-4-26B."""
    plan = _arbiter.compute_plan(mode)  # type: ignore
    return {
        "mode": plan.mode,
        "primary_ngl": plan.primary_ngl,
        "vision_ngl": plan.vision_ngl,
        "primary_vram_gb": plan.primary_vram_gb,
        "vision_vram_gb": plan.vision_vram_gb,
        "total_vram_allocated_gb": plan.total_vram_allocated_gb,
        "available_vram_gb": plan.available_vram_gb,
    }


@app.get("/v1/bie/metrics")
async def bie_metrics() -> dict[str, Any]:
    """Expose live engine metrics, hardware state, and configuration."""
    engine = _require_engine()
    return {
        "status": "ready",
        "model": engine.model_name,
        "n_ctx": getattr(engine, "n_ctx", 32768),
        "runtime_config": getattr(engine, "runtime_config", {}),
        "kv_current_tokens": getattr(getattr(engine, "kv_manager", None), "current_len", 0),
        "hot_pin_state": getattr(getattr(engine, "hot_pin", None), "pinned", False),
        "prompt_cache": _cache.stats(),
        "admission": _admission.stats(),
    }


@app.get("/v1/bie/telemetry")
async def bie_telemetry() -> dict[str, Any]:
    """Expose real-time performance telemetry, percentiles, cache, admission, and heatmap."""
    engine = _require_engine()
    telemetry_eng = getattr(engine, "telemetry", None)
    base: dict[str, Any]
    if telemetry_eng is not None:
        summary = telemetry_eng.get_summary()
        heatmap = telemetry_eng.render_ascii_heatmap()
        base = {**summary, "ascii_heatmap": heatmap}
    else:
        base = {"uptime_seconds": 0.0, "total_requests_recorded": 0}
    base["prompt_cache"] = _cache.stats()
    base["admission"] = _admission.stats()
    return base


@app.delete("/v1/bie/cache")
async def bie_clear_cache() -> dict[str, Any]:
    """Flush cached prompt responses."""
    cleared = _cache.clear()
    return {"cleared_entries": cleared, "cache": _cache.stats()}


@app.post("/v1/bie/guard")
async def bie_guard_check(request: GuardCheckRequest) -> dict[str, Any]:
    """Audit prompt/output against epistemic drift and hallucination."""
    return _guard.check(request.output, request.prompt)


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    """Generate vector embeddings for input strings."""
    engine = _engine
    inputs = [request.input] if isinstance(request.input, str) else list(request.input)

    if engine is not None and hasattr(engine, "embed"):
        vectors = engine.embed(inputs)
    else:
        vectors = _fallback_embeddings.embed(inputs)

    total_tok = sum(_estimate_tokens(t) for t in inputs)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": idx, "embedding": vec}
            for idx, vec in enumerate(vectors)
        ],
        "model": request.model,
        "usage": {"prompt_tokens": total_tok, "total_tokens": total_tok},
    }


@app.post("/v1/completions", response_model=None)
async def completions(request: CompletionRequest) -> dict[str, Any] | StreamingResponse:
    engine = _require_engine()
    request_id = f"cmpl-{uuid.uuid4().hex}"
    options = {"temperature": request.temperature, "max_tokens": request.max_tokens}

    if request.stream:
        if not _admission.admit_or_reject(request_id, estimated_tokens=request.max_tokens):
            raise HTTPException(
                status_code=429,
                detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
            )

        def _guarded_stream() -> Iterator[str]:
            try:
                with _inference_lock:
                    yield from _sse(engine.stream(request.prompt, **options), request_id, request.model)
            finally:
                _admission.release()

        return StreamingResponse(_guarded_stream(), media_type="text/event-stream")

    cache_key: str | None = None
    if _cache.is_cacheable(
        temperature=request.temperature,
        stream=request.stream,
        cache_bypass=not request.use_cache,
    ):
        cache_key = _cache.build_key(
            "completions",
            {
                "model": request.model,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "schema_constraint": request.schema_constraint,
            },
        )
        cached = _cache.get(cache_key)
        if cached is not None:
            cached["id"] = request_id
            cached["cached"] = True
            return cached

    with _admission.lease(request_id, estimated_tokens=request.max_tokens) as admitted:
        if not admitted:
            raise HTTPException(
                status_code=429,
                detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
            )
        try:
            if request.schema_constraint is not None:
                result = await _run_inference_sync(
                    engine.complete, request.prompt, schema=request.schema_constraint, **options
                )
            else:
                result = await _run_inference_sync(engine.complete, request.prompt, **options)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(exc), "type": "server_error"}},
            ) from exc

    text = (
        result.get("choices", [{}])[0].get("text", "")
        if isinstance(result, dict)
        else str(result)
    )

    if request.schema_constraint:
        val_res = GrammarGuard.validate(text, schema=request.schema_constraint)
        if isinstance(result, dict):
            result["schema_validation"] = val_res

    p_tokens = _estimate_tokens(request.prompt)
    c_tokens = _estimate_tokens(text)
    payload: dict[str, Any] = {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        },
    }
    if cache_key is not None:
        _cache.put(cache_key, payload, tokens_saved=p_tokens + c_tokens)
    return payload


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any] | StreamingResponse:
    engine = _require_engine()
    raw_messages: list[dict[str, Any]] = []
    has_vision = False
    has_video = False
    effective_model = request.model

    for m in request.messages:
        c = m.content
        msg_dict: dict[str, Any] = {"role": m.role}
        if m.name:
            msg_dict["name"] = m.name
        if m.tool_call_id:
            msg_dict["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            msg_dict["tool_calls"] = m.tool_calls
        if isinstance(c, list):
            text, images, videos = _video_engine.extract_video_and_image_payloads(c)
            if videos:
                has_video = True
                text = _video_engine.format_video_prompt(text, videos)
            elif images:
                has_vision = True
                text = _vision_engine.format_vision_prompt(text, len(images))
            msg_dict["content"] = text
        elif c is None:
            msg_dict["content"] = ""
        else:
            msg_dict["content"] = str(c)
        raw_messages.append(msg_dict)

    if has_video and request.model == DEFAULT_PRIMARY_MODEL:
        effective_model = DEFAULT_VIDEO_UNDERSTANDING_MODEL
    elif has_vision and request.model == DEFAULT_PRIMARY_MODEL:
        effective_model = DEFAULT_VISION_MODEL

    # Skill 63 Cyber Sentinel Firewall check on user/tool turns
    security_report: dict[str, Any] = {"threat_level": "clean", "matched_rules": [], "sanitized": False}
    for idx, msg_dict in enumerate(raw_messages):
        if msg_dict.get("role") in ("user", "tool") and isinstance(msg_dict.get("content"), str):
            insp = _firewall.inspect(msg_dict["content"])
            if not insp.safe:
                security_report["matched_rules"] = sorted(
                    set(security_report["matched_rules"]) | set(insp.matched_rules)
                )
                security_report["sanitized"] = True
                if insp.threat_level == "malicious":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": f"Blocked by Cyber Sentinel Firewall: {', '.join(insp.matched_rules)}",
                                "type": "cyber_sentinel_firewall_block",
                            }
                        },
                    )
                security_report["threat_level"] = "suspicious"
                raw_messages[idx] = {**msg_dict, "content": insp.sanitized_prompt}

    schema_def: dict[str, Any] | None = None
    if request.response_format:
        rf_type = request.response_format.get("type")
        if rf_type in ("json_object", "json_schema"):
            schema_def = request.response_format.get("schema") or (
                (request.response_format.get("json_schema") or {}).get("schema")
            )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    chat_cache_key: str | None = None
    if _cache.is_cacheable(
        temperature=request.temperature,
        stream=request.stream,
        cache_bypass=not request.use_cache,
    ):
        chat_cache_key = _cache.build_key(
            "chat_completions",
            {
                "model": effective_model,
                "messages": raw_messages,
                "tools": request.tools,
                "tool_choice": request.tool_choice,
                "schema_def": schema_def,
                "response_format": request.response_format,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "task_type": request.task_type,
                "apply_skill81": request.apply_skill81,
                "guard": request.guard,
            },
        )
        cached_chat = _cache.get(chat_cache_key)
        if cached_chat is not None:
            cached_chat["id"] = request_id
            cached_chat["cached"] = True
            return cached_chat

    use_native_chat = (
        hasattr(engine.model, "create_chat_completion")
        and callable(engine.model.create_chat_completion)
    )

    if use_native_chat and not request.stream:
        chat_messages = raw_messages
        if request.apply_skill81 and hasattr(engine, "ceiling_extractor"):
            chat_messages = engine.ceiling_extractor.extract(
                chat_messages,
                model_profile={"model_id": effective_model},
                task_type=request.task_type,
            )
        n_ctx = getattr(engine, "n_ctx", 8192) or 8192
        approx_prompt_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in chat_messages) + (
            len(json.dumps(request.tools, separators=(",", ":"), ensure_ascii=False)) if request.tools else 0
        )
        approx_prompt_tokens = max(1, approx_prompt_chars // 3)
        safe_max_tokens = max(64, min(request.max_tokens, max(64, n_ctx - approx_prompt_tokens)))

        chat_kwargs: dict[str, Any] = {
            "messages": chat_messages,
            "temperature": request.temperature,
            "max_tokens": safe_max_tokens,
            "repeat_penalty": 1.05,
        }
        if request.tools:
            chat_kwargs["tools"] = request.tools
            if request.tool_choice is not None:
                chat_kwargs["tool_choice"] = request.tool_choice
        if schema_def and engine._supports_grammar(engine.model.create_chat_completion):
            grammar = GrammarGuard.to_llama_grammar(schema_def)
            if grammar is not None:
                chat_kwargs["grammar"] = grammar

        with _admission.lease(request_id, estimated_tokens=safe_max_tokens) as admitted:
            if not admitted:
                raise HTTPException(
                    status_code=429,
                    detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
                )
            try:
                native_res = await _run_inference_sync(engine.model.create_chat_completion, **chat_kwargs)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"message": str(exc), "type": "server_error"}},
                ) from exc

        choice_msg = (native_res.get("choices") or [{}])[0].get("message") or {}
        raw_text = str(choice_msg.get("content") or "")
        clean_text, reasoning_content = PromptTemplateEngine.parse_reasoning(raw_text)

        tool_calls = list(choice_msg.get("tool_calls") or [])
        if not tool_calls and (request.tools or "<tool_call>" in raw_text):
            tool_calls = ToolCallingEngine.parse_tool_calls(clean_text or raw_text)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": clean_text}
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        finish_reason = "stop"
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        usage = native_res.get("usage") or {}
        p_tokens = int(usage.get("prompt_tokens") or approx_prompt_tokens)
        c_tokens = int(usage.get("completion_tokens") or _estimate_tokens(raw_text))
        response_payload: dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": effective_model,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_msg,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": p_tokens + c_tokens,
            },
            "security": security_report,
        }
        if schema_def or (request.response_format and request.response_format.get("type") == "json_object"):
            response_payload["schema_validation"] = GrammarGuard.validate(clean_text, schema=schema_def)
        if request.guard:
            user_prompt = str(request.messages[-1].content or "") if request.messages else ""
            response_payload["guard"] = _guard.check(clean_text, user_prompt)
        if chat_cache_key is not None:
            _cache.put(chat_cache_key, response_payload, tokens_saved=p_tokens + c_tokens)
        return response_payload

    if request.stream and hasattr(engine, "chat_stream"):
        if not _admission.admit_or_reject(request_id, estimated_tokens=request.max_tokens):
            raise HTTPException(
                status_code=429,
                detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
            )
        try:
            delta_stream = engine.chat_stream(
                raw_messages,
                task_type=request.task_type,
                apply_skill81=request.apply_skill81,
                tools=request.tools,
                schema=schema_def,
                security_scan=False,  # already scanned above
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            def _guarded_chat_stream() -> Iterator[str]:
                try:
                    with _inference_lock:
                        yield from _sse_deltas(delta_stream, request_id, effective_model)
                finally:
                    _admission.release()

            return StreamingResponse(
                _guarded_chat_stream(),
                media_type="text/event-stream",
            )
        except ValueError as exc:
            _admission.release()
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
            ) from exc

    # Delegate to full BIEEngine.chat when available (handles native templates, GBNF, reasoning)
    if not request.stream and hasattr(engine, "chat") and callable(engine.chat):
        with _admission.lease(request_id, estimated_tokens=request.max_tokens) as admitted:
            if not admitted:
                raise HTTPException(
                    status_code=429,
                    detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
                )
            try:
                chat_res = await _run_inference_sync(
                    engine.chat,
                    raw_messages,
                    task_type=request.task_type,
                    apply_skill81=request.apply_skill81,
                    guard=request.guard,
                    tools=request.tools,
                    schema=schema_def,
                    security_scan=False,  # already scanned above
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                chat_res["model"] = effective_model
                chat_res["security"] = security_report
                if chat_cache_key is not None:
                    u = chat_res.get("usage", {})
                    _cache.put(chat_cache_key, chat_res, tokens_saved=int(u.get("total_tokens", 0)))
                return chat_res
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"message": str(exc), "type": "server_error"}},
                ) from exc

    # Tool calling injection fallback for minimal mock engines without .chat()
    if request.tools:
        raw_messages = ToolCallingEngine.inject_tools(raw_messages, request.tools)

    if request.apply_skill81 and hasattr(engine, "ceiling_extractor"):
        processed_messages = engine.ceiling_extractor.extract(
            raw_messages,
            model_profile={"model_id": effective_model},
            task_type=request.task_type,
        )
    else:
        processed_messages = raw_messages

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in processed_messages)
    options = {"temperature": request.temperature, "max_tokens": request.max_tokens}

    if request.stream:
        if not _admission.admit_or_reject(request_id, estimated_tokens=request.max_tokens):
            raise HTTPException(
                status_code=429,
                detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
            )

        def _guarded_fallback_stream() -> Iterator[str]:
            try:
                with _inference_lock:
                    yield from _sse(engine.stream(prompt, **options), request_id, effective_model)
            finally:
                _admission.release()

        return StreamingResponse(_guarded_fallback_stream(), media_type="text/event-stream")

    with _admission.lease(request_id, estimated_tokens=request.max_tokens) as admitted:
        if not admitted:
            raise HTTPException(
                status_code=429,
                detail={"error": {"message": "Inference queue saturated.", "type": "rate_limit_exceeded"}},
            )
        try:
            if schema_def is not None:
                result = await _run_inference_sync(engine.complete, prompt, schema=schema_def, **options)
            else:
                result = await _run_inference_sync(engine.complete, prompt, **options)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(exc), "type": "server_error"}},
            ) from exc
    raw_text = (
        result.get("choices", [{}])[0].get("text", "")
        if isinstance(result, dict)
        else str(result)
    )
    clean_text, reasoning_content = PromptTemplateEngine.parse_reasoning(raw_text)

    p_tokens = _estimate_tokens(prompt)
    c_tokens = _estimate_tokens(raw_text)

    # Tool calling detection
    tool_calls = (
        ToolCallingEngine.parse_tool_calls(clean_text)
        if (request.tools or "<tool_call>" in raw_text)
        else []
    )
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": clean_text}
    if reasoning_content:
        assistant_msg["reasoning_content"] = reasoning_content
    finish_reason = "stop"

    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    response_payload: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": effective_model,
        "choices": [
            {
                "index": 0,
                "message": assistant_msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        },
        "security": security_report,
    }

    if schema_def or (request.response_format and request.response_format.get("type") == "json_object"):
        response_payload["schema_validation"] = GrammarGuard.validate(clean_text, schema=schema_def)

    if request.guard:
        user_prompt = request.messages[-1].content if request.messages else prompt
        response_payload["guard"] = _guard.check(clean_text, user_prompt)

    return response_payload


def _convert_anthropic_to_openai_messages(request: AnthropicMessagesRequest) -> list[ChatMessage]:
    converted: list[ChatMessage] = []
    if request.system:
        if isinstance(request.system, str):
            converted.append(ChatMessage(role="system", content=request.system))
        elif isinstance(request.system, list):
            sys_txt = "\n".join(
                str(b.get("text", "")) for b in request.system if isinstance(b, dict) and b.get("text")
            )
            if sys_txt:
                converted.append(ChatMessage(role="system", content=sys_txt))

    for msg in request.messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            converted.append(ChatMessage(role=role, content=content))
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            multimodal_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[ChatMessage] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    txt = str(block.get("text", ""))
                    text_parts.append(txt)
                    multimodal_parts.append({"type": "text", "text": txt})
                elif btype == "image":
                    src = block.get("source") or {}
                    if src.get("type") == "base64" and src.get("data"):
                        media_type = src.get("media_type", "image/png")
                        multimodal_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{src['data']}"},
                            }
                        )
                    elif src.get("url"):
                        multimodal_parts.append({"type": "image_url", "image_url": {"url": src["url"]}})
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name", "unknown"),
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
                elif btype == "tool_result":
                    tr_content = block.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = "\n".join(
                            str(item.get("text", "")) for item in tr_content if isinstance(item, dict)
                        )
                    tool_results.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=block.get("tool_use_id"),
                            content=str(tr_content),
                        )
                    )

            has_images = any(p.get("type") == "image_url" for p in multimodal_parts)
            if tool_calls or text_parts or has_images:
                converted.append(
                    ChatMessage(
                        role=role,
                        content=multimodal_parts if has_images else ("\n".join(text_parts) if text_parts else None),
                        tool_calls=tool_calls or None,
                    )
                )
            converted.extend(tool_results)

    return converted


def _anthropic_sse_from_openai(
    openai_payload: dict[str, Any], message_id: str, model: str
) -> Iterator[str]:
    usage = openai_payload.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    choice_msg = (openai_payload.get("choices") or [{}])[0].get("message") or {}
    text = str(choice_msg.get("content") or "")

    start_evt = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(start_evt)}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    if text:
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tok}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@app.post("/v1/messages", response_model=None)
async def anthropic_messages(request: AnthropicMessagesRequest) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API compatibility endpoint for Claude-compatible local agents."""
    openai_messages = _convert_anthropic_to_openai_messages(request)
    if not openai_messages:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "messages must contain at least one valid turn", "type": "invalid_request_error"}},
        )

    openai_tools: list[dict[str, Any]] | None = None
    if request.tools:
        openai_tools = []
        for tool in request.tools:
            if "function" in tool:
                openai_tools.append(tool)
            else:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", "tool"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                        },
                    }
                )

    oai_req = ChatCompletionRequest(
        model=request.model,
        messages=openai_messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=False,
        apply_skill81=request.apply_skill81,
        tools=openai_tools,
    )
    oai_res = await chat_completions(oai_req)
    assert isinstance(oai_res, dict)

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    if request.stream:
        return StreamingResponse(
            _anthropic_sse_from_openai(oai_res, message_id, oai_res.get("model", request.model)),
            media_type="text/event-stream",
        )

    choice = (oai_res.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict[str, Any]] = []
    if msg.get("reasoning_content"):
        content_blocks.append({"type": "thinking", "thinking": msg["reasoning_content"]})
    if msg.get("content"):
        content_blocks.append({"type": "text", "text": msg["content"]})

    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments", "{}")
        try:
            parsed_input = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed_input = {"raw": raw_args}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                "name": fn.get("name", "unknown"),
                "input": parsed_input,
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = oai_res.get("usage") or {}
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": oai_res.get("model", request.model),
        "content": content_blocks,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
        "cached": bool(oai_res.get("cached", False)),
    }
