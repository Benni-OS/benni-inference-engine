"""OpenAI-compatible serving layer for the BIE runtime."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bie.core.embeddings import EmbeddingsEngine
from bie.core.engine import BIEEngine
from bie.core.grammar_guard import GrammarGuard
from bie.core.vision import VisionEngine
from bie.core.vram_arbiter import VRAMArbiter
from bie.serve.tool_caller import ToolCallingEngine
from bie.skill81.hallucination_guard import HallucinationGuard

VERSION = "0.4.0"
DEFAULT_PRIMARY_MODEL = "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
DEFAULT_VISION_MODEL = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

app = FastAPI(title="benni-inference-engine", version=VERSION)
_engine: BIEEngine | None = None
_guard = HallucinationGuard()
_fallback_embeddings = EmbeddingsEngine()
_vision_engine = VisionEngine()
_arbiter = VRAMArbiter()


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | list[Any]


class CompletionRequest(BaseModel):
    model: str = DEFAULT_PRIMARY_MODEL
    prompt: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False
    schema_constraint: dict[str, Any] | None = None


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


class EmbeddingRequest(BaseModel):
    model: str = DEFAULT_PRIMARY_MODEL
    input: str | list[str]


class GuardCheckRequest(BaseModel):
    prompt: str
    output: str


def configure_engine(engine: BIEEngine | None) -> None:
    """Inject or clear the runtime engine; intended for startup and tests."""
    global _engine
    _engine = engine


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
        ],
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
    }


@app.get("/v1/bie/telemetry")
async def bie_telemetry() -> dict[str, Any]:
    """Expose real-time performance telemetry, percentiles, and expert heatmap."""
    engine = _require_engine()
    telemetry_eng = getattr(engine, "telemetry", None)
    if telemetry_eng is not None:
        summary = telemetry_eng.get_summary()
        heatmap = telemetry_eng.render_ascii_heatmap()
        return {**summary, "ascii_heatmap": heatmap}
    return {"uptime_seconds": 0.0, "total_requests_recorded": 0}


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
        return StreamingResponse(
            _sse(engine.stream(request.prompt, **options), request_id, request.model),
            media_type="text/event-stream",
        )
    result = engine.complete(request.prompt, **options)
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
    return {
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


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any] | StreamingResponse:
    engine = _require_engine()
    raw_messages: list[dict[str, Any]] = []
    has_vision = False
    effective_model = request.model

    for m in request.messages:
        c = m.content
        if isinstance(c, list):
            has_vision = True
            text, images = _vision_engine.extract_image_payloads(c)
            if images:
                text = _vision_engine.format_vision_prompt(text, len(images))
            raw_messages.append({"role": m.role, "content": text})
        else:
            raw_messages.append({"role": m.role, "content": str(c)})

    if has_vision and request.model == DEFAULT_PRIMARY_MODEL:
        effective_model = DEFAULT_VISION_MODEL

    # Tool calling injection if requested
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
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    options = {"temperature": request.temperature, "max_tokens": request.max_tokens}

    if request.stream:
        return StreamingResponse(
            _sse(engine.stream(prompt, **options), request_id, effective_model),
            media_type="text/event-stream",
        )

    result = engine.complete(prompt, **options)
    text = (
        result.get("choices", [{}])[0].get("text", "")
        if isinstance(result, dict)
        else str(result)
    )

    p_tokens = _estimate_tokens(prompt)
    c_tokens = _estimate_tokens(text)

    # Tool calling detection
    tool_calls = ToolCallingEngine.parse_tool_calls(text) if (request.tools or "<tool_call>" in text) else []
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
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
    }

    if request.response_format and request.response_format.get("type") == "json_object":
        schema_def = request.response_format.get("schema")
        response_payload["schema_validation"] = GrammarGuard.validate(text, schema=schema_def)

    if request.guard:
        user_prompt = request.messages[-1].content if request.messages else prompt
        response_payload["guard"] = _guard.check(text, user_prompt)

    return response_payload
