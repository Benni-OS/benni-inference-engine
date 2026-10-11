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

from bie.core.engine import BIEEngine
from bie.skill81.hallucination_guard import HallucinationGuard

VERSION = "0.3.0"
app = FastAPI(title="benni-inference-engine", version=VERSION)
_engine: BIEEngine | None = None
_guard = HallucinationGuard()


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str = "qwen3-30b-a3b"
    prompt: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-30b-a3b"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False
    task_type: str = "general"
    apply_skill81: bool = False
    guard: bool = False


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


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "engine": "benni-inference-engine", "version": VERSION}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    _require_engine()
    return {"status": "ready", "engine": "benni-inference-engine", "version": VERSION}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    model = _engine.model_name if _engine else "qwen3-30b-a3b"
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "benni-os"}]}


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


@app.post("/v1/bie/guard")
async def bie_guard_check(request: GuardCheckRequest) -> dict[str, Any]:
    """Audit prompt/output against epistemic drift and hallucination."""
    return _guard.check(request.output, request.prompt)


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
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.apply_skill81 and hasattr(engine, "ceiling_extractor"):
        processed_messages = engine.ceiling_extractor.extract(
            raw_messages,
            model_profile={"model_id": request.model},
            task_type=request.task_type,
        )
    else:
        processed_messages = raw_messages

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in processed_messages)
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    options = {"temperature": request.temperature, "max_tokens": request.max_tokens}

    if request.stream:
        return StreamingResponse(
            _sse(engine.stream(prompt, **options), request_id, request.model),
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
    response_payload: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        },
    }

    if request.guard:
        user_prompt = request.messages[-1].content if request.messages else prompt
        response_payload["guard"] = _guard.check(text, user_prompt)

    return response_payload
