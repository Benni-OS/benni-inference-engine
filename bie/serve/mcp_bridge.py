"""MCP Bridge — Project Benjamin and Antigravity native tool integration.

Provides direct in-memory or HTTP client access as well as a standalone
JSON-RPC 2.0 stdio server implementing the Model Context Protocol (MCP).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx

from bie.core.moe_scheduler import MoEScheduler
from bie.skill81.hallucination_guard import HallucinationGuard


class BIETool:
    """Native MCP tool and client interface for Project Benjamin and Benni OS."""

    BIE_BASE_URL: str = "http://localhost:8080"
    _in_memory_engine: Any | None = None
    _guard: HallucinationGuard = HallucinationGuard()

    DEFAULT_PRIMARY_MODEL: str = "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    DEFAULT_VISION_MODEL: str = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

    @classmethod
    def set_in_memory_engine(cls, engine: Any | None) -> None:
        """Attach a local BIEEngine instance for zero-HTTP direct in-process execution."""
        cls._in_memory_engine = engine

    @classmethod
    async def complete(
        cls,
        prompt: str,
        task_type: str = "general",
        model: str = DEFAULT_PRIMARY_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Route to local BIE engine with Skill 81 active."""
        if cls._in_memory_engine is not None and getattr(cls._in_memory_engine, "model", None) is not None:
            # Direct in-process execution
            res = cls._in_memory_engine.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            if isinstance(res, dict) and "choices" in res:
                return str(res["choices"][0].get("text", ""))
            return str(res)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{cls.BIE_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "task_type": task_type,
                    "apply_skill81": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

    @classmethod
    async def chat(
        cls,
        messages: list[dict[str, Any]],
        task_type: str = "general",
        model: str = DEFAULT_PRIMARY_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        apply_skill81: bool = True,
    ) -> dict[str, Any]:
        """Execute chat completion over HTTP or in-process."""
        if cls._in_memory_engine is not None and getattr(cls._in_memory_engine, "model", None) is not None:
            if hasattr(cls._in_memory_engine, "chat"):
                return cls._in_memory_engine.chat(
                    messages,
                    task_type=task_type,
                    apply_skill81=apply_skill81,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{cls.BIE_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "task_type": task_type,
                    "apply_skill81": apply_skill81,
                },
            )
            resp.raise_for_status()
            return resp.json()

    @classmethod
    async def vision_infer(
        cls,
        image: str,
        prompt: str = "Analyze this image in detail.",
        model: str = DEFAULT_VISION_MODEL,
    ) -> str:
        """Route multimodal image and text query to Gemma-4 vision."""
        from pathlib import Path
        url_ref = image if image.startswith("data:") or image.startswith("http") else str(Path(image).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url_ref}},
                ],
            }
        ]
        res = await cls.chat(messages, task_type="vision", model=model)
        if "choices" in res:
            return str(res["choices"][0]["message"]["content"])
        return str(res)

    @classmethod
    def vram_arbiter_plan(cls, mode: str = "cooperative_split") -> dict[str, Any]:
        from bie.core.vram_arbiter import VRAMArbiter
        plan = VRAMArbiter().compute_plan(mode)  # type: ignore
        return {
            "mode": plan.mode,
            "primary_ngl": plan.primary_ngl,
            "vision_ngl": plan.vision_ngl,
            "primary_vram_gb": plan.primary_vram_gb,
            "vision_vram_gb": plan.vision_vram_gb,
            "total_allocated_gb": plan.total_vram_allocated_gb,
            "available_vram_gb": plan.available_vram_gb,
        }

    @classmethod
    def guard_check(cls, output: str, prompt: str) -> dict[str, Any]:
        """Audit text for epistemic drift and hallucination."""
        return cls._guard.check(output, prompt)

    @classmethod
    def hardware_plan(cls, model_name: str) -> dict[str, Any]:
        """Calculate deterministic offload placement."""
        return MoEScheduler(model_name).compute()


# ---------------------------------------------------------------------------
# Standalone MCP Server implementation (JSON-RPC 2.0 stdio)
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "bie_complete",
        "description": "Generate text completions via local Benni Inference Engine.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt text"},
                "model": {"type": "string", "default": "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"},
                "task_type": {
                    "type": "string",
                    "enum": ["DELTA", "CHARLIE", "BRAVO", "ALPHA", "FOXTROT", "general"],
                    "default": "general",
                },
                "temperature": {"type": "number", "default": 0.7},
                "max_tokens": {"type": "integer", "default": 2048},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "bie_vision_infer",
        "description": "Execute multimodal image understanding using Gemma-4-26B vision model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Local path, URL, or data:image base64 URI"},
                "prompt": {"type": "string", "default": "Analyze this image in detail.", "description": "Question or analysis prompt"},
                "model": {"type": "string", "default": "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "bie_vram_arbiter",
        "description": "Calculate optimal GPU layer allocation between Coder-35B and Gemma-4-26B.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["primary_exclusive", "vision_exclusive", "cooperative_split"],
                    "default": "cooperative_split",
                    "description": "Offload allocation mode",
                },
            },
        },
    },
    {
        "name": "bie_guard_check",
        "description": "Audit an output against prompt for hallucinations, drift, and unanchored claims (SA-81D).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Original prompt or user query"},
                "output": {"type": "string", "description": "Model output text to be checked"},
            },
            "required": ["prompt", "output"],
        },
    },
    {
        "name": "bie_hardware_plan",
        "description": "Compute optimal GPU layers (ngl) and RAM expert slots for a model based on detected hardware.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model ID or GGUF name"},
            },
            "required": ["model_name"],
        },
    },
]


def run_mcp_server() -> None:
    """Run an interactive MCP JSON-RPC 2.0 server over stdin/stdout."""
    import asyncio

    async def _handle_message(req: dict[str, Any]) -> dict[str, Any] | None:
        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "benni-inference-engine", "version": "0.3.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": MCP_TOOLS},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})

            try:
                if tool_name == "bie_complete":
                    res = await BIETool.complete(
                        prompt=args["prompt"],
                        task_type=args.get("task_type", "general"),
                        model=args.get("model", BIETool.DEFAULT_PRIMARY_MODEL),
                        temperature=float(args.get("temperature", 0.7)),
                        max_tokens=int(args.get("max_tokens", 2048)),
                    )
                    content = [{"type": "text", "text": res}]
                elif tool_name == "bie_vision_infer":
                    res = await BIETool.vision_infer(
                        image=args["image"],
                        prompt=args.get("prompt", "Analyze this image in detail."),
                        model=args.get("model", BIETool.DEFAULT_VISION_MODEL),
                    )
                    content = [{"type": "text", "text": res}]
                elif tool_name == "bie_vram_arbiter":
                    res = BIETool.vram_arbiter_plan(mode=args.get("mode", "cooperative_split"))
                    content = [{"type": "text", "text": json.dumps(res, indent=2)}]
                elif tool_name == "bie_guard_check":
                    res = BIETool.guard_check(output=args["output"], prompt=args["prompt"])
                    content = [{"type": "text", "text": json.dumps(res, indent=2)}]
                elif tool_name == "bie_hardware_plan":
                    res = BIETool.hardware_plan(args["model_name"])
                    content = [{"type": "text", "text": json.dumps(res, indent=2)}]
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Method {tool_name} not found"},
                    }
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": content}}
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": str(exc)}]},
                }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unhandled method {method}"},
        }

    async def _loop():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = await _handle_message(payload)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

    try:
        asyncio.run(_loop())
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == "__main__":
    run_mcp_server()
