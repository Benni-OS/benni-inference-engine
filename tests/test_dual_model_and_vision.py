"""Unit tests for Wave 4: Dual-Model Architecture, Vision Engine, and VRAM Arbiter."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from bie.core.dual_model import DualModelOrchestrator
from bie.core.engine import BIEEngine
from bie.core.router import TaskRouter
from bie.core.vision import ImagePayload, VisionEngine
from bie.core.vram_arbiter import AllocationPlan, VRAMArbiter
from bie.serve.mcp_bridge import BIETool


def test_vision_engine_extract_base64(tmp_path: Path) -> None:
    engine = VisionEngine()
    b64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    message_content = [
        {"type": "text", "text": "Descreva este pixel"},
        {"type": "image_url", "image_url": {"url": b64_img}},
    ]
    prompt, images = engine.extract_image_payloads(message_content)
    assert prompt == "Descreva este pixel"
    assert len(images) == 1
    assert images[0].source_type == "base64"
    assert images[0].mime_type == "image/png"
    assert images[0].byte_size > 0


def test_vision_engine_extract_local_file(tmp_path: Path) -> None:
    sample_img = tmp_path / "test.jpg"
    sample_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    engine = VisionEngine()
    message_content = [
        {"type": "text", "text": "Inspecione imagem local"},
        {"type": "image_url", "image_url": {"url": str(sample_img)}},
    ]
    prompt, images = engine.extract_image_payloads(message_content)
    assert prompt == "Inspecione imagem local"
    assert len(images) == 1
    assert images[0].source_type == "local_file"
    assert images[0].mime_type == "image/jpeg"
    assert images[0].byte_size == 36


def test_vision_engine_format_prompt() -> None:
    res = VisionEngine.format_vision_prompt("O que é isso?", 2)
    assert "[IMAGE_1]" in res
    assert "[IMAGE_2]" in res
    assert "O que é isso?" in res


def test_vram_arbiter_exclusive_plans() -> None:
    arbiter = VRAMArbiter(total_vram_gb=8.0)
    p_plan = arbiter.compute_plan("primary_exclusive")
    assert p_plan.mode == "primary_exclusive"
    assert p_plan.primary_ngl > 20
    assert p_plan.vision_ngl == 0

    v_plan = arbiter.compute_plan("vision_exclusive")
    assert v_plan.mode == "vision_exclusive"
    assert v_plan.primary_ngl == 0
    assert v_plan.vision_ngl > 20


def test_vram_arbiter_cooperative_split() -> None:
    arbiter = VRAMArbiter(total_vram_gb=8.0)
    c_plan = arbiter.compute_plan("cooperative_split")
    assert c_plan.mode == "cooperative_split"
    assert c_plan.primary_ngl > 0
    assert c_plan.vision_ngl > 0
    assert c_plan.total_vram_allocated_gb <= 8.0


def test_dual_model_orchestrator_resolution() -> None:
    orch = DualModelOrchestrator()
    assert orch.primary_model_name == "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    assert orch.vision_model_name == "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

    # Pure text resolves to primary coder
    t_msgs = [{"role": "user", "content": "Escreva uma função em Rust"}]
    assert orch.resolve_target_model(t_msgs) == orch.primary_model_name

    # Vision payload resolves to Gemma-4 vision
    v_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explique o diagrama"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    assert orch.resolve_target_model(v_msgs) == orch.vision_model_name


def test_task_router_vision_routing() -> None:
    router = TaskRouter()
    decision = router.route("Analise o diagrama da arquitetura [IMAGE_1]")
    assert decision.selected_model == "gemma-4-26b-vision"

    code_decision = router.route("Escreva um parser zero-copy em C++")
    assert code_decision.selected_model == "cyber-tiel-coder-35b"


def test_bie_engine_defaults_and_vision_method() -> None:
    engine = BIEEngine()
    assert engine.model_name == "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    assert engine.vision_engine is not None
    assert engine.arbiter is not None
    assert engine.dual_orchestrator is not None

    # Mock model call to test vision convenience method
    class MockModel:
        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert "[IMAGE_1]" in prompt
            return {"choices": [{"text": "Mock vision output"}]}

    engine.model = MockModel()
    res = engine.vision(
        b"raw-image-bytes",
        prompt="Descreva a foto",
    )
    assert res["choices"][0]["message"]["content"] == "Mock vision output"


def test_mcp_bridge_vram_arbiter() -> None:
    plan = BIETool.vram_arbiter_plan("cooperative_split")
    assert plan["mode"] == "cooperative_split"
    assert plan["primary_ngl"] > 0
    assert plan["vision_ngl"] > 0
