"""Dual-Model Orchestrator for BIE.

Manages cooperative dispatch between the Primary Sovereign Coder (Cyber-Tiel-Coder-35B)
and the Secondary Vision Model (Gemma-4-26B).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import BIEEngine

from .vision import VisionEngine
from .vram_arbiter import VRAMArbiter


class DualModelOrchestrator:
    """Coordinates execution between default Coder and Vision models."""

    DEFAULT_PRIMARY_MODEL: str = "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    DEFAULT_VISION_MODEL: str = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

    def __init__(
        self,
        primary_model: str | None = None,
        vision_model: str | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path("bie/models")
        self.primary_model_name = primary_model or self.DEFAULT_PRIMARY_MODEL
        self.vision_model_name = vision_model or self.DEFAULT_VISION_MODEL

        self.vision_engine = VisionEngine()
        self.arbiter = VRAMArbiter()

        # Engine handles (lazy loaded or injected)
        self.primary_engine: BIEEngine | None = None
        self.vision_engine_handle: BIEEngine | None = None

    def resolve_target_model(self, messages: list[dict[str, Any]] | str) -> str:
        """Inspect prompt content and choose between primary coder and secondary vision model."""
        if isinstance(messages, str):
            if "[IMAGE_" in messages or "data:image/" in messages:
                return self.vision_model_name
            return self.primary_model_name

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        return self.vision_model_name
            elif isinstance(content, str):
                if "[IMAGE_" in content or "data:image/" in content:
                    return self.vision_model_name

        return self.primary_model_name

    def dispatch(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route request to appropriate engine based on multimodal inspection."""
        target_model = self.resolve_target_model(messages)
        is_vision = target_model == self.vision_model_name

        engine = self.vision_engine_handle if is_vision else self.primary_engine

        # If engine handle not active, return structured routing plan for caller
        if engine is None:
            return {
                "dispatched_model": target_model,
                "is_vision": is_vision,
                "plan": self.arbiter.compute_plan("vision_exclusive" if is_vision else "primary_exclusive"),
                "status": "routed",
            }

        return engine.chat(messages, **kwargs)
