"""Dynamic LoRA / Adapter Manager for BIE.

Manages registration, validation, scaling, and application of fine-tuned LoRA
adapters on top of the base GGUF model without reloading base weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoRAAdapter:
    adapter_id: str
    path: Path
    scale: float = 1.0
    description: str = ""
    enabled: bool = True


class LoRAManager:
    """Catalogs and manages active LoRA adapters for inference calls."""

    def __init__(self, adapters_dir: str | Path | None = None) -> None:
        self.adapters_dir = Path(adapters_dir) if adapters_dir else Path("models/lora")
        self.registry: dict[str, LoRAAdapter] = {}

    def register(
        self,
        adapter_id: str,
        path: str | Path,
        scale: float = 1.0,
        description: str = "",
    ) -> LoRAAdapter:
        """Register a new LoRA adapter."""
        p = Path(path)
        adapter = LoRAAdapter(
            adapter_id=adapter_id,
            path=p,
            scale=max(0.0, min(2.0, scale)),
            description=description,
            enabled=True,
        )
        self.registry[adapter_id] = adapter
        return adapter

    def set_scale(self, adapter_id: str, scale: float) -> bool:
        """Update the scaling factor of an existing adapter."""
        if adapter_id in self.registry:
            self.registry[adapter_id].scale = max(0.0, min(2.0, scale))
            return True
        return False

    def set_enabled(self, adapter_id: str, enabled: bool) -> bool:
        """Enable or disable a specific adapter."""
        if adapter_id in self.registry:
            self.registry[adapter_id].enabled = enabled
            return True
        return False

    def get_active_adapters(self) -> list[LoRAAdapter]:
        """Return list of currently enabled adapters."""
        return [ad for ad in self.registry.values() if ad.enabled]

    def get_backend_lora_args(self) -> dict[str, Any]:
        """Generate constructor / inference parameters for llama.cpp."""
        active = self.get_active_adapters()
        if not active:
            return {}

        primary = active[0]
        return {
            "lora_path": str(primary.path),
            "lora_scale": primary.scale,
            "active_loras": [
                {"id": ad.adapter_id, "path": str(ad.path), "scale": ad.scale}
                for ad in active
            ],
        }

    def list_adapters(self) -> list[dict[str, Any]]:
        """Return serialized list of all registered adapters."""
        return [
            {
                "id": ad.adapter_id,
                "path": str(ad.path),
                "scale": ad.scale,
                "description": ad.description,
                "enabled": ad.enabled,
            }
            for ad in self.registry.values()
        ]
