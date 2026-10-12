"""Cooperative VRAM Arbiter for BIE.

Manages GPU layer allocation between the Primary Coder model (Cyber-Tiel-Coder-35B)
and the Secondary Vision model (Gemma-4-26B) on consumer 8GB GPUs without OOM.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Literal

ArbiterMode = Literal["primary_exclusive", "vision_exclusive", "cooperative_split"]


@dataclass
class AllocationPlan:
    mode: ArbiterMode
    primary_ngl: int
    vision_ngl: int
    primary_vram_gb: float
    vision_vram_gb: float
    total_vram_allocated_gb: float
    available_vram_gb: float


class VRAMArbiter:
    """Arbitrates GPU layer placement between Coder and Vision models."""

    def __init__(
        self,
        total_vram_gb: float | None = None,
        primary_dense_gb: float = 0.17,
        primary_total_layers: int = 41,
        vision_dense_gb: float = 0.20,
        vision_total_layers: int = 30,
        safety_reserve_gb: float = 1.2,
    ) -> None:
        self.total_vram_gb = total_vram_gb or self._detect_vram()
        self.primary_dense_gb = primary_dense_gb
        self.primary_total_layers = primary_total_layers
        self.vision_dense_gb = vision_dense_gb
        self.vision_total_layers = vision_total_layers
        self.safety_reserve_gb = safety_reserve_gb

    @staticmethod
    def _detect_vram() -> float:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return max(0.0, float(res.stdout.strip().splitlines()[0]) / 1024.0)
        except Exception:
            return 8.0  # Default to 8GB baseline

    def compute_plan(self, mode: ArbiterMode = "primary_exclusive") -> AllocationPlan:
        """Compute optimal layer allocation for the chosen execution mode."""
        usable = max(0.0, (self.total_vram_gb - self.safety_reserve_gb) * 0.90)

        if mode == "primary_exclusive":
            ngl_primary = min(self.primary_total_layers, int(usable / self.primary_dense_gb))
            return AllocationPlan(
                mode="primary_exclusive",
                primary_ngl=ngl_primary,
                vision_ngl=0,
                primary_vram_gb=round(ngl_primary * self.primary_dense_gb, 2),
                vision_vram_gb=0.0,
                total_vram_allocated_gb=round(ngl_primary * self.primary_dense_gb, 2),
                available_vram_gb=round(self.total_vram_gb, 2),
            )

        if mode == "vision_exclusive":
            ngl_vision = min(self.vision_total_layers, int(usable / self.vision_dense_gb))
            return AllocationPlan(
                mode="vision_exclusive",
                primary_ngl=0,
                vision_ngl=ngl_vision,
                primary_vram_gb=0.0,
                vision_vram_gb=round(ngl_vision * self.vision_dense_gb, 2),
                total_vram_allocated_gb=round(ngl_vision * self.vision_dense_gb, 2),
                available_vram_gb=round(self.total_vram_gb, 2),
            )

        # cooperative_split: 60% budget to Coder, 40% budget to Vision
        coder_budget = usable * 0.60
        vision_budget = usable * 0.40

        ngl_primary = min(self.primary_total_layers, int(coder_budget / self.primary_dense_gb))
        ngl_vision = min(self.vision_total_layers, int(vision_budget / self.vision_dense_gb))

        p_gb = round(ngl_primary * self.primary_dense_gb, 2)
        v_gb = round(ngl_vision * self.vision_dense_gb, 2)

        return AllocationPlan(
            mode="cooperative_split",
            primary_ngl=ngl_primary,
            vision_ngl=ngl_vision,
            primary_vram_gb=p_gb,
            vision_vram_gb=v_gb,
            total_vram_allocated_gb=round(p_gb + v_gb, 2),
            available_vram_gb=round(self.total_vram_gb, 2),
        )
