"""Deterministic hardware-aware planning for local MoE inference."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any

import psutil

from .scheduler_quant_aware import derive_from_gguf


@dataclass(frozen=True)
class ModelProfile:
    """Hardware-relevant model facts; values must come from model metadata."""

    total_layers: int
    dense_layer_gb: float
    expert_cache_gb_per_slot: float
    expert_slots: int
    moe_offload_pattern: str = r"\.ffn_.*_exps\.=CPU"


@dataclass(frozen=True)
class HardwareBudget:
    """Measured memory and safety reserves used by the planner."""

    vram_available_gb: float
    ram_available_gb: float
    vram_safety_margin: float = 0.85
    vram_reserve_gb: float = 1.0
    ram_reserve_gb: float = 2.0
    kv_reserve_gb: float = 1.5


DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "qwen3-30b-a3b": ModelProfile(
        total_layers=48,
        dense_layer_gb=0.16,
        expert_cache_gb_per_slot=0.12,
        expert_slots=128,
    ),
    "qwen3-8b": ModelProfile(
        total_layers=36,
        dense_layer_gb=0.14,
        expert_cache_gb_per_slot=0.0,
        expert_slots=0,
    ),
    "qwen3-235b-a22b": ModelProfile(
        total_layers=96,
        dense_layer_gb=0.35,
        expert_cache_gb_per_slot=0.25,
        expert_slots=256,
    ),
    "kimi-k2-32b": ModelProfile(
        total_layers=64,
        dense_layer_gb=0.22,
        expert_cache_gb_per_slot=0.0,
        expert_slots=0,
    ),
    "deepseek-v4": ModelProfile(
        total_layers=61,
        dense_layer_gb=0.40,
        expert_cache_gb_per_slot=0.30,
        expert_slots=256,
    ),
    "cyber-tiel-coder-35b": ModelProfile(
        total_layers=41,
        dense_layer_gb=0.17,
        expert_cache_gb_per_slot=0.12,
        expert_slots=128,
    ),
    "gemma-4-26b": ModelProfile(
        total_layers=30,
        dense_layer_gb=0.20,
        expert_cache_gb_per_slot=0.10,
        expert_slots=64,
    ),
}


class MoEScheduler:
    """Plan GPU/CPU placement without silently overcommitting hardware."""

    def __init__(
        self,
        model_name: str,
        *,
        profile: ModelProfile | None = None,
        budget: HardwareBudget | None = None,
    ) -> None:
        self.model_name = model_name
        self.profile = profile or self._profile_for(model_name)
        self._budget = budget

    def _load_registry_profile(self, key: str) -> ModelProfile | None:
        """Search profiles in bie/models/profiles and bie/models/registry.json."""
        root_dir = Path(__file__).parents[1]
        profiles_dir = root_dir / "models" / "profiles"

        # Check json files in profiles directory
        if profiles_dir.is_dir():
            for p_file in profiles_dir.glob("*.json"):
                stem = p_file.stem.lower()
                if stem in key or key in stem:
                    try:
                        data = json.loads(p_file.read_text(encoding="utf-8"))
                        moe = data.get("moe_offload", {})
                        if moe:
                            return ModelProfile(
                                total_layers=data.get("total_layers", 48),
                                dense_layer_gb=data.get("dense_layer_gb", 0.16),
                                expert_cache_gb_per_slot=data.get("expert_cache_gb_per_slot", 0.12),
                                expert_slots=data.get("expert_slots", 128),
                                moe_offload_pattern=moe.get("pattern", r"\.ffn_.*_exps\.=CPU"),
                            )
                    except (json.JSONDecodeError, OSError):
                        pass

        # Check registry.json
        reg_file = root_dir / "models" / "registry.json"
        if reg_file.is_file():
            try:
                reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
                for entry in reg_data.get("models", []):
                    entry_id = str(entry.get("id", "")).lower()
                    if entry_id in key or key in entry_id:
                        is_moe = entry.get("type") == "moe"
                        return ModelProfile(
                            total_layers=48 if is_moe else 36,
                            dense_layer_gb=0.18 if is_moe else 0.14,
                            expert_cache_gb_per_slot=0.12 if is_moe else 0.0,
                            expert_slots=128 if is_moe else 0,
                        )
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _profile_from_gguf_file(self, model_name: str) -> ModelProfile | None:
        """Attempt to read metadata directly from GGUF file if available."""
        candidate = Path(model_name)
        if not candidate.is_file():
            model_dir = Path(os.getenv("BIE_MODEL_DIR", "models"))
            test_paths = [
                model_dir / model_name,
                model_dir / f"{model_name}.gguf",
                Path(__file__).parents[1] / "models" / model_name,
                Path(__file__).parents[1] / "models" / f"{model_name}.gguf",
            ]
            candidate = next((p for p in test_paths if p.is_file()), candidate)

        if candidate.is_file():
            try:
                gguf_info = derive_from_gguf(candidate)
                layers = int(gguf_info.get("block_count", 0))
                dense = float(gguf_info.get("dense_layer_gb", 0.14))
                if layers > 0:
                    is_moe = "expert" in candidate.name.lower() or "a3b" in candidate.name.lower() or "a4b" in candidate.name.lower() or "a22b" in candidate.name.lower()
                    return ModelProfile(
                        total_layers=layers,
                        dense_layer_gb=dense,
                        expert_cache_gb_per_slot=0.12 if is_moe else 0.0,
                        expert_slots=128 if is_moe else 0,
                    )
            except Exception:
                pass
        return None

    def _profile_for(self, model_name: str) -> ModelProfile:
        key = model_name.lower().removesuffix(".gguf")
        for name, profile in DEFAULT_PROFILES.items():
            if name in key:
                return profile

        # Search dynamic profiles
        registry_profile = self._load_registry_profile(key)
        if registry_profile is not None:
            return registry_profile

        # GGUF file metadata inspection fallback
        file_profile = self._profile_from_gguf_file(model_name)
        if file_profile is not None:
            return file_profile

        return ModelProfile(
            total_layers=0,
            dense_layer_gb=0.0,
            expert_cache_gb_per_slot=0.0,
            expert_slots=0,
        )

    def detect_vram(self) -> float:
        """Return free VRAM in GiB; return zero when nvidia-smi is unavailable."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            value = result.stdout.strip().splitlines()[0]
            return max(0.0, float(value) / 1024.0)
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return 0.0

    def detect_ram(self) -> float:
        """Return available system RAM in GiB."""
        return max(0.0, psutil.virtual_memory().available / (1024**3))

    def _hardware_budget(self) -> HardwareBudget:
        return self._budget or HardwareBudget(
            vram_available_gb=self.detect_vram(),
            ram_available_gb=self.detect_ram(),
        )

    @staticmethod
    def _env_int(name: str) -> int | None:
        value = os.getenv(name)
        if value is None:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        return max(0, parsed)

    def compute(self) -> dict[str, Any]:
        """Return an auditable, conservative placement plan."""
        budget = self._hardware_budget()
        warnings: list[str] = []
        gpu_override = self._env_int("BIE_GPU_LAYERS")
        cache_override = self._env_int("BIE_EXPERT_CACHE_SLOTS")

        if self.profile.total_layers <= 0:
            warnings.append("unknown_model_profile: GPU placement disabled")
        usable_vram = max(
            0.0,
            (budget.vram_available_gb - budget.vram_reserve_gb - budget.kv_reserve_gb)
            * budget.vram_safety_margin,
        )
        calculated_ngl = (
            min(self.profile.total_layers, floor(usable_vram / self.profile.dense_layer_gb))
            if self.profile.dense_layer_gb > 0
            else 0
        )
        ngl = min(self.profile.total_layers, gpu_override) if gpu_override is not None else calculated_ngl
        if budget.vram_available_gb <= 0 and ngl > 0:
            warnings.append("no_cuda_vram: GPU placement forced to zero")
            ngl = 0

        usable_ram = max(0.0, budget.ram_available_gb - budget.ram_reserve_gb)
        calculated_slots = (
            floor(usable_ram / self.profile.expert_cache_gb_per_slot)
            if self.profile.expert_cache_gb_per_slot > 0
            else 0
        )
        slots = min(self.profile.expert_slots, calculated_slots)
        if cache_override is not None:
            slots = min(self.profile.expert_slots, cache_override)

        return {
            "vram_available_gb": round(budget.vram_available_gb, 2),
            "ram_available_gb": round(budget.ram_available_gb, 2),
            "ngl": max(0, ngl),
            "moe_offload_pattern": self.profile.moe_offload_pattern,
            "expert_cache_slots": max(0, slots),
            "model_profile": "known" if self.profile.total_layers else "unknown",
            "planner_source": "override" if gpu_override is not None else "calculated",
            "warnings": warnings,
        }
