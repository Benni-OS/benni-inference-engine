"""Quantized KV Cache Support for BIE.

Reduces KV Cache VRAM footprint by up to 70% using quantized Key-Value representations
(q4_0, q8_0, f16). Integrates directly with llama.cpp backend configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

KVQuantType = Literal["f16", "q8_0", "q4_0", "q4_1"]

# Effective bits per weight/element for KV cache representations
KV_QUANT_BITS: dict[KVQuantType, float] = {
    "f16": 16.0,
    "q8_0": 8.5,
    "q4_1": 5.0,
    "q4_0": 4.5,
}

# llama.cpp GGML type mapping integers
GGML_TYPE_MAP: dict[KVQuantType, int] = {
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q8_0": 8,
}


@dataclass(frozen=True)
class KVFootprintReport:
    quant_type: KVQuantType
    bytes_per_token: float
    mb_at_4k: float
    mb_at_8k: float
    mb_at_16k: float
    mb_at_32k: float
    vram_savings_vs_f16_pct: float


class KVQuantManager:
    """Calculates and applies Quantized KV Cache configurations."""

    def __init__(
        self,
        n_layers: int = 48,
        n_heads_kv: int = 8,
        head_dim: int = 128,
        default_quant: KVQuantType = "q4_0",
    ) -> None:
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.head_dim = head_dim
        self.active_quant = default_quant

    def bytes_per_token(self, quant: KVQuantType | None = None) -> float:
        """Calculate bytes per token for both Key and Value matrices."""
        q = quant or self.active_quant
        bits = KV_QUANT_BITS.get(q, 16.0)
        # Factor 2 accounts for Key + Value caches
        return 2.0 * self.n_layers * self.n_heads_kv * self.head_dim * (bits / 8.0)

    def compute_footprint(self, quant: KVQuantType | None = None) -> KVFootprintReport:
        """Compute full VRAM footprint metrics across context sizes."""
        q = quant or self.active_quant
        bpt = self.bytes_per_token(q)
        bpt_f16 = self.bytes_per_token("f16")
        savings = round(((bpt_f16 - bpt) / bpt_f16) * 100.0, 1)

        def to_mb(tokens: int) -> float:
            return round((bpt * tokens) / (1024 * 1024), 1)

        return KVFootprintReport(
            quant_type=q,
            bytes_per_token=round(bpt, 2),
            mb_at_4k=to_mb(4096),
            mb_at_8k=to_mb(8192),
            mb_at_16k=to_mb(16384),
            mb_at_32k=to_mb(32768),
            vram_savings_vs_f16_pct=savings,
        )

    def get_backend_params(self, quant: KVQuantType | None = None) -> dict[str, Any]:
        """Return parameters to inject into llama-cpp-python / llama-server."""
        q = quant or self.active_quant
        ggml_code = GGML_TYPE_MAP.get(q, 1)
        return {
            "type_k": ggml_code,
            "type_v": ggml_code,
            "cache_type_k": q,
            "cache_type_v": q,
        }

    def recommend_for_vram(self, target_ctx: int, available_vram_gb: float) -> KVQuantType:
        """Select the highest quality KV quant that safely fits inside available VRAM."""
        for q in ("f16", "q8_0", "q4_1", "q4_0"):
            needed_gb = (self.bytes_per_token(q) * target_ctx) / (1024**3)
            if needed_gb <= available_vram_gb:
                return q  # type: ignore
        return "q4_0"
