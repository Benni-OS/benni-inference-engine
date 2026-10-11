"""Hot Pin — adaptive expert pinning. Learns usage patterns and pins hot experts in RAM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HotPinManager:
    """Tracks expert activation frequency across inference calls.

    After warmup_tokens, identifies the top-N most-used experts per layer.
    Supports frequency decay, state export/import (.bie_hot_pin.json),
    and generation of pinning maps for MoE offloading.
    """

    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        hot_n: int = 14,
        warmup_tokens: int = 32,
        decay_factor: float = 0.95,
        state_file: str | Path | None = None,
    ) -> None:
        self.n_layers = max(1, n_layers)
        self.n_experts = max(1, n_experts)
        self.hot_n = min(self.n_experts, max(1, hot_n))
        self.warmup_tokens = warmup_tokens
        self.decay_factor = decay_factor
        self.state_file = Path(state_file) if state_file else Path(".bie_hot_pin.json")
        self.freq: list[list[float]] = [[0.0] * self.n_experts for _ in range(self.n_layers)]
        self.token_count = 0
        self.pinned = False
        self.pinned_map: dict[int, list[int]] = {}

    def record(self, layer: int, expert_ids: list[int]) -> None:
        """Record expert activations for a token in a given layer."""
        if 0 <= layer < self.n_layers:
            for eid in expert_ids:
                if 0 <= eid < self.n_experts:
                    self.freq[layer][eid] += 1.0
            self.token_count += 1
            if not self.pinned and self.token_count >= self.warmup_tokens:
                self._pin()

    def apply_decay(self) -> None:
        """Apply decay to prevent historical bias and adapt to shifting query topics."""
        for layer in range(self.n_layers):
            for eid in range(self.n_experts):
                self.freq[layer][eid] *= self.decay_factor

    def get_hot_experts_for_layer(self, layer: int) -> list[int]:
        """Return the top-N hot expert indices for a specific layer."""
        if not (0 <= layer < self.n_layers):
            return []
        layer_freqs = self.freq[layer]
        # Sort experts by frequency descending, break ties by expert index ascending
        ranked = sorted(range(self.n_experts), key=lambda i: (layer_freqs[i], -i), reverse=True)
        return ranked[: self.hot_n]

    def compute_all_hot_experts(self) -> dict[int, list[int]]:
        """Return top-N hot experts for all layers."""
        return {layer: self.get_hot_experts_for_layer(layer) for layer in range(self.n_layers)}

    def _pin(self) -> None:
        """Lock in current top-N experts per layer and flag as pinned."""
        self.pinned_map = self.compute_all_hot_experts()
        self.pinned = True

    def export_state(self) -> dict[str, Any]:
        """Export serializable hot-pin state."""
        return {
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "hot_n": self.hot_n,
            "warmup_tokens": self.warmup_tokens,
            "token_count": self.token_count,
            "pinned": self.pinned,
            "pinned_map": {str(k): v for k, v in self.pinned_map.items()},
            "freq": self.freq,
        }

    def save(self, target_path: str | Path | None = None) -> Path:
        """Persist pinning state to disk."""
        path = Path(target_path) if target_path else self.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_state(), f, indent=2)
        return path

    def load(self, source_path: str | Path | None = None) -> bool:
        """Load pinning state from disk."""
        path = Path(source_path) if source_path else self.state_file
        if not path.is_file():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.n_layers = data.get("n_layers", self.n_layers)
            self.n_experts = data.get("n_experts", self.n_experts)
            self.hot_n = data.get("hot_n", self.hot_n)
            self.warmup_tokens = data.get("warmup_tokens", self.warmup_tokens)
            self.token_count = data.get("token_count", 0)
            self.pinned = data.get("pinned", False)
            raw_map = data.get("pinned_map", {})
            self.pinned_map = {int(k): list(v) for k, v in raw_map.items()}
            self.freq = data.get("freq", [[0.0] * self.n_experts for _ in range(self.n_layers)])
            return True
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return False
