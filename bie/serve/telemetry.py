"""Real-Time Telemetry & MoE Expert Heatmap Engine for BIE.

Captures request latencies (P50, P90, P99), decode tokens/second, and aggregates
MoE expert activation distributions across all transformer layers.
"""

from __future__ import annotations

import statistics
import time
from typing import Any


class TelemetryEngine:
    """Collects real-time inference diagnostics and MoE heatmap statistics."""

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self.latencies_s: list[float] = []
        self.ttft_s: list[float] = []
        self.decode_rates: list[float] = []
        self.expert_heatmap: dict[int, dict[int, int]] = {}
        self.start_time = time.time()
        self.total_tokens_processed = 0

    def record_request(
        self,
        latency_s: float,
        ttft_s: float | None = None,
        decode_tok_s: float | None = None,
        tokens_count: int = 0,
    ) -> None:
        """Record metrics for an executed completion request."""
        self.latencies_s.append(latency_s)
        if len(self.latencies_s) > self.max_history:
            self.latencies_s.pop(0)

        if ttft_s is not None:
            self.ttft_s.append(ttft_s)
            if len(self.ttft_s) > self.max_history:
                self.ttft_s.pop(0)

        if decode_tok_s is not None:
            self.decode_rates.append(decode_tok_s)
            if len(self.decode_rates) > self.max_history:
                self.decode_rates.pop(0)

        self.total_tokens_processed += tokens_count

    def record_expert_activations(self, layer: int, expert_ids: list[int]) -> None:
        """Increment activation counters for specific layer experts."""
        if layer not in self.expert_heatmap:
            self.expert_heatmap[layer] = {}
        layer_dict = self.expert_heatmap[layer]
        for eid in expert_ids:
            layer_dict[eid] = layer_dict.get(eid, 0) + 1

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = min(len(sorted_v) - 1, int(len(sorted_v) * p))
        return round(sorted_v[idx], 4)

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated latency percentiles and performance metrics."""
        uptime = max(1.0, time.time() - self.start_time)
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests_recorded": len(self.latencies_s),
            "total_tokens_processed": self.total_tokens_processed,
            "overall_tokens_per_second": round(self.total_tokens_processed / uptime, 2),
            "latency_p50_s": self._percentile(self.latencies_s, 0.50),
            "latency_p90_s": self._percentile(self.latencies_s, 0.90),
            "latency_p99_s": self._percentile(self.latencies_s, 0.99),
            "mean_decode_tok_s": round(statistics.mean(self.decode_rates), 1) if self.decode_rates else 0.0,
            "mean_ttft_s": round(statistics.mean(self.ttft_s), 4) if self.ttft_s else 0.0,
        }

    def render_ascii_heatmap(self, max_layers: int = 12, max_experts: int = 16) -> str:
        """Render a compact ASCII heatmap representation of expert usage."""
        if not self.expert_heatmap:
            return "[No MoE expert activations recorded]"

        lines: list[str] = ["MoE Expert Activation Heatmap (Layers vs Experts):"]
        chars = " .:-=+*#%@"

        # Find max frequency for normalization
        max_val = max(
            (val for layer in self.expert_heatmap.values() for val in layer.values()),
            default=1,
        )

        for l in range(min(max_layers, len(self.expert_heatmap))):
            layer_map = self.expert_heatmap.get(l, {})
            row_chars = []
            for e in range(max_experts):
                count = layer_map.get(e, 0)
                intensity = int((count / max_val) * (len(chars) - 1)) if max_val > 0 else 0
                row_chars.append(chars[intensity])
            lines.append(f"L{l:02d} |{''.join(row_chars)}|")

        return "\n".join(lines)
