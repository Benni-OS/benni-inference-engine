#!/usr/bin/env python3
"""8GB-16GB Consumer GPU Zero-RAM-Spill Benchmark & KV-Cache Compression Planner.

Demonstrates how benni-inference-engine (BIE) fits 30B-35B MoE models and 32K+
context windows inside 8GB-16GB consumer GPUs without spilling KV-cache into
system RAM:
  1. KV-cache compression math (f16 vs q8_0 vs q4_1 vs q4_0)
  2. Zero-RAM-spill layer + KV budget planning across 8GB / 12GB / 16GB tiers
  3. Cooperative VRAM arbitration between Coder-35B and Vision-26B

Usage:
    python scripts/benchmark_8gb_vram.py
    python scripts/benchmark_8gb_vram.py --vram-gb 8.0 --model-vram-gb 5.5
    python scripts/benchmark_8gb_vram.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Allow running directly as `python scripts/benchmark_8gb_vram.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bie.core.kv_quant import KVQuantManager, KVQuantType
from bie.core.scheduler_quant_aware import derive_dense_layer_gb, vram_estimate_gb
from bie.core.vram_arbiter import VRAMArbiter
from bie.tools.doctor import estimate_context_limits, probe_gpu


@dataclass(frozen=True)
class TierBenchmarkRow:
    gpu_vram_gb: float
    model_vram_gb: float
    safety_reserve_gb: float
    kv_budget_gb: float
    fp16_max_ctx: int
    q8_0_max_ctx: int
    q4_0_max_ctx: int
    fp16_32k_spill_gb: float
    q4_0_32k_vram_gb: float
    zero_ram_spill_at_32k_with_q4: bool


def compute_zero_spill_matrix(
    tiers_gb: tuple[float, ...] = (8.0, 12.0, 16.0),
    model_vram_gb: float = 5.5,
    safety_reserve_gb: float = 0.8,
    n_layers: int = 48,
    n_heads_kv: int = 8,
    head_dim: int = 128,
) -> list[TierBenchmarkRow]:
    """Compute zero-RAM-spill context capacity across consumer GPU VRAM tiers."""
    kv_mgr = KVQuantManager(n_layers=n_layers, n_heads_kv=n_heads_kv, head_dim=head_dim)
    fp16_32k_gb = (kv_mgr.bytes_per_token("f16") * 32768) / (1024**3)
    q4_32k_gb = (kv_mgr.bytes_per_token("q4_0") * 32768) / (1024**3)

    rows: list[TierBenchmarkRow] = []
    for vram in tiers_gb:
        estimates = {
            e.quant_type: e
            for e in estimate_context_limits(
                total_vram_gb=vram,
                model_vram_gb=model_vram_gb,
                safety_reserve_gb=safety_reserve_gb,
                n_layers=n_layers,
                n_heads_kv=n_heads_kv,
                head_dim=head_dim,
            )
        }
        kv_budget = round(max(0.0, vram - model_vram_gb - safety_reserve_gb), 2)
        fp16_spill = round(max(0.0, fp16_32k_gb - kv_budget), 2)
        rows.append(
            TierBenchmarkRow(
                gpu_vram_gb=vram,
                model_vram_gb=model_vram_gb,
                safety_reserve_gb=safety_reserve_gb,
                kv_budget_gb=kv_budget,
                fp16_max_ctx=estimates["f16"].max_context_tokens,
                q8_0_max_ctx=estimates["q8_0"].max_context_tokens,
                q4_0_max_ctx=estimates["q4_0"].max_context_tokens,
                fp16_32k_spill_gb=fp16_spill,
                q4_0_32k_vram_gb=round(q4_32k_gb, 2),
                zero_ram_spill_at_32k_with_q4=estimates["q4_0"].fits_32k,
            )
        )
    return rows


def run_benchmark(
    vram_gb: float = 8.0,
    model_vram_gb: float = 5.5,
    safety_reserve_gb: float = 0.8,
    n_layers: int = 48,
    n_heads_kv: int = 8,
    head_dim: int = 128,
    iterations: int = 500,
) -> dict[str, Any]:
    """Run deterministic VRAM planning and KV-cache compression benchmark."""
    t0 = time.perf_counter()
    kv_mgr = KVQuantManager(n_layers=n_layers, n_heads_kv=n_heads_kv, head_dim=head_dim)

    footprints = {}
    for q in ("f16", "q8_0", "q4_1", "q4_0"):
        q_typed: KVQuantType = q  # type: ignore[assignment]
        footprints[q] = asdict(kv_mgr.compute_footprint(q_typed))

    # Measure planner throughput (micro-benchmark of zero-spill scheduler math)
    arbiter = VRAMArbiter(total_vram_gb=vram_gb)
    for _ in range(iterations):
        arbiter.compute_plan("cooperative_split")
        kv_mgr.recommend_for_vram(32768, max(0.0, vram_gb - model_vram_gb - safety_reserve_gb))
    planner_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # Example layer-offload sizing for a 30B-A3B Q4_K_M MoE model (~17.5 GB total, 48 layers)
    dense_gb = derive_dense_layer_gb(model_size_gb=17.5, num_layers=n_layers, quant="Q4_K_M")
    offload_ngl = min(n_layers, int(model_vram_gb / dense_gb)) if dense_gb > 0 else 0
    estimated_vram = vram_estimate_gb(
        dense_layer_gb=dense_gb,
        n_gpu_layers=offload_ngl,
        kv_cache_gb=round(footprints["q4_0"]["mb_at_32k"] / 1024.0, 2),
    )

    matrix = [
        asdict(r)
        for r in compute_zero_spill_matrix(
            tiers_gb=(8.0, 12.0, 16.0),
            model_vram_gb=model_vram_gb,
            safety_reserve_gb=safety_reserve_gb,
            n_layers=n_layers,
            n_heads_kv=n_heads_kv,
            head_dim=head_dim,
        )
    ]

    live_gpu = probe_gpu()

    return {
        "benchmark": "bie-8gb-zero-ram-spill",
        "live_gpu": asdict(live_gpu),
        "config": {
            "target_vram_gb": vram_gb,
            "model_vram_gb": model_vram_gb,
            "safety_reserve_gb": safety_reserve_gb,
            "n_layers": n_layers,
            "n_heads_kv": n_heads_kv,
            "head_dim": head_dim,
        },
        "kv_footprints": footprints,
        "moe_offload_plan": {
            "dense_layer_gb": dense_gb,
            "offloaded_gpu_layers": offload_ngl,
            "total_layers": n_layers,
            "estimated_total_vram_gb": estimated_vram,
            "fits_target_vram": estimated_vram <= (vram_gb - safety_reserve_gb),
        },
        "tier_matrix": matrix,
        "planner_benchmark_ms": planner_ms,
        "planner_iterations": iterations,
    }


def render_benchmark(result: dict[str, Any]) -> str:
    cfg = result["config"]
    plan = result["moe_offload_plan"]
    fp = result["kv_footprints"]

    lines = [
        "==========================================================================",
        "  BIE 8GB-16GB VRAM Zero-RAM-Spill Benchmark & KV Compression Report",
        "==========================================================================",
        f"Live GPU      : {result['live_gpu']['name']}",
        f"Target Config : {cfg['target_vram_gb']} GB VRAM | Model Budget {cfg['model_vram_gb']} GB | Reserve {cfg['safety_reserve_gb']} GB",
        f"Planner Speed : {result['planner_iterations']} iterations in {result['planner_benchmark_ms']} ms",
        "",
        "1. KV-Cache Compression Math (48L, 8 KV-Heads, 128 Head-Dim):",
        f"  {'Quant':<8} {'Bytes/Tok':>10} {'8K (MB)':>10} {'16K (MB)':>10} {'32K (MB)':>10} {'Savings':>9}",
        "  " + "-" * 62,
    ]
    for q in ("f16", "q8_0", "q4_1", "q4_0"):
        row = fp[q]
        lines.append(
            f"  {q:<8} {row['bytes_per_token']:>10.1f} {row['mb_at_4k'] * 2:>10.1f} "
            f"{row['mb_at_16k']:>10.1f} {row['mb_at_32k']:>10.1f} {row['vram_savings_vs_f16_pct']:>8.1f}%"
        )

    lines.extend(
        [
            "",
            "2. Consumer GPU Zero-RAM-Spill Context Matrix:",
            f"  {'GPU':>6} {'KV Budget':>10} {'FP16 Max':>10} {'q8_0 Max':>10} {'q4_0 Max':>10} {'32K q4_0':>10}",
            "  " + "-" * 62,
        ]
    )
    for row in result["tier_matrix"]:
        status = "ZERO-SPILL" if row["zero_ram_spill_at_32k_with_q4"] else "SPILL"
        lines.append(
            f"  {row['gpu_vram_gb']:>4.0f}GB {row['kv_budget_gb']:>9.2f}G "
            f"{row['fp16_max_ctx']:>10,} {row['q8_0_max_ctx']:>10,} "
            f"{row['q4_0_max_ctx']:>10,} {status:>10}"
        )

    lines.extend(
        [
            "",
            "3. MoE Offload Plan (30B-A3B Q4_K_M):",
            f"  Dense layer size : {plan['dense_layer_gb']} GB/layer",
            f"  GPU layers (ngl) : {plan['offloaded_gpu_layers']} / {plan['total_layers']}",
            f"  Estimated VRAM   : {plan['estimated_total_vram_gb']} GB (with 32K q4_0 KV)",
            "==========================================================================",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BIE 8GB VRAM Zero-RAM-Spill Benchmark")
    parser.add_argument("--vram-gb", type=float, default=8.0, help="Target GPU VRAM in GB")
    parser.add_argument("--model-vram-gb", type=float, default=5.5, help="Model VRAM footprint in GB")
    parser.add_argument("--reserve-gb", type=float, default=0.8, help="OS/CUDA safety reserve in GB")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted table")
    args = parser.parse_args(argv)

    result = run_benchmark(
        vram_gb=args.vram_gb,
        model_vram_gb=args.model_vram_gb,
        safety_reserve_gb=args.reserve_gb,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_benchmark(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
