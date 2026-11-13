"""BIE Doctor & VRAM Probe — Hardware diagnostics and zero-RAM-spill context estimator.

Inspects local GPU VRAM, system RAM, runtime dependencies, and computes maximum
context lengths across FP16, FP8/INT8 (q8_0), and INT4 (q4_0/q4_1) KV-Cache
quantization without spilling into system RAM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

import psutil

from bie.core.kv_quant import KVQuantManager, KVQuantType
from bie.core.vram_arbiter import VRAMArbiter


@dataclass(frozen=True)
class GPUInfo:
    available: bool
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    driver_version: str | None = None


@dataclass(frozen=True)
class ContextEstimate:
    quant_type: KVQuantType
    label: str
    bytes_per_token: float
    kv_budget_gb: float
    max_context_tokens: int
    fits_8k: bool
    fits_32k: bool
    fits_64k: bool
    ram_spill_at_32k: bool
    vram_savings_vs_f16_pct: float


KV_LABELS: dict[KVQuantType, str] = {
    "f16": "FP16 (Unquantized)",
    "q8_0": "FP8 / INT8 (q8_0)",
    "q4_1": "INT4+ (q4_1)",
    "q4_0": "INT4 (q4_0)",
}


def probe_gpu() -> GPUInfo:
    """Query nvidia-smi for GPU name, VRAM total/free/used, and driver version."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                return GPUInfo(
                    available=True,
                    name=parts[0],
                    vram_total_gb=round(float(parts[1]) / 1024.0, 2),
                    vram_free_gb=round(float(parts[2]) / 1024.0, 2),
                    vram_used_gb=round(float(parts[3]) / 1024.0, 2),
                    driver_version=parts[4] if len(parts) > 4 else None,
                )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return GPUInfo(
        available=False,
        name="No NVIDIA GPU detected (CPU/Simulated mode)",
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        vram_used_gb=0.0,
        driver_version=None,
    )


def probe_dependencies() -> dict[str, bool]:
    """Check availability of core and optional BIE runtime dependencies."""
    packages = ("fastapi", "uvicorn", "pydantic", "httpx", "rich", "psutil", "llama_cpp", "openai")
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in packages}


def estimate_context_limits(
    total_vram_gb: float,
    model_vram_gb: float = 5.5,
    safety_reserve_gb: float = 0.8,
    n_layers: int = 48,
    n_heads_kv: int = 8,
    head_dim: int = 128,
) -> list[ContextEstimate]:
    """Estimate maximum zero-RAM-spill context length per KV-cache quantization."""
    kv_budget_gb = round(max(0.0, total_vram_gb - model_vram_gb - safety_reserve_gb), 3)
    kv_budget_bytes = kv_budget_gb * (1024**3)
    kv_mgr = KVQuantManager(n_layers=n_layers, n_heads_kv=n_heads_kv, head_dim=head_dim)

    estimates: list[ContextEstimate] = []
    for q in ("f16", "q8_0", "q4_1", "q4_0"):
        q_typed: KVQuantType = q  # type: ignore[assignment]
        report = kv_mgr.compute_footprint(q_typed)
        bpt = kv_mgr.bytes_per_token(q_typed)
        max_tokens = int(kv_budget_bytes // bpt) if bpt > 0 and kv_budget_bytes > 0 else 0
        estimates.append(
            ContextEstimate(
                quant_type=q_typed,
                label=KV_LABELS[q_typed],
                bytes_per_token=round(bpt, 2),
                kv_budget_gb=kv_budget_gb,
                max_context_tokens=max_tokens,
                fits_8k=max_tokens >= 8192,
                fits_32k=max_tokens >= 32768,
                fits_64k=max_tokens >= 65536,
                ram_spill_at_32k=max_tokens < 32768,
                vram_savings_vs_f16_pct=report.vram_savings_vs_f16_pct,
            )
        )
    return estimates


def run_doctor(
    vram_gb: float | None = None,
    model_vram_gb: float = 5.5,
    safety_reserve_gb: float = 0.8,
    n_layers: int = 48,
    n_heads_kv: int = 8,
    head_dim: int = 128,
) -> dict[str, Any]:
    """Run full BIE system diagnostics and return a structured report."""
    gpu = probe_gpu()
    vm = psutil.virtual_memory()
    deps = probe_dependencies()

    effective_vram_gb = vram_gb if vram_gb is not None else (gpu.vram_total_gb if gpu.available else 8.0)
    simulated = vram_gb is not None or not gpu.available

    estimates = estimate_context_limits(
        total_vram_gb=effective_vram_gb,
        model_vram_gb=model_vram_gb,
        safety_reserve_gb=safety_reserve_gb,
        n_layers=n_layers,
        n_heads_kv=n_heads_kv,
        head_dim=head_dim,
    )

    arbiter = VRAMArbiter(total_vram_gb=effective_vram_gb)
    primary_plan = arbiter.compute_plan("primary_exclusive")
    coop_plan = arbiter.compute_plan("cooperative_split")

    kv_mgr = KVQuantManager(n_layers=n_layers, n_heads_kv=n_heads_kv, head_dim=head_dim)
    kv_budget = max(0.0, effective_vram_gb - model_vram_gb - safety_reserve_gb)
    recommended_kv_32k = kv_mgr.recommend_for_vram(32768, kv_budget)

    status = "ready" if deps.get("fastapi") and deps.get("psutil") else "degraded"
    if not deps.get("llama_cpp"):
        status = "cpu_or_external_backend" if status == "ready" else status

    return {
        "status": status,
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
        },
        "gpu": asdict(gpu),
        "probe_config": {
            "effective_vram_gb": round(effective_vram_gb, 2),
            "simulated_vram": simulated,
            "model_vram_gb": model_vram_gb,
            "safety_reserve_gb": safety_reserve_gb,
            "kv_budget_gb": round(kv_budget, 3),
            "n_layers": n_layers,
            "n_heads_kv": n_heads_kv,
            "head_dim": head_dim,
        },
        "recommended_kv_quant_32k": recommended_kv_32k,
        "context_estimates": [asdict(e) for e in estimates],
        "vram_arbiter": {
            "primary_exclusive": asdict(primary_plan),
            "cooperative_split": asdict(coop_plan),
        },
        "dependencies": deps,
    }


def render_doctor_report(report: dict[str, Any]) -> str:
    """Format the doctor report as a clean terminal string."""
    gpu = report["gpu"]
    sys_info = report["system"]
    cfg = report["probe_config"]
    deps = report["dependencies"]

    lines = [
        "======================================================================",
        "  BIE DOCTOR — Sovereign Consumer-GPU Inference Diagnostics",
        "======================================================================",
        f"Status        : {report['status'].upper()}",
        f"Python / OS   : {sys_info['python']} ({sys_info['platform']})",
        f"System RAM    : {sys_info['ram_available_gb']} GB free / {sys_info['ram_total_gb']} GB total",
        f"GPU           : {gpu['name']}",
    ]
    if gpu["available"]:
        lines.append(
            f"Live VRAM     : {gpu['vram_free_gb']} GB free / {gpu['vram_total_gb']} GB total "
            f"(used: {gpu['vram_used_gb']} GB, driver: {gpu['driver_version'] or 'n/a'})"
        )
    else:
        lines.append(
            f"Simulated VRAM: {cfg['effective_vram_gb']} GB target "
            "(pass --vram-gb to override)"
        )

    lines.extend(
        [
            "",
            f"Zero-RAM-Spill KV Budget: {cfg['kv_budget_gb']} GB "
            f"(Total {cfg['effective_vram_gb']} GB - Model {cfg['model_vram_gb']} GB - Reserve {cfg['safety_reserve_gb']} GB)",
            f"Recommended KV Quant (32K): {report['recommended_kv_quant_32k']}",
            "",
            "KV-Cache Context Matrix (Zero RAM Spill):",
            f"  {'KV Quant':<20} {'Bytes/Tok':>10} {'Max Context':>13} {'8K':>5} {'32K':>5} {'Savings':>9}",
            "  " + "-" * 66,
        ]
    )

    for est in report["context_estimates"]:
        f8 = "OK" if est["fits_8k"] else "SPILL"
        f32 = "OK" if est["fits_32k"] else "SPILL"
        lines.append(
            f"  {est['label']:<20} {est['bytes_per_token']:>10.1f} "
            f"{est['max_context_tokens']:>13,} {f8:>5} {f32:>5} "
            f"{est['vram_savings_vs_f16_pct']:>8.1f}%"
        )

    p_plan = report["vram_arbiter"]["primary_exclusive"]
    c_plan = report["vram_arbiter"]["cooperative_split"]
    lines.extend(
        [
            "",
            "VRAM Arbiter Plans:",
            f"  primary_exclusive : coder_ngl={p_plan['primary_ngl']} ({p_plan['primary_vram_gb']} GB VRAM)",
            f"  cooperative_split : coder_ngl={c_plan['primary_ngl']} ({c_plan['primary_vram_gb']} GB) + "
            f"vision_ngl={c_plan['vision_ngl']} ({c_plan['vision_vram_gb']} GB)",
            "",
            "Dependencies:",
            "  " + ", ".join(f"{k}={'OK' if v else 'MISSING'}" for k, v in deps.items()),
            "======================================================================",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bie-probe",
        description="BIE Doctor & VRAM Probe — inspect hardware and zero-RAM-spill context limits.",
    )
    parser.add_argument("--vram-gb", type=float, default=None, help="Override GPU VRAM in GB (e.g., 8, 12, 16)")
    parser.add_argument("--model-vram-gb", type=float, default=5.5, help="Model weights VRAM footprint in GB")
    parser.add_argument("--reserve-gb", type=float, default=0.8, help="OS/CUDA safety reserve in GB")
    parser.add_argument("--layers", type=int, default=48, help="Transformer layer count")
    parser.add_argument("--kv-heads", type=int, default=8, help="KV attention heads (GQA)")
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_doctor(
        vram_gb=args.vram_gb,
        model_vram_gb=args.model_vram_gb,
        safety_reserve_gb=args.reserve_gb,
        n_layers=args.layers,
        n_heads_kv=args.kv_heads,
        head_dim=args.head_dim,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_doctor_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
