#!/usr/bin/env python3
"""BIE VRAM Profiler — Mede e estima VRAM por camada, KV cache, throughput PCIe.

Objetivo: Baseline e simulação preditiva para otimização de offload (32GB VRAM -> 8GB).
Suporta medição live em hardware (nvidia-smi) e análise estática offline via GGUF metadata.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from bie.core.scheduler_quant_aware import derive_from_gguf, vram_estimate_gb


def run_bie_benchmark(model: str = "Qwen3-8B-Q4_K_M", n_ctx: int = 8192) -> dict[str, Any] | None:
    """Rodar benchmark do BIE e coletar métricas de VRAM."""
    cmd = [
        sys.executable,
        "-m",
        "bie.serve",
        "bench",
        "--model",
        model,
        "--model-path",
        f"bie/models/{model}.gguf",
        "--context",
        str(n_ctx),
        "--n-gpu-layers",
        "22",
        "--reps",
        "3",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", check=False)
    if result.returncode != 0:
        print(f"Erro no benchmark: {result.stderr[:500]}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"JSON inválido no benchmark: {result.stdout[:200]}")
        return None


def _loaded_vram_mib(model_path: str, n_gpu_layers: int, n_ctx: int) -> int | None:
    """Carrega o modelo e reporta a VRAM usada enquanto o modelo está vivo."""
    code = f"""
import subprocess, time
from llama_cpp import Llama
llm = Llama(model_path=r"{model_path}", n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, verbose=False)
llm.create_chat_completion(messages=[{{"role": "user", "content": "test"}}], max_tokens=2)
time.sleep(0.3)
out = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=False,
)
print(out.stdout.strip())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(f"  erro load ngl={n_gpu_layers}: {result.stderr.strip()[:200]}")
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def measure_vram_per_layer(model: str = "Qwen3-8B-Q4_K_M") -> list[dict[str, Any]]:
    """Medir VRAM usada por n_gpu_layers de 1..36 (load real do modelo)."""
    layers: list[dict[str, Any]] = []
    model_path = f"bie/models/{model}.gguf"
    previous = None
    for layer_id in range(36):
        used = _loaded_vram_mib(model_path, layer_id + 1, 8192)
        if used is None:
            continue
        delta = used - previous if previous is not None else used
        layers.append({
            "layer": layer_id,
            "ngl": layer_id + 1,
            "vram_used_mib": used,
            "vram_delta_mib": delta,
        })
        print(f"Layer {layer_id} (ngl={layer_id + 1}): {used} MiB (delta +{delta})")
        previous = used
    return layers


def get_vram_usage() -> int:
    """Obter uso atual de VRAM via nvidia-smi."""
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
        return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def measure_pcie_throughput() -> str:
    """Medir throughput PCIe durante inferência (nvidia-smi dmon)."""
    cmd = ["nvidia-smi", "dmon", "-c", "10", "-s", "p"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=12)
        return result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi dmon unavailable"


def profile_kv_cache(model: str = "Qwen3-8B-Q4_K_M", contexts: list[int] | None = None) -> list[dict[str, Any]]:
    """Medir VRAM por tamanho de contexto (KV cache), fixando ngl=22."""
    if contexts is None:
        contexts = [8192, 16384, 32768]
    results: list[dict[str, Any]] = []
    model_path = f"bie/models/{model}.gguf"
    reference = _loaded_vram_mib(model_path, 22, 2048)
    if reference is None:
        return results
    for ctx in contexts:
        used = _loaded_vram_mib(model_path, 22, ctx)
        if used is None:
            continue
        kv_mib = max(used - reference, 0)
        results.append({"context": ctx, "vram_used_mib": used, "kv_cache_mib": kv_mib})
        print(f"Context {ctx}: {used} MiB (KV vs ctx2048: +{kv_mib} MiB)")
    return results


def profile_offline(gguf_path: str | Path) -> dict[str, Any]:
    """Estima matematicamente consumo de VRAM e planos de offload sem GPU."""
    path = Path(gguf_path)
    info = derive_from_gguf(path)
    dense_gb = float(info["dense_layer_gb"])
    block_count = int(info["block_count"])

    # Estimativa de footprint de KV cache por token (~2 bytes por element em fp16 ou 0.5 em q4)
    # KV per token approx = 2 * n_layers * n_heads * head_dim * sizeof(type)
    kv_per_1k = round(dense_gb * 0.15, 3)

    tiers = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    gpu_recommendations = {}
    for vram in tiers:
        usable = max(0.0, (vram - 1.0 - 0.75) * 0.85)  # 1GB OS reserve, 0.75GB KV
        ngl = min(block_count, int(usable / dense_gb)) if dense_gb > 0 else 0
        gpu_recommendations[f"{int(vram)}GB_GPU"] = {
            "recommended_ngl": ngl,
            "offload_pct": round((ngl / block_count) * 100, 1) if block_count else 0,
            "estimated_vram_gb": vram_estimate_gb(dense_gb, ngl, kv_cache_gb=0.75),
        }

    return {
        "model": path.name,
        "metadata": info,
        "dense_layer_gb": dense_gb,
        "total_layers": block_count,
        "gpu_recommendations": gpu_recommendations,
        "kv_estimates": {
            "ctx_2048": round(kv_per_1k * 2, 2),
            "ctx_8192": round(kv_per_1k * 8, 2),
            "ctx_16384": round(kv_per_1k * 16, 2),
            "ctx_32768": round(kv_per_1k * 32, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="BIE VRAM Profiler")
    parser.add_argument("--model", default="Qwen3-8B-Q4_K_M")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--offline", action="store_true", help="Calcular matematicamente sem invocar GPU")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default="output/vram_profile.json")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else Path(f"bie/models/{args.model}.gguf")

    if args.offline:
        if not model_path.is_file():
            print(f"Erro: Arquivo GGUF não encontrado para análise offline: {model_path}")
            sys.exit(1)
        res = profile_offline(model_path)
        if args.json:
            print(json.dumps(res))
        else:
            print(json.dumps(res, indent=2))
        return

    print("=" * 60)
    print("BIE VRAM Profiler — Baseline para Otimização (32GB -> 8GB)")
    print("=" * 60)

    print("\n[1/4] Benchmark baseline (bie.serve bench)...")
    benchmark = run_bie_benchmark(args.model)
    if benchmark:
        print(f"Decode: {benchmark.get('decode_tok_s', 'N/A')} tok/s")
        print(f"VRAM pico: {benchmark.get('vram_peak_mib', 'N/A')} MiB")
        print(f"VRAM delta: {benchmark.get('vram_delta_mib', 'N/A')} MiB")

    print("\n[2/4] VRAM por n_gpu_layers...")
    layers = measure_vram_per_layer(args.model)
    print(f"Camadas medidas: {len(layers)}")

    print("\n[3/4] KV cache por contexto...")
    kv_cache = profile_kv_cache(args.model)

    print("\n[4/4] Throughput PCIe...")
    pcie = measure_pcie_throughput()
    print(pcie[:500])

    output = {"benchmark": benchmark, "layers": layers, "kv_cache": kv_cache, "pcie_throughput": pcie[:1000]}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados salvos em {output_path}")


if __name__ == "__main__":
    main()