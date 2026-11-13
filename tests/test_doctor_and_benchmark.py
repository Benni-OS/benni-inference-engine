"""Unit tests for BIE Doctor (VRAM Probe), 8GB Zero-RAM-Spill Benchmark, and OpenAI Drop-in Quickstart."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from bie.serve.__main__ import main as cli_main
from bie.tools.doctor import (
    estimate_context_limits,
    render_doctor_report,
    run_doctor,
)
from bie.tools.doctor import (
    main as doctor_main,
)
from scripts.benchmark_8gb_vram import (
    compute_zero_spill_matrix,
    render_benchmark,
    run_benchmark,
)
from scripts.benchmark_8gb_vram import (
    main as bench_main,
)


def _load_quickstart_module():
    path = Path(__file__).resolve().parent.parent / "examples" / "quickstart_openai_dropin.py"
    spec = importlib.util.spec_from_file_location("quickstart_openai_dropin", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_estimate_context_limits_8gb_zero_spill() -> None:
    estimates = {e.quant_type: e for e in estimate_context_limits(total_vram_gb=8.0, model_vram_gb=5.5, safety_reserve_gb=0.8)}
    assert estimates["f16"].kv_budget_gb == 1.7
    assert estimates["f16"].max_context_tokens == 9284
    assert estimates["f16"].ram_spill_at_32k is True

    assert estimates["q8_0"].max_context_tokens == 17476
    assert estimates["q8_0"].vram_savings_vs_f16_pct == 46.9

    assert estimates["q4_0"].max_context_tokens == 33010
    assert estimates["q4_0"].fits_32k is True
    assert estimates["q4_0"].ram_spill_at_32k is False
    assert estimates["q4_0"].vram_savings_vs_f16_pct == 71.9


def test_estimate_context_limits_zero_budget_edge_case() -> None:
    estimates = estimate_context_limits(total_vram_gb=4.0, model_vram_gb=5.5, safety_reserve_gb=0.8)
    assert all(e.kv_budget_gb == 0.0 for e in estimates)
    assert all(e.max_context_tokens == 0 for e in estimates)
    assert all(e.ram_spill_at_32k is True for e in estimates)


def test_run_doctor_and_render_report() -> None:
    report = run_doctor(vram_gb=8.0)
    assert report["probe_config"]["effective_vram_gb"] == 8.0
    assert report["recommended_kv_quant_32k"] == "q4_0"
    assert len(report["context_estimates"]) == 4
    assert "primary_exclusive" in report["vram_arbiter"]
    assert "cooperative_split" in report["vram_arbiter"]

    text = render_doctor_report(report)
    assert "BIE DOCTOR" in text
    assert "Zero-RAM-Spill KV Budget" in text
    assert "INT4 (q4_0)" in text


def test_doctor_cli_entrypoints_json(capsys) -> None:
    rc1 = doctor_main(["--vram-gb", "12.0", "--json"])
    assert rc1 == 0
    payload1 = json.loads(capsys.readouterr().out)
    assert payload1["probe_config"]["effective_vram_gb"] == 12.0
    assert payload1["recommended_kv_quant_32k"] == "q8_0"

    rc2 = cli_main(["doctor", "--vram-gb", "16.0", "--json"])
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["probe_config"]["effective_vram_gb"] == 16.0
    assert payload2["recommended_kv_quant_32k"] == "f16"


def test_benchmark_8gb_vram_matrix_and_cli(capsys) -> None:
    matrix = compute_zero_spill_matrix()
    assert len(matrix) == 3
    assert matrix[0].gpu_vram_gb == 8.0
    assert matrix[0].zero_ram_spill_at_32k_with_q4 is True
    assert matrix[2].gpu_vram_gb == 16.0
    assert matrix[2].fp16_max_ctx > 32768

    result = run_benchmark(vram_gb=8.0, iterations=10)
    assert result["benchmark"] == "bie-8gb-zero-ram-spill"
    assert result["kv_footprints"]["q4_0"]["vram_savings_vs_f16_pct"] == 71.9
    rendered = render_benchmark(result)
    assert "ZERO-SPILL" in rendered

    rc = bench_main(["--vram-gb", "8.0", "--json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["config"]["target_vram_gb"] == 8.0


def test_quickstart_openai_dropin_helpers_and_offline_guard(capsys) -> None:
    mod = _load_quickstart_module()
    tools = mod.build_tool_definitions()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "estimate_kv_budget"

    # Point to an unreachable port to verify clean user-facing error handling
    rc = mod.main(["--base-url", "http://127.0.0.1:59999/v1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "bie doctor" in err or "Install the OpenAI SDK" in err
