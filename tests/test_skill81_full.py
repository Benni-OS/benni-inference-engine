"""Comprehensive tests for Skill 81 full implementations: CeilingExtractor, HallucinationGuard, KVManager, HotPinManager."""

import pytest
from pathlib import Path

from bie.skill81.ceiling_extractor import CeilingExtractor
from bie.skill81.hallucination_guard import HallucinationGuard
from bie.core.kv_manager import KVManager
from bie.core.hot_pin import HotPinManager


def test_ceiling_extractor_task_frames():
    extractor = CeilingExtractor()
    messages = [{"role": "user", "content": "Develop an inference engine."}]

    # BRAVO (code)
    bravo_res = extractor.extract(messages, task_type="BRAVO")
    assert any("[ROLE LOCK]: Principal Software Systems Engineer" in m["content"] for m in bravo_res)
    assert any("[SA-81C REASONING]" in m["content"] for m in bravo_res)

    # DELTA (research)
    delta_res = extractor.extract(messages, task_type="DELTA")
    assert any("Chief Research Analyst" in m["content"] for m in delta_res)

    # ALPHA (synthesis)
    alpha_res = extractor.extract(messages, task_type="ALPHA")
    assert any("Executive Synthesis Director" in m["content"] for m in alpha_res)


def test_ceiling_extractor_profile_optimizations():
    extractor = CeilingExtractor()
    messages = [{"role": "user", "content": "hello"}]
    profile = {
        "skill81_profile": {
            "optimizations": ["format_lock_strict", "explicit_success_criterion"]
        }
    }
    res = extractor.extract(messages, model_profile=profile)
    system_text = res[0]["content"]
    assert "[FORMAT LOCK]" in system_text
    assert "[SUCCESS CRITERION]" in system_text


def test_ceiling_extractor_cot_triggers():
    extractor = CeilingExtractor()
    msg_step = [{"role": "user", "content": "explique o passo a passo da otimizacao"}]
    res = extractor.extract(msg_step, task_type="general")
    assert "[SA-81C REASONING]" in res[0]["content"]


def test_hallucination_guard_verified_claim():
    guard = HallucinationGuard()
    prompt = "Qual a latência medida no benchmark da RTX 4060?"
    output = "[VERIFIED] A latência medida no benchmark da RTX 4060 foi de 14ms conforme demonstrado nos testes."
    report = guard.check(output, prompt)
    assert report["passed"] is True
    assert report["claim_breakdown"]["verified"] >= 1
    assert report["drift_detected"] is False
    assert report["confidence_score"] > 0.5


def test_hallucination_guard_drift_and_unanchored():
    guard = HallucinationGuard()
    prompt = "Explique a história da culinária italiana."
    # Output drifting to high-frequency trading and unanchored file references
    output = "A taxa de transferência PCIe atingiu 99.8% com throughput de 128gb/s no arquivo /etc/secret_trade.py."
    report = guard.check(output, prompt)
    assert report["drift_detected"] is True
    assert len(report["unanchored_claims"]) >= 1
    assert report["confidence_score"] < 0.8


def test_kv_manager_token_estimation_and_cleanup():
    kv = KVManager(ctx_size=1024)
    raw = "test     multiple    spaces \n\n\n\n lines and ------------ dashes"
    cleaned = kv.clean_text_deadweight(raw)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "------------" not in cleaned


def test_kv_manager_causal_compression():
    kv = KVManager(ctx_size=100, compression_threshold=0.5)
    long_msg = "x" * 600
    messages = [
        {"role": "system", "content": "Core system prompt"},
        {"role": "user", "content": "Initial question"},
        {"role": "assistant", "content": long_msg},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]
    compressed = kv.compress(messages)
    assert len(compressed) == 5
    assert compressed[0]["content"] == "Core system prompt"
    # Intermediate assistant message compressed
    assert "SA-81B: intermediate turn collapsed" in compressed[2]["content"]
    assert compressed[3]["content"] == "Recent question"


def test_hot_pin_manager_lifecycle(tmp_path):
    mgr = HotPinManager(n_layers=4, n_experts=8, hot_n=2, warmup_tokens=5)
    for _ in range(5):
        mgr.record(layer=0, expert_ids=[2, 5])
        mgr.record(layer=1, expert_ids=[0, 7])

    assert mgr.pinned is True
    top_l0 = mgr.get_hot_experts_for_layer(0)
    assert 2 in top_l0 and 5 in top_l0
    assert len(top_l0) == 2

    # Test decay
    initial_freq = mgr.freq[0][2]
    mgr.apply_decay()
    assert mgr.freq[0][2] < initial_freq

    # Test persistence
    state_file = tmp_path / ".bie_hot_pin.json"
    saved_path = mgr.save(state_file)
    assert saved_path.is_file()

    new_mgr = HotPinManager(n_layers=4, n_experts=8)
    assert new_mgr.load(saved_path) is True
    assert new_mgr.pinned is True
    assert new_mgr.pinned_map[0] == top_l0
