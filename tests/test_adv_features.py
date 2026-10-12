"""Comprehensive unit tests for the 10 advanced BIE features:
Speculative Decoding, Grammar Guard, Task Router, Quantized KV, Tool Calling,
Embeddings, Admission Controller, LoRA Manager, Telemetry, and Bench Matrix.
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from bie.core.speculative import SpeculativeEngine
from bie.core.grammar_guard import GrammarGuard
from bie.core.router import TaskRouter
from bie.core.kv_quant import KVQuantManager
from bie.serve.tool_caller import ToolCallingEngine
from bie.core.embeddings import EmbeddingsEngine
from bie.core.admission import AdmissionController
from bie.core.lora_manager import LoRAManager
from bie.serve.telemetry import TelemetryEngine
from bie.tools.bench_matrix import BenchMatrixRunner
from bie.serve.openai_compat import app, configure_engine


# --- 1. Speculative Decoding ---
def test_speculative_decoding():
    target = lambda prompt, **kw: {"choices": [{"text": "verified answer"}]}
    draft = lambda prompt, **kw: {"choices": [{"text": "candidate token"}]}

    engine = SpeculativeEngine(target_model=target, draft_model=draft, gamma=2)
    res = engine.generate("test prompt", max_tokens=10)

    assert res["speculative"] is True
    assert res["tokens_generated"] > 0
    assert engine.stats.total_drafted_tokens > 0
    assert engine.stats.speedup_ratio >= 1.0


# --- 2. Grammar Guard (Structured Outputs) ---
def test_grammar_guard_repair_and_validation():
    broken_json = '```json\n{"name": "Alice", "age": 30,}\n```'
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    report = GrammarGuard.validate(broken_json, schema=schema, auto_repair=True)
    assert report["valid"] is True
    assert report["data"]["name"] == "Alice"
    assert report["data"]["age"] == 30

    # Test missing required field
    missing_json = '{"name": "Bob"}'
    report_fail = GrammarGuard.validate(missing_json, schema=schema)
    assert report_fail["valid"] is False
    assert "Missing required property: 'age'" in report_fail["error"]


# --- 3. Task Router ---
def test_task_router_policies():
    router = TaskRouter(local_context_limit=4096, default_privacy_mode="hybrid")

    # BRAVO formation routes to local
    decision_bravo = router.route("def solve(): pass", task_type="BRAVO")
    assert decision_bravo.target == "local"
    assert decision_bravo.privacy_score == 1.0

    # Huge prompt exceeding 90% context routes to cloud
    huge_prompt = "word " * 4000
    decision_cloud = router.route(huge_prompt, task_type="general")
    assert decision_cloud.target == "cloud"
    assert decision_cloud.estimated_cost_usd > 0.0

    # Sovereign privacy override forces local
    decision_sovereign = router.route(huge_prompt, privacy_override="sovereign")
    assert decision_sovereign.target == "local"

    telem = router.telemetry()
    assert telem["total_requests"] >= 3


# --- 4. Quantized KV Cache ---
def test_kv_quant_manager():
    kv_q = KVQuantManager(n_layers=48, n_heads_kv=8, head_dim=128, default_quant="q4_0")
    report = kv_q.compute_footprint("q4_0")

    assert report.quant_type == "q4_0"
    assert report.vram_savings_vs_f16_pct > 60.0
    assert report.mb_at_32k < 2000  # Stays well within consumer VRAM (1.7GB vs ~6GB in FP16)

    params = kv_q.get_backend_params("q4_0")
    assert params["type_k"] == 2
    assert params["cache_type_k"] == "q4_0"

    rec = kv_q.recommend_for_vram(target_ctx=32768, available_vram_gb=1.0)
    assert rec in ("q4_0", "q4_1")


# --- 5. Tool Calling Engine ---
def test_tool_calling_engine():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_stock",
                "description": "Get ticker price",
                "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
            },
        }
    ]
    messages = [{"role": "user", "content": "Price of AAPL?"}]
    injected = ToolCallingEngine.inject_tools(messages, tools)
    assert any("# TOOLS AVAILABLE" in m["content"] for m in injected)

    sample_output = '<tool_call>{"name": "lookup_stock", "arguments": {"ticker": "AAPL"}}</tool_call>'
    calls = ToolCallingEngine.parse_tool_calls(sample_output)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup_stock"
    assert json.loads(calls[0]["function"]["arguments"])["ticker"] == "AAPL"


# --- 6. Embeddings Engine ---
def test_embeddings_engine():
    engine = EmbeddingsEngine(default_dimension=64)
    v1, v2 = engine.embed(["machine learning model", "deep learning artificial intelligence"])
    assert len(v1) == 64
    assert len(v2) == 64

    sim = EmbeddingsEngine.cosine_similarity(v1, v2)
    assert -1.0 <= sim <= 1.0

    nearest = engine.find_nearest(v1, [v1, v2], top_k=1)
    assert nearest[0][0] == 0
    assert nearest[0][1] == pytest.approx(1.0, abs=1e-3)


# --- 7. Admission Controller ---
def test_admission_controller():
    adm = AdmissionController(max_concurrent=1, max_queue_depth=2)
    assert adm.admit_or_reject("req-1", priority="normal") is True
    assert adm.can_admit_immediately() is False

    # Second request queued
    assert adm.admit_or_reject("req-2", priority="critical") is True
    assert len(adm.queue) == 1

    # Release promotes req-2
    adm.release()
    assert adm.active_count == 1
    assert len(adm.queue) == 0

    stats = adm.stats()
    assert stats["completed_requests"] == 1


# --- 8. LoRA Manager ---
def test_lora_manager():
    mgr = LoRAManager()
    adapter = mgr.register("coder-adapter", "models/lora/coder.gguf", scale=0.8)
    assert adapter.adapter_id == "coder-adapter"

    mgr.set_scale("coder-adapter", 1.2)
    assert mgr.registry["coder-adapter"].scale == 1.2

    backend_args = mgr.get_backend_lora_args()
    assert backend_args["lora_scale"] == 1.2
    assert "coder.gguf" in backend_args["lora_path"]


# --- 9. Telemetry Engine ---
def test_telemetry_engine():
    telem = TelemetryEngine()
    telem.record_request(latency_s=0.05, ttft_s=0.01, decode_tok_s=25.0, tokens_count=10)
    telem.record_request(latency_s=0.10, ttft_s=0.02, decode_tok_s=20.0, tokens_count=15)
    telem.record_expert_activations(layer=0, expert_ids=[1, 3, 3])

    summary = telem.get_summary()
    assert summary["total_requests_recorded"] == 2
    assert summary["total_tokens_processed"] == 25
    assert summary["latency_p50_s"] > 0.0

    heatmap = telem.render_ascii_heatmap()
    assert "L00" in heatmap


# --- 10. Bench Matrix Runner ---
def test_bench_matrix_runner():
    runner = BenchMatrixRunner("qwen3-8b", contexts=[1024, 2048], max_tokens=16, reps=1)
    res = runner.run_matrix()
    assert len(res["matrix_results"]) == 2
    assert res["matrix_results"][0]["context_size"] == 1024
    md = runner.format_markdown_table(res)
    assert "| 1024 |" in md


# --- 11. OpenAI Compat API Extensions ---
class FakeAdvEngine:
    def __init__(self):
        self.model_name = "qwen3-8b"
        self.model = object()
        self.telemetry = TelemetryEngine()

    def complete(self, prompt, **kwargs):
        return {"choices": [{"text": '<tool_call>{"name": "fetch", "arguments": {"id": 1}}</tool_call>'}]}

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_openai_compat_embeddings_and_telemetry():
    client = TestClient(app)
    configure_engine(FakeAdvEngine())

    # /v1/embeddings
    res_emb = client.post("/v1/embeddings", json={"input": "test text", "model": "qwen3-8b"})
    assert res_emb.status_code == 200
    assert len(res_emb.json()["data"]) == 1
    assert len(res_emb.json()["data"][0]["embedding"]) == 3

    # /v1/bie/telemetry
    res_telem = client.get("/v1/bie/telemetry")
    assert res_telem.status_code == 200
    assert "uptime_seconds" in res_telem.json()

    # /v1/chat/completions with tools
    res_chat = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "fetch id 1"}],
            "tools": [{"type": "function", "function": {"name": "fetch"}}],
        },
    )
    assert res_chat.status_code == 200
    choice = res_chat.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "fetch"

    configure_engine(None)
