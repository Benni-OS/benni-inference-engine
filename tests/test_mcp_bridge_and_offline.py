"""Tests for MCP Bridge, BIETool in-memory fallback, and offline VRAM profiler."""

import asyncio
from pathlib import Path
from bie.serve.mcp_bridge import BIETool, MCP_TOOLS
from bie.tools.vram_profiler import profile_offline


class DummyEngine:
    def __init__(self):
        self.model = object()

    def complete(self, prompt, **kwargs):
        return {"choices": [{"text": "In-memory complete result"}]}

    def chat(self, messages, **kwargs):
        return {
            "choices": [{"message": {"role": "assistant", "content": "In-memory chat result"}}],
            "usage": {"total_tokens": 42},
        }


def test_bie_tool_in_memory():
    async def _run():
        dummy = DummyEngine()
        BIETool.set_in_memory_engine(dummy)

        res_complete = await BIETool.complete("hello", task_type="general")
        assert res_complete == "In-memory complete result"

        res_chat = await BIETool.chat([{"role": "user", "content": "hi"}], task_type="BRAVO")
        assert res_chat["choices"][0]["message"]["content"] == "In-memory chat result"

        BIETool.set_in_memory_engine(None)

    asyncio.run(_run())


def test_bie_tool_guard_and_hardware():
    guard_res = BIETool.guard_check(output="[VERIFIED] Fact", prompt="Fact")
    assert "passed" in guard_res
    assert guard_res["passed"] is True

    hw_res = BIETool.hardware_plan("qwen3-30b-a3b")
    assert "ngl" in hw_res
    assert "expert_cache_slots" in hw_res


def test_mcp_tools_schema():
    tool_names = [t["name"] for t in MCP_TOOLS]
    assert "bie_complete" in tool_names
    assert "bie_guard_check" in tool_names
    assert "bie_hardware_plan" in tool_names


def test_profile_offline_synthetic_gguf(tmp_path, monkeypatch):
    import struct
    # Create minimal synthetic GGUF file
    gguf_file = tmp_path / "test-model-Q4_K_M.gguf"
    # GGUF header: magic b'GGUF', version 3, tensor_count 0, kv_count 2
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
    # KV 1: general.architecture = "llama"
    kv1_key = b"general.architecture"
    kv1 = struct.pack("<Q", len(kv1_key)) + kv1_key + struct.pack("<I", 8) + struct.pack("<Q", 5) + b"llama"
    # KV 2: llama.block_count = 32 (u32: type 4)
    kv2_key = b"llama.block_count"
    kv2 = struct.pack("<Q", len(kv2_key)) + kv2_key + struct.pack("<I", 4) + struct.pack("<I", 32)
    # Dummy weight payload to reach ~1GB
    payload = b"\x00" * (1024 * 1024)  # 1MB for test
    gguf_file.write_bytes(header + kv1 + kv2 + payload)

    report = profile_offline(gguf_file)
    assert report["total_layers"] == 32
    assert "gpu_recommendations" in report
    assert "kv_estimates" in report
    assert "8GB_GPU" in report["gpu_recommendations"]
