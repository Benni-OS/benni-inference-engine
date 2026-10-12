# benni-inference-engine — ROADMAP

## Phase 1 — Core Engine (BRAVO Formation)
> Target: 1 week | Priority: SHIP | Status: ✅ COMPLETED

- [x] `engine.py` — llama-cpp-python CUDA load + forward + streaming
- [x] `moe_scheduler.py` — auto VRAM/RAM detection + `-ngl` + `-ot` flag generation
- [x] `openai_compat.py` — FastAPI server, streaming support, OpenAI spec compliance
- [x] `bie serve` CLI — working end-to-end with Qwen3-30B-A3B and multi-model support
- [x] Hardware benchmark: RTX 4060 + Ryzen 5 5600X + 32GB RAM (verified baseline)

## Phase 2 — Skill 81 + Hot Pin
> Target: 1 week after Phase 1 | Status: ✅ COMPLETED

- [x] `ceiling_extractor.py` — SA-81A/B/C/E full implementation (Role lock, task frames, forced CoT)
- [x] `hallucination_guard.py` — SA-81D claim classifier, epistemic markers, drift detection
- [x] `hot_pin.py` — expert frequency tracking + adaptive pinning with EWMA decay & JSON persistence
- [x] `kv_manager.py` — sliding window compression, dead-weight cleanup, causal pruning
- [x] SA-81E model profiles: Qwen3, Kimi K2, DeepSeek V4, Cyber-Tiel-Coder-35B, Gemma-4-26B

## Phase 3 — Project Benjamin Native + Open Source Launch
> Target: 2 weeks after Phase 2 | Status: 🟢 ACTIVE / ADVANCED

- [x] `mcp_bridge.py` — full Project Benjamin integration + standalone JSON-RPC 2.0 stdio server
- [x] Task routing policy (`router.py` — DELTA/CHARLIE/BRAVO → local; ALPHA/FOXTROT → cloud; sovereign mode)
- [x] Tool calling engine (`tool_caller.py` — `<tool_call>` parsing & OpenAI function calls)
- [x] Vector embeddings pipeline (`embeddings.py` & `/v1/embeddings`)
- [x] Concurrency & Admission controller (`admission.py` with priority queue)
- [x] Model Hub Downloader (`hub_downloader.py` & `bie pull`)
- [ ] PyPI publish: `pip install benni-inference-engine`
- [ ] Product Hunt launch alongside mcp-forge + operator-gateway
- [ ] Community benchmark submissions (open issue template)

## Phase 4 — Multi-Model + Advanced Systems
> Roadmap & Production Enhancements | Status: 🟢 IN PROGRESS

- [x] Speculative decoding (`speculative.py` — draft model rejection sampling)
- [x] Quantized KV Cache (`kv_quant.py` — q4_0, q8_0 with >70% VRAM savings)
- [x] Structured Outputs & JSON Schema Guard (`grammar_guard.py` with auto-repair)
- [x] Real-time Telemetry & MoE Heatmap (`telemetry.py` & `/v1/bie/telemetry`)
- [x] Continuous Batching Multi-Slot Multiplexer (`continuous_batcher.py`)
- [x] Paged Cache Memory Allocator (`paged_cache.py`)
- [x] Prompt Injection Firewall (`injection_firewall.py` — Skill 63 Cyber Sentinel)
- [x] Self-Correction & Reflection Loop (`reflection_loop.py`)
- [x] Dynamic LoRA Adapter Manager (`lora_manager.py`)
- [x] Matrix Benchmark Suite (`bench_matrix.py` & `bie matrix`)
- [x] Windows & Linux CUDA native runtime support
