"""CLI do BIE serve — subcomandos bench, serve e health.

Fallback: usa llama.cpp diretamente se o BIEEngine não estiver disponível.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from pathlib import Path
from typing import Any

try:
    from bie.core.engine import BIEEngine as _BIEEngine

    ENGINE_CLS = _BIEEngine
except ImportError:
    ENGINE_CLS = None


class _FallbackEngine:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        n_ctx: int = 32768,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else Path(f"models/{model_name}.gguf")
        self.n_ctx = n_ctx
        self.n_threads = n_threads or 0
        self.n_batch = n_batch
        self.verbose = verbose
        self.model: Any | None = None

    def load(self) -> dict[str, Any]:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            verbose=self.verbose,
        )
        return {}

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return self.model(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Any:
        for chunk in self.model(prompt, stream=True, **kwargs):
            yield chunk["choices"][0]["text"]

    def unload(self) -> None:
        self.model = None


def _engine_cls() -> type[Any]:
    if ENGINE_CLS is None:
        return _FallbackEngine
    return ENGINE_CLS


def _gpu_vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def _token_count(llm: Any, text: str) -> int:
    tokenize = getattr(llm, "tokenize", None)
    if tokenize is not None:
        try:
            return len(tokenize(text.encode("utf-8")))
        except (ValueError, RuntimeError):
            pass
    return max(1, len(text) // 4)


def _rss_self_mib() -> int | None:
    """Working set (MiB) of the current process via self-measurement.

    Cross-process RSS reads are unreliable on some Windows setups, so the
    bench reports its own peak directly instead of being observed externally.
    """
    if sys.platform != "win32":
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PMC),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = k32.GetCurrentProcess()
        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize // (1024 * 1024)) if ok else None
    except (AttributeError, OSError, ValueError):
        return None


def _rss_poller(stop: threading.Event, peak: list[int | None]) -> None:
    while not stop.is_set():
        value = _rss_self_mib()
        if value is not None and (peak[0] is None or value > peak[0]):
            peak[0] = value
        stop.wait(0.25)


def _vram_poller(stop: threading.Event, peak: list[int]) -> None:
    while not stop.is_set():
        value = _gpu_vram_mib()
        if value is not None and value > peak[0]:
            peak[0] = value
        stop.wait(0.25)


def _build_engine(args: argparse.Namespace) -> Any:
    if args.engine == "llama-server":
        from bie.serve.llama_server_adapter import LlamaServerAdapter

        return LlamaServerAdapter(args.model)
    cls = _engine_cls()
    return cls(
        args.model,
        model_path=args.model_path,
        n_ctx=args.context,
        n_threads=args.threads,
        n_batch=args.batch,
        verbose=args.verbose,
    )


def _apply_ngl_override(args: argparse.Namespace) -> None:
    if getattr(args, "n_gpu_layers", None):
        os.environ["BIE_GPU_LAYERS"] = str(args.n_gpu_layers)


def _bench_prompt(llm: Any, target_tokens: int) -> str:
    seed = "The history of computing is the history of compression. "
    while _token_count(llm, seed) < target_tokens:
        seed += seed
    return seed


def _run_rep(engine: Any, llm: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    t0 = time.time()
    lats: list[float] = []
    prev = time.time()
    generated = 0
    for _ in engine.stream(prompt, max_tokens=max_tokens, temperature=0.0):
        now = time.time()
        lats.append(now - prev)
        prev = now
        generated += 1
    total = time.time() - t0
    ttft = lats[0] if lats else total
    prompt_tokens = _token_count(llm, prompt)
    prefill = prompt_tokens / ttft if ttft > 0 else 0.0
    gen_time = max(total - ttft, 1e-9)
    decode = (generated - 1) / gen_time if generated > 1 else 0.0
    return {
        "ttft_s": round(ttft, 3),
        "prefill_est_tok_s": round(prefill, 1),
        "decode_tok_s": round(decode, 1),
        "gen_tokens": generated,
        "lats": lats,
    }


def cmd_bench(args: argparse.Namespace) -> int:
    vram_baseline = _gpu_vram_mib()
    _apply_ngl_override(args)
    engine = _build_engine(args)
    stop = threading.Event()
    rss_peak: list[int | None] = [None]
    rss_poller = threading.Thread(target=_rss_poller, args=(stop, rss_peak))
    rss_poller.start()
    t0 = time.time()
    rc = engine.load()
    load_s = round(time.time() - t0, 2)
    llm = engine.model
    prompt = _bench_prompt(llm, args.prompt_tokens)

    _ = engine.stream(prompt, max_tokens=8, temperature=0.0)
    for _ in engine.stream(prompt, max_tokens=8, temperature=0.0):
        pass

    peak: list[int] = [vram_baseline or 0]
    poller = threading.Thread(target=_vram_poller, args=(stop, peak))
    poller.start()
    reps = []
    for _ in range(args.reps):
        reps.append(_run_rep(engine, llm, prompt, args.max_tokens))
    stop.set()
    poller.join(timeout=2)
    rss_poller.join(timeout=2)
    engine.unload()

    decode_rates = [r["decode_tok_s"] for r in reps]
    prefill_rates = [r["prefill_est_tok_s"] for r in reps]
    lats = sorted(x for r in reps for x in r["lats"][1:])
    p50 = lats[len(lats) // 2] * 1000 if lats else 0.0
    p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))] * 1000 if lats else 0.0
    vram_after = _gpu_vram_mib()
    model_label = Path(args.model_path).name if args.model_path else args.model
    result = {
        "model": model_label,
        "model_path": str(Path(args.model_path)) if args.model_path else None,
        "n_ctx": args.context,
        "ngl": rc.get("ngl"),
        "load_time_s": load_s,
        "prompt_tokens": _token_count(llm, prompt) if llm is not None else None,
        "reps": reps,
        "decode_tok_s": round(statistics.mean(decode_rates), 1),
        "decode_rates": decode_rates,
        "prefill_est_tok_s": round(statistics.mean(prefill_rates), 1),
        "p50_ms_per_tok": round(p50, 1),
        "p95_ms_per_tok": round(p95, 1),
        "ram_peak_mib": rss_peak[0],
        "vram_peak_mib": peak[0],
        "vram_delta_mib": (vram_after - vram_baseline) if vram_baseline and vram_after else None,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    _apply_ngl_override(args)
    engine = _build_engine(args)
    rc = engine.load()
    print(json.dumps(rc), flush=True)
    import uvicorn

    from bie.serve.openai_compat import app, configure_engine

    configure_engine(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    base = f"http://{args.host}:{args.port}"
    for path in ("/health", "/ready", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=args.timeout) as resp:
                print(path, resp.status, resp.read().decode())
        except urllib.error.HTTPError as exc:
            print(path, exc.code, exc.read().decode())
        except OSError as exc:
            print(path, "unreachable:", exc)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from bie.core.moe_scheduler import MoEScheduler
    from bie.core.scheduler_quant_aware import derive_from_gguf

    scheduler = MoEScheduler(args.model)
    plan = scheduler.compute()
    result: dict[str, Any] = {"model": args.model, "hardware_plan": plan}

    model_path = Path(args.model_path) if args.model_path else Path(f"bie/models/{args.model}.gguf")
    if model_path.is_file():
        try:
            result["gguf_metadata"] = derive_from_gguf(model_path)
        except Exception as exc:
            result["gguf_metadata_error"] = str(exc)

    print(json.dumps(result, indent=2))
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    from bie.skill81.hallucination_guard import HallucinationGuard

    prompt = args.prompt or ""
    output = args.output or ""
    if args.prompt_file and Path(args.prompt_file).is_file():
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if args.output_file and Path(args.output_file).is_file():
        output = Path(args.output_file).read_text(encoding="utf-8")

    guard = HallucinationGuard()
    res = guard.check(output, prompt)
    print(json.dumps(res, indent=2))
    return 0 if res.get("passed", True) else 1


def cmd_mcp(args: argparse.Namespace) -> int:
    from bie.serve.mcp_bridge import run_mcp_server

    run_mcp_server()
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    from bie.tools.bench_matrix import BenchMatrixRunner

    runner = BenchMatrixRunner(
        args.model,
        contexts=args.contexts,
        max_tokens=args.max_tokens,
        reps=args.reps,
    )
    res = runner.run_matrix()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(runner.format_markdown_table(res))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bie.serve")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, fn in (
        ("bench", "rodar benchmark de decode/prefill/VRAM", cmd_bench),
        ("serve", "servir o modelo via uvicorn", cmd_serve),
        ("health", "checar /health, /ready e /v1/models", cmd_health),
        ("inspect", "inspecionar metadados de modelo e plano de offload", cmd_inspect),
        ("guard", "verificar texto contra alucinação e drift (SA-81D)", cmd_guard),
        ("mcp", "iniciar servidor nativo MCP via stdio JSON-RPC 2.0", cmd_mcp),
        ("matrix", "rodar matriz de benchmark em múltiplos contextos", cmd_matrix),
    ):
        p = sub.add_parser(name, help=help_text)
        if name in ("bench", "serve", "health", "inspect", "matrix"):
            p.add_argument("--model", default="qwen3-8b")
            p.add_argument("--model-path", default=None)
            p.add_argument("--context", type=int, default=4096)
            p.add_argument("--threads", type=int, default=None)
            p.add_argument("--batch", type=int, default=512)
            p.add_argument("--n-gpu-layers", type=int, default=None)
            p.add_argument("--verbose", action="store_true")
            p.add_argument(
                "--engine",
                choices=["bie", "llama-server"],
                default="bie",
                help="experimental: use a native llama-server over HTTP instead of the bundled backend",
            )
        if name == "bench":
            p.add_argument("--prompt-tokens", type=int, default=512)
            p.add_argument("--max-tokens", type=int, default=128)
            p.add_argument("--reps", type=int, default=3)
            p.add_argument("--json", action="store_true")
        elif name == "matrix":
            p.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096, 8192])
            p.add_argument("--max-tokens", type=int, default=32)
            p.add_argument("--reps", type=int, default=1)
            p.add_argument("--json", action="store_true")
        elif name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8080)
        elif name == "health":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8080)
            p.add_argument("--timeout", type=float, default=5.0)
        elif name == "guard":
            p.add_argument("--prompt", default="", help="Prompt original")
            p.add_argument("--output", default="", help="Texto de saída gerado")
            p.add_argument("--prompt-file", default=None, help="Caminho para arquivo com o prompt")
            p.add_argument("--output-file", default=None, help="Caminho para arquivo com o output")
        p.set_defaults(fn=fn)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())