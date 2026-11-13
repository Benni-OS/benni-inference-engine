"""Drop-In OpenAI SDK Quickstart for benni-inference-engine (BIE).

Demonstrates:
  1. Chat completions (standard + streaming)
  2. Structured tool / function calling
  3. L2-normalized vector embeddings

Prerequisites:
    # 1. Inspect local GPU VRAM & zero-RAM-spill context limits
    bie doctor

    # 2. Start the OpenAI-compatible BIE server
    bie serve --port 8080

    # 3. Run this drop-in client script
    python examples/quickstart_openai_dropin.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


def build_client(base_url: str = "http://localhost:8080/v1", api_key: str = "bie") -> Any:
    """Create an OpenAI SDK client pointed at the local BIE server."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def build_tool_definitions() -> list[dict[str, Any]]:
    """Return an example OpenAI-format tool definition for function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "estimate_kv_budget",
                "description": "Estimate remaining VRAM budget for quantized KV-cache.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpu_vram_gb": {"type": "number", "description": "Total GPU VRAM in GB"},
                        "model_vram_gb": {"type": "number", "description": "Model weights in VRAM (GB)"},
                    },
                    "required": ["gpu_vram_gb", "model_vram_gb"],
                },
            },
        }
    ]


def run_dropin_demo(
    base_url: str = "http://localhost:8080/v1",
    model: str = "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf",
) -> None:
    """Execute a complete drop-in OpenAI SDK workflow against BIE."""
    client = build_client(base_url=base_url)

    print(f"Connecting to BIE OpenAI-compatible endpoint at {base_url}...")
    models = client.models.list()
    print("Active models:", [m.id for m in models.data])

    print("\n1) Streaming chat completion:")
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise systems engineer."},
            {"role": "user", "content": "Explain why INT4 KV-cache prevents RAM spill on 8GB GPUs."},
        ],
        max_tokens=160,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print("\n")

    print("2) Function / Tool Calling:")
    tool_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Calculate KV budget for an 8GB GPU with 5.5GB weights."}],
        tools=build_tool_definitions(),
        tool_choice="auto",
        max_tokens=128,
    )
    msg = tool_resp.choices[0].message
    print("Assistant content:", msg.content)
    if getattr(msg, "tool_calls", None):
        print("Tool calls:", [tc.function.name for tc in msg.tool_calls])

    print("\n3) Vector Embeddings:")
    emb = client.embeddings.create(input="Zero-RAM-spill MoE inference on 8GB consumer GPUs")
    print(f"Embedding dimensions: {len(emb.data[0].embedding)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BIE OpenAI Drop-In Quickstart")
    parser.add_argument("--base-url", default=os.environ.get("BIE_BASE_URL", "http://localhost:8080/v1"))
    parser.add_argument("--model", default="Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf")
    args = parser.parse_args(argv)

    try:
        run_dropin_demo(base_url=args.base_url, model=args.model)
        return 0
    except ImportError:
        print("Install the OpenAI SDK first: pip install openai", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Could not reach BIE server at {args.base_url}: {exc}", file=sys.stderr)
        print("Start BIE first:\n  bie doctor\n  bie serve --port 8080", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
