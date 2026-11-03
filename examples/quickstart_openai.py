"""Quickstart: Querying benni-inference-engine with the OpenAI SDK.

Prerequisite:
    1. Start the BIE server:
       bie serve --model qwen3-8b --port 8080

    2. Run this script:
       python examples/quickstart_openai.py
"""

from openai import OpenAI


def main():
    # BIE exposes standard OpenAI-compatible endpoints on /v1
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="bie"  # Any string works for local authentication
    )

    print("Listing available models from BIE...")
    models = client.models.list()
    for m in models.data:
        print(f" - Model ID: {m.id}")

    print("\nSending chat completion request...")
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[
            {"role": "system", "content": "You are a concise, high-performance assistant."},
            {"role": "user", "content": "Explain what makes MoE architecture efficient for local inference."}
        ],
        temperature=0.7,
        max_tokens=256
    )

    print("\nResponse from BIE:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
