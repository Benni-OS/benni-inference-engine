"""Prompt Template Engine for BIE.

Converts message arrays into model-native chat formatting strings
(ChatML, Llama-3, DeepSeek, Alpaca, raw).
"""

from __future__ import annotations

import re
from typing import Any, Literal

TemplateStyle = Literal["chatml", "llama3", "deepseek", "gemma", "alpaca", "raw"]


class PromptTemplateEngine:
    """Renders conversation messages into exact model-specific prompt formats."""

    _THINK_BLOCK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)

    @classmethod
    def detect_style(cls, model_name: str) -> TemplateStyle:
        """Detect formatting template from model ID or file name."""
        name = model_name.lower()
        if "gemma" in name:
            return "gemma"
        if "llama-3" in name or "llama3" in name:
            return "llama3"
        if "deepseek" in name:
            return "deepseek"
        if "alpaca" in name or "vicuna" in name:
            return "alpaca"
        if "qwen" in name or "kimi" in name or "cyber-tiel" in name:
            return "chatml"
        return "chatml"

    @classmethod
    def parse_reasoning(cls, text: str) -> tuple[str, str | None]:
        """Split ``<think>...</think>`` traces from the final assistant content."""
        if not text:
            return "", None
        thinks = [m.strip() for m in cls._THINK_BLOCK_RE.findall(text) if m.strip()]
        if thinks:
            cleaned = cls._THINK_BLOCK_RE.sub("", text).strip()
            reasoning = "\n\n".join(thinks)
            return (cleaned if cleaned else text.strip()), reasoning
        if "</think>" in text:
            head, tail = text.rsplit("</think>", 1)
            reasoning = head.replace("<think>", "").strip()
            cleaned = tail.strip()
            return (cleaned if cleaned else text.strip()), (reasoning or None)
        return text, None

    @classmethod
    def render(
        cls,
        messages: list[dict[str, Any]],
        style: TemplateStyle = "chatml",
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages according to selected template style."""
        if not messages:
            return ""

        if style == "chatml":
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

        if style == "llama3":
            parts = ["<|begin_of_text|>"]
            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            if add_generation_prompt:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        if style == "deepseek":
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                if role == "system":
                    parts.append(f"{content}\n\n")
                elif role == "user":
                    parts.append(f"<｜User｜>{content}")
                elif role == "assistant":
                    parts.append(f"<｜Assistant｜>{content}")
            if add_generation_prompt:
                parts.append("<｜Assistant｜>")
            return "".join(parts)

        if style == "gemma":
            parts = []
            system_prefix = ""
            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                if role == "system":
                    system_prefix += f"{content}\n\n"
                    continue
                turn_role = "model" if role == "assistant" else "user"
                if turn_role == "user" and system_prefix:
                    content = f"{system_prefix}{content}"
                    system_prefix = ""
                parts.append(f"<start_of_turn>{turn_role}\n{content}<end_of_turn>")
            if system_prefix:
                parts.append(f"<start_of_turn>user\n{system_prefix.strip()}<end_of_turn>")
            if add_generation_prompt:
                parts.append("<start_of_turn>model\n")
            return "\n".join(parts)

        if style == "alpaca":
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                if role == "system":
                    parts.append(f"{content}\n\n")
                elif role == "user":
                    parts.append(f"### Instruction:\n{content}\n\n")
                elif role == "assistant":
                    parts.append(f"### Response:\n{content}\n\n")
            if add_generation_prompt:
                parts.append("### Response:\n")
            return "".join(parts)

        # Raw fallback
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        return "\n".join(parts)
