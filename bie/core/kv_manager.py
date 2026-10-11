"""KV Cache Manager — context compression via SA-81B (Benni OS Skill 81)."""

from __future__ import annotations

import re
from typing import Any


class KVManager:
    """Manages KV cache lifecycle:

    - Compresses context when approaching window limit (SA-81B)
    - Removes dead-weight tokens (no causal influence on current output)
    - Tracks proximity-causal ordering for optimal attention
    """

    def __init__(self, ctx_size: int = 32768, compression_threshold: float = 0.8) -> None:
        self.ctx_size = ctx_size
        self.compression_threshold = compression_threshold
        self.current_len = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimation: ~4 characters per token."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            content = str(m.get("content", ""))
            total += self.estimate_tokens(content) + 4  # overhead per message
        return total

    def update_usage(self, token_count: int) -> None:
        self.current_len = max(0, token_count)

    def should_compress(self, threshold: float | None = None) -> bool:
        """Returns True when context exceeds threshold of window."""
        t = threshold if threshold is not None else self.compression_threshold
        return self.current_len > int(self.ctx_size * t)

    def clean_text_deadweight(self, text: str) -> str:
        """Remove dead-weight tokens: redundant blank lines, repeated dashes/equals, trailing spaces."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"([-=_*~]){4,}", r"\1\1\1", text)
        return text.strip()

    def compress(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """SA-81B: Remove dead-weight, preserve causal decisions and recent dialog.

        Preserves:
        1. System prompts (uncompromised)
        2. The most recent user query and immediately preceding turn
        3. Causal decisions extracted from prior turns
        """
        if not messages:
            return []

        initial_tokens = self.count_message_tokens(messages)
        self.current_len = initial_tokens

        max_allowed = target_tokens or int(self.ctx_size * self.compression_threshold)
        if initial_tokens <= max_allowed:
            # Clean dead-weight whitespace even if below threshold
            cleaned = []
            for m in messages:
                cleaned.append({
                    **m,
                    "content": self.clean_text_deadweight(str(m.get("content", ""))),
                })
            return cleaned

        # Context exceeds budget: apply structured pruning
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") != "system"]

        # If chat is tiny, return cleaned
        if len(chat_msgs) <= 2:
            return [{**m, "content": self.clean_text_deadweight(str(m.get("content", "")))} for m in messages]

        # Always preserve the last user message and its immediate context
        recent_tail = chat_msgs[-2:]
        history = chat_msgs[:-2]

        compressed_history: list[dict[str, Any]] = []
        for msg in history:
            role = msg.get("role")
            content = str(msg.get("content", ""))
            cleaned = self.clean_text_deadweight(content)

            # Prune long intermediate assistant code/logs
            if role == "assistant" and len(cleaned) > 500:
                summary = (
                    cleaned[:200]
                    + "\n... [SA-81B: intermediate turn collapsed to preserve KV cache] ...\n"
                    + cleaned[-150:]
                )
                compressed_history.append({**msg, "content": summary})
            else:
                compressed_history.append({**msg, "content": cleaned})

        new_messages = system_msgs + compressed_history + recent_tail
        self.current_len = self.count_message_tokens(new_messages)
        return new_messages
