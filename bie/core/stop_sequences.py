"""Streaming Tokenizer & Stop Sequence Matcher for BIE.

Buffers stream chunks to prevent emitting partial stop sequences across token boundaries
(e.g. preventing 'Observation:' or '<|im_end|>' from leaking to clients).
"""

from __future__ import annotations

from typing import Iterator


class StopSequenceMatcher:
    """Detects and suppresses multi-character stop sequences in token streams."""

    def __init__(self, stop_sequences: list[str] | None = None) -> None:
        self.stop_sequences = [s for s in (stop_sequences or []) if s]
        self.max_stop_len = max((len(s) for s in self.stop_sequences), default=0)
        self.buffer = ""
        self.stopped = False

    def process_chunk(self, chunk: str) -> str:
        """Process incoming token chunk and yield text that is guaranteed not part of a stop sequence."""
        if self.stopped or not chunk:
            return ""

        if not self.stop_sequences:
            return chunk

        self.buffer += chunk

        # 1. Check for complete stop sequence match
        for seq in self.stop_sequences:
            idx = self.buffer.find(seq)
            if idx != -1:
                self.stopped = True
                emit = self.buffer[:idx]
                self.buffer = ""
                return emit

        # 2. Check if the buffer ends with a prefix of any stop sequence
        longest_prefix_len = 0
        for seq in self.stop_sequences:
            for i in range(1, len(seq)):
                prefix = seq[:i]
                if self.buffer.endswith(prefix):
                    longest_prefix_len = max(longest_prefix_len, len(prefix))

        # Safe text to emit
        if longest_prefix_len > 0:
            safe_len = len(self.buffer) - longest_prefix_len
            emit = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            return emit

        # No prefix match: emit entire buffer
        emit = self.buffer
        self.buffer = ""
        return emit

    def flush(self) -> str:
        """Flush any remaining buffered text when stream ends."""
        if self.stopped:
            return ""
        emit = self.buffer
        self.buffer = ""
        return emit

    def filter_stream(self, token_iterator: Iterator[str]) -> Iterator[str]:
        """Wrap an iterator of tokens with stop sequence filtering."""
        for chunk in token_iterator:
            out = self.process_chunk(chunk)
            if out:
                yield out
            if self.stopped:
                break
        rem = self.flush()
        if rem:
            yield rem
