"""Continuous Batching & Request Slot Multiplexer for BIE.

Implements multi-slot concurrent generation by coordinating step-by-step
token generation across active slots without stalling new incoming requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SlotState(str, Enum):
    IDLE = "idle"
    PREFILLING = "prefilling"
    GENERATING = "generating"
    COMPLETED = "completed"


@dataclass
class RequestSlot:
    slot_id: int
    request_id: str | None = None
    prompt: str = ""
    generated_text: str = ""
    max_tokens: int = 128
    tokens_generated: int = 0
    state: SlotState = SlotState.IDLE
    start_time: float = 0.0


class ContinuousBatcher:
    """Coordinates concurrent inference across fixed slots."""

    def __init__(self, num_slots: int = 4) -> None:
        self.num_slots = max(1, num_slots)
        self.slots: list[RequestSlot] = [
            RequestSlot(slot_id=i) for i in range(self.num_slots)
        ]
        self.queue: list[dict[str, Any]] = []

    def get_idle_slot(self) -> RequestSlot | None:
        """Find first available idle slot."""
        for slot in self.slots:
            if slot.state == SlotState.IDLE:
                return slot
        return None

    def add_request(self, request_id: str, prompt: str, max_tokens: int = 128) -> bool:
        """Assign request to an idle slot or add to waiting queue."""
        slot = self.get_idle_slot()
        if slot is not None:
            slot.request_id = request_id
            slot.prompt = prompt
            slot.generated_text = ""
            slot.max_tokens = max_tokens
            slot.tokens_generated = 0
            slot.state = SlotState.PREFILLING
            slot.start_time = time.time()
            return True

        self.queue.append({
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
        })
        return False

    def step(self, generate_token_fn: Callable[[RequestSlot], str]) -> dict[str, Any]:
        """Perform one forward step across all active generating slots."""
        step_outputs: dict[str, str] = {}
        completed: list[str] = []

        for slot in self.slots:
            if slot.state in (SlotState.PREFILLING, SlotState.GENERATING):
                token = generate_token_fn(slot)
                slot.generated_text += token
                slot.tokens_generated += 1
                slot.state = SlotState.GENERATING

                req_id = slot.request_id or f"slot_{slot.slot_id}"
                step_outputs[req_id] = token

                # Completion checks
                if slot.tokens_generated >= slot.max_tokens or token in ("<|im_end|>", "</s>", ""):
                    slot.state = SlotState.COMPLETED
                    completed.append(req_id)

                    # Free slot and drain queue
                    slot.state = SlotState.IDLE
                    slot.request_id = None
                    if self.queue:
                        next_req = self.queue.pop(0)
                        slot.request_id = next_req["request_id"]
                        slot.prompt = next_req["prompt"]
                        slot.generated_text = ""
                        slot.max_tokens = next_req["max_tokens"]
                        slot.tokens_generated = 0
                        slot.state = SlotState.PREFILLING
                        slot.start_time = time.time()

        return {
            "active_slots": sum(1 for s in self.slots if s.state != SlotState.IDLE),
            "step_outputs": step_outputs,
            "completed_requests": completed,
            "queue_depth": len(self.queue),
        }
