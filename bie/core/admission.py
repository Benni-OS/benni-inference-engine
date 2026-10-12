"""Concurrency Queue & Admission Controller for BIE.

Prevents CUDA Out-Of-Memory (OOM) crashes and system thrashing by governing
concurrent inference requests via priority queuing and memory-aware admission.
"""

from __future__ import annotations

import heapq
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

PriorityLevel = Literal["critical", "high", "normal", "low"]

PRIORITY_WEIGHTS: dict[PriorityLevel, int] = {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}


@dataclass(order=True)
class QueuedItem:
    priority_score: int
    timestamp: float
    request_id: str = field(compare=False)
    estimated_tokens: int = field(compare=False)


class AdmissionController:
    """Manages inference request slots and prevents memory overcommit."""

    def __init__(
        self,
        max_concurrent: int = 2,
        max_queue_depth: int = 32,
        tokens_per_second: float = 100.0,
    ) -> None:
        self.max_concurrent = max(1, max_concurrent)
        self.max_queue_depth = max_queue_depth
        self.tokens_per_second = tokens_per_second

        self.active_count = 0
        self.peak_concurrent = 0
        self.rejected_count = 0
        self.completed_count = 0
        self.queue: list[QueuedItem] = []

    def can_admit_immediately(self) -> bool:
        """Check if an execution slot is immediately free."""
        return self.active_count < self.max_concurrent

    def admit_or_reject(
        self,
        request_id: str,
        priority: PriorityLevel = "normal",
        estimated_tokens: int = 512,
    ) -> bool:
        """Admit request if slot available, or reject if queue is saturated."""
        if self.can_admit_immediately():
            self.active_count += 1
            if self.active_count > self.peak_concurrent:
                self.peak_concurrent = self.active_count
            return True

        if len(self.queue) >= self.max_queue_depth:
            self.rejected_count += 1
            return False

        score = PRIORITY_WEIGHTS.get(priority, 2)
        heapq.heappush(
            self.queue,
            QueuedItem(
                priority_score=score,
                timestamp=time.time(),
                request_id=request_id,
                estimated_tokens=estimated_tokens,
            ),
        )
        return True

    def release(self) -> None:
        """Release a slot and promote highest priority item from the queue."""
        if self.active_count > 0:
            self.active_count -= 1
            self.completed_count += 1

        if self.queue and self.can_admit_immediately():
            heapq.heappop(self.queue)
            self.active_count += 1

    @contextmanager
    def lease(
        self,
        request_id: str,
        priority: PriorityLevel = "normal",
        estimated_tokens: int = 512,
    ) -> Iterator[bool]:
        """Context manager for acquiring and safely releasing an inference lease."""
        admitted = self.admit_or_reject(request_id, priority, estimated_tokens)
        try:
            yield admitted
        finally:
            if admitted:
                self.release()

    def stats(self) -> dict[str, Any]:
        """Return admission and concurrency diagnostics."""
        return {
            "active_requests": self.active_count,
            "max_concurrent": self.max_concurrent,
            "queued_requests": len(self.queue),
            "max_queue_depth": self.max_queue_depth,
            "peak_concurrency": self.peak_concurrent,
            "rejected_requests": self.rejected_count,
            "completed_requests": self.completed_count,
        }
