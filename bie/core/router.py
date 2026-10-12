"""Hybrid Task Routing Engine for BIE.

Directs tasks between local BIE inference and cloud upstream providers based on
formation rules, privacy requirements, token budgets, and model capabilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class RouteDecision:
    target: Literal["local", "cloud"]
    reason: str
    task_type: str
    estimated_tokens: int
    privacy_score: float  # 1.0 = fully local/sovereign, 0.0 = third-party cloud
    estimated_cost_usd: float


class TaskRouter:
    """Evaluates prompts and formations to select the optimal inference target."""

    # Cost per 1k tokens benchmark (Cloud reference: $0.003 / 1k tokens)
    CLOUD_COST_PER_1K: float = 0.003

    def __init__(
        self,
        local_context_limit: int = 32768,
        default_privacy_mode: str = "sovereign",  # "sovereign" | "hybrid" | "cloud_preferred"
    ) -> None:
        self.local_context_limit = local_context_limit
        self.privacy_mode = default_privacy_mode
        self.local_requests_count = 0
        self.cloud_requests_count = 0
        self.total_cost_saved_usd = 0.0

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def route(
        self,
        prompt: str,
        task_type: str = "general",
        *,
        privacy_override: str | None = None,
        force_target: Literal["local", "cloud"] | None = None,
    ) -> RouteDecision:
        """Determine whether to process locally or dispatch to cloud."""
        if force_target is not None:
            return RouteDecision(
                target=force_target,
                reason=f"Explicitly forced to {force_target}",
                task_type=task_type,
                estimated_tokens=self.estimate_tokens(prompt),
                privacy_score=1.0 if force_target == "local" else 0.0,
                estimated_cost_usd=0.0 if force_target == "local" else 0.005,
            )

        privacy = privacy_override or self.privacy_mode
        tokens = self.estimate_tokens(prompt)

        # 1. Sovereign Privacy Enforcement (Rule Tier 0)
        if privacy == "sovereign":
            self.local_requests_count += 1
            cost_saved = (tokens / 1000.0) * self.CLOUD_COST_PER_1K
            self.total_cost_saved_usd += cost_saved
            return RouteDecision(
                target="local",
                reason="Sovereign mode active: strictly local execution required.",
                task_type=task_type,
                estimated_tokens=tokens,
                privacy_score=1.0,
                estimated_cost_usd=0.0,
            )

        # 2. Context Window Boundary Check
        if tokens > int(self.local_context_limit * 0.9):
            self.cloud_requests_count += 1
            cost = (tokens / 1000.0) * self.CLOUD_COST_PER_1K
            return RouteDecision(
                target="cloud",
                reason=f"Prompt tokens ({tokens}) exceed 90% of local context window ({self.local_context_limit}).",
                task_type=task_type,
                estimated_tokens=tokens,
                privacy_score=0.2,
                estimated_cost_usd=round(cost, 4),
            )

        # 3. Formation-Specific Policies
        # DELTA (research), BRAVO (code), CHARLIE (creative) -> Local optimal
        if task_type in ("DELTA", "BRAVO", "CHARLIE", "general"):
            self.local_requests_count += 1
            cost_saved = (tokens / 1000.0) * self.CLOUD_COST_PER_1K
            self.total_cost_saved_usd += cost_saved
            return RouteDecision(
                target="local",
                reason=f"Task formation '{task_type}' mapped to local sovereign execution.",
                task_type=task_type,
                estimated_tokens=tokens,
                privacy_score=1.0,
                estimated_cost_usd=0.0,
            )

        # ALPHA (heavy executive synthesis) or FOXTROT (heavy forensics)
        if task_type in ("ALPHA", "FOXTROT") and privacy == "hybrid":
            self.cloud_requests_count += 1
            cost = (tokens / 1000.0) * self.CLOUD_COST_PER_1K
            return RouteDecision(
                target="cloud",
                reason=f"Formation '{task_type}' with hybrid policy routed to frontier cloud reasoning.",
                task_type=task_type,
                estimated_tokens=tokens,
                privacy_score=0.5,
                estimated_cost_usd=round(cost, 4),
            )

        # Default fallback to local
        self.local_requests_count += 1
        return RouteDecision(
            target="local",
            reason="Default local BIE execution.",
            task_type=task_type,
            estimated_tokens=tokens,
            privacy_score=1.0,
            estimated_cost_usd=0.0,
        )

    def telemetry(self) -> dict[str, Any]:
        """Return routing telemetry metrics."""
        total = self.local_requests_count + self.cloud_requests_count
        local_ratio = round(self.local_requests_count / total, 3) if total > 0 else 1.0
        return {
            "total_requests": total,
            "local_requests": self.local_requests_count,
            "cloud_requests": self.cloud_requests_count,
            "local_ratio": local_ratio,
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
        }
