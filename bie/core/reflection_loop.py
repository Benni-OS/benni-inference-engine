"""Self-Correction & Autonomous Reflection Loop for BIE.

Executes autonomous iterative refinement when an initial generation fails
HallucinationGuard (SA-81D) or GrammarGuard (schema validation).
"""

from __future__ import annotations

from typing import Any, Callable

from ..skill81.hallucination_guard import HallucinationGuard
from .grammar_guard import GrammarGuard


class ReflectionLoop:
    """Iteratively refines model outputs using automated critique prompts."""

    def __init__(
        self,
        max_attempts: int = 2,
        guard: HallucinationGuard | None = None,
        temperature_decay: float = 0.7,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.guard = guard or HallucinationGuard()
        self.temperature_decay = temperature_decay

    def execute_with_reflection(
        self,
        generate_fn: Callable[[list[dict[str, Any]], float], str],
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        initial_temperature: float = 0.7,
        check_guard: bool = False,
    ) -> dict[str, Any]:
        """Generate response with self-correction feedback loop if validation fails."""
        current_messages = [dict(m) for m in messages]
        temp = initial_temperature
        critique_history: list[str] = []

        last_output = ""
        passed = False

        for attempt in range(self.max_attempts + 1):
            last_output = generate_fn(current_messages, temp)
            issues: list[str] = []

            # 1. Check Schema validation
            if schema:
                val = GrammarGuard.validate(last_output, schema=schema)
                if not val["valid"]:
                    issues.append(f"Schema error: {val['error']}")

            # 2. Check Hallucination Guard (if enabled)
            if check_guard:
                last_user_prompt = messages[-1].get("content", "") if messages else ""
                guard_report = self.guard.check(last_output, last_user_prompt)
                if not guard_report["passed"]:
                    if guard_report["drift_detected"]:
                        issues.append("Response drifted off-topic from original user prompt.")
                    if guard_report["unanchored_claims"]:
                        issues.append(f"Unanchored claims detected: {', '.join(guard_report['unanchored_claims'][:2])}")

            if not issues or attempt == self.max_attempts:
                passed = len(issues) == 0
                break

            # Construct critique feedback
            critique = (
                f"[SELF-CORRECTION FEEDBACK]: Your previous response had the following issues:\n"
                + "\n".join(f"- {iss}" for iss in issues)
                + "\nPlease regenerate your response correcting all errors."
            )
            critique_history.append(critique)

            # Append assistant response and critique to conversation
            current_messages.append({"role": "assistant", "content": last_output})
            current_messages.append({"role": "user", "content": critique})
            temp = max(0.1, temp * self.temperature_decay)

        return {
            "output": last_output,
            "passed": passed,
            "attempts": len(critique_history) + 1,
            "critiques": critique_history,
        }
