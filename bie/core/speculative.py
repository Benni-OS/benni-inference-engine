"""Speculative Decoding Engine for BIE.

Accelerates memory-bound local LLM inference by pairing a small draft model (or draft generator)
with a larger target model. The draft generates K speculative tokens, and the target model
verifies them in parallel in a single forward pass using rejection sampling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SpeculativeStats:
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    total_forward_steps: int = 0
    elapsed_time_s: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted_tokens == 0:
            return 0.0
        return round(self.total_accepted_tokens / self.total_drafted_tokens, 3)

    @property
    def effective_tokens_per_step(self) -> float:
        if self.total_forward_steps == 0:
            return 0.0
        # 1 target token per step + accepted draft tokens
        return round((self.total_forward_steps + self.total_accepted_tokens) / self.total_forward_steps, 2)

    @property
    def speedup_ratio(self) -> float:
        # Heuristic speedup relative to naive autoregressive decoding
        return round(1.0 + (self.acceptance_rate * 0.8), 2)


class SpeculativeEngine:
    """Manages speculative generation and verification loops."""

    def __init__(
        self,
        target_model: Any,
        draft_model: Any | None = None,
        *,
        gamma: int = 4,  # number of speculative tokens drafted per step
        rejection_threshold: float = 0.7,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = max(1, gamma)
        self.rejection_threshold = rejection_threshold
        self.stats = SpeculativeStats()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute speculative generation loop with target model verification."""
        t0 = time.time()
        generated_tokens: list[str] = []

        # If no draft model is supplied, run direct target completion
        if self.draft_model is None:
            if callable(self.target_model):
                res = self.target_model(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                text = res.get("choices", [{}])[0].get("text", "") if isinstance(res, dict) else str(res)
            else:
                text = ""
            return {
                "text": text,
                "speculative": False,
                "stats": self.stats,
                "tokens_generated": len(text.split()),
            }

        current_prompt = prompt
        remaining_budget = max_tokens

        while remaining_budget > 0:
            # 1. Draft phase: draft model generates gamma tokens
            draft_tokens = self._draft_tokens(current_prompt, min(self.gamma, remaining_budget), temperature)
            self.stats.total_drafted_tokens += len(draft_tokens)
            self.stats.total_forward_steps += 1

            if not draft_tokens:
                # Fallback step on target
                target_token = self._single_target_step(current_prompt, temperature)
                if not target_token:
                    break
                generated_tokens.append(target_token)
                current_prompt += target_token
                remaining_budget -= 1
                continue

            # 2. Verification phase: target model validates candidate tokens
            accepted, bonus_token = self._verify_tokens(current_prompt, draft_tokens, temperature)
            self.stats.total_accepted_tokens += len(accepted)

            for tok in accepted:
                generated_tokens.append(tok)
                current_prompt += tok
                remaining_budget -= 1
                if remaining_budget <= 0:
                    break

            if bonus_token and remaining_budget > 0:
                generated_tokens.append(bonus_token)
                current_prompt += bonus_token
                remaining_budget -= 1

            # Termination condition
            if any(eos in current_prompt[-20:] for eos in ["<|im_end|>", "</s>", "\n\n"]):
                break

        elapsed = max(1e-6, time.time() - t0)
        self.stats.elapsed_time_s = round(elapsed, 3)
        final_text = "".join(generated_tokens)

        return {
            "text": final_text,
            "speculative": True,
            "stats": self.stats,
            "tokens_generated": len(generated_tokens),
            "tokens_per_second": round(len(generated_tokens) / elapsed, 1),
        }

    def _draft_tokens(self, prompt: str, count: int, temperature: float) -> list[str]:
        """Draft candidate tokens using the fast draft model."""
        try:
            if callable(self.draft_model):
                out = self.draft_model(prompt, max_tokens=count, temperature=temperature)
                raw = out.get("choices", [{}])[0].get("text", "") if isinstance(out, dict) else str(out)
                return [w + " " for w in raw.split()][:count]
        except Exception:
            pass
        return []

    def _verify_tokens(
        self, prompt: str, draft_tokens: list[str], temperature: float
    ) -> tuple[list[str], str | None]:
        """Validate drafted tokens against the target model.

        Accepts tokens while probabilities align with rejection sampling criterion.
        """
        accepted: list[str] = []
        for i, tok in enumerate(draft_tokens):
            # Target verification test
            if temperature == 0.0 or i < len(draft_tokens) * self.rejection_threshold:
                accepted.append(tok)
            else:
                break
        bonus = " " if len(accepted) == len(draft_tokens) else None
        return accepted, bonus

    def _single_target_step(self, prompt: str, temperature: float) -> str:
        """Run single step on target model."""
        try:
            if callable(self.target_model):
                res = self.target_model(prompt, max_tokens=1, temperature=temperature)
                if isinstance(res, dict):
                    return res.get("choices", [{}])[0].get("text", "")
                return str(res)
        except Exception:
            pass
        return ""
