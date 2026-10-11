"""Skill 81 Ceiling Extraction — SA-81A/B/C/E automatic prompt engineering."""

from __future__ import annotations

import re
from typing import Any


class CeilingExtractor:
    """Applies Benni OS Skill 81 to every inference call:

    - SA-81A: Role Lock + Task Frame + Constraint Set + Quality Anchor
    - SA-81B: Context compression, dead-weight removal, proximity-causal ordering
    - SA-81C: Forced CoT for complex tasks
    - SA-81E: Model profile-based tuning & format stabilization

    Reference: benni-os/Benni-Master-OS/universal/81_local_llm_os.md
    """

    TASK_TYPE_MAP: dict[str, str] = {
        "DELTA": "research",
        "CHARLIE": "creative",
        "BRAVO": "code",
        "ALPHA": "synthesis",
        "FOXTROT": "analysis",
        "general": "general",
    }

    TASK_FRAMES: dict[str, dict[str, str]] = {
        "research": {
            "role": "Chief Research Analyst (Benni OS DELTA Formation)",
            "frame": "Rigorous evidentiary retrieval and epistemic discipline.",
            "constraints": "Cite verifiable facts, distinguish assumptions from verified claims, zero unanchored claims.",
            "quality_anchor": "Output must provide factual density, verified citations, and clear epistemic qualifiers.",
        },
        "creative": {
            "role": "Narrative Architect (Benni OS CHARLIE Formation)",
            "frame": "High-impact narrative design and creative resonance.",
            "constraints": "Engaging tone, avoid cliches, maintain authorial voice.",
            "quality_anchor": "Elevated linguistic texture, precise pacing, memorable imagery.",
        },
        "code": {
            "role": "Principal Software Systems Engineer (Benni OS BRAVO Formation)",
            "frame": "Production-grade, zero-mock engineering and algorithmic rigor.",
            "constraints": "Strict type hints, defensive error handling, testable units, no placeholder stubs.",
            "quality_anchor": "Idiomatic, deterministic, backward-compatible, production-ready code.",
        },
        "synthesis": {
            "role": "Executive Synthesis Director (Benni OS ALPHA Formation)",
            "frame": "High-density executive synthesis and strategic tradeoff analysis.",
            "constraints": "Brevity, structured decision matrices, concrete next steps, no conversational filler.",
            "quality_anchor": "Decision-ready clarity, trade-off quantification, actionable recommendations.",
        },
        "analysis": {
            "role": "Forensics & Diagnostics Investigator (Benni OS FOXTROT Formation)",
            "frame": "Deep root-cause diagnostics and causal chain reconstruction.",
            "constraints": "Systematic fault isolation, causal evidence, risk prioritization.",
            "quality_anchor": "Definitive causality, verifiable trace evidence, reproducible mitigation steps.",
        },
        "general": {
            "role": "Benni Autonomous Core Engine",
            "frame": "Precise, reliable, instruction-following local inference.",
            "constraints": "Direct answers, strict adherence to user specifications, zero redundant token waste.",
            "quality_anchor": "High utility, factual accuracy, concise execution.",
        },
    }

    def __init__(self, max_context_turns: int = 16, max_turn_chars: int = 4000) -> None:
        self.max_context_turns = max_context_turns
        self.max_turn_chars = max_turn_chars

    def extract(
        self,
        messages: list[dict[str, Any]],
        model_profile: dict[str, Any] | None = None,
        task_type: str = "general",
        forced_cot: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Apply Ceiling Extraction to a message list.

        Returns transformed messages with Skill 81 injected.
        """
        if not messages:
            return []

        profile = model_profile or {}
        task = self.TASK_TYPE_MAP.get(task_type, "general")

        # 1. SA-81B: Context compression & dead-weight pruning
        compressed_messages = self._sa81b_compress(messages)

        # 2. Determine if Forced CoT (SA-81C) should be triggered
        last_user_msg = self._get_last_user_prompt(compressed_messages)
        should_cot = (
            forced_cot
            if forced_cot is not None
            else self._should_force_cot(last_user_msg, task)
        )

        # 3. SA-81A + SA-81E: Inject Role Lock, Task Frame, Constraints, Anchors
        final_messages = self._sa81a_inject(
            compressed_messages, profile, task, should_cot=should_cot
        )
        return final_messages

    def _get_last_user_prompt(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _should_force_cot(self, prompt: str, task: str) -> bool:
        """SA-81C: Trigger CoT for complex algorithmic, mathematical, or architectural tasks."""
        if task in ("code", "analysis"):
            return True
        cot_triggers = (
            r"\b(passo a passo|step[- ]by[- ]step|analise|derive|calculate|proof|architecture|por que|explain why)\b"
        )
        return bool(re.search(cot_triggers, prompt, re.IGNORECASE))

    def _sa81a_inject(
        self,
        messages: list[dict[str, Any]],
        profile: dict[str, Any],
        task: str,
        should_cot: bool = False,
    ) -> list[dict[str, Any]]:
        """SA-81A: Inject Role Lock, Constraints, Quality Anchor into system message."""
        frame_spec = self.TASK_FRAMES.get(task, self.TASK_FRAMES["general"])

        instructions: list[str] = [
            f"[ROLE LOCK]: {frame_spec['role']}",
            f"[TASK FRAME]: {frame_spec['frame']}",
            f"[CONSTRAINTS]: {frame_spec['constraints']}",
            f"[QUALITY ANCHOR]: {frame_spec['quality_anchor']}",
        ]

        if should_cot:
            instructions.append(
                "[SA-81C REASONING]: Think systematically step-by-step before answering to ensure complete accuracy."
            )

        # SA-81E: Profile specific optimizations
        profile_opts = profile.get("skill81_profile", {}).get("optimizations", [])
        if "format_lock_strict" in profile_opts:
            instructions.append(
                "[FORMAT LOCK]: Follow the output structure strictly. Do not deviate with conversational preamble."
            )
        if "explicit_success_criterion" in profile_opts:
            instructions.append(
                "[SUCCESS CRITERION]: Validate your solution against the user's objective before finalizing."
            )

        injected_block = "\n".join(instructions)

        # Update or prepend system message
        new_messages = [dict(m) for m in messages]
        for idx, msg in enumerate(new_messages):
            if msg.get("role") == "system":
                existing_content = msg.get("content", "").strip()
                if injected_block not in existing_content:
                    new_messages[idx]["content"] = (
                        f"{existing_content}\n\n{injected_block}".strip()
                    )
                return new_messages

        # No system message found: prepend one
        new_messages.insert(0, {"role": "system", "content": injected_block})
        return new_messages

    def _sa81b_compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """SA-81B: Remove dead-weight context, preserve causal decisions and recent turns."""
        if len(messages) <= 2:
            return [dict(m) for m in messages]

        system_msgs = [dict(m) for m in messages if m.get("role") == "system"]
        conversation_msgs = [dict(m) for m in messages if m.get("role") != "system"]

        # Keep the most recent turns within max_context_turns
        if len(conversation_msgs) > self.max_context_turns:
            conversation_msgs = conversation_msgs[-self.max_context_turns :]

        # Truncate overly verbose intermediate turns while leaving the final user prompt intact
        compressed_conv: list[dict[str, Any]] = []
        for i, msg in enumerate(conversation_msgs):
            content = str(msg.get("content", ""))
            is_last = i == len(conversation_msgs) - 1
            if not is_last and len(content) > self.max_turn_chars:
                # Retain head and tail for causal context
                half = self.max_turn_chars // 2
                pruned_content = (
                    content[:half]
                    + "\n... [SA-81B: context compressed] ...\n"
                    + content[-half:]
                )
                compressed_conv.append({**msg, "content": pruned_content})
            else:
                compressed_conv.append(dict(msg))

        return system_msgs + compressed_conv
