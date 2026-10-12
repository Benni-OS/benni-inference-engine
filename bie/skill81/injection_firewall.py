"""Guardrails & Prompt Injection Firewall for BIE (Skill 63 / SA-81 Security Gate).

Inspects incoming prompts for adversarial jailbreaks, role-override attacks,
base64 obfuscation, unicode exploits, and prompt extraction attempts.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, Literal

ThreatLevel = Literal["clean", "suspicious", "malicious"]


@dataclass
class InspectionResult:
    safe: bool
    threat_level: ThreatLevel
    matched_rules: list[str]
    sanitized_prompt: str


class PromptInjectionFirewall:
    """Pre-inference firewall defending local LLMs against malicious adversarial inputs."""

    # Explicit jailbreak and bypass signatures
    JAILBREAK_PATTERNS = [
        (re.compile(r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b", re.I), "rule_ignore_instructions"),
        (re.compile(r"\b(you\s+are\s+now|act\s+as)\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|dan|evil)\b", re.I), "rule_dan_jailbreak"),
        (re.compile(r"\b(developer\s+mode|god\s+mode|root\s+access|admin\s+override)\b", re.I), "rule_privilege_override"),
        (re.compile(r"\b(output|print|reveal|leak|repeat)\s+(the\s+)?(system\s+prompt|initial\s+instructions|hidden\s+rules)\b", re.I), "rule_system_prompt_leak"),
    ]

    # Delimiter and role spoofing attempts
    ROLE_SPOOF_PATTERNS = [
        (re.compile(r"<\|im_start\|>\s*(system|developer)", re.I), "rule_chatml_role_spoof"),
        (re.compile(r"\[(SYSTEM|ROLE LOCK|TASK FRAME)\]", re.I), "rule_skill81_frame_spoof"),
        (re.compile(r"^\s*system:\s*", re.I | re.M), "rule_raw_system_prefix"),
    ]

    # Base64 payload detection (strings > 32 chars of pure base64)
    B64_REGEX = re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b")

    def inspect(self, prompt: str) -> InspectionResult:
        """Scan input prompt and return structured safety diagnostic."""
        if not prompt or not prompt.strip():
            return InspectionResult(safe=True, threat_level="clean", matched_rules=[], sanitized_prompt="")

        matched_rules: list[str] = []
        sanitized = prompt

        # 1. Check Jailbreak Patterns
        for pattern, rule_name in self.JAILBREAK_PATTERNS:
            if pattern.search(prompt):
                matched_rules.append(rule_name)

        # 2. Check Role Spoofing Patterns
        for pattern, rule_name in self.ROLE_SPOOF_PATTERNS:
            if pattern.search(prompt):
                matched_rules.append(rule_name)
                # Strip out spoofed delimiter from sanitized prompt
                sanitized = pattern.sub("[REDACTED_SPOOF]", sanitized)

        # 3. Check for Hidden Unicode Zero-Width Characters
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"]
        if any(c in prompt for c in zero_width_chars):
            matched_rules.append("rule_hidden_zero_width_unicode")
            for c in zero_width_chars:
                sanitized = sanitized.replace(c, "")

        # 4. Check Base64 payloads and decode inspection
        b64_matches = self.B64_REGEX.findall(prompt)
        for b64_cand in b64_matches:
            try:
                decoded = base64.b64decode(b64_cand).decode("utf-8", errors="ignore")
                for pattern, _ in self.JAILBREAK_PATTERNS:
                    if pattern.search(decoded):
                        matched_rules.append("rule_base64_obfuscated_jailbreak")
                        break
            except Exception:
                pass

        # Threat level assessment
        if any("dan" in r or "override" in r or "obfuscated" in r for r in matched_rules):
            threat: ThreatLevel = "malicious"
            safe = False
        elif matched_rules:
            threat = "suspicious"
            safe = False
        else:
            threat = "clean"
            safe = True

        return InspectionResult(
            safe=safe,
            threat_level=threat,
            matched_rules=matched_rules,
            sanitized_prompt=sanitized,
        )
