"""Skill 81 SA-81D — Hallucination & Drift Monitor."""

from __future__ import annotations

import re
from typing import Any


class HallucinationGuard:
    """Classifies output claims as [VERIFIED / INFERENCE / HYPOTHESIS / UNVERIFIED].

    Detects semantic and lexical drift from the original task.
    Identifies unanchored statistical, numeric, and citation assertions.
    Calibrates confidence scores based on evidentiary grounding.

    Reference: benni-os/Benni-Master-OS/universal/81_local_llm_os.md SA-81D
    """

    # Indicators for epistemic classification
    VERIFIED_PATTERNS = re.compile(
        r"\[VERIFIED\]|\b(conforme demonstrado|medido em|testado em|documentado no arquivo|de acordo com o c[oó]digo|according to|measured at|verified by)\b",
        re.IGNORECASE,
    )
    HYPOTHESIS_PATTERNS = re.compile(
        r"\[HYPOTHESIS\]|\b(hip[oó]tese|possivelmente|talvez|pode ser que|supondo que|hypothetically|presumably|might be)\b",
        re.IGNORECASE,
    )
    INFERENCE_PATTERNS = re.compile(
        r"\[INFERENCE\]|\b(portanto|consequentemente|deduz-se que|indica que|suggests that|therefore|consequently|implies that)\b",
        re.IGNORECASE,
    )

    # Patterns indicating high-specificity claims prone to hallucination if not anchored in prompt
    SPECIFIC_METRIC_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|ms|gb|mb|ghz|tb|tok/s|toks|requests/s|x\b)",
        re.IGNORECASE,
    )
    FILE_PATH_PATTERN = re.compile(
        r"(?:[a-zA-Z]:[\\/]|(?:\.{1,2}[\\/])|[a-zA-Z0-9_\-\.]+\.(?:py|json|md|yaml|yml|toml|ts|js|cpp|c|cu))\b",
        re.IGNORECASE,
    )

    def __init__(self, drift_threshold: float = 0.25, confidence_threshold: float = 0.65) -> None:
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold

    def _tokenize_keywords(self, text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\u00C0-\u00FF]{3,}\b", text.lower())
        stopwords = {
            "para", "com", "uma", "esse", "este", "essa", "esta", "mais", "como",
            "pelo", "pela", "que", "dos", "das", "the", "and", "that", "this",
            "with", "from", "for", "are", "was", "were", "been", "have", "has",
        }
        return {w for w in words if w not in stopwords}

    def _calculate_drift(self, output: str, prompt: str) -> float:
        """Calculate drift score (0.0 = aligned, 1.0 = completely disjoint)."""
        prompt_words = self._tokenize_keywords(prompt)
        output_words = self._tokenize_keywords(output)

        if not prompt_words or not output_words:
            return 0.0

        overlap = prompt_words.intersection(output_words)
        # Ratio of prompt concepts addressed in the output
        prompt_coverage = len(overlap) / len(prompt_words)
        drift = max(0.0, 1.0 - prompt_coverage)
        return round(drift, 3)

    def _extract_unanchored_claims(self, output: str, prompt: str) -> list[str]:
        """Detect specific metrics or file references not present in the prompt or code context."""
        unanchored: list[str] = []
        prompt_lower = prompt.lower()

        # Check file references
        found_files = self.FILE_PATH_PATTERN.findall(output)
        for f in found_files:
            if f.lower() not in prompt_lower and not f.startswith("."):
                unanchored.append(f"unanchored_file_reference: {f}")

        # Check numeric metrics that claim certainty without being in prompt
        sentences = re.split(r"[.\n;]", output)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if self.SPECIFIC_METRIC_PATTERN.search(s_clean):
                if not any(kw in prompt_lower for kw in self._tokenize_keywords(s_clean)):
                    if not self.HYPOTHESIS_PATTERNS.search(s_clean) and not self.INFERENCE_PATTERNS.search(s_clean):
                        # Strong claim without grounding or epistemic tag
                        unanchored.append(f"unanchored_assertive_metric: {s_clean[:80]}...")

        return unanchored[:10]

    def check(self, output: str, original_prompt: str) -> dict[str, Any]:
        """Audit output against original prompt and return SA-81D diagnostic."""
        if not output or not output.strip():
            return {
                "passed": False,
                "drift_detected": True,
                "unanchored_claims": ["empty_output"],
                "confidence_score": 0.0,
                "claim_breakdown": {"verified": 0, "inference": 0, "hypothesis": 0, "unverified": 0},
                "drift_score": 1.0,
            }

        sentences = [s.strip() for s in re.split(r"[.\n]+", output) if s.strip()]
        total_claims = max(1, len(sentences))

        verified_count = 0
        inference_count = 0
        hypothesis_count = 0

        for s in sentences:
            if self.VERIFIED_PATTERNS.search(s):
                verified_count += 1
            elif self.HYPOTHESIS_PATTERNS.search(s):
                hypothesis_count += 1
            elif self.INFERENCE_PATTERNS.search(s):
                inference_count += 1

        unverified_count = max(0, total_claims - (verified_count + inference_count + hypothesis_count))

        unanchored = self._extract_unanchored_claims(output, original_prompt)
        drift_score = self._calculate_drift(output, original_prompt)
        drift_detected = drift_score > self.drift_threshold

        # Confidence calculation calibrated by grounding & epistemic transparency
        grounded_ratio = (verified_count * 1.0 + inference_count * 0.8 + hypothesis_count * 0.7) / total_claims
        penalty = min(0.5, len(unanchored) * 0.1) + (0.2 if drift_detected else 0.0)
        confidence = max(0.1, min(1.0, round(grounded_ratio - penalty + 0.3, 2)))

        passed = confidence >= self.confidence_threshold and not (drift_detected and len(unanchored) > 2)

        return {
            "passed": passed,
            "drift_detected": drift_detected,
            "unanchored_claims": unanchored,
            "confidence_score": confidence,
            "claim_breakdown": {
                "verified": verified_count,
                "inference": inference_count,
                "hypothesis": hypothesis_count,
                "unverified": unverified_count,
            },
            "drift_score": drift_score,
        }
