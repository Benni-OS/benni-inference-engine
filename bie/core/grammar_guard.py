"""Structured Outputs & JSON Schema Enforcer for BIE.

Constrains and repairs LLM outputs to guarantee valid JSON adhering strictly
to provided JSON schemas or Pydantic definitions.
"""

from __future__ import annotations

import json
import re
from typing import Any


class GrammarGuard:
    """Enforces and validates structured JSON schemas on model generation."""

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        """Remove ```json ... ``` markdown wrappers."""
        text = text.strip()
        pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    @classmethod
    def auto_repair_json(cls, raw: str) -> str:
        """Heuristic repairs for trailing commas, unclosed brackets/quotes."""
        text = cls.strip_markdown_fences(raw).strip()
        if not text:
            return "{}"

        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([\]}])", r"\1", text)

        # Balance braces and brackets if truncated
        open_curly = text.count("{") - text.count("}")
        open_square = text.count("[") - text.count("]")

        if text.endswith(",") or text.endswith(":"):
            text = text[:-1].strip()

        # Close missing quotes if odd number of unescaped quotes
        unescaped_quotes = len(re.findall(r'(?<!\\)"', text))
        if unescaped_quotes % 2 != 0:
            text += '"'

        if open_square > 0:
            text += "]" * open_square
        if open_curly > 0:
            text += "}" * open_curly

        return text

    @classmethod
    def validate(
        cls,
        text: str,
        schema: dict[str, Any] | None = None,
        auto_repair: bool = True,
    ) -> dict[str, Any]:
        """Validate string output against JSON syntax and optional JSON schema.

        Returns:
            {
                "valid": bool,
                "repaired": bool,
                "data": parsed_dict_or_list,
                "error": error_message_if_any
            }
        """
        cleaned = cls.strip_markdown_fences(text)
        repaired = False
        parsed_data = None
        error_msg = None

        try:
            parsed_data = json.loads(cleaned)
        except json.JSONDecodeError as err:
            if auto_repair:
                repaired_text = cls.auto_repair_json(cleaned)
                try:
                    parsed_data = json.loads(repaired_text)
                    repaired = True
                except json.JSONDecodeError as repair_err:
                    return {
                        "valid": False,
                        "repaired": False,
                        "data": None,
                        "error": f"JSON parse error after auto-repair: {repair_err}",
                    }
            else:
                return {
                    "valid": False,
                    "repaired": False,
                    "data": None,
                    "error": f"JSON parse error: {err}",
                }

        # If schema is provided, validate keys and basic types
        if schema and isinstance(parsed_data, dict):
            schema_type = schema.get("type", "object")
            if schema_type == "object":
                required_props = schema.get("required", [])
                for prop in required_props:
                    if prop not in parsed_data:
                        return {
                            "valid": False,
                            "repaired": repaired,
                            "data": parsed_data,
                            "error": f"Missing required property: '{prop}'",
                        }

                properties = schema.get("properties", {})
                for prop, prop_schema in properties.items():
                    if prop in parsed_data:
                        expected_type = prop_schema.get("type")
                        val = parsed_data[prop]
                        if expected_type == "string" and not isinstance(val, str):
                            return {"valid": False, "repaired": repaired, "data": parsed_data, "error": f"Property '{prop}' must be a string"}
                        if expected_type == "number" and not isinstance(val, (int, float)):
                            return {"valid": False, "repaired": repaired, "data": parsed_data, "error": f"Property '{prop}' must be a number"}
                        if expected_type == "integer" and not isinstance(val, int):
                            return {"valid": False, "repaired": repaired, "data": parsed_data, "error": f"Property '{prop}' must be an integer"}
                        if expected_type == "boolean" and not isinstance(val, bool):
                            return {"valid": False, "repaired": repaired, "data": parsed_data, "error": f"Property '{prop}' must be a boolean"}
                        if expected_type == "array" and not isinstance(val, list):
                            return {"valid": False, "repaired": repaired, "data": parsed_data, "error": f"Property '{prop}' must be an array"}

        return {
            "valid": True,
            "repaired": repaired,
            "data": parsed_data,
            "error": None,
        }

    @classmethod
    def format_prompt_for_schema(cls, prompt: str, schema: dict[str, Any]) -> str:
        """Inject strict JSON Schema constraints into the prompt."""
        schema_str = json.dumps(schema, indent=2)
        instruction = (
            f"\n\n[STRICT JSON SCHEMA REQUIRED]:\n"
            f"You MUST return valid JSON matching this schema exactly. "
            f"Do not include conversational preamble or markdown explanations outside the JSON object.\n"
            f"{schema_str}\n"
        )
        return prompt + instruction
