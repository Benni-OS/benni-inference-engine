"""OpenAI-compatible Tool & Function Calling Engine for BIE.

Formats tools into ChatML instructions and parses model-generated tool calls into
standard OpenAI JSON objects: `choices[0].message.tool_calls`.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


class ToolCallingEngine:
    """Handles tool definition injection and tool call response extraction."""

    TOOL_CALL_REGEX = re.compile(
        r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE
    )
    JSON_BLOCK_REGEX = re.compile(
        r"```(?:json)?\s*(\{\s*\"(?:name|function|tool)\"[\s\S]*?\})\s*```", re.IGNORECASE
    )

    @classmethod
    def format_tools_system_message(cls, tools: list[dict[str, Any]]) -> str:
        """Render registered tools into system instructions."""
        tools_doc = json.dumps(tools, indent=2)
        return (
            "\n\n# TOOLS AVAILABLE\n"
            "You have access to the following tools. If you need to invoke a tool, output a `<tool_call>` XML tag "
            "containing a JSON object with `name` and `arguments`.\n\n"
            f"{tools_doc}\n\n"
            "Example invocation:\n"
            '<tool_call>{"name": "function_name", "arguments": {"param1": "value"}}</tool_call>\n'
        )

    @classmethod
    def inject_tools(
        cls, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Append tool documentation to the system message or create one."""
        if not tools:
            return messages

        tool_prompt = cls.format_tools_system_message(tools)
        new_msgs = [dict(m) for m in messages]

        for m in new_msgs:
            if m.get("role") == "system":
                m["content"] = str(m.get("content", "")) + tool_prompt
                return new_msgs

        new_msgs.insert(0, {"role": "system", "content": tool_prompt.strip()})
        return new_msgs

    @classmethod
    def parse_tool_calls(cls, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from model generated output."""
        tool_calls: list[dict[str, Any]] = []

        # 1. Search for <tool_call> tags
        matches = cls.TOOL_CALL_REGEX.findall(text)
        for raw_json in matches:
            call = cls._normalize_tool_call_json(raw_json)
            if call:
                tool_calls.append(call)

        # 2. Search for ```json blocks if no tags found
        if not tool_calls:
            json_matches = cls.JSON_BLOCK_REGEX.findall(text)
            for raw_json in json_matches:
                call = cls._normalize_tool_call_json(raw_json)
                if call:
                    tool_calls.append(call)

        # 3. Direct JSON search if output is pure JSON
        if not tool_calls and text.strip().startswith("{") and text.strip().endswith("}"):
            call = cls._normalize_tool_call_json(text.strip())
            if call:
                tool_calls.append(call)

        return tool_calls

    @classmethod
    def _normalize_tool_call_json(cls, raw: str) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None

            name = data.get("name") or data.get("tool") or data.get("function")
            if not name or not isinstance(name, str):
                return None

            args = data.get("arguments") or data.get("parameters") or {}
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)

            call_id = f"call_{uuid.uuid4().hex[:12]}"
            return {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str,
                },
            }
        except (json.JSONDecodeError, TypeError):
            return None
