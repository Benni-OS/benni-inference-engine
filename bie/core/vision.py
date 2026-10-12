"""Multimodal Vision Engine for BIE.

Processes image inputs (URLs, base64 data URIs, local file paths) and prepares
vision embeddings and prompt tokens for Gemma-4-26B multimodal inference.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImagePayload:
    source_type: str  # "base64" | "local_file" | "url"
    mime_type: str
    data_bytes: bytes
    width: int | None = None
    height: int | None = None
    byte_size: int = 0


class VisionEngine:
    """Preprocesses and normalizes image inputs for multimodal models."""

    B64_DATA_URI_PATTERN = re.compile(r"^data:(image\/[a-zA-Z0-9_\-\.\+]+);base64,(.*)$", re.DOTALL)

    def extract_image_payloads(self, message_content: str | list[dict[str, Any]]) -> tuple[str, list[ImagePayload]]:
        """Parse text prompt and extract image payloads from standard OpenAI message format."""
        if isinstance(message_content, str):
            return message_content, []

        text_parts: list[str] = []
        images: list[ImagePayload] = []

        for item in message_content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text_parts.append(str(item.get("text", "")))
            elif item_type == "image_url":
                url_obj = item.get("image_url", {})
                url_str = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                payload = self._resolve_image(url_str)
                if payload:
                    images.append(payload)

        prompt_text = " ".join(text_parts)
        return prompt_text, images

    def _resolve_image(self, url_or_path: str) -> ImagePayload | None:
        """Resolve base64, local file or URL into ImagePayload."""
        if not url_or_path:
            return None

        # 1. Base64 data URI
        match = self.B64_DATA_URI_PATTERN.match(url_or_path)
        if match:
            mime = match.group(1)
            raw_b64 = match.group(2)
            try:
                raw_bytes = base64.b64decode(raw_b64)
                return ImagePayload(
                    source_type="base64",
                    mime_type=mime,
                    data_bytes=raw_bytes,
                    byte_size=len(raw_bytes),
                )
            except Exception:
                return None

        # 2. Local File
        local_p = Path(url_or_path)
        if local_p.is_file():
            try:
                raw_bytes = local_p.read_bytes()
                ext = local_p.suffix.lower().lstrip(".")
                mime = "image/jpeg" if ext in ("jpg", "jpeg") else (f"image/{ext}" if ext in ("png", "webp", "gif") else "image/jpeg")
                return ImagePayload(
                    source_type="local_file",
                    mime_type=mime,
                    data_bytes=raw_bytes,
                    byte_size=len(raw_bytes),
                )
            except Exception:
                return None

        # 3. Remote URL reference
        return ImagePayload(
            source_type="url",
            mime_type="image/jpeg",
            data_bytes=url_or_path.encode("utf-8"),
            byte_size=len(url_or_path),
        )

    @staticmethod
    def format_vision_prompt(prompt: str, images_count: int) -> str:
        """Inject multimodal image tokens into the user prompt."""
        if images_count <= 0:
            return prompt
        image_tokens = "\n".join(f"[IMAGE_{i+1}]" for i in range(images_count))
        return f"{image_tokens}\n\n{prompt}".strip()
