"""Video Understanding & Ingestion Engine for BIE.

Processes video inputs (MP4, WebM, local files, URLs, base64 data URIs),
performs temporal frame sampling with metadata extraction, and formats
tokens for multimodal Video-LLMs (Qwen2.5-VL, MiniCPM-V, InternVL).
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vision import ImagePayload


@dataclass
class VideoFrame:
    index: int
    timestamp_sec: float
    timecode: str
    image_payload: ImagePayload


@dataclass
class VideoPayload:
    source_type: str  # "local_file" | "base64" | "url"
    mime_type: str
    file_path: Path | None = None
    data_bytes: bytes | None = None
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    sampled_frames: list[VideoFrame] = field(default_factory=list)


class VideoEngine:
    """Preprocesses and temporally samples video inputs for Video-LLMs."""

    B64_VIDEO_URI_PATTERN = re.compile(
        r"^data:(video\/[a-zA-Z0-9_\-\.\+]+);base64,(.*)$", re.DOTALL
    )

    def __init__(self, ffmpeg_bin: str | None = None) -> None:
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_bin = shutil.which("ffprobe") or "ffprobe"

    def probe_video_metadata(self, video_path: Path) -> dict[str, Any]:
        """Extract metadata (duration, fps, resolution) using ffprobe or fallback."""
        try:
            cmd = [
                self.ffprobe_bin,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                streams = data.get("streams", [])
                format_info = data.get("format", {})
                w = int(streams[0].get("width", 0)) if streams else 0
                h = int(streams[0].get("height", 0)) if streams else 0
                dur_str = (streams[0].get("duration") if streams else None) or format_info.get("duration", "0")
                dur = float(dur_str) if dur_str else 0.0
                fps_str = streams[0].get("r_frame_rate", "24/1") if streams else "24/1"
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / max(1.0, float(den))
                else:
                    fps = float(fps_str)
                return {
                    "width": w,
                    "height": h,
                    "duration_sec": dur,
                    "fps": fps,
                }
        except Exception:
            pass
        return {"width": 1280, "height": 720, "duration_sec": 5.0, "fps": 24.0}

    def sample_frames_ffmpeg(
        self,
        video_path: Path,
        sampling_fps: float = 1.0,
        max_frames: int = 16,
    ) -> list[VideoFrame]:
        """Extract frames at regular intervals using ffmpeg."""
        frames: list[VideoFrame] = []
        if not video_path.is_file():
            return frames

        meta = self.probe_video_metadata(video_path)
        duration = max(0.1, meta["duration_sec"])

        with tempfile.TemporaryDirectory(prefix="bie_video_sample_") as tmp_dir:
            out_pattern = Path(tmp_dir) / "frame_%04d.jpg"
            # ffmpeg command to sample at specified fps or uniform distribution
            step = max(1.0 / max_frames, duration / max_frames) if duration * sampling_fps > max_frames else (1.0 / sampling_fps)
            vf_filter = f"fps=1/{step:.3f},scale='min(1280,iw)':-1"

            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(video_path),
                "-vf", vf_filter,
                "-vframes", str(max_frames),
                "-q:v", "2",
                str(out_pattern),
            ]
            subprocess.run(cmd, capture_output=True, check=False, timeout=30)

            extracted = sorted(Path(tmp_dir).glob("frame_*.jpg"))
            for idx, img_p in enumerate(extracted):
                raw_bytes = img_p.read_bytes()
                curr_time = min(duration, idx * step)
                minutes = int(curr_time // 60)
                seconds = curr_time % 60
                timecode = f"{minutes:02d}:{seconds:05.2f}"

                payload = ImagePayload(
                    source_type="local_file",
                    mime_type="image/jpeg",
                    data_bytes=raw_bytes,
                    width=meta["width"],
                    height=meta["height"],
                    byte_size=len(raw_bytes),
                )
                frames.append(
                    VideoFrame(
                        index=idx + 1,
                        timestamp_sec=curr_time,
                        timecode=timecode,
                        image_payload=payload,
                    )
                )

        return frames

    def resolve_video(
        self,
        url_or_path: str,
        sampling_fps: float = 1.0,
        max_frames: int = 16,
    ) -> VideoPayload | None:
        """Resolve base64, local file or URL into VideoPayload with sampled frames."""
        if not url_or_path:
            return None

        # 1. Base64 video URI
        b64_match = self.B64_VIDEO_URI_PATTERN.match(url_or_path)
        if b64_match:
            mime = b64_match.group(1)
            raw_b64 = b64_match.group(2)
            try:
                raw_bytes = base64.b64decode(raw_b64)
                ext = mime.split("/")[-1] if "/" in mime else "mp4"
                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = Path(tmp.name)
                try:
                    meta = self.probe_video_metadata(tmp_path)
                    frames = self.sample_frames_ffmpeg(tmp_path, sampling_fps, max_frames)
                    return VideoPayload(
                        source_type="base64",
                        mime_type=mime,
                        data_bytes=raw_bytes,
                        duration_sec=meta["duration_sec"],
                        fps=meta["fps"],
                        width=meta["width"],
                        height=meta["height"],
                        total_frames=len(frames),
                        sampled_frames=frames,
                    )
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            except Exception:
                return None

        # 2. Local File
        local_p = Path(url_or_path)
        if local_p.is_file():
            ext = local_p.suffix.lower().lstrip(".")
            mime = f"video/{ext}" if ext in ("mp4", "webm", "mkv", "avi", "mov") else "video/mp4"
            meta = self.probe_video_metadata(local_p)
            frames = self.sample_frames_ffmpeg(local_p, sampling_fps, max_frames)
            return VideoPayload(
                source_type="local_file",
                mime_type=mime,
                file_path=local_p,
                duration_sec=meta["duration_sec"],
                fps=meta["fps"],
                width=meta["width"],
                height=meta["height"],
                total_frames=len(frames),
                sampled_frames=frames,
            )

        # 3. URL reference
        return VideoPayload(
            source_type="url",
            mime_type="video/mp4",
            data_bytes=url_or_path.encode("utf-8"),
            duration_sec=0.0,
            fps=24.0,
            sampled_frames=[],
        )

    def extract_video_and_image_payloads(
        self, message_content: str | list[dict[str, Any]]
    ) -> tuple[str, list[ImagePayload], list[VideoPayload]]:
        """Parse text prompt and extract image and video payloads from standard message format."""
        if isinstance(message_content, str):
            return message_content, [], []

        text_parts: list[str] = []
        images: list[ImagePayload] = []
        videos: list[VideoPayload] = []

        for item in message_content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text_parts.append(str(item.get("text", "")))
            elif item_type == "image_url":
                url_obj = item.get("image_url", {})
                url_str = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                # Delegate to image resolution
                from .vision import VisionEngine
                img = VisionEngine()._resolve_image(url_str)
                if img:
                    images.append(img)
            elif item_type in ("video_url", "video"):
                video_obj = item.get(item_type, {})
                video_url = video_obj.get("url", "") if isinstance(video_obj, dict) else str(video_obj)
                fps = float(video_obj.get("fps", 1.0)) if isinstance(video_obj, dict) else 1.0
                max_frames = int(video_obj.get("max_frames", 16)) if isinstance(video_obj, dict) else 16
                video_payload = self.resolve_video(video_url, sampling_fps=fps, max_frames=max_frames)
                if video_payload:
                    videos.append(video_payload)

        prompt_text = " ".join(text_parts)
        return prompt_text, images, videos

    @staticmethod
    def format_video_prompt(
        prompt: str,
        videos: list[VideoPayload],
        target_model: str = "qwen2.5-vl",
    ) -> str:
        """Format video tokens and timecodes for Qwen2.5-VL / MiniCPM-V prompt structure."""
        if not videos:
            return prompt

        sections: list[str] = []
        for v_idx, video in enumerate(videos):
            sections.append(
                f"[VIDEO_{v_idx + 1} | Duration: {video.duration_sec:.1f}s | Resolution: {video.width}x{video.height}]"
            )
            for f in video.sampled_frames:
                sections.append(f"<|frame_{f.index:03d} at {f.timecode}|>")

        # Inject Qwen/Vision sequence wrapper
        tokens_block = "\n".join(sections)
        return f"<|vision_start|><|video_pad|>{tokens_block}<|vision_end|>\n\n{prompt}".strip()
