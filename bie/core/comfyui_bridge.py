"""ComfyUI Bridge & Workflow Dispatcher for BIE.

Connects BIE with local ComfyUI installations (e.g. C:\\Users\\T-GAMER\\Documents\\ComfyUI),
submits headless API workflows for Wan2.1-14B (GGUF + TeaCache) and LTX-Video,
monitors generation progress via WebSocket/HTTP, and retrieves rendered MP4s.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComfyUIConfig:
    server_address: str = "127.0.0.1:8188"
    timeout_sec: float = 300.0
    client_id: str = "bie-comfy-client"


class ComfyUIBridge:
    """Manages headless execution of video generation graphs on ComfyUI."""

    DEFAULT_COMFY_PATH = Path("C:/Users/T-GAMER/Documents/ComfyUI")

    def __init__(
        self,
        config: ComfyUIConfig | None = None,
        comfy_root: Path | str | None = None,
    ) -> None:
        self.config = config or ComfyUIConfig()
        self.comfy_root = Path(comfy_root) if comfy_root else self.DEFAULT_COMFY_PATH

    @property
    def base_url(self) -> str:
        return f"http://{self.config.server_address}"

    def is_alive(self) -> bool:
        """Probe if ComfyUI HTTP server is running and reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats", headers={"User-Agent": "BIE-Bridge/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_system_stats(self) -> dict[str, Any]:
        """Fetch system stats and devices from ComfyUI."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return {"error": str(exc), "status": "offline"}

    def build_wan2_1_workflow(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, deformed, watermark",
        width: int = 832,
        height: int = 480,
        frames: int = 49,
        steps: int = 30,
        fps: int = 16,
        cfg_scale: float = 5.0,
        enable_teacache: bool = True,
        teacache_threshold: float = 0.15,
        target_fps: int = 60,
    ) -> dict[str, Any]:
        """Generate API prompt format graph for Wan2.1 GGUF + TeaCache + RIFE 60fps."""
        workflow: dict[str, Any] = {
            # 1. Wan2.1 GGUF Model Loader
            "1": {
                "inputs": {
                    "unet_name": "Wan2.1-T2V-14B-Q4_K_M.gguf",
                },
                "class_type": "UnetLoaderGGUF",
            },
            # 2. TeaCache Acceleration Node
            "2": {
                "inputs": {
                    "model": ["1", 0],
                    "enable": enable_teacache,
                    "rel_l1_thresh": teacache_threshold,
                },
                "class_type": "TeaCacheForWanVideo",
            },
            # 3. UMT5-XXL Text Encoder Loader
            "3": {
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn.safetensors",
                    "type": "wan",
                },
                "class_type": "CLIPLoader",
            },
            # 4. Positive Prompt Encoding
            "4": {
                "inputs": {
                    "text": prompt,
                    "clip": ["3", 0],
                },
                "class_type": "CLIPTextEncode",
            },
            # 5. Negative Prompt Encoding
            "5": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["3", 0],
                },
                "class_type": "CLIPTextEncode",
            },
            # 6. Wan VAE Loader
            "6": {
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors",
                },
                "class_type": "VAELoader",
            },
            # 7. Empty Wan Latent Video Canvas
            "7": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": frames,
                    "batch_size": 1,
                },
                "class_type": "WanVideoEmptyLatent",
            },
            # 8. KSampler / DiT Denoising
            "8": {
                "inputs": {
                    "model": ["2", 0],
                    "seed": int(time.time() * 1000) % (2**31),
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": "uni_pc",
                    "scheduler": "normal",
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["7", 0],
                    "denoise": 1.0,
                },
                "class_type": "KSampler",
            },
            # 9. Tiled VAE Decode (prevents OOM on 8GB GPU)
            "9": {
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["6", 0],
                    "tile_size": 256,
                    "overlap": 32,
                    "temporal_size": 8,
                },
                "class_type": "VAEDecodeTiled",
            },
            # 10. RIFE Frame Interpolation (16fps -> 60fps)
            "10": {
                "inputs": {
                    "images": ["9", 0],
                    "multiplier": max(2, int(target_fps / fps)),
                    "ckpt_name": "rife46.pth",
                    "clear_cache_after_n_frames": 10,
                },
                "class_type": "RIFE VFI",
            },
            # 11. Video Combine & Export (VideoHelperSuite)
            "11": {
                "inputs": {
                    "images": ["10", 0],
                    "frame_rate": target_fps,
                    "loop_count": 0,
                    "filename_prefix": "BIE_Wan2.1_Cinema",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 18,
                },
                "class_type": "VHS_VideoCombine",
            },
        }
        return workflow

    def build_ltx_workflow(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, stuttering, distorted",
        width: int = 768,
        height: int = 512,
        frames: int = 49,
        steps: int = 25,
        fps: int = 24,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.5,
        target_fps: int = 60,
    ) -> dict[str, Any]:
        """Generate API prompt format graph for LTX-Video GGUF + STG + RIFE 60fps."""
        workflow: dict[str, Any] = {
            "1": {
                "inputs": {
                    "unet_name": "ltx-video-2b-v0.9.5-Q4_K_M.gguf",
                },
                "class_type": "UnetLoaderGGUF",
            },
            "2": {
                "inputs": {
                    "clip_name": "t5xxl_fp8_e4m3fn.safetensors",
                    "type": "ltxv",
                },
                "class_type": "CLIPLoader",
            },
            "3": {
                "inputs": {"text": prompt, "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"text": negative_prompt, "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
            "5": {
                "inputs": {"vae_name": "ltx_video_vae.safetensors"},
                "class_type": "VAELoader",
            },
            "6": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": frames,
                    "batch_size": 1,
                },
                "class_type": "LTXVideoEmptyLatent",
            },
            "7": {
                "inputs": {
                    "model": ["1", 0],
                    "stg_scale": stg_scale,
                },
                "class_type": "LTXVSpatialTemporalGuidance",
            },
            "8": {
                "inputs": {
                    "model": ["7", 0],
                    "seed": int(time.time() * 1000) % (2**31),
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["6", 0],
                    "denoise": 1.0,
                },
                "class_type": "KSampler",
            },
            "9": {
                "inputs": {"samples": ["8", 0], "vae": ["5", 0]},
                "class_type": "VAEDecodeTiled",
            },
            "10": {
                "inputs": {
                    "images": ["9", 0],
                    "multiplier": max(2, int(target_fps / fps)),
                    "ckpt_name": "rife46.pth",
                },
                "class_type": "RIFE VFI",
            },
            "11": {
                "inputs": {
                    "images": ["10", 0],
                    "frame_rate": target_fps,
                    "filename_prefix": "BIE_LTX_Realtime",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 18,
                },
                "class_type": "VHS_VideoCombine",
            },
        }
        return workflow

    def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Post the workflow to ComfyUI /prompt endpoint."""
        payload = {
            "prompt": workflow,
            "client_id": self.config.client_id,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "BIE-Bridge/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def poll_history(self, prompt_id: str, max_wait_sec: float = 300.0) -> dict[str, Any]:
        """Poll /history until the execution completes."""
        start_t = time.time()
        while time.time() - start_t < max_wait_sec:
            try:
                req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    history = json.loads(resp.read().decode("utf-8"))
                    if prompt_id in history:
                        entry = history[prompt_id]
                        if "outputs" in entry:
                            return entry
            except Exception:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {max_wait_sec}s.")

    def extract_output_videos(self, history_entry: dict[str, Any]) -> list[str]:
        """Extract output video paths or filenames from ComfyUI history outputs."""
        video_paths: list[str] = []
        outputs = history_entry.get("outputs", {})
        for _, node_output in outputs.items():
            # Check VideoHelperSuite output format
            if "gifs" in node_output:
                for item in node_output["gifs"]:
                    fn = item.get("filename")
                    sub = item.get("subfolder", "")
                    if fn:
                        full_p = self.comfy_root / "output" / sub / fn
                        video_paths.append(str(full_p))
            # Generic files output
            if "images" in node_output:
                for item in node_output["images"]:
                    fn = item.get("filename", "")
                    if fn.endswith((".mp4", ".webm", ".mkv")):
                        sub = item.get("subfolder", "")
                        full_p = self.comfy_root / "output" / sub / fn
                        video_paths.append(str(full_p))
        return video_paths
