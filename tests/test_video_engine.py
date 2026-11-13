"""Unit tests for BIE Video Engine, Video Generator (Wan2.1 & LTX), and Video Upscaler."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bie.core.video_engine import VideoEngine
from bie.core.video_generator import VideoGenerator, VideoGenParams
from bie.core.video_upscaler import UpscaleConfig, VideoUpscaler
from bie.serve.openai_compat import app


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Generate a lightweight synthetic 1-second test video clip using ffmpeg."""
    video_path = tmp_path / "test_sample.mp4"
    generator = VideoGenerator(output_dir=tmp_path)
    generator._render_clip(
        output_path=video_path,
        width=320,
        height=240,
        fps=16,
        duration=1.0,
        prompt="Test clip",
    )
    return video_path


def test_video_engine_extract_payloads(sample_video: Path) -> None:
    engine = VideoEngine()
    content = [
        {"type": "text", "text": "Analyze this shot:"},
        {"type": "video_url", "video_url": {"url": str(sample_video), "fps": 1.0, "max_frames": 4}},
    ]
    prompt, images, videos = engine.extract_video_and_image_payloads(content)
    assert prompt == "Analyze this shot:"
    assert len(images) == 0
    assert len(videos) == 1
    assert videos[0].source_type == "local_file"
    assert videos[0].width == 320
    assert videos[0].height == 240
    assert len(videos[0].sampled_frames) > 0


def test_video_engine_format_prompt(sample_video: Path) -> None:
    engine = VideoEngine()
    content = [
        {"type": "text", "text": "What happens here?"},
        {"type": "video_url", "video_url": {"url": str(sample_video), "fps": 1.0, "max_frames": 2}},
    ]
    prompt, _, videos = engine.extract_video_and_image_payloads(content)
    formatted = VideoEngine.format_video_prompt(prompt, videos)
    assert "<|vision_start|><|video_pad|>" in formatted
    assert "<|vision_end|>" in formatted
    assert "What happens here?" in formatted
    assert "<|frame_001 at" in formatted


def test_video_generator_param_validation() -> None:
    gen = VideoGenerator()
    params = VideoGenParams(
        prompt="A cinematic drone shot over mountains",
        model="wan2.1-t2v-14b",
        width=835,  # non-multiple of 16
        height=483,
    )
    validated = gen.validate_params(params)
    assert validated.width % 16 == 0
    assert validated.height % 16 == 0
    assert validated.model == "wan2.1-t2v-14b"


def test_video_generator_i2v_requires_image() -> None:
    gen = VideoGenerator()
    params = VideoGenParams(
        prompt="Animate this portrait",
        model="wan2.1-i2v-14b",
        image_url=None,
    )
    with pytest.raises(ValueError, match="Image-to-Video"):
        gen.validate_params(params)


def test_video_generator_plan_teacache() -> None:
    gen = VideoGenerator()
    params = VideoGenParams(
        prompt="Cyberpunk street",
        model="wan2.1-t2v-14b",
        steps=30,
        enable_teacache=True,
    )
    plan = gen.compute_generation_plan(params)
    assert plan["teacache_enabled"] is True
    assert plan["effective_steps_with_teacache"] < 30
    assert plan["vram_allocation_gb"] <= 6.5
    assert plan["ram_allocation_gb"] <= 24.0


def test_video_generator_ltx_plan() -> None:
    gen = VideoGenerator()
    params = VideoGenParams(
        prompt="Running athlete in golden hour",
        model="ltx-video-0.9.5",
        steps=25,
    )
    plan = gen.compute_generation_plan(params)
    assert plan["model"] == "LTX-Video-0.9.5"
    assert plan["vram_allocation_gb"] <= 6.0


def test_video_generator_task_lifecycle(tmp_path: Path) -> None:
    gen = VideoGenerator(output_dir=tmp_path)
    params = VideoGenParams(
        prompt="Testing video generation",
        model="wan2.1-t2v-14b",
        width=160,
        height=160,
        frames=16,
        fps=16,
        steps=5,
        upscale_after=False,
    )
    task = gen.create_task(params)
    assert task.status == "queued"
    assert gen.get_task(task.task_id) is not None

    completed = gen.execute_task(task.task_id)
    assert completed.status == "completed"
    assert completed.progress == 1.0
    assert completed.output_path is not None
    assert Path(completed.output_path).is_file()


def test_video_upscaler_pipeline_plan(sample_video: Path) -> None:
    upscaler = VideoUpscaler()
    cfg = UpscaleConfig(scale=2, target_fps=60, mode="fast_cinema")
    plan = upscaler.compute_pipeline_plan(sample_video, cfg)
    assert plan["scale_factor"] == 2
    assert plan["target"]["width"] == 640
    assert plan["target"]["height"] == 480
    assert plan["target"]["fps"] == 60


def test_video_upscaler_execution(sample_video: Path, tmp_path: Path) -> None:
    upscaler = VideoUpscaler()
    out_file = tmp_path / "sample_upscaled.mp4"
    cfg = UpscaleConfig(scale=2, target_fps=32, mode="fast_cinema", codec="libx264")
    result_path = upscaler.upscale(sample_video, output_path=out_file, config=cfg)
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_openai_compat_video_endpoints(sample_video: Path, tmp_path: Path) -> None:
    client = TestClient(app)

    # 1. Models list
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    model_ids = [m["id"] for m in resp.json()["data"]]
    assert "wan2.1-t2v-14b" in model_ids
    assert "ltx-video-0.9.5" in model_ids
    assert "qwen2.5-vl-7b-video" in model_ids

    # 2. Video models catalog
    v_resp = client.get("/v1/videos/models")
    assert v_resp.status_code == 200
    assert "wan2.1-t2v-14b" in v_resp.json()["video_generation"]

    # 3. Video generation endpoint
    gen_payload = {
        "prompt": "Solar flare eruption",
        "model": "wan2.1-t2v-14b",
        "width": 160,
        "height": 160,
        "frames": 16,
        "fps": 16,
        "steps": 5,
        "upscale_after": False,
    }
    gen_resp = client.post("/v1/videos/generations", json=gen_payload)
    assert gen_resp.status_code == 200
    task_data = gen_resp.json()
    assert task_data["status"] == "completed"
    task_id = task_data["task_id"]

    # 4. Video task query
    query_resp = client.get(f"/v1/videos/{task_id}")
    assert query_resp.status_code == 200
    assert query_resp.json()["status"] == "completed"

    # 5. Video upscale endpoint
    upscale_payload = {
        "video_path": str(sample_video),
        "scale": 2,
        "target_fps": 32,
        "codec": "libx264",
    }
    up_resp = client.post("/v1/videos/upscale", json=upscale_payload)
    assert up_resp.status_code == 200
    assert up_resp.json()["status"] == "completed"
    assert Path(up_resp.json()["output_path"]).is_file()


def test_comfyui_bridge_workflow_generation() -> None:
    from bie.core.comfyui_bridge import ComfyUIBridge

    bridge = ComfyUIBridge()
    # 1. Wan2.1 workflow
    wan_wf = bridge.build_wan2_1_workflow(
        prompt="Cyberpunk drone shot",
        width=832,
        height=480,
        enable_teacache=True,
        target_fps=60,
    )
    assert "1" in wan_wf
    assert wan_wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert wan_wf["2"]["class_type"] == "TeaCacheForWanVideo"
    assert wan_wf["4"]["inputs"]["text"] == "Cyberpunk drone shot"
    assert wan_wf["10"]["class_type"] == "RIFE VFI"
    assert wan_wf["11"]["inputs"]["frame_rate"] == 60

    # 2. LTX-Video workflow
    ltx_wf = bridge.build_ltx_workflow(
        prompt="Sports car high speed",
        target_fps=60,
    )
    assert ltx_wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert ltx_wf["7"]["class_type"] == "LTXVSpatialTemporalGuidance"
    assert ltx_wf["10"]["class_type"] == "RIFE VFI"


def test_comfyui_bridge_extract_output_videos() -> None:
    from bie.core.comfyui_bridge import ComfyUIBridge

    bridge = ComfyUIBridge()
    simulated_history = {
        "outputs": {
            "11": {
                "gifs": [
                    {"filename": "BIE_Wan2.1_Cinema_0001.mp4", "subfolder": "wan21", "type": "output"}
                ]
            }
        }
    }
    videos = bridge.extract_output_videos(simulated_history)
    assert len(videos) == 1
    assert "BIE_Wan2.1_Cinema_0001.mp4" in videos[0]


def test_mcp_bridge_tools_contain_video() -> None:
    from bie.serve.mcp_bridge import MCP_TOOLS

    names = [t["name"] for t in MCP_TOOLS]
    assert "bie_video_generate" in names
    assert "bie_video_upscale" in names
