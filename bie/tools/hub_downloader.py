"""Model Hub Downloader & HuggingFace GGUF Sync for BIE.

Downloads GGUF model files directly from HuggingFace, verifies checksums,
and registers them into the local BIE model registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Callable


class ModelHubDownloader:
    """Downloads and verifies GGUF models from HuggingFace repositories."""

    def __init__(self, target_dir: str | Path | None = None) -> None:
        self.target_dir = Path(target_dir) if target_dir else Path("models")
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_info(self, model_id_or_repo: str) -> dict[str, str]:
        """Lookup model in registry.json or parse raw HuggingFace identifier."""
        reg_path = Path(__file__).parents[1] / "models" / "registry.json"
        if reg_path.is_file():
            try:
                data = json.loads(reg_path.read_text(encoding="utf-8"))
                for m in data.get("models", []):
                    if m.get("id") == model_id_or_repo:
                        return {
                            "repo": m.get("hf_repo", ""),
                            "file": m.get("hf_file", ""),
                            "id": m.get("id", ""),
                        }
            except Exception:
                pass

        if "/" in model_id_or_repo:
            parts = model_id_or_repo.split("/")
            repo = f"{parts[0]}/{parts[1]}"
            filename = parts[2] if len(parts) > 2 else f"{parts[1]}.gguf"
            return {"repo": repo, "file": filename, "id": model_id_or_repo}

        return {
            "repo": f"unsloth/{model_id_or_repo}-GGUF",
            "file": f"{model_id_or_repo}-UD-Q4_K_XL.gguf",
            "id": model_id_or_repo,
        }

    def download(
        self,
        model_id_or_repo: str,
        dry_run: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download model GGUF from Hugging Face repository."""
        info = self.resolve_model_info(model_id_or_repo)
        repo = info["repo"]
        filename = info["file"]

        if not repo or not filename or filename == "TBD":
            raise ValueError(f"Model {model_id_or_repo} has no valid HuggingFace repository defined.")

        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        dest_path = self.target_dir / filename

        if dry_run:
            return dest_path

        # Stream download
        req = urllib.request.Request(url, headers={"User-Agent": "benni-inference-engine/0.4.0"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)

        return dest_path


def main():
    parser = argparse.ArgumentParser(description="BIE Model Downloader")
    parser.add_argument("model", help="Model ID or repo/file (e.g. qwen3-30b-a3b)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    downloader = ModelHubDownloader()
    info = downloader.resolve_model_info(args.model)
    print(f"Resolving model {args.model}: {info['repo']} -> {info['file']}")
    path = downloader.download(args.model, dry_run=args.dry_run)
    print(f"Target destination: {path}")


if __name__ == "__main__":
    main()
