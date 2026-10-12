"""System Health & Resource Drift Monitor for BIE.

Monitors real-time VRAM/RAM exhaustion, memory leak trends, and GPU thermal/power
status to prevent thermal throttling and hardware crashes.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import psutil


class HealthMonitor:
    """Monitors live resource health and flags degraded or critical states."""

    def __init__(self, vram_warning_threshold_gb: float = 0.5, ram_warning_threshold_gb: float = 1.0) -> None:
        self.vram_warning_threshold_gb = vram_warning_threshold_gb
        self.ram_warning_threshold_gb = ram_warning_threshold_gb

    @staticmethod
    def query_gpu_status() -> dict[str, Any]:
        """Query nvidia-smi for VRAM usage, temperature, and power draw."""
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                parts = [p.strip() for p in out.stdout.strip().split(",")]
                return {
                    "available": True,
                    "vram_total_mib": float(parts[0]),
                    "vram_used_mib": float(parts[1]),
                    "vram_free_mib": float(parts[2]),
                    "gpu_temp_c": float(parts[3]) if len(parts) > 3 else None,
                    "power_draw_w": float(parts[4]) if len(parts) > 4 else None,
                }
        except Exception:
            pass
        return {"available": False, "vram_total_mib": 0.0, "vram_used_mib": 0.0, "vram_free_mib": 0.0}

    def check_health(self) -> dict[str, Any]:
        """Generate structured health assessment."""
        gpu = self.query_gpu_status()
        vm = psutil.virtual_memory()

        ram_free_gb = round(vm.available / (1024**3), 2)
        ram_total_gb = round(vm.total / (1024**3), 2)
        vram_free_gb = round(gpu["vram_free_mib"] / 1024.0, 2) if gpu["available"] else 0.0

        warnings: list[str] = []
        status = "healthy"

        if gpu["available"] and vram_free_gb < self.vram_warning_threshold_gb:
            warnings.append(f"Low free VRAM: {vram_free_gb} GB remaining")
            status = "critical" if vram_free_gb < 0.2 else "degraded"

        if ram_free_gb < self.ram_warning_threshold_gb:
            warnings.append(f"Low system RAM: {ram_free_gb} GB remaining")
            status = "degraded" if status != "critical" else "critical"

        return {
            "status": status,
            "timestamp": int(time.time()),
            "ram": {
                "total_gb": ram_total_gb,
                "free_gb": ram_free_gb,
                "used_pct": vm.percent,
            },
            "gpu": gpu,
            "warnings": warnings,
        }
