"""Hardware and GPU monitoring subsystem.
Supports NVIDIA CUDA, Apple Silicon MPS, and CPU fallback.
Provides background sampling, peak memory tracking, and rolling history.
"""
from __future__ import annotations

import collections
import logging
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, Any, Optional

import psutil
import torch

logger = logging.getLogger("inference.monitor.gpu")


@dataclass
class HardwareSnapshot:
    device_type: str
    device_name: str
    gpu_utilization_pct: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_memory_pct: float
    ram_used_mb: float
    ram_total_mb: float
    ram_pct: float
    cpu_utilization_pct: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "gpu_memory_used_mb": round(self.gpu_memory_used_mb, 2),
            "gpu_memory_total_mb": round(self.gpu_memory_total_mb, 2),
            "gpu_memory_pct": round(self.gpu_memory_pct, 1),
            "ram_used_mb": round(self.ram_used_mb, 2),
            "ram_total_mb": round(self.ram_total_mb, 2),
            "ram_pct": round(self.ram_pct, 1),
            "cpu_utilization_pct": round(self.cpu_utilization_pct, 1),
            "timestamp": self.timestamp,
        }


class HardwareMonitor:
    """Thread-safe hardware sampler recording GPU compute utilization and memory."""

    def __init__(
        self,
        device_override: Optional[str] = None,
        sample_interval: float = 0.5,
        history_len: int = 120,
    ) -> None:
        self.sample_interval = sample_interval
        self.history_len = history_len
        self._history: Deque[HardwareSnapshot] = collections.deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.device_type = self._detect_device(device_override)
        self.device_name = self._detect_device_name()
        self.total_vram_mb = self._detect_total_vram_mb()
        self._last_mps_util_time = 0.0
        self._cached_mps_util = 0.0

        # Initial sample
        snap = self.sample()
        with self._lock:
            self._history.append(snap)

    def _detect_device(self, override: Optional[str] = None) -> str:
        if override and override != "auto":
            return override.lower()
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _detect_device_name(self) -> str:
        if self.device_type == "cuda":
            return torch.cuda.get_device_name(0)
        elif self.device_type == "mps":
            # Attempt to read Apple chip model
            chip = platform.processor()
            if not chip or chip == "arm":
                try:
                    chip = subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                    ).strip()
                except Exception:
                    chip = "Apple Silicon GPU"
            return f"{chip} (MPS)"
        else:
            return platform.processor() or "CPU"

    def _detect_total_vram_mb(self) -> float:
        if self.device_type == "cuda":
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory / (1024 * 1024))
        elif self.device_type == "mps":
            # Apple Silicon unified memory
            try:
                raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                return float(int(raw.strip()) / (1024 * 1024))
            except Exception:
                vm = psutil.virtual_memory()
                return float(vm.total / (1024 * 1024))
        else:
            vm = psutil.virtual_memory()
            return float(vm.total / (1024 * 1024))

    def _get_mps_utilization(self) -> float:
        """Sample macOS Apple Silicon GPU utilization % from IOAccelerator with rate limiting."""
        now = time.time()
        if now - self._last_mps_util_time < 1.0:
            return self._cached_mps_util
        self._last_mps_util_time = now
        try:
            out = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                text=True,
                timeout=0.5,
                stderr=subprocess.DEVNULL,
            )
            m = re.search(r'"Device Utilization %"=(\d+)', out)
            if m:
                self._cached_mps_util = float(m.group(1))
                return self._cached_mps_util
        except Exception:
            pass
        return self._cached_mps_util

    def _get_cuda_utilization(self) -> float:
        """Query CUDA utilization via pynvml if present or fallback."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(rates.gpu)
        except Exception:
            return 0.0

    def sample(self) -> HardwareSnapshot:
        """Take an instantaneous hardware snapshot."""
        ram = psutil.virtual_memory()
        ram_used_mb = (ram.total - ram.available) / (1024 * 1024)
        ram_total_mb = ram.total / (1024 * 1024)
        ram_pct = ram.percent
        cpu_util = psutil.cpu_percent(interval=None)

        gpu_util = 0.0
        gpu_used_mb = 0.0
        gpu_total_mb = self.total_vram_mb

        if self.device_type == "cuda":
            gpu_util = self._get_cuda_utilization()
            gpu_used_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        elif self.device_type == "mps":
            gpu_util = self._get_mps_utilization()
            # PyTorch MPS memory
            try:
                curr = torch.mps.current_allocated_memory()
                driver = torch.mps.driver_allocated_memory()
                gpu_used_mb = max(curr, driver) / (1024 * 1024)
            except Exception:
                gpu_used_mb = ram_used_mb * 0.25
        else:
            # CPU fallback: GPU util matches CPU, VRAM matches RAM
            gpu_util = cpu_util
            gpu_used_mb = ram_used_mb

        gpu_mem_pct = (gpu_used_mb / gpu_total_mb * 100.0) if gpu_total_mb > 0 else 0.0

        return HardwareSnapshot(
            device_type=self.device_type,
            device_name=self.device_name,
            gpu_utilization_pct=gpu_util,
            gpu_memory_used_mb=gpu_used_mb,
            gpu_memory_total_mb=gpu_total_mb,
            gpu_memory_pct=gpu_mem_pct,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            ram_pct=ram_pct,
            cpu_utilization_pct=cpu_util,
        )

    def _sampling_loop(self) -> None:
        while self._running:
            try:
                snap = self.sample()
                with self._lock:
                    self._history.append(snap)
            except Exception as e:
                logger.debug("Hardware sample error: %s", e)
            time.sleep(self.sample_interval)

    def start(self) -> None:
        """Start the background monitor daemon."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True, name="HWMonitorThread")
        self._thread.start()
        logger.info("Hardware monitor started for device: %s (%s)", self.device_type, self.device_name)

    def stop(self) -> None:
        """Stop background sampling."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self) -> HardwareSnapshot:
        """Get the most recent hardware snapshot."""
        with self._lock:
            if self._history:
                return self._history[-1]
        return self.sample()

    def get_summary(self) -> Dict[str, Any]:
        """Compute rolling window summary statistics and peak memory."""
        with self._lock:
            history = list(self._history)

        if not history:
            latest = self.sample()
            return {
                "latest": latest.to_dict(),
                "peak_gpu_memory_mb": round(latest.gpu_memory_used_mb, 2),
                "avg_gpu_utilization_pct": round(latest.gpu_utilization_pct, 1),
                "history_samples": 1,
            }

        peak_gpu_mem = max(s.gpu_memory_used_mb for s in history)
        avg_gpu_util = sum(s.gpu_utilization_pct for s in history) / len(history)
        avg_cpu_util = sum(s.cpu_utilization_pct for s in history) / len(history)

        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "latest": history[-1].to_dict(),
            "peak_gpu_memory_mb": round(peak_gpu_mem, 2),
            "avg_gpu_utilization_pct": round(avg_gpu_util, 1),
            "avg_cpu_utilization_pct": round(avg_cpu_util, 1),
            "history_samples": len(history),
            "history": [s.to_dict() for s in history[-30:]],  # Last 30 points for sparklines
        }
