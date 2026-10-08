"""Phase 4: Distributed Training & Scaling Infrastructure.
Models and profiles multi-GPU scaling dynamics:
- Ring-AllReduce communication overhead
- FSDP / ZeRO parameter & gradient sharding
- Scaling efficiency, step times, and bandwidth saturation curves across 1, 2, 4, 8 GPU topologies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, List

from frontier_platform.config import DistributedConfig

logger = logging.getLogger("platform.distributed.scaling")


@dataclass
class DistributedProfileResult:
    num_gpus: int
    strategy: str
    interconnect: str
    step_time_ms: float
    compute_time_ms: float
    comm_time_ms: float
    opt_time_ms: float
    throughput_tokens_sec: float
    scaling_efficiency_pct: float
    speedup: float
    comm_volume_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_gpus": self.num_gpus,
            "strategy": self.strategy,
            "interconnect": self.interconnect,
            "step_time_ms": round(self.step_time_ms, 2),
            "compute_time_ms": round(self.compute_time_ms, 2),
            "comm_time_ms": round(self.comm_time_ms, 2),
            "opt_time_ms": round(self.opt_time_ms, 2),
            "throughput_tokens_sec": round(self.throughput_tokens_sec, 1),
            "scaling_efficiency_pct": round(self.scaling_efficiency_pct, 1),
            "speedup": round(self.speedup, 2),
            "comm_volume_mb": round(self.comm_volume_mb, 2),
        }


class DistributedScalingProfiler:
    """Analytical & simulation profiler for distributed training topologies."""

    def __init__(
        self,
        model_params: int = 494_000_000,
        bytes_per_param: int = 2,  # FP16
        seq_length: int = 512,
        global_batch_size: int = 16,
    ) -> None:
        self.model_params = model_params
        self.bytes_per_param = bytes_per_param
        self.seq_length = seq_length
        self.global_batch_size = global_batch_size

        # Model gradient buffer size in Bytes
        self.grad_size_bytes = model_params * bytes_per_param
        self.grad_size_mb = self.grad_size_bytes / (1024 * 1024)

    def profile_topology(
        self,
        num_gpus: int,
        strategy: str = "ddp",
        bandwidth_gbps: float = 900.0,  # NVLink = 900, PCIe Gen4 = 64
        latency_us: float = 5.0,
    ) -> DistributedProfileResult:
        """Calculate exact computation, Ring-AllReduce communication, and scaling efficiency."""
        # Baseline single-GPU compute time (Forward + Backward)
        # Empirical constant calibrated for modern Transformer compute kernels
        t_fwd_single = 0.035 * (self.global_batch_size / 2) * (self.model_params / 5e8)  # seconds
        t_bwd_single = 2.0 * t_fwd_single
        t_opt_single = 0.008 * (self.model_params / 5e8)
        t_step_1gpu = (t_fwd_single + t_bwd_single + t_opt_single) * 1000.0  # ms

        if num_gpus == 1:
            tokens_per_step = self.global_batch_size * self.seq_length
            tps = tokens_per_step / (t_step_1gpu / 1000.0)
            return DistributedProfileResult(
                num_gpus=1,
                strategy=strategy,
                interconnect="Local GPU",
                step_time_ms=t_step_1gpu,
                compute_time_ms=(t_fwd_single + t_bwd_single) * 1000.0,
                comm_time_ms=0.0,
                opt_time_ms=t_opt_single * 1000.0,
                throughput_tokens_sec=tps,
                scaling_efficiency_pct=100.0,
                speedup=1.00,
                comm_volume_mb=0.0,
            )

        # Multi-GPU compute time: linear reduction with local batch size B / N
        local_bsz = max(1, self.global_batch_size // num_gpus)
        t_compute_sec = (t_fwd_single + t_bwd_single) * (local_bsz / self.global_batch_size)

        # Communication modeling:
        # Ring-AllReduce data transferred per rank = 2 * (N - 1) / N * M
        comm_bytes = 2.0 * ((num_gpus - 1) / num_gpus) * self.grad_size_bytes
        comm_mb = comm_bytes / (1024 * 1024)

        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8.0  # Convert Gbps to Bytes/s
        transfer_time_sec = comm_bytes / bandwidth_bytes_per_sec
        latency_time_sec = 2.0 * (num_gpus - 1) * (latency_us * 1e-6)
        t_comm_sec = transfer_time_sec + latency_time_sec

        # FSDP / ZeRO-3 adds AllGather for forward parameters (+50% comm time overhead)
        if strategy.lower() in ("fsdp", "zero3"):
            t_comm_sec *= 1.5
            comm_mb *= 1.5

        # In overlapping implementations, ~65% of backward communication is hidden by compute
        overlap_factor = 0.35
        effective_comm_sec = t_comm_sec * overlap_factor

        t_opt_sec = t_opt_single / (num_gpus if strategy.lower() in ("fsdp", "zero2", "zero3") else 1.0)

        total_step_sec = t_compute_sec + effective_comm_sec + t_opt_sec
        total_step_ms = total_step_sec * 1000.0

        speedup = t_step_1gpu / total_step_ms
        scaling_efficiency = (speedup / num_gpus) * 100.0
        tokens_per_step = self.global_batch_size * self.seq_length
        throughput_tokens_sec = tokens_per_step / total_step_sec

        interconnect_name = "NVLink (900 GB/s)" if bandwidth_gbps >= 500 else "PCIe Gen4 (64 GB/s)"

        return DistributedProfileResult(
            num_gpus=num_gpus,
            strategy=strategy,
            interconnect=interconnect_name,
            step_time_ms=total_step_ms,
            compute_time_ms=t_compute_sec * 1000.0,
            comm_time_ms=effective_comm_sec * 1000.0,
            opt_time_ms=t_opt_sec * 1000.0,
            throughput_tokens_sec=throughput_tokens_sec,
            scaling_efficiency_pct=min(100.0, scaling_efficiency),
            speedup=speedup,
            comm_volume_mb=comm_mb,
        )

    def run_scaling_benchmark(
        self,
        gpu_counts: List[int] = [1, 2, 4, 8],
        strategy: str = "ddp",
        bandwidth_gbps: float = 900.0,
    ) -> List[DistributedProfileResult]:
        """Profile scaling across multiple GPU configurations."""
        return [
            self.profile_topology(n, strategy=strategy, bandwidth_gbps=bandwidth_gbps)
            for n in gpu_counts
        ]
