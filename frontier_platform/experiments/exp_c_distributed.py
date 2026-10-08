"""Experiment C: Distributed Scaling & Communication Topology Comparison.
Compares:
- 1 GPU vs 2 GPU vs 4 GPU vs 8 GPU topologies
- NVLink (900 GB/s) vs PCIe Gen4 (64 GB/s) interconnect dynamics
Measures:
- Scaling efficiency (%)
- Ring-AllReduce communication overhead (ms)
- Global token throughput (tokens/s) and TFLOPS
- Speedup multiplier.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

from frontier_platform.distributed.scaling import DistributedScalingProfiler

logger = logging.getLogger("platform.experiments.exp_c")


def run_experiment_c(
    model_params: int = 494_000_000,
    gpu_counts: List[int] = [1, 2, 4, 8],
) -> Dict[str, Any]:
    """Execute Experiment C profiling distributed training scaling laws."""
    logger.info("=== Running Experiment C: Distributed Scaling (1, 2, 4, 8 GPUs) ===")
    profiler = DistributedScalingProfiler(model_params=model_params)

    # 1. High-speed NVLink profile
    nvlink_results = profiler.run_scaling_benchmark(gpu_counts=gpu_counts, bandwidth_gbps=900.0)

    # 2. Standard PCIe Gen4 profile
    pcie_results = profiler.run_scaling_benchmark(gpu_counts=gpu_counts, bandwidth_gbps=64.0)

    summary: Dict[str, Any] = {
        "nvlink_900gbps": {str(r.num_gpus): r.to_dict() for r in nvlink_results},
        "pcie_64gbps": {str(r.num_gpus): r.to_dict() for r in pcie_results},
    }

    for r in nvlink_results:
        logger.info(
            "NVLink %d GPUs -> Step: %.1fms | Comm: %.1fms | Throughput: %.1f tok/s | Eff: %.1f%% | Speedup: %.2fx",
            r.num_gpus, r.step_time_ms, r.comm_time_ms, r.throughput_tokens_sec, r.scaling_efficiency_pct, r.speedup
        )

    return summary
