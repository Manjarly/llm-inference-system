"""Visualization utilities for LLM inference benchmarks."""
from __future__ import annotations

import os
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt


def plot_batching_results(data: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Generate comparative visualization for batching policies."""
    policies = list(data.keys())
    tps = [data[p]["tokens_per_sec"] for p in policies]
    ttft = [data[p]["ttft_avg_ms"] for p in policies]
    duration = [data[p]["duration_s"] for p in policies]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    colors = ["#94a3b8", "#38bdf8", "#10b981"]

    # 1. Throughput
    bars1 = ax1.bar([p.capitalize() for p in policies], tps, color=colors, width=0.5)
    ax1.set_title("Generation Throughput (Tokens/s)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Tokens / Sec")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + max(tps) * 0.02, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    # 2. TTFT
    bars2 = ax2.bar([p.capitalize() for p in policies], ttft, color=colors, width=0.5)
    ax2.set_title("Time to First Token (TTFT ms)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Milliseconds (Lower is better)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + max(ttft) * 0.02, f"{yval:.1f}ms", ha="center", va="bottom", fontweight="bold")

    # 3. Total Duration
    bars3 = ax3.bar([p.capitalize() for p in policies], duration, color=colors, width=0.5)
    ax3.set_title("Total Batch Completion Time (s)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Seconds (Lower is better)")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + max(duration) * 0.02, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Batching Strategy Benchmark Comparison (Workload: 8 Concurrent Requests)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Batching benchmark plot saved to {output_path}")


def plot_quantization_results(data: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Generate comparative visualization for quantization modes."""
    modes = list(data.keys())
    mem = [data[m]["memory_mb"] for m in modes]
    compression = [data[m]["compression_ratio"] for m in modes]
    savings = [data[m]["savings_pct"] for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    colors = ["#64748b", "#3b82f6", "#8b5cf6", "#ec4899"]

    # 1. Model Memory (MB)
    bars1 = ax1.bar([m.upper() for m in modes], mem, color=colors[:len(modes)], width=0.5)
    ax1.set_title("Model Memory Footprint (MB)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Megabytes (MB)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + max(mem) * 0.02, f"{yval:.1f} MB", ha="center", va="bottom", fontweight="bold")

    # 2. Compression Ratio
    bars2 = ax2.bar([m.upper() for m in modes], compression, color=colors[:len(modes)], width=0.5)
    ax2.set_title("VRAM Compression Multiplier", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Compression Factor (x)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + max(compression) * 0.02, f"{yval:.2f}x", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Quantization Memory Footprint & Compression Analysis", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Quantization benchmark plot saved to {output_path}")
