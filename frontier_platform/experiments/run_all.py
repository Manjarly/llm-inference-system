"""Master experimental pipeline executing Experiments A, B, C, and D,
generating matplotlib research plots, and compiling experimental tables.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from frontier_platform.experiments.exp_a_peft import run_experiment_a
from frontier_platform.experiments.exp_b_alignment import run_experiment_b
from frontier_platform.experiments.exp_c_distributed import run_experiment_c
from frontier_platform.experiments.exp_d_inference import run_experiment_d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("platform.experiments.master")


def plot_exp_a(res_a: Dict[str, Any], output_path: str) -> None:
    """Plot Experiment A: Memory vs Training Time vs Quality."""
    strats = ["FULL", "LORA", "QLORA"]
    keys = ["full", "lora", "qlora"]
    vram = [res_a[k]["peak_memory_mb"] for k in keys]
    time_s = [res_a[k]["training_time_s"] for k in keys]
    acc = [res_a[k]["eval_accuracy_pct"] for k in keys]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    colors = ["#ef4444", "#3b82f6", "#10b981"]

    # 1. VRAM Consumption
    b1 = ax1.bar(strats, vram, color=colors, width=0.55)
    ax1.set_title("Peak Training VRAM (MB)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Megabytes (Lower is better)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, y + max(vram) * 0.02, f"{y:.1f} MB", ha="center", va="bottom", fontweight="bold")

    # 2. Training Time
    b2 = ax2.bar(strats, time_s, color=colors, width=0.55)
    ax2.set_title("Training Duration (s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Seconds (Lower is better)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b2:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, y + max(time_s) * 0.02, f"{y:.1f}s", ha="center", va="bottom", fontweight="bold")

    # 3. Accuracy Retention
    b3 = ax3.bar(strats, acc, color=colors, width=0.55)
    ax3.set_title("Post-Adaptation Accuracy (%)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Accuracy % (Higher is better)")
    ax3.set_ylim(0, 105)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b3:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, y + 2, f"{y:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Experiment A: Full Fine-Tuning vs LoRA vs QLoRA Trade-off Frontiers", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved Experiment A plot to %s", output_path)


def plot_exp_b(res_b: Dict[str, Any], output_path: str) -> None:
    """Plot Experiment B: Radar / Spider chart comparing Alignment Stages."""
    categories = ["Quality", "Safety", "Robustness", "Win-Rate vs Base"]
    N = len(categories)

    stages = ["base", "sft", "dpo", "sft_plus_dpo"]
    stage_labels = ["Base Model", "SFT", "DPO (Direct)", "SFT + DPO"]
    colors = ["#94a3b8", "#38bdf8", "#8b5cf6", "#10b981"]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=180)

    for idx, (stg, label) in enumerate(zip(stages, stage_labels)):
        data = res_b[stg]
        values = [
            data["quality_accuracy_pct"],
            data["safety_refusal_pct"],
            data["robustness_score"],
            data["win_rate_vs_base_pct"],
        ]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=label, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=10)
    plt.title("Experiment B: Alignment Stages Multi-Dimensional Scorecard", size=13, fontweight="bold", y=1.1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved Experiment B plot to %s", output_path)


def plot_exp_c(res_c: Dict[str, Any], output_path: str) -> None:
    """Plot Experiment C: Scaling Efficiency and Throughput across GPUs."""
    nvlink = res_c["nvlink_900gbps"]
    pcie = res_c["pcie_64gbps"]

    gpus = [1, 2, 4, 8]
    nv_eff = [nvlink[str(n)]["scaling_efficiency_pct"] for n in gpus]
    pcie_eff = [pcie[str(n)]["scaling_efficiency_pct"] for n in gpus]
    nv_tps = [nvlink[str(n)]["throughput_tokens_sec"] for n in gpus]
    pcie_tps = [pcie[str(n)]["throughput_tokens_sec"] for n in gpus]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    # 1. Scaling Efficiency (%)
    ax1.plot(gpus, nv_eff, marker="o", linewidth=2.5, color="#10b981", label="NVLink (900 GB/s)")
    ax1.plot(gpus, pcie_eff, marker="s", linewidth=2.5, color="#f59e0b", label="PCIe Gen4 (64 GB/s)")
    ax1.axhline(100.0, linestyle="--", color="#64748b", alpha=0.7, label="Linear Ideal (100%)")
    ax1.set_title("Distributed Scaling Efficiency (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Scaling Efficiency %")
    ax1.set_ylim(50, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # 2. Global Token Throughput
    width = 0.35
    x = np.arange(len(gpus))
    ax2.bar(x - width/2, nv_tps, width, label="NVLink", color="#10b981")
    ax2.bar(x + width/2, pcie_tps, width, label="PCIe Gen4", color="#f59e0b")
    ax2.set_title("Global Training Throughput (Tokens/sec)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Tokens / Sec")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} GPU" for n in gpus])
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    plt.suptitle("Experiment C: Multi-GPU Distributed Scaling Dynamics & Interconnect Impact", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved Experiment C plot to %s", output_path)


def plot_exp_d(res_d: Dict[str, Any], output_path: str) -> None:
    """Plot Experiment D: Comprehensive Inference Serving Architecture Comparison across backends."""
    names = list(res_d.keys())
    short_names = [k.replace("PyTorch ", "").replace(" Engine", "").replace(" Quantized", " Q") for k in names]
    tps = [res_d[k]["tokens_per_sec"] for k in names]
    ttft = [res_d[k]["ttft_ms"] for k in names]
    tpot = [res_d[k]["tpot_ms"] for k in names]
    vram = [res_d[k]["vram_mb"] for k in names]
    frag = [res_d[k].get("kv_fragmentation_pct", 20.0) for k in names]
    conc = [res_d[k].get("max_concurrency", 16) for k in names]
    tpd = [res_d[k]["tokens_per_dollar"] for k in names]
    cost = [res_d[k]["cost_per_1m_tokens_usd"] for k in names]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    colors = ["#ef4444", "#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"][:len(names)]

    # 1. Pareto Frontier: TTFT vs Throughput
    ax1 = axes[0, 0]
    for i, name in enumerate(names):
        ax1.scatter(ttft[i], tps[i], s=260, color=colors[i], edgecolors="black", linewidth=1.5, zorder=5)
        ax1.annotate(
            f"{short_names[i]}\n({tps[i]:.1f} tok/s)",
            (ttft[i], tps[i]),
            textcoords="offset points",
            xytext=(10, -5 if i % 2 == 0 else 8),
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_title("Pareto Frontier: TTFT Latency vs Throughput", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time To First Token (TTFT ms) [Log Scale] - Lower is better", fontsize=10)
    ax1.set_ylabel("Throughput (Tokens/s) - Higher is better", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xscale("log")

    # 2. TPOT vs VRAM Footprint
    ax2 = axes[0, 1]
    w = 0.35
    x = np.arange(len(short_names))
    ax2_twin = ax2.twinx()
    ax2.bar(x - w/2, tpot, w, label="TPOT (ms)", color=colors, alpha=0.85, edgecolor="black")
    ax2_twin.plot(x + w/2, vram, marker="o", color="#0f172a", linewidth=2.5, label="VRAM (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=9.5, fontweight="bold", rotation=15)
    ax2.set_ylabel("Time Per Output Token (ms) - Lower is better", fontsize=10)
    ax2_twin.set_ylabel("Memory Footprint (MB)", fontsize=10)
    ax2.set_title("Inter-Token Latency (TPOT) & VRAM Footprint", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. KV Cache Memory Fragmentation & Concurrency Capacity
    ax3 = axes[1, 0]
    bars3 = ax3.bar(short_names, frag, color=colors, width=0.55, edgecolor="black")
    ax3.set_title("KV Cache Memory Fragmentation (%)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Memory Wasted / Internal Fragmentation %", fontsize=10)
    ax3.set_ylim(0, 85)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for i, bar in enumerate(bars3):
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Economic Efficiency: Cost / 1M Tokens & Tokens / $
    ax4 = axes[1, 1]
    bars4 = ax4.bar(short_names, cost, color=colors, width=0.55, edgecolor="black")
    ax4.set_title("Serving Cost per 1M Tokens ($ USD)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("USD / 1M Output Tokens - Lower is better", fontsize=10)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    for i, bar in enumerate(bars4):
        y = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, y + 0.5, f"${y:.2f}\n({tpd[i]:,.0f} tok/$)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Experiment D: High-Throughput Serving Architecture Benchmark", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved Experiment D plot to %s", output_path)


def run_master_experiments() -> Dict[str, Any]:
    """Execute all experiments A, B, C, D and compile results."""
    logger.info("=================================================================")
    logger.info("  STARTING FRONTIER PLATFORM SCIENTIFIC BENCHMARK SUITE")
    logger.info("=================================================================")

    # 1. Experiment A
    res_a = run_experiment_a(num_samples=10, num_epochs=1)

    # 2. Experiment B
    res_b = run_experiment_b(num_samples=8)

    # 3. Experiment C
    res_c = run_experiment_c(gpu_counts=[1, 2, 4, 8])

    # 4. Experiment D
    res_d = run_experiment_d()

    # Base paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    paper_dir = os.path.join(root_dir, "paper")
    plots_dir = os.path.join(paper_dir, "plots")
    tables_dir = os.path.join(paper_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Generate plots
    plot_exp_a(res_a, os.path.join(plots_dir, "exp_a_peft_comparison.png"))
    plot_exp_b(res_b, os.path.join(plots_dir, "exp_b_alignment_radar.png"))
    plot_exp_c(res_c, os.path.join(plots_dir, "exp_c_distributed_scaling.png"))
    plot_exp_d(res_d, os.path.join(plots_dir, "exp_d_inference_pareto.png"))

    # Save compiled JSON results
    all_results = {
        "experiment_a": res_a,
        "experiment_b": res_b,
        "experiment_c": res_c,
        "experiment_d": res_d,
    }
    json_path = os.path.join(tables_dir, "experimental_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved compiled experimental results to %s", json_path)

    return all_results


if __name__ == "__main__":
    run_master_experiments()
