"""Experiment A: Parameter-Efficient Fine-Tuning Comparison.
Compares Full Fine-Tuning vs LoRA vs QLoRA across:
- GPU VRAM consumption (MB)
- Training step time and epoch duration (s)
- Computational training cost ($)
- Evaluation accuracy (%) and final loss.
"""
from __future__ import annotations

import logging
import time
import numpy as np
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

import torch
from transformers import AutoTokenizer

from frontier_platform.config import SFTConfig
from frontier_platform.data.dataset import DatasetCurator
from frontier_platform.training.sft import SFTTrainer, SFTTrainMetrics
from frontier_platform.evaluation.quality import evaluate_quality
from frontier_platform.profiling.gpu_profiler import MemoryProfiler
from frontier_platform.profiling.cost_model import CostModel

logger = logging.getLogger("platform.experiments.exp_a")


def run_experiment_a(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    num_samples: int = 40,
    num_epochs: int = 1,
) -> Dict[str, Any]:
    """Execute Experiment A comparing Full FT, LoRA, and QLoRA."""
    logger.info("=== Running Experiment A: Full FT vs LoRA vs QLoRA ===")
    curator = DatasetCurator(seed=42)
    sft_data = curator.generate_sft_dataset(num_samples=num_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    strategies = ["full", "lora", "qlora"]
    results: Dict[str, Any] = {}

    profiler = MemoryProfiler()
    cost_model = CostModel()

    for strat in strategies:
        logger.info("Evaluating Strategy: %s...", strat.upper())
        cfg = SFTConfig(
            model_id=model_id,
            strategy=strat,
            batch_size=1 if strat == "full" else 2,
            gradient_accumulation_steps=2,
            num_epochs=num_epochs,
            learning_rate=1e-5 if strat == "full" else 2e-4,
            max_grad_norm=0.5,
            lora_r=16,
            lora_alpha=32,
        )

        trainer = SFTTrainer(cfg)
        try:
            train_metrics = trainer.train(sft_data)
            device = trainer.device
            q_eval = evaluate_quality(trainer.model, tokenizer, device=device)
            acc_val = q_eval.accuracy_pct
            ppl_val = q_eval.perplexity
            loss_val = train_metrics.final_loss
            time_val = train_metrics.training_time_s
            tps_val = train_metrics.tokens_per_second
            peak_val = train_metrics.peak_memory_mb
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                logger.warning("Strategy %s reached hardware memory ceiling (OOM): %s", strat.upper(), err)
                loss_val = float("nan")
                time_val = 0.0
                tps_val = 0.0
                peak_val = (trainer.param_stats["total_params"] * 16) / (1024 * 1024)  # Empirical 16 bytes/param AdamW ceiling
                acc_val = 0.0
                ppl_val = float("inf")
                train_metrics = SFTTrainMetrics(
                    strategy=strat,
                    total_params=trainer.param_stats["total_params"],
                    trainable_params=trainer.param_stats["trainable_params"],
                    trainable_pct=trainer.param_stats["trainable_pct"],
                    final_loss=loss_val,
                    training_time_s=time_val,
                    tokens_per_second=tps_val,
                    peak_memory_mb=peak_val,
                )
            else:
                raise err

        # Theoretical & practical memory breakdown
        mem_breakdown = profiler.profile_training(
            strategy=strat,
            precision="fp16",
            lora_trainable_pct=train_metrics.trainable_pct,
        )

        cost_report = cost_model.evaluate_training(
            tokens_per_sec=train_metrics.tokens_per_second,
            total_tokens=int(train_metrics.tokens_per_second * train_metrics.training_time_s),
            num_gpus=1,
        )

        results[strat] = {
            "strategy": strat.upper(),
            "trainable_params": train_metrics.trainable_params,
            "trainable_pct": train_metrics.trainable_pct,
            "peak_memory_mb": max(train_metrics.peak_memory_mb, mem_breakdown.total_training_memory_mb),
            "training_time_s": train_metrics.training_time_s,
            "tokens_per_sec": train_metrics.tokens_per_second,
            "training_cost_usd": cost_report.training_cost_usd,
            "final_loss": train_metrics.final_loss,
            "eval_accuracy_pct": acc_val,
            "eval_perplexity": ppl_val,
        }

        logger.info(
            "%s -> VRAM: %.1f MB | Time: %.1fs | Cost: $%.4f | Loss: %.4f | Acc: %.1f%%",
            strat.upper(),
            results[strat]["peak_memory_mb"],
            results[strat]["training_time_s"],
            results[strat]["training_cost_usd"],
            results[strat]["final_loss"],
            results[strat]["eval_accuracy_pct"],
        )

        # Clean memory between runs
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results
