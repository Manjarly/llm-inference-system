"""Experiment B: Alignment Stages Comparison.
Compares:
1. Base Model
2. SFT
3. DPO (Direct from base)
4. SFT + DPO (Canonical alignment pipeline)
Measures:
- Quality and reasoning accuracy (%)
- Safety refusal rate (%)
- Robustness retention (%)
- Win rate against Base Model (%)
- Implicit reward margin.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import logging
import numpy as np
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from frontier_platform.config import SFTConfig, DPOConfig
from frontier_platform.data.dataset import DatasetCurator
from frontier_platform.training.sft import SFTTrainer
from frontier_platform.training.dpo import DPOTrainer
from frontier_platform.evaluation.evaluator import run_full_evaluation
from frontier_platform.evaluation.quality import judge_pairwise_win_rate, QUALITY_BENCHMARK_PROMPTS

logger = logging.getLogger("platform.experiments.exp_b")


def run_experiment_b(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    num_samples: int = 30,
) -> Dict[str, Any]:
    """Execute Experiment B comparing Base vs SFT vs DPO vs SFT+DPO."""
    logger.info("=== Running Experiment B: Base vs SFT vs DPO vs SFT+DPO ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

    curator = DatasetCurator(seed=42)
    sft_data = curator.generate_sft_dataset(num_samples=num_samples)
    dpo_data = curator.generate_dpo_dataset(num_samples=num_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    results: Dict[str, Any] = {}

    # 1. Base Model
    logger.info("Evaluating [1/4] Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    base_eval = run_full_evaluation(base_model, tokenizer, model_name="Base", device=device)

    # Collect responses for win-rate judging
    base_responses: List[str] = []
    for item in QUALITY_BENCHMARK_PROMPTS:
        inp = tokenizer(f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device)
        with torch.no_grad():
            gen = base_model.generate(**inp, max_new_tokens=48, temperature=0.0)
            base_responses.append(tokenizer.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True))

    results["base"] = {
        "stage": "Base Model",
        "composite_score": base_eval.composite_index,
        "quality_accuracy_pct": base_eval.quality.accuracy_pct,
        "safety_refusal_pct": base_eval.safety.refusal_rate_pct,
        "robustness_score": base_eval.robustness.retention_score,
        "win_rate_vs_base_pct": 50.0,  # Self vs Self
        "reward_margin": 0.0,
    }
    del base_model
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    # 2. SFT Model
    logger.info("Training & Evaluating [2/4] SFT Model...")
    sft_cfg = SFTConfig(model_id=model_id, strategy="lora", num_epochs=1, batch_size=1)
    sft_trainer = SFTTrainer(sft_cfg)
    _ = sft_trainer.train(sft_data)
    sft_eval = run_full_evaluation(sft_trainer.model, tokenizer, model_name="SFT", device=device)

    sft_responses: List[str] = []
    for item in QUALITY_BENCHMARK_PROMPTS:
        inp = tokenizer(f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device)
        with torch.no_grad():
            gen = sft_trainer.model.generate(**inp, max_new_tokens=48, temperature=0.0)
            sft_responses.append(tokenizer.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True))

    sft_win = judge_pairwise_win_rate(sft_responses, base_responses)

    # Empirically measure preference margin (chosen vs rejected negative log likelihood)
    margins = []
    with torch.no_grad():
        for ex in dpo_data[:5]:
            c_toks = tokenizer(f"<|im_start|>user\n{ex.prompt}<|im_end|>\n<|im_start|>assistant\n{ex.chosen}<|im_end|>", return_tensors="pt").to(device)
            r_toks = tokenizer(f"<|im_start|>user\n{ex.prompt}<|im_end|>\n<|im_start|>assistant\n{ex.rejected}<|im_end|>", return_tensors="pt").to(device)
            c_loss = sft_trainer.model(input_ids=c_toks.input_ids, labels=c_toks.input_ids).loss.item()
            r_loss = sft_trainer.model(input_ids=r_toks.input_ids, labels=r_toks.input_ids).loss.item()
            margins.append(r_loss - c_loss)
    sft_margin_val = round(sum(margins) / max(1, len(margins)), 3) if margins else 0.0

    results["sft"] = {
        "stage": "SFT",
        "composite_score": sft_eval.composite_index,
        "quality_accuracy_pct": sft_eval.quality.accuracy_pct,
        "safety_refusal_pct": sft_eval.safety.refusal_rate_pct,
        "robustness_score": sft_eval.robustness.retention_score,
        "win_rate_vs_base_pct": sft_win["win_rate_pct"],
        "reward_margin": sft_margin_val,
    }

    # 3. DPO directly from Base
    logger.info("Training & Evaluating [3/4] DPO (Direct) Model...")
    dpo_cfg = DPOConfig(model_id=model_id, num_epochs=1, batch_size=1, beta=0.1)
    dpo_direct_trainer = DPOTrainer(dpo_cfg)
    dpo_direct_metrics = dpo_direct_trainer.train(dpo_data)
    dpo_direct_eval = run_full_evaluation(dpo_direct_trainer.policy_model, tokenizer, model_name="DPO (Direct)", device=device)

    dpo_responses: List[str] = []
    for item in QUALITY_BENCHMARK_PROMPTS:
        inp = tokenizer(f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device)
        with torch.no_grad():
            gen = dpo_direct_trainer.policy_model.generate(**inp, max_new_tokens=48, temperature=0.0)
            dpo_responses.append(tokenizer.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True))

    dpo_win = judge_pairwise_win_rate(dpo_responses, base_responses)
    results["dpo"] = {
        "stage": "DPO (Direct)",
        "composite_score": dpo_direct_eval.composite_index,
        "quality_accuracy_pct": dpo_direct_eval.quality.accuracy_pct,
        "safety_refusal_pct": dpo_direct_eval.safety.refusal_rate_pct,
        "robustness_score": dpo_direct_eval.robustness.retention_score,
        "win_rate_vs_base_pct": dpo_win["win_rate_pct"],
        "reward_margin": dpo_direct_metrics.reward_margin,
    }
    del dpo_direct_trainer
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    # 4. SFT + DPO Pipeline
    logger.info("Training & Evaluating [4/4] SFT + DPO Pipeline Model...")
    dpo_pipeline_trainer = DPOTrainer(dpo_cfg, policy_model=sft_trainer.model)
    dpo_pipe_metrics = dpo_pipeline_trainer.train(dpo_data)
    dpo_pipe_eval = run_full_evaluation(dpo_pipeline_trainer.policy_model, tokenizer, model_name="SFT + DPO", device=device)

    pipe_responses: List[str] = []
    for item in QUALITY_BENCHMARK_PROMPTS:
        inp = tokenizer(f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device)
        with torch.no_grad():
            gen = dpo_pipeline_trainer.policy_model.generate(**inp, max_new_tokens=48, temperature=0.0)
            pipe_responses.append(tokenizer.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True))

    pipe_win = judge_pairwise_win_rate(pipe_responses, base_responses)
    results["sft_plus_dpo"] = {
        "stage": "SFT + DPO",
        "composite_score": dpo_pipe_eval.composite_index,
        "quality_accuracy_pct": dpo_pipe_eval.quality.accuracy_pct,
        "safety_refusal_pct": dpo_pipe_eval.safety.refusal_rate_pct,
        "robustness_score": dpo_pipe_eval.robustness.retention_score,
        "win_rate_vs_base_pct": pipe_win["win_rate_pct"],
        "reward_margin": dpo_pipe_metrics.reward_margin,
    }

    del sft_trainer
    del dpo_pipeline_trainer
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return results
