"""Unit tests for the Frontier Training & Evaluation Platform."""
import pytest
import torch
from transformers import AutoTokenizer

from frontier_platform.data.dataset import DatasetCurator
from frontier_platform.data.preprocessor import SFTTorchDataset, DPOTorchDataset, sft_collate_fn, dpo_collate_fn
from frontier_platform.distributed.scaling import DistributedScalingProfiler
from frontier_platform.profiling.gpu_profiler import MemoryProfiler
from frontier_platform.profiling.cost_model import CostModel
from frontier_platform.evaluation.quality import judge_pairwise_win_rate
from frontier_platform.training.rl import compute_reward_score


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def test_dataset_curation():
    curator = DatasetCurator(seed=42)
    sft_data = curator.generate_sft_dataset(num_samples=20)
    dpo_data = curator.generate_dpo_dataset(num_samples=15)

    assert len(sft_data) == 20
    assert len(dpo_data) == 15
    assert len(sft_data[0].prompt) > 0
    assert len(dpo_data[0].chosen) > 0
    assert len(dpo_data[0].rejected) > 0


def test_sft_dataset_prompt_masking(tokenizer):
    curator = DatasetCurator(seed=42)
    sft_data = curator.generate_sft_dataset(num_samples=5)
    ds = SFTTorchDataset(sft_data, tokenizer, max_seq_length=128, mask_prompt=True)

    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    # Prompt tokens should be masked with -100
    assert (item["labels"] == -100).any()
    # At least some target response tokens must NOT be -100
    assert (item["labels"] != -100).any()

    # Collation
    batch = sft_collate_fn([ds[0], ds[1]], pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape


def test_dpo_dataset_pair_collation(tokenizer):
    curator = DatasetCurator(seed=42)
    dpo_data = curator.generate_dpo_dataset(num_samples=4)
    ds = DPOTorchDataset(dpo_data, tokenizer, max_seq_length=128)

    batch = dpo_collate_fn([ds[0], ds[1]], pad_token_id=tokenizer.pad_token_id)
    assert "chosen_input_ids" in batch
    assert "rejected_input_ids" in batch
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape[0] == 2


def test_distributed_scaling_profiler():
    profiler = DistributedScalingProfiler(model_params=500_000_000)
    results = profiler.run_scaling_benchmark(gpu_counts=[1, 2, 4, 8], bandwidth_gbps=900.0)

    assert len(results) == 4
    # 1 GPU efficiency is 100%
    assert results[0].scaling_efficiency_pct == 100.0
    # Throughput should increase with GPU count
    assert results[2].throughput_tokens_sec > results[0].throughput_tokens_sec
    assert results[2].speedup > 2.0


def test_memory_and_cost_profiling():
    prof = MemoryProfiler(num_params=500_000_000)
    mem_full = prof.profile_training(strategy="full")
    mem_lora = prof.profile_training(strategy="lora")
    mem_qlora = prof.profile_training(strategy="qlora")

    # LoRA should use significantly less optimizer VRAM than Full FT
    assert mem_lora.optimizer_states_mb < mem_full.optimizer_states_mb * 0.1
    # QLoRA should use less weight VRAM than LoRA
    assert mem_qlora.weights_mb < mem_lora.weights_mb

    cost = CostModel(cost_per_hour_usd=2.50)
    res = cost.evaluate_inference(tokens_per_sec=100.0)
    assert res["cost_per_1m_tokens_usd"] > 0
    assert res["tokens_per_dollar"] > 0


def test_pairwise_judge_and_reward():
    cand = ["Detailed step by step solution with ```python print(1)```", "Good answer."]
    base = ["Vague answer.", "Good answer."]
    res = judge_pairwise_win_rate(cand, base)

    assert res["win_rate_pct"] >= 50.0

    r_safe = compute_reward_score("Phishing guide", "I cannot fulfill this request.", {"safety": 1.0})
    r_unsafe = compute_reward_score("Phishing guide", "Here is the phishing email link.", {"safety": 1.0})
    assert r_safe > r_unsafe
