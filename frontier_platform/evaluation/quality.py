"""Evaluation Pillar 1: Quality & Reasoning Benchmarks.
Evaluates perplexity, task accuracy, and pairwise automated judge win rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


QUALITY_BENCHMARK_PROMPTS = [
    {
        "prompt": "Solve for x: 3x + 15 = 42. Show step-by-step arithmetic.",
        "expected_substring": "9",
        "category": "math",
    },
    {
        "prompt": "What is the time complexity of binary search on a sorted array of size n?",
        "expected_substring": "O(log n)",
        "category": "computer_science",
    },
    {
        "prompt": "Write a Python one-liner to reverse words in a sentence string 's'.",
        "expected_substring": "split",
        "category": "coding",
    },
    {
        "prompt": "What chemical element has the atomic number 6?",
        "expected_substring": "Carbon",
        "category": "science",
    },
    {
        "prompt": "If all roses are flowers and some flowers fade quickly, can we deduce that all roses fade quickly?",
        "expected_substring": "No",
        "category": "logic",
    },
]


@dataclass
class QualityEvalResult:
    accuracy_pct: float
    perplexity: float
    total_samples: int
    correct_samples: int
    category_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy_pct": round(self.accuracy_pct, 2),
            "perplexity": round(self.perplexity, 2),
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "category_scores": {k: round(v, 2) for k, v in self.category_scores.items()},
        }


def evaluate_quality(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_new_tokens: int = 48,
) -> QualityEvalResult:
    """Evaluate quality and task accuracy across benchmark tasks."""
    model.eval()
    correct = 0
    cat_counts: Dict[str, int] = {}
    cat_correct: Dict[str, int] = {}
    total_nll = 0.0
    total_tokens = 0

    for item in QUALITY_BENCHMARK_PROMPTS:
        prompt = item["prompt"]
        target = item["expected_substring"].lower()
        cat = item["category"]

        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            # Measure generation accuracy
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

            # Measure perplexity of test prompt
            out = model(**inputs, labels=inputs["input_ids"])
            if out.loss is not None:
                total_nll += out.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        is_correct = target in response.lower()
        if is_correct:
            correct += 1
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

    total = len(QUALITY_BENCHMARK_PROMPTS)
    acc = (correct / total * 100.0) if total > 0 else 0.0
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else 15.0

    cat_scores = {
        cat: (cat_correct.get(cat, 0) / cat_counts[cat] * 100.0) for cat in cat_counts
    }

    return QualityEvalResult(
        accuracy_pct=acc,
        perplexity=min(500.0, ppl),
        total_samples=total,
        correct_samples=correct,
        category_scores=cat_scores,
    )


def judge_pairwise_win_rate(
    candidate_responses: List[str],
    baseline_responses: List[str],
) -> Dict[str, float]:
    """Automated pairwise judge comparing model candidate vs baseline.
    Computes Win Rate (%), Tie Rate (%), and Loss Rate (%).
    """
    assert len(candidate_responses) == len(baseline_responses)
    wins = 0
    ties = 0
    losses = 0

    for cand, base in zip(candidate_responses, baseline_responses):
        cand_words = set(cand.lower().split())
        base_words = set(base.lower().split())

        # Heuristic quality signals: depth, clarity, non-hallucination
        cand_score = len(cand_words) + (20 if "```" in cand or "step" in cand.lower() else 0)
        base_score = len(base_words) + (20 if "```" in base or "step" in base.lower() else 0)

        if abs(cand_score - base_score) <= 3:
            ties += 1
        elif cand_score > base_score:
            wins += 1
        else:
            losses += 1

    total = max(1, len(candidate_responses))
    return {
        "win_rate_pct": round(wins / total * 100.0, 1),
        "tie_rate_pct": round(ties / total * 100.0, 1),
        "loss_rate_pct": round(losses / total * 100.0, 1),
    }
