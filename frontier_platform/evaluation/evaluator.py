"""Master Multi-Dimensional Evaluation Suite.
Coordinates Quality, Safety, and Robustness benchmarks into a unified assessment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from frontier_platform.evaluation.quality import evaluate_quality, QualityEvalResult
from frontier_platform.evaluation.safety import evaluate_safety, SafetyEvalResult
from frontier_platform.evaluation.robustness import evaluate_robustness, RobustnessEvalResult

logger = logging.getLogger("platform.evaluation")


@dataclass
class ComprehensiveEvalReport:
    model_name: str
    quality: QualityEvalResult
    safety: SafetyEvalResult
    robustness: RobustnessEvalResult
    composite_index: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quality": self.quality.to_dict(),
            "safety": self.safety.to_dict(),
            "robustness": self.robustness.to_dict(),
            "composite_index": round(self.composite_index, 2),
            "radar_vector": {
                "Quality": round(self.quality.accuracy_pct, 1),
                "Safety": round(self.safety.safety_score, 1),
                "Robustness": round(self.robustness.retention_score, 1),
                "Perplexity_Norm": round(max(0.0, 100.0 - min(100.0, self.quality.perplexity * 2)), 1),
            },
        }


def run_full_evaluation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    model_name: str = "CandidateModel",
    device: Optional[torch.device] = None,
) -> ComprehensiveEvalReport:
    """Run Quality, Safety, and Robustness benchmarks across a model."""
    if device is None:
        device = next(model.parameters()).device

    logger.info("Starting Multi-Dimensional Evaluation for %s on %s...", model_name, device)

    q_res = evaluate_quality(model, tokenizer, device=device)
    s_res = evaluate_safety(model, tokenizer, device=device)
    r_res = evaluate_robustness(model, tokenizer, device=device)

    # Composite weighted benchmark score (0-100)
    composite = 0.45 * q_res.accuracy_pct + 0.35 * s_res.safety_score + 0.20 * r_res.retention_score

    logger.info(
        "Eval Finished for %s -> Composite=%.1f | Quality=%.1f%% | Safety=%.1f%% | Robustness=%.1f%%",
        model_name, composite, q_res.accuracy_pct, s_res.safety_score, r_res.retention_score
    )

    return ComprehensiveEvalReport(
        model_name=model_name,
        quality=q_res,
        safety=s_res,
        robustness=r_res,
        composite_index=composite,
    )
