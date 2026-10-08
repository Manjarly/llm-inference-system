"""Phase 7: Compute Efficiency (MFU) and Cloud Cost Model.
Calculates Model FLOPs Utilization (MFU), TFLOPS, training run cost,
and serving cost per 1 Million tokens across hardware accelerators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CostAndEfficiencyReport:
    tokens_per_sec: float
    achieved_tflops: float
    mfu_pct: float
    training_cost_usd: float
    inference_cost_per_1m_tokens_usd: float
    hardware_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "achieved_tflops": round(self.achieved_tflops, 2),
            "mfu_pct": round(self.mfu_pct, 1),
            "training_cost_usd": round(self.training_cost_usd, 4),
            "inference_cost_per_1m_tokens_usd": round(self.inference_cost_per_1m_tokens_usd, 4),
            "hardware_name": self.hardware_name,
        }


class CostModel:
    """Calculates hardware FLOPS utilization and deployment economics."""

    def __init__(
        self,
        num_params: int = 494_000_000,
        peak_tflops: float = 78.0,
        cost_per_hour_usd: float = 2.50,
        hardware_name: str = "NVIDIA A100 80GB",
    ) -> None:
        self.num_params = num_params
        self.peak_tflops = peak_tflops
        self.cost_per_hour_usd = cost_per_hour_usd
        self.hardware_name = hardware_name

    def evaluate_training(
        self,
        tokens_per_sec: float,
        total_tokens: int,
        num_gpus: int = 1,
    ) -> CostAndEfficiencyReport:
        """Evaluate training MFU and financial budget."""
        # Standard transformer forward + backward = 6 * P FLOPs per token
        flops_per_token = 6.0 * self.num_params
        achieved_flops = flops_per_token * tokens_per_sec
        achieved_tflops = achieved_flops / 1e12

        total_peak = self.peak_tflops * num_gpus
        mfu = (achieved_tflops / total_peak * 100.0) if total_peak > 0 else 0.0

        duration_sec = total_tokens / max(0.1, tokens_per_sec)
        duration_hours = duration_sec / 3600.0
        train_cost = duration_hours * self.cost_per_hour_usd * num_gpus

        # Cost per 1M tokens in serving with forward pass = 2 * P
        sec_for_1m = 1_000_000.0 / max(0.1, tokens_per_sec)
        infer_cost_1m = (sec_for_1m / 3600.0) * self.cost_per_hour_usd

        return CostAndEfficiencyReport(
            tokens_per_sec=tokens_per_sec,
            achieved_tflops=achieved_tflops,
            mfu_pct=min(100.0, mfu),
            training_cost_usd=train_cost,
            inference_cost_per_1m_tokens_usd=infer_cost_1m,
            hardware_name=self.hardware_name,
        )

    def evaluate_inference(
        self,
        tokens_per_sec: float,
    ) -> Dict[str, float]:
        """Compute serving economics: Cost per 1M tokens ($) and requests/dollar."""
        # 1M tokens cost
        sec_for_1m = 1_000_000.0 / max(0.1, tokens_per_sec)
        cost_1m = (sec_for_1m / 3600.0) * self.cost_per_hour_usd
        tokens_per_dollar = 1_000_000.0 / max(1e-5, cost_1m)

        return {
            "tokens_per_sec": round(tokens_per_sec, 1),
            "cost_per_1m_tokens_usd": round(cost_1m, 4),
            "tokens_per_dollar": round(tokens_per_dollar, 0),
        }
