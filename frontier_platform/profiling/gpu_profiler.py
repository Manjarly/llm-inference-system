"""Phase 7: Deep GPU VRAM & Memory Allocation Profiler.
Breaks down memory footprints into Weights, Gradients, Optimizer States (AdamW),
Activation Buffers, and KV-cache across Full FT, LoRA, and QLoRA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MemoryBreakdown:
    weights_mb: float
    gradients_mb: float
    optimizer_states_mb: float
    activations_mb: float
    total_training_memory_mb: float
    kv_cache_inference_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights_mb": round(self.weights_mb, 2),
            "gradients_mb": round(self.gradients_mb, 2),
            "optimizer_states_mb": round(self.optimizer_states_mb, 2),
            "activations_mb": round(self.activations_mb, 2),
            "total_training_memory_mb": round(self.total_training_memory_mb, 2),
            "kv_cache_inference_mb": round(self.kv_cache_inference_mb, 2),
        }


class MemoryProfiler:
    """Profiles theoretical and empirical memory footprints for LLMs."""

    def __init__(
        self,
        num_params: int = 494_000_000,
        num_layers: int = 24,
        hidden_dim: int = 896,
        num_heads: int = 14,
        seq_length: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.num_params = num_params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.batch_size = batch_size

    def profile_training(
        self,
        strategy: str = "lora",
        precision: str = "fp16",
        lora_trainable_pct: float = 0.5,
    ) -> MemoryBreakdown:
        """Compute exact training memory breakdown."""
        bytes_to_mb = 1.0 / (1024 * 1024)

        # 1. Weights
        if strategy == "qlora":
            w_bytes = self.num_params * 0.5  # 4-bit = 0.5 bytes
        elif precision in ("fp16", "bf16"):
            w_bytes = self.num_params * 2.0
        else:
            w_bytes = self.num_params * 4.0
        weights_mb = w_bytes * bytes_to_mb

        # 2. Gradients & Optimizer States
        if strategy == "full":
            trainable_params = self.num_params
            grad_bytes = trainable_params * 2.0  # FP16 grads
            # AdamW stores 2 states (FP32) + optional FP32 master weight = 12-16 bytes/param
            opt_bytes = trainable_params * 12.0
        else:  # LoRA / QLoRA
            trainable_params = self.num_params * (lora_trainable_pct / 100.0)
            grad_bytes = trainable_params * 2.0
            opt_bytes = trainable_params * 12.0

        gradients_mb = grad_bytes * bytes_to_mb
        optimizer_mb = opt_bytes * bytes_to_mb

        # 3. Activations (Standard Transformer with self-attention and MLP)
        # Act ~ b * s * h * L * (34 + 5 * a * s / h) bytes per layer
        b, s, h, l = self.batch_size, self.seq_length, self.hidden_dim, self.num_layers
        act_bytes = b * s * h * l * 18 * 2  # in FP16
        activations_mb = act_bytes * bytes_to_mb

        # 4. KV-cache for inference (batch_size * 2 * n_layers * n_heads * d_k * seq_len * 2 bytes)
        head_dim = h // self.num_heads
        kv_bytes = b * 2 * l * self.num_heads * head_dim * s * 2
        kv_mb = kv_bytes * bytes_to_mb

        total_mb = weights_mb + gradients_mb + optimizer_mb + activations_mb

        return MemoryBreakdown(
            weights_mb=weights_mb,
            gradients_mb=gradients_mb,
            optimizer_states_mb=optimizer_mb,
            activations_mb=activations_mb,
            total_training_memory_mb=total_mb,
            kv_cache_inference_mb=kv_mb,
        )
