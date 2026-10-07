"""Quantization orchestrator and memory profiler.
Converts standard PyTorch Transformer models into INT8, INT4, FP16, or BF16.
Profiles model memory footprints and calculates compression ratios.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from inference.quantization.int8 import Int8Linear
from inference.quantization.int4 import Int4Linear

logger = logging.getLogger("inference.quantization")


@dataclass
class ModelMemoryProfile:
    total_params: int
    param_memory_mb: float
    buffer_memory_mb: float
    total_memory_mb: float
    dtype_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_params": self.total_params,
            "param_memory_mb": round(self.param_memory_mb, 2),
            "buffer_memory_mb": round(self.buffer_memory_mb, 2),
            "total_memory_mb": round(self.total_memory_mb, 2),
            "dtype_breakdown": {k: round(v, 2) for k, v in self.dtype_breakdown.items()},
        }


def profile_model_memory(model: nn.Module) -> ModelMemoryProfile:
    """Accurately compute total parameters and memory footprint (MB) of a PyTorch model."""
    param_count = 0
    param_bytes = 0
    buffer_bytes = 0
    dtype_bytes: Dict[str, int] = {}

    for p in model.parameters():
        param_count += p.numel()
        b = p.numel() * p.element_size()
        param_bytes += b
        dt_str = str(p.dtype).replace("torch.", "")
        dtype_bytes[dt_str] = dtype_bytes.get(dt_str, 0) + b

    for buf in model.buffers():
        b = buf.numel() * buf.element_size()
        buffer_bytes += b
        dt_str = str(buf.dtype).replace("torch.", "")
        dtype_bytes[dt_str] = dtype_bytes.get(dt_str, 0) + b

    to_mb = 1.0 / (1024 * 1024)
    return ModelMemoryProfile(
        total_params=param_count,
        param_memory_mb=param_bytes * to_mb,
        buffer_memory_mb=buffer_bytes * to_mb,
        total_memory_mb=(param_bytes + buffer_bytes) * to_mb,
        dtype_breakdown={k: v * to_mb for k, v in dtype_bytes.items()},
    )


def replace_linear_modules(
    module: nn.Module,
    quant_mode: str,
    target_dtype: torch.dtype,
    group_size: int = 64,
    quantize_lm_head: bool = False,
    prefix: str = "",
) -> int:
    """Recursively traverse model and replace nn.Linear with quantized linear modules.
    Returns the number of linear layers replaced.
    """
    replaced_count = 0

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip lm_head by default unless explicitly permitted
        if "lm_head" in name.lower() and not quantize_lm_head:
            continue

        if isinstance(child, nn.Linear):
            if quant_mode == "int8":
                new_layer = Int8Linear.from_float(child, dtype=target_dtype)
                setattr(module, name, new_layer)
                replaced_count += 1
            elif quant_mode == "int4":
                new_layer = Int4Linear.from_float(child, group_size=group_size, dtype=target_dtype)
                setattr(module, name, new_layer)
                replaced_count += 1
        else:
            replaced_count += replace_linear_modules(
                child,
                quant_mode=quant_mode,
                target_dtype=target_dtype,
                group_size=group_size,
                quantize_lm_head=quantize_lm_head,
                prefix=full_name,
            )

    return replaced_count


def quantize_model(
    model: nn.Module,
    mode: str = "none",
    group_size: int = 64,
    quantize_lm_head: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Apply quantization strategy to a model and return (model, quant_stats).
    Supported modes:
      - 'none' / 'fp32': float32 precision
      - 'fp16': float16 precision
      - 'bf16': bfloat16 precision
      - 'int8': dynamic 8-bit weights with per-channel scaling
      - 'int4': group-wise 4-bit packed weights (W4A16)
    """
    mode = mode.lower()
    profile_before = profile_model_memory(model)
    layers_replaced = 0

    if mode in ("none", "fp32"):
        model = model.to(dtype=torch.float32)
    elif mode == "fp16":
        model = model.to(dtype=torch.float16)
    elif mode == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif mode in ("int8", "int4"):
        target_dtype = torch.float16
        layers_replaced = replace_linear_modules(
            model,
            quant_mode=mode,
            target_dtype=target_dtype,
            group_size=group_size,
            quantize_lm_head=quantize_lm_head,
        )
    else:
        logger.warning("Unsupported quantization mode '%s', falling back to unchanged.", mode)

    if device is not None:
        model = model.to(device)

    profile_after = profile_model_memory(model)
    compression_ratio = (
        (profile_before.total_memory_mb / profile_after.total_memory_mb)
        if profile_after.total_memory_mb > 0
        else 1.0
    )
    savings_pct = (
        (1.0 - (profile_after.total_memory_mb / profile_before.total_memory_mb)) * 100.0
        if profile_before.total_memory_mb > 0
        else 0.0
    )

    stats = {
        "mode": mode,
        "layers_replaced": layers_replaced,
        "original_memory_mb": round(profile_before.total_memory_mb, 2),
        "quantized_memory_mb": round(profile_after.total_memory_mb, 2),
        "compression_ratio": round(compression_ratio, 2),
        "savings_pct": round(savings_pct, 1),
        "profile": profile_after.to_dict(),
    }

    logger.info(
        "Quantization [%s]: %.2f MB -> %.2f MB (%.1fx compression, %.1f%% savings, %d layers converted)",
        mode,
        profile_before.total_memory_mb,
        profile_after.total_memory_mb,
        compression_ratio,
        savings_pct,
        layers_replaced,
    )
    return model, stats
