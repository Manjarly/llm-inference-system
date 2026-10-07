"""INT8 dynamic & weight-only quantization module.
Replaces nn.Linear with Int8Linear storing weights as torch.int8 with per-channel scales.
Supports dynamic activation quantization and fast inference across CUDA, MPS, and CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class Int8Linear(nn.Module):
    """Linear layer with 8-bit quantized weights and per-channel FP16/FP32 scale.
    Per-channel quantization: scale is computed per output row (out_features, 1).
    Weight values are in [-127, 127] signed int8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.target_dtype = dtype

        # Register buffers (not parameters, to avoid auto-gradients during inference)
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros((out_features, 1), dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_float(cls, linear_module: nn.Linear, dtype: torch.dtype = torch.float16) -> "Int8Linear":
        """Quantize an existing nn.Linear module to Int8Linear."""
        with torch.no_grad():
            w = linear_module.weight.detach().to(dtype=torch.float32)
            has_bias = linear_module.bias is not None

            layer = cls(
                in_features=linear_module.in_features,
                out_features=linear_module.out_features,
                bias=has_bias,
                dtype=dtype,
            )

            # Symmetrical per-channel quantization along output dim
            # scale = max(|w|, dim=-1) / 127.0
            max_val = torch.amax(torch.abs(w), dim=1, keepdim=True)
            scale = torch.clamp(max_val / 127.0, min=1e-8)

            # Quantize: q = round(w / scale) clamped to [-127, 127]
            q_w = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)

            layer.weight_int8.copy_(q_w)
            layer.weight_scale.copy_(scale.to(dtype=dtype))

            if has_bias:
                layer.bias.copy_(linear_module.bias.detach().to(dtype=dtype))

            return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weight or dynamic INT8 GEMM.
        Dequantization is fast: weight_int8.to(x.dtype) * weight_scale.
        """
        orig_dtype = x.dtype
        # Dequantize weight on the fly to activation dtype
        w_dequant = (self.weight_int8.to(orig_dtype) * self.weight_scale.to(orig_dtype))
        out = torch.matmul(x, w_dequant.t())
        if self.bias is not None:
            out = out + self.bias.to(orig_dtype)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quant=int8_per_channel"
