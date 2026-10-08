"""INT4 Group-Wise Quantization Module (W4A16).
Quantizes weights into 4-bit integers packed into uint8 bytes (2 weights per byte).
Uses group-wise affine scaling (scales + zero_points) for high numerical fidelity.
Reduces weight memory footprint by ~75% compared to FP16 (~87.5% compared to FP32).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class Int4Linear(nn.Linear):
    """Linear layer storing 4-bit weights packed into uint8 with group-wise scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 64,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.target_dtype = dtype
        self.weight = nn.Parameter(torch.empty(0, 0, dtype=dtype), requires_grad=False)

        # If in_features is not divisible by group_size, pad conceptually
        self.num_groups_per_row = (in_features + group_size - 1) // group_size
        self.padded_in_features = self.num_groups_per_row * group_size

        # Packed weight: 2 x 4-bit elements per uint8 byte
        packed_in_features = self.padded_in_features // 2
        self.register_buffer(
            "weight_packed",
            torch.zeros((out_features, packed_in_features), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.zeros((out_features, self.num_groups_per_row), dtype=dtype),
        )
        self.register_buffer(
            "zeros",
            torch.zeros((out_features, self.num_groups_per_row), dtype=dtype),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_float(
        cls,
        linear_module: nn.Linear,
        group_size: int = 64,
        dtype: torch.dtype = torch.float16,
    ) -> "Int4Linear":
        """Quantize an existing nn.Linear module into Int4Linear."""
        with torch.no_grad():
            w = linear_module.weight.detach().to(dtype=torch.float32)
            out_features, in_features = w.shape
            has_bias = linear_module.bias is not None

            layer = cls(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
                group_size=group_size,
                dtype=dtype,
            )

            # Pad weights if needed to match padded_in_features
            if in_features < layer.padded_in_features:
                pad_len = layer.padded_in_features - in_features
                w_padded = torch.nn.functional.pad(w, (0, pad_len), value=0.0)
            else:
                w_padded = w

            # Reshape into groups: (out_features, num_groups, group_size)
            w_grouped = w_padded.view(out_features, layer.num_groups_per_row, group_size)

            min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
            max_val = torch.amax(w_grouped, dim=-1, keepdim=True)

            # Scale and zero point for 4-bit unsigned [0, 15]
            scale = torch.clamp((max_val - min_val) / 15.0, min=1e-8)
            zero = torch.clamp(torch.round(-min_val / scale), 0.0, 15.0)

            # Quantize elements
            q_grouped = torch.clamp(
                torch.round((w_grouped - min_val) / scale), 0.0, 15.0
            ).to(torch.uint8)

            # Flatten back to (out_features, padded_in_features)
            q_flat = q_grouped.view(out_features, layer.padded_in_features)

            # Pack two 4-bit values into one uint8: lower 4 bits and upper 4 bits
            # even indices: lower bits, odd indices: upper bits
            even = q_flat[:, 0::2]
            odd = q_flat[:, 1::2]
            packed = (even & 0x0F) | ((odd & 0x0F) << 4)

            layer.weight_packed.copy_(packed)
            layer.scales.copy_(scale.squeeze(-1).to(dtype=dtype))
            layer.zeros.copy_(zero.squeeze(-1).to(dtype=dtype))

            if has_bias:
                layer.bias.copy_(linear_module.bias.detach().to(dtype=dtype))

            return layer

    def dequantize(self, target_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize the packed 4-bit weight to target floating point tensor."""
        # Unpack: even = packed & 0x0F, odd = (packed >> 4) & 0x0F
        even = (self.weight_packed & 0x0F).to(target_dtype)
        odd = ((self.weight_packed >> 4) & 0x0F).to(target_dtype)

        # Interleave even and odd back into (out_features, padded_in_features)
        out_f, half_in_f = self.weight_packed.shape
        unpacked = torch.empty(
            (out_f, half_in_f * 2), dtype=target_dtype, device=self.weight_packed.device
        )
        unpacked[:, 0::2] = even
        unpacked[:, 1::2] = odd

        # Group-wise dequantization: (q - zero) * scale
        unpacked_grouped = unpacked.view(self.out_features, self.num_groups_per_row, self.group_size)
        scale_grouped = self.scales.to(target_dtype).unsqueeze(-1)
        zero_grouped = self.zeros.to(target_dtype).unsqueeze(-1)

        w_dequant_grouped = (unpacked_grouped - zero_grouped) * scale_grouped
        w_dequant = w_dequant_grouped.view(self.out_features, self.padded_in_features)

        # Slice away any padding
        if self.in_features < self.padded_in_features:
            w_dequant = w_dequant[:, :self.in_features]

        return w_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly group dequantization."""
        orig_dtype = x.dtype
        w = self.dequantize(orig_dtype)
        out = torch.matmul(x, w.t())
        if self.bias is not None:
            out = out + self.bias.to(orig_dtype)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}, quant=int4_groupwise"
        )
