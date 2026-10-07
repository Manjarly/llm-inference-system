"""Unit tests for quantization modules."""
import torch
import torch.nn as nn
import pytest

from inference.quantization.int8 import Int8Linear
from inference.quantization.int4 import Int4Linear
from inference.quantization.quantizer import quantize_model, profile_model_memory


def test_int8_linear_conversion():
    lin = nn.Linear(64, 128, bias=True)
    int8_lin = Int8Linear.from_float(lin, dtype=torch.float32)

    x = torch.randn(4, 64, dtype=torch.float32)
    out_orig = lin(x)
    out_int8 = int8_lin(x)

    # Int8 quantization should have low numerical error (< 0.05 absolute mean diff)
    diff = (out_orig - out_int8).abs().mean().item()
    assert diff < 0.05
    assert int8_lin.weight_int8.dtype == torch.int8
    assert int8_lin.weight_int8.shape == (128, 64)


def test_int4_linear_packing_and_forward():
    lin = nn.Linear(128, 64, bias=True)
    int4_lin = Int4Linear.from_float(lin, group_size=32, dtype=torch.float32)

    x = torch.randn(2, 128, dtype=torch.float32)
    out_orig = lin(x)
    out_int4 = int4_lin(x)

    diff = (out_orig - out_int4).abs().mean().item()
    assert diff < 0.15  # INT4 has higher error than INT8 but should remain well bounded
    assert int4_lin.weight_packed.dtype == torch.uint8
    # 128 features packed 2-per-byte = 64 bytes
    assert int4_lin.weight_packed.shape == (64, 64)


def test_profile_model_memory():
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.Linear(200, 50),
    )
    prof = profile_model_memory(model)
    # (100*200 + 200) + (200*50 + 50) = 20,200 + 10,050 = 30,250 params
    assert prof.total_params == 30250
    assert prof.total_memory_mb > 0


def test_quantize_model_replaces_layers():
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    q_model, stats = quantize_model(model, mode="int4", group_size=32)
    assert stats["layers_replaced"] == 2
    assert stats["compression_ratio"] > 1.0
    assert stats["savings_pct"] > 50.0
