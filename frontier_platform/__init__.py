"""Frontier LLM Training & Evaluation Platform.
A unified engineering infrastructure covering data preprocessing, SFT (Full/LoRA/QLoRA),
DPO, RL/GRPO alignment, distributed scaling, multi-dimensional evaluation,
inference serving, and GPU/cost profiling.
"""
import os
import sys
import numpy as np

# Ensure compatibility across numpy and scipy versions
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "1.0.0"
