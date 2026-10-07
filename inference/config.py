"""Configuration objects for the inference engine and server."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


SCHEDULER_POLICIES = ("continuous", "static", "none")
QUANT_MODES = ("none", "fp16", "bf16", "int8", "int4", "dynamic_int8", "bnb_int8", "bnb_nf4")


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    device: str = "auto"             # auto | cuda | mps | cpu
    dtype: str = "auto"              # auto | float32 | float16 | bfloat16
    quantization: str = "none"       # see QUANT_MODES
    int4_group_size: int = 64
    quantize_lm_head: bool = False
    attn_implementation: str = "sdpa"  # sdpa | eager
    trust_remote_code: bool = False
    max_context_len: int = 2048


@dataclass
class SchedulerConfig:
    policy: str = "continuous"       # continuous | static | none
    max_batch_size: int = 16
    max_batch_tokens: int = 16384    # cap on B * T held in the KV cache
    max_prefill_tokens: int = 4096   # cap on prompt tokens prefilled per engine step
    batch_wait_ms: float = 5.0       # static policy: time to wait for a batch to fill
    max_queue_size: int = 1024

    def __post_init__(self) -> None:
        if self.policy not in SCHEDULER_POLICIES:
            raise ValueError(f"policy must be one of {SCHEDULER_POLICIES}, got {self.policy!r}")
        if self.policy == "none":
            self.max_batch_size = 1


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    default_max_new_tokens: int = 128
    gpu_monitor_interval_s: float = 0.5
    seed: Optional[int] = 0

    def to_dict(self) -> dict:
        return asdict(self)
