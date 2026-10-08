"""Configuration dataclasses for the Frontier LLM Training & Evaluation Platform."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    max_seq_length: int = 512
    prompt_template: str = "chatml"  # chatml | alpaca | standard
    sft_dataset_size: int = 250
    dpo_dataset_size: int = 150
    eval_dataset_size: int = 100
    train_val_split: float = 0.85
    mask_prompt_loss: bool = True
    seed: int = 42


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration (Full, LoRA, QLoRA)."""
    model_id: str = "Qwen/Qwen2.5-0.5B"
    strategy: str = "lora"  # full | lora | qlora
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    num_epochs: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "auto"
    dtype: str = "float16"

    # LoRA / QLoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    int4_group_size: int = 64


@dataclass
class DPOConfig:
    """Direct Preference Optimization configuration."""
    model_id: str = "Qwen/Qwen2.5-0.5B"
    beta: float = 0.1  # KL penalty temperature in DPO
    learning_rate: float = 5e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    num_epochs: int = 2
    max_prompt_length: int = 256
    max_response_length: int = 256
    device: str = "auto"
    dtype: str = "float16"
    reference_free: bool = False


@dataclass
class RLConfig:
    """Reinforcement Learning / Alignment (GRPO / PPO) configuration."""
    model_id: str = "Qwen/Qwen2.5-0.5B"
    strategy: str = "grpo"  # grpo | ppo
    learning_rate: float = 3e-5
    num_epochs: int = 2
    group_size: int = 4  # G candidate completions sampled per prompt
    ppo_clip_eps: float = 0.2
    kl_coef: float = 0.04
    temperature: float = 0.8
    max_new_tokens: int = 64
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "correctness": 0.5,
            "helpfulness": 0.3,
            "safety": 0.2,
        }
    )


@dataclass
class DistributedConfig:
    """Distributed training & scaling configuration."""
    world_size: int = 4  # simulated or physical GPUs (1, 2, 4, 8)
    strategy: str = "ddp"  # ddp | fsdp | zero1 | zero2 | zero3
    interconnect_bandwidth_gbps: float = 900.0  # NVLink = 900, PCIe = 64
    interconnect_latency_us: float = 5.0
    gradient_compression: bool = False
    pipeline_parallel_stages: int = 1
    tensor_parallel_size: int = 1


@dataclass
class EvalConfig:
    """Multi-dimensional evaluation suite configuration."""
    eval_batch_size: int = 4
    temperature: float = 0.0
    max_new_tokens: int = 64
    dimensions: List[str] = field(default_factory=lambda: ["quality", "safety", "robustness"])
    robustness_noise_types: List[str] = field(default_factory=lambda: ["typo", "paraphrase", "distractor"])


@dataclass
class ProfilingConfig:
    """Hardware profiling & cost analysis configuration."""
    cost_per_gpu_hour_usd: float = 2.50  # e.g., A100 $2.50/hr, H100 $3.50/hr
    peak_device_tflops: float = 78.0  # FP16 Tensor Core TFLOPS (M3/MPS ~ 15-20, A100 = 312)
    sample_interval_s: float = 0.5


@dataclass
class PlatformConfig:
    """Master platform configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
