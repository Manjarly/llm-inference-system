"""Phase 1: Supervised Fine-Tuning (SFT) Engine.
Supports Full Fine-Tuning, Parameter-Efficient Fine-Tuning (LoRA),
and Quantized Low-Rank Adaptation (QLoRA).
Instrumented with step-level loss, gradient clipping, VRAM profiling, and throughput tracking.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from frontier_platform.config import SFTConfig
from frontier_platform.data.dataset import SFTExample
from frontier_platform.data.preprocessor import SFTTorchDataset, sft_collate_fn
from inference.quantization.quantizer import quantize_model

logger = logging.getLogger("platform.training.sft")


@dataclass
class SFTTrainMetrics:
    strategy: str
    total_params: int
    trainable_params: int
    trainable_pct: float
    final_loss: float
    training_time_s: float
    tokens_per_second: float
    peak_memory_mb: float
    history_loss: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "trainable_pct": round(self.trainable_pct, 3),
            "final_loss": round(self.final_loss, 4),
            "training_time_s": round(self.training_time_s, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "history_loss": [round(x, 4) for x in self.history_loss],
        }


class SFTTrainer:
    """Supervised fine-tuning trainer supporting Full FT, LoRA, and QLoRA."""

    def __init__(self, config: SFTConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32

        logger.info(
            "Initializing SFTTrainer: model=%s, strategy=%s, device=%s",
            config.model_id, config.strategy, self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model, self.param_stats = self._setup_model()

    def _resolve_device(self, dev: str) -> torch.device:
        if dev == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(dev)

    def _setup_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        raw_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
        )

        if self.config.strategy == "full":
            # Full fine-tuning: all weights require gradients
            model = raw_model.to(self.device)
            for p in model.parameters():
                p.requires_grad = True
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

        elif self.config.strategy == "lora":
            # LoRA adaptation: freeze base, inject rank-r adapters
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
            )
            model = get_peft_model(raw_model, peft_config).to(self.device)

        elif self.config.strategy == "qlora":
            # QLoRA: Quantize base linear layers to 4-bit, inject FP16 LoRA adapters
            q_model, _ = quantize_model(raw_model, mode="int4", group_size=self.config.int4_group_size)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
            )
            model = get_peft_model(q_model, peft_config).to(self.device)
        else:
            raise ValueError(f"Unknown SFT strategy: {self.config.strategy}")

        tot_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = (train_params / tot_params * 100.0) if tot_params > 0 else 0.0

        logger.info(
            "Model parameters [%s]: %d total, %d trainable (%.2f%%)",
            self.config.strategy, tot_params, train_params, pct
        )
        return model, {
            "total_params": tot_params,
            "trainable_params": train_params,
            "trainable_pct": pct,
        }

    def train(self, dataset_examples: List[SFTExample]) -> SFTTrainMetrics:
        """Execute SFT training loop with loss logging, memory tracking, and schedulers."""
        train_dataset = SFTTorchDataset(
            examples=dataset_examples,
            tokenizer=self.tokenizer,
            max_seq_length=256,
            mask_prompt=True,
        )
        collate_fn = lambda b: sft_collate_fn(b, pad_token_id=self.tokenizer.pad_token_id)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        self.model.train()
        history_loss: List[float] = []
        total_tokens_trained = 0
        peak_vram_mb = 0.0

        t0 = time.time()
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                total_tokens_trained += attention_mask.sum().item()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                current_loss = loss.item() * self.config.gradient_accumulation_steps
                history_loss.append(current_loss)

                # Track peak memory
                if self.device.type == "mps":
                    curr_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
                    peak_vram_mb = max(peak_vram_mb, curr_mem)
                elif self.device.type == "cuda":
                    curr_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                    peak_vram_mb = max(peak_vram_mb, curr_mem)

        elapsed = max(0.001, time.time() - t0)
        tps = total_tokens_trained / elapsed
        final_loss = history_loss[-1] if history_loss else 0.0

        if peak_vram_mb == 0.0:
            # Estimate peak VRAM from parameter count and buffers if device query was 0
            peak_vram_mb = (self.param_stats["total_params"] * 2) / (1024 * 1024) * 1.5

        return SFTTrainMetrics(
            strategy=self.config.strategy,
            total_params=self.param_stats["total_params"],
            trainable_params=self.param_stats["trainable_params"],
            trainable_pct=self.param_stats["trainable_pct"],
            final_loss=final_loss,
            training_time_s=elapsed,
            tokens_per_second=tps,
            peak_memory_mb=peak_vram_mb,
            history_loss=history_loss,
        )
