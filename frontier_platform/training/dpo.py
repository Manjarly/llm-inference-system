"""Phase 2: Direct Preference Optimization (DPO) Engine.
Implements the closed-form DPO objective aligning a policy model against
a frozen reference model without training an auxiliary reward model.
Tracks implicit reward margins, preference accuracies, and log-ratio divergences.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from frontier_platform.config import DPOConfig
from frontier_platform.data.dataset import DPOExample
from frontier_platform.data.preprocessor import DPOTorchDataset, dpo_collate_fn

logger = logging.getLogger("platform.training.dpo")


@dataclass
class DPOTrainMetrics:
    final_loss: float
    reward_margin: float
    accuracy: float
    training_time_s: float
    history_loss: List[float] = field(default_factory=list)
    history_margin: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_loss": round(self.final_loss, 4),
            "reward_margin": round(self.reward_margin, 3),
            "accuracy": round(self.accuracy, 3),
            "training_time_s": round(self.training_time_s, 2),
            "history_loss": [round(x, 4) for x in self.history_loss],
            "history_margin": [round(x, 3) for x in self.history_margin],
        }


def compute_sequence_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log-probabilities for target response tokens (where labels != -100)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Log softmax across vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log prob of true labels
    # For padded/masked positions where label is -100, replace with 0 for gather then mask
    mask = (shift_labels != -100)
    clamped_labels = shift_labels.clone()
    clamped_labels[~mask] = 0

    per_token_logps = torch.gather(log_probs, dim=2, index=clamped_labels.unsqueeze(2)).squeeze(2)
    per_token_logps = per_token_logps * mask.float()

    return per_token_logps.sum(dim=-1)


class DPOTrainer:
    """Direct Preference Optimization (DPO) trainer."""

    def __init__(self, config: DPOConfig, policy_model: Optional[nn.Module] = None) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32

        logger.info("Initializing DPOTrainer: model=%s, beta=%.2f", config.model_id, config.beta)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup policy model (trainable)
        if policy_model is not None:
            self.policy_model = policy_model.to(self.device)
        else:
            raw_model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=self.dtype,
            )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.policy_model = get_peft_model(raw_model, peft_config).to(self.device)

        if hasattr(self.policy_model, "gradient_checkpointing_enable"):
            self.policy_model.gradient_checkpointing_enable()

        # Setup reference model (frozen snapshot)
        # If policy_model supports disable_adapter, we avoid duplicating 500M params in memory!
        if hasattr(self.policy_model, "disable_adapter"):
            self.ref_model = None
            logger.info("Using adapter disabling for zero-memory reference model in DPO")
        else:
            self.ref_model = copy.deepcopy(self.policy_model).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

    def _resolve_device(self, dev: str) -> torch.device:
        if dev == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(dev)

    def train(self, dataset_examples: List[DPOExample]) -> DPOTrainMetrics:
        """Run DPO optimization loop over paired chosen/rejected examples."""
        train_dataset = DPOTorchDataset(
            examples=dataset_examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_prompt_length + self.config.max_response_length,
        )
        collate_fn = lambda b: dpo_collate_fn(b, pad_token_id=self.tokenizer.pad_token_id)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(dataloader) * self.config.num_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

        self.policy_model.train()
        history_loss: List[float] = []
        history_margins: List[float] = []
        correct_preferences = 0
        total_pairs = 0

        t0 = time.time()
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(dataloader):
                c_ids = batch["chosen_input_ids"].to(self.device)
                c_mask = batch["chosen_attention_mask"].to(self.device)
                c_labels = batch["chosen_labels"].to(self.device)

                r_ids = batch["rejected_input_ids"].to(self.device)
                r_mask = batch["rejected_attention_mask"].to(self.device)
                r_labels = batch["rejected_labels"].to(self.device)

                # 1. Forward policy model
                policy_chosen_logps = compute_sequence_logps(self.policy_model, c_ids, c_mask, c_labels)
                policy_rejected_logps = compute_sequence_logps(self.policy_model, r_ids, r_mask, r_labels)

                # 2. Forward frozen reference model
                with torch.no_grad():
                    if hasattr(self.policy_model, "disable_adapter"):
                        with self.policy_model.disable_adapter():
                            ref_chosen_logps = compute_sequence_logps(self.policy_model, c_ids, c_mask, c_labels)
                            ref_rejected_logps = compute_sequence_logps(self.policy_model, r_ids, r_mask, r_labels)
                    else:
                        ref_chosen_logps = compute_sequence_logps(self.ref_model, c_ids, c_mask, c_labels)
                        ref_rejected_logps = compute_sequence_logps(self.ref_model, r_ids, r_mask, r_labels)

                # 3. Compute implicit rewards: beta * log(pi / pi_ref)
                chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
                rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
                reward_margin = chosen_rewards - rejected_rewards

                # 4. DPO Loss: -log sigmoid(r_w - r_l)
                losses = -F.logsigmoid(reward_margin)
                loss = losses.mean() / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()

                # Metrics logging
                history_loss.append(loss.item() * self.config.gradient_accumulation_steps)
                history_margins.append(reward_margin.mean().item())

                correct_preferences += (reward_margin > 0).sum().item()
                total_pairs += len(reward_margin)

        elapsed = max(0.001, time.time() - t0)
        accuracy = (correct_preferences / total_pairs) if total_pairs > 0 else 0.0

        return DPOTrainMetrics(
            final_loss=history_loss[-1] if history_loss else 0.0,
            reward_margin=sum(history_margins) / len(history_margins) if history_margins else 0.0,
            accuracy=accuracy,
            training_time_s=elapsed,
            history_loss=history_loss,
            history_margin=history_margins,
        )
