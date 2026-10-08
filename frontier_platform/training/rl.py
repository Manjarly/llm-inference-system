"""Phase 3: Reinforcement Learning Policy Alignment (GRPO / PPO).
Implements Group Relative Policy Optimization (GRPO), eliminating the need for
a separate critic network while enabling online sampling and verifiable reward alignment.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from frontier_platform.config import RLConfig

logger = logging.getLogger("platform.training.rl")


@dataclass
class RLTrainMetrics:
    mean_reward: float
    reward_variance: float
    kl_divergence: float
    policy_loss: float
    training_time_s: float
    history_rewards: List[float] = field(default_factory=list)
    history_kl: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_reward": round(self.mean_reward, 3),
            "reward_variance": round(self.reward_variance, 4),
            "kl_divergence": round(self.kl_divergence, 4),
            "policy_loss": round(self.policy_loss, 4),
            "training_time_s": round(self.training_time_s, 2),
            "history_rewards": [round(x, 3) for x in self.history_rewards],
            "history_kl": [round(x, 4) for x in self.history_kl],
        }


def compute_reward_score(prompt: str, completion: str, weights: Dict[str, float]) -> float:
    """Multi-dimensional reward evaluation: correctness, helpfulness, and safety."""
    score = 0.0

    # 1. Verification of correctness & structure
    has_structure = ("1." in completion or "-" in completion or "```" in completion or "\n\n" in completion)
    length_penalty = 1.0 if len(completion.split()) >= 10 else 0.4
    correctness_score = (1.0 if has_structure else 0.6) * length_penalty
    score += weights.get("correctness", 0.5) * correctness_score

    # 2. Helpfulness & conciseness (penalizing repetitive loops)
    words = completion.lower().split()
    unique_ratio = len(set(words)) / max(1, len(words))
    helpfulness_score = min(1.0, unique_ratio * 1.2)
    score += weights.get("helpfulness", 0.3) * helpfulness_score

    # 3. Safety compliance (rewarding proper refusal of hazardous queries)
    is_harmful_prompt = any(kw in prompt.lower() for kw in ["phishing", "firewall bypass", "malware", "exploit"])
    is_safe_refusal = any(kw in completion.lower() for kw in ["cannot", "i cannot", "unauthorized", "mitigate", "prevention"])
    if is_harmful_prompt:
        safety_score = 1.0 if is_safe_refusal else 0.0
    else:
        safety_score = 1.0 if not any(kw in completion.lower() for kw in ["offensive", "unethical"]) else 0.5
    score += weights.get("safety", 0.2) * safety_score

    return score


class RLPolicyTrainer:
    """Group Relative Policy Optimization (GRPO) Trainer."""

    def __init__(self, config: RLConfig, policy_model: Optional[nn.Module] = None) -> None:
        self.config = config
        self.device = self._resolve_device(config.model_id)
        self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32

        logger.info("Initializing RLPolicyTrainer (GRPO): group_size=%d", config.group_size)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if policy_model is not None:
            self.policy_model = policy_model.to(self.device)
        else:
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=self.dtype
            ).to(self.device)

        # Frozen reference model for KL divergence regularizer
        self.ref_model = copy.deepcopy(self.policy_model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def _resolve_device(self, _: str) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train_step(self, prompts: List[str]) -> Tuple[float, float, float]:
        """Perform one GRPO policy update over a batch of prompts with group sampling."""
        optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

        all_step_rewards: List[float] = []
        total_kl = 0.0
        total_loss = 0.0

        for prompt in prompts:
            # 1. Sample G candidate completions from policy
            p_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            p_len = p_inputs["input_ids"].shape[1]

            group_outputs: List[str] = []
            group_rewards: List[float] = []
            group_tokens: List[torch.Tensor] = []

            with torch.no_grad():
                for _ in range(self.config.group_size):
                    gen_ids = self.policy_model.generate(
                        **p_inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    completion_text = self.tokenizer.decode(gen_ids[0, p_len:], skip_special_tokens=True)
                    r = compute_reward_score(prompt, completion_text, self.config.reward_weights)
                    group_outputs.append(completion_text)
                    group_rewards.append(r)
                    group_tokens.append(gen_ids)

            # 2. Compute group-relative normalized advantages
            mean_r = sum(group_rewards) / len(group_rewards)
            var_r = sum((r - mean_r) ** 2 for r in group_rewards) / max(1, len(group_rewards))
            std_r = math.sqrt(var_r) + 1e-6
            advantages = [(r - mean_r) / std_r for r in group_rewards]
            all_step_rewards.extend(group_rewards)

            # 3. Policy gradient update with PPO clipping & KL regularization
            for i, gen_ids in enumerate(group_tokens):
                adv = torch.tensor(advantages[i], device=self.device, dtype=torch.float32)

                # Policy forward
                policy_logits = self.policy_model(gen_ids).logits
                with torch.no_grad():
                    ref_logits = self.ref_model(gen_ids).logits

                # Compute completion token log-probs
                shift_policy = policy_logits[0, p_len - 1 : -1, :]
                shift_ref = ref_logits[0, p_len - 1 : -1, :]
                target_ids = gen_ids[0, p_len:]

                policy_logp = F.log_softmax(shift_policy, dim=-1)
                ref_logp = F.log_softmax(shift_ref, dim=-1)

                p_token_logps = torch.gather(policy_logp, 1, target_ids.unsqueeze(1)).squeeze(1)
                r_token_logps = torch.gather(ref_logp, 1, target_ids.unsqueeze(1)).squeeze(1)

                # Ratio: pi / pi_old (here single step pi_old is detached ref/init)
                ratio = torch.exp(p_token_logps - p_token_logps.detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip_eps, 1.0 + self.config.ppo_clip_eps) * adv
                policy_obj = torch.min(surr1, surr2).mean()

                # KL penalty: approx KL = log(pi) - log(pi_ref)
                kl = (p_token_logps - r_token_logps).mean()
                loss = -(policy_obj - self.config.kl_coef * kl)

                loss.backward()
                total_loss += loss.item()
                total_kl += kl.item()

        optimizer.step()
        optimizer.zero_grad()

        mean_reward = sum(all_step_rewards) / max(1, len(all_step_rewards))
        avg_kl = total_kl / max(1, len(prompts) * self.config.group_size)
        avg_loss = total_loss / max(1, len(prompts) * self.config.group_size)
        return mean_reward, avg_kl, avg_loss

    def train(self, prompts: List[str], num_iterations: int = 3) -> RLTrainMetrics:
        """Run RL alignment iterations."""
        history_r: List[float] = []
        history_kl: List[float] = []
        t0 = time.time()

        for it in range(num_iterations):
            mean_r, kl, loss = self.train_step(prompts)
            history_r.append(mean_r)
            history_kl.append(kl)
            logger.info("RL Iteration %d: Mean Reward=%.3f, KL=%.4f, Loss=%.4f", it + 1, mean_r, kl, loss)

        elapsed = max(0.001, time.time() - t0)
        final_mean_r = history_r[-1] if history_r else 0.0
        r_var = sum((r - final_mean_r) ** 2 for r in history_r) / max(1, len(history_r))

        return RLTrainMetrics(
            mean_reward=final_mean_r,
            reward_variance=r_var,
            kl_divergence=history_kl[-1] if history_kl else 0.0,
            policy_loss=loss,
            training_time_s=elapsed,
            history_rewards=history_r,
            history_kl=history_kl,
        )
