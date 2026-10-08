"""Data preprocessing and tokenization pipeline.
Implements ChatML instruction formatting, label masking (-100 on prompts),
and paired tokenization for Direct Preference Optimization (DPO).
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from frontier_platform.data.dataset import SFTExample, DPOExample


def format_chatml(prompt: str, completion: Optional[str] = None) -> str:
    """Format into standard ChatML conversation representation."""
    text = f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
    if completion is not None:
        text += f"{completion.strip()}<|im_end|>"
    return text


class SFTTorchDataset(Dataset):
    """PyTorch Dataset for SFT with prompt loss masking (-100)."""

    def __init__(
        self,
        examples: List[SFTExample],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        mask_prompt: bool = True,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt

        self.encoded_items: List[Dict[str, torch.Tensor]] = []
        self._preprocess()

    def _preprocess(self) -> None:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for ex in self.examples:
            prompt_header = format_chatml(ex.prompt, completion=None)
            full_text = format_chatml(ex.prompt, completion=ex.completion)

            prompt_ids = self.tokenizer.encode(prompt_header, add_special_tokens=False)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

            # Truncate if exceeds max length
            if len(full_ids) > self.max_seq_length:
                full_ids = full_ids[: self.max_seq_length]

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            # Mask out prompt tokens so loss is computed strictly on the assistant response
            if self.mask_prompt:
                prompt_len = min(len(prompt_ids), len(labels))
                labels[:prompt_len] = -100

            self.encoded_items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.encoded_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_items[idx]


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Dynamic padding collator for SFT batches."""
    max_len = max(len(item["input_ids"]) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DPOTorchDataset(Dataset):
    """PyTorch Dataset for DPO preference pairs (chosen vs rejected)."""

    def __init__(
        self,
        examples: List[DPOExample],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.encoded_items: List[Dict[str, torch.Tensor]] = []
        self._preprocess()

    def _preprocess(self) -> None:
        for ex in self.examples:
            prompt_header = format_chatml(ex.prompt, completion=None)
            chosen_text = format_chatml(ex.prompt, completion=ex.chosen)
            rejected_text = format_chatml(ex.prompt, completion=ex.rejected)

            prompt_ids = self.tokenizer.encode(prompt_header, add_special_tokens=False)
            chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)[: self.max_seq_length]
            rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)[: self.max_seq_length]

            # Chosen tensors
            c_input_ids = torch.tensor(chosen_ids, dtype=torch.long)
            c_attention_mask = torch.ones_like(c_input_ids)
            c_labels = c_input_ids.clone()
            c_labels[: min(len(prompt_ids), len(c_labels))] = -100

            # Rejected tensors
            r_input_ids = torch.tensor(rejected_ids, dtype=torch.long)
            r_attention_mask = torch.ones_like(r_input_ids)
            r_labels = r_input_ids.clone()
            r_labels[: min(len(prompt_ids), len(r_labels))] = -100

            self.encoded_items.append({
                "chosen_input_ids": c_input_ids,
                "chosen_attention_mask": c_attention_mask,
                "chosen_labels": c_labels,
                "rejected_input_ids": r_input_ids,
                "rejected_attention_mask": r_attention_mask,
                "rejected_labels": r_labels,
            })

    def __len__(self) -> int:
        return len(self.encoded_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_items[idx]


def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collator for DPO preference batches."""
    bsz = len(batch)
    max_c_len = max(len(b["chosen_input_ids"]) for b in batch)
    max_r_len = max(len(b["rejected_input_ids"]) for b in batch)

    c_ids = torch.full((bsz, max_c_len), pad_token_id, dtype=torch.long)
    c_mask = torch.zeros((bsz, max_c_len), dtype=torch.long)
    c_labels = torch.full((bsz, max_c_len), -100, dtype=torch.long)

    r_ids = torch.full((bsz, max_r_len), pad_token_id, dtype=torch.long)
    r_mask = torch.zeros((bsz, max_r_len), dtype=torch.long)
    r_labels = torch.full((bsz, max_r_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        cl = len(b["chosen_input_ids"])
        c_ids[i, :cl] = b["chosen_input_ids"]
        c_mask[i, :cl] = b["chosen_attention_mask"]
        c_labels[i, :cl] = b["chosen_labels"]

        rl = len(b["rejected_input_ids"])
        r_ids[i, :rl] = b["rejected_input_ids"]
        r_mask[i, :rl] = b["rejected_attention_mask"]
        r_labels[i, :rl] = b["rejected_labels"]

    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }
