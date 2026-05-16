"""VQA-style dataset + collator.

manifest.json 형식:
  [
    {"image": "path/to/img.jpg", "question": "...", "answer": "..."},
    ...
  ]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, PreTrainedTokenizerBase

from .config import IGNORE_INDEX, IMAGE_TOKEN, SYSTEM_PROMPT


def _build_messages(question: str, answer: str | None = None):
    """Qwen2.5 chat template 형식의 messages list.

    user 메시지에 <image>\\n 을 prepend 하여 이미지 위치를 명시한다.
    """
    user_content = f"{IMAGE_TOKEN}\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def encode_for_training(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    max_length: int = 512,
):
    """학습용: full conversation + instruction-only label masking.

    답변(assistant) 토큰만 loss를 받고, 그 이전(system+user)은 IGNORE_INDEX 처리.
    """
    full_msgs = _build_messages(question, answer)
    prompt_msgs = _build_messages(question, answer=None)

    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    full = tokenizer(
        full_text, max_length=max_length, truncation=True, return_tensors="pt"
    )
    prompt = tokenizer(prompt_text, truncation=True, return_tensors="pt")

    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]
    labels = input_ids.clone()
    prompt_len = min(prompt["input_ids"].shape[1], len(labels))
    labels[:prompt_len] = IGNORE_INDEX

    return input_ids, attention_mask, labels


def encode_for_inference(
    tokenizer: PreTrainedTokenizerBase, question: str, max_length: int = 512
):
    """추론용: prompt까지만 (assistant 응답 시작 직전)."""
    prompt_msgs = _build_messages(question, answer=None)
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(
        prompt_text, max_length=max_length, truncation=True, return_tensors="pt"
    )
    return enc["input_ids"][0], enc["attention_mask"][0]


class VQADataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: CLIPImageProcessor,
        max_length: int = 256,
    ):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["image"]).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        input_ids, attention_mask, labels = encode_for_training(
            self.tokenizer, s["question"], s["answer"], self.max_length
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


@dataclass
class VQACollator:
    """가변 길이 텍스트를 우측 패딩, 이미지는 단순 stack."""

    pad_token_id: int

    def __call__(self, batch: List[dict]):
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            ids = item["input_ids"]
            am = item["attention_mask"]
            lb = item["labels"]
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat(
                    [ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]
                )
                am = torch.cat([am, torch.zeros(pad_len, dtype=am.dtype)])
                lb = torch.cat(
                    [lb, torch.full((pad_len,), IGNORE_INDEX, dtype=lb.dtype)]
                )
            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lb)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        }
