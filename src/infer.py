"""추론 헬퍼 — Gradio 앱과 CLI 양쪽에서 재사용."""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
from PIL import Image

from .config import VISION_MODEL, GenerationConfig
from .dataset import encode_for_inference
from .model import MiniLLaVA


class VLMInference:
    """학습된 MiniLLaVA를 감싸 단일 (image, question) → answer 호출 인터페이스 제공."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        vision_model: str = VISION_MODEL,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[infer] device={self.device}, vision={vision_model}")
        self.model = MiniLLaVA(
            vision_model_name=vision_model,
            freeze_vision=True,
            freeze_llm=True,
            torch_dtype=torch_dtype,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[infer] loading projector → {checkpoint_path}")
            self.model.load_projector(checkpoint_path, map_location="cpu")
        else:
            if checkpoint_path:
                print(f"[infer] WARNING: projector ckpt not found: {checkpoint_path}")
            else:
                print("[infer] WARNING: no projector checkpoint — random init.")

        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"[infer] loading LoRA adapter → {lora_adapter_path}")
            self.model.load_lora_adapter(lora_adapter_path)
        elif lora_adapter_path:
            print(f"[infer] WARNING: LoRA adapter not found: {lora_adapter_path}")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        question: str,
        gen_cfg: Optional[GenerationConfig] = None,
    ) -> dict:
        gen_cfg = gen_cfg or GenerationConfig()

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.model.image_processor(image, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)

        input_ids, attention_mask = encode_for_inference(
            self.model.tokenizer, question
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        t0 = time.time()
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            do_sample=gen_cfg.do_sample,
        )
        elapsed = time.time() - t0

        text = self.model.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return {"answer": text, "elapsed": elapsed}
