"""MiniLLaVA — CLIP-ViT + MultiModalProjector + Qwen2.5 Causal LM.

LLaVA-1.5의 핵심 아키텍처를 직접 구현. HuggingFace의 LlavaForConditionalGeneration
같은 고수준 클래스를 사용하지 않고, 텍스트/이미지 임베딩 융합과 splice 로직을
저수준에서 직접 다룬다.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from .config import IGNORE_INDEX, IMAGE_TOKEN, LLM_MODEL, VISION_MODEL


class MultiModalProjector(nn.Module):
    """CLIP의 시각 특징을 LLM의 임베딩 공간으로 매핑하는 2-layer MLP.

    LLaVA-1.5의 'mlp2x_gelu' projector를 그대로 따른다.
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MiniLLaVA(nn.Module):
    """Vision-Language Model.

    - CLIP-ViT는 항상 frozen (강력한 사전학습 시각 표현 활용)
    - LLM은 기본 frozen (LLaVA Stage 1 alignment)
    - Stage 1: projector 만 학습
    - Stage 2: projector + LoRA(all-linear, r=16) 동시 학습
    """

    def __init__(
        self,
        vision_model_name: str = VISION_MODEL,
        llm_model_name: str = LLM_MODEL,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        use_qlora: bool = False,
        qlora_compute_dtype: str = "bf16",
        qlora_use_double_quant: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.vision = CLIPVisionModel.from_pretrained(vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        # v4 신규: QLoRA 4-bit NF4 양자화 (bitsandbytes) — 1.5B LLM 을 8GB VRAM 에 fit
        # base LLM 가중치를 4-bit 로 양자화 후 LoRA adapter 만 bf16 으로 학습.
        # use_qlora=True 시 LoRA 활성화는 호출자 (train.py) 책임.
        llm_kwargs: dict = {}
        if use_qlora:
            from transformers import BitsAndBytesConfig

            compute_dtype = (
                torch.bfloat16 if qlora_compute_dtype == "bf16" else torch.float16
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=qlora_use_double_quant,
            )
            llm_kwargs["quantization_config"] = bnb_config
            # 4-bit base 는 .to(cuda) 로 옮길 수 없어 로드 시점에 device 배치 필요.
            # 단일 GPU 라 {"": 0} 으로 전체를 GPU 0 에 고정 — "auto" 의 CPU offload
            # 분기 / accelerate hook 복잡도를 피하는 단일 GPU QLoRA 표준 패턴.
            llm_kwargs["device_map"] = {"": 0}
            print(
                f"[qlora] 4-bit NF4 + compute_dtype={qlora_compute_dtype}"
                f" + double_quant={qlora_use_double_quant}"
            )

        # transformers 5.x 는 dtype=, 4.x 는 torch_dtype= — 둘 다 지원하기 위해 동적 분기
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name, dtype=torch_dtype, **llm_kwargs
            )
        except TypeError:  # transformers 4.x fallback
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name, torch_dtype=torch_dtype, **llm_kwargs
            )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self._is_qlora = use_qlora

        # 이미지 placeholder 토큰 해석.
        # v4 는 Qwen2.5 내장 <|image_pad|> 사용 → vocab 에 이미 존재 → resize 불필요.
        # vocab 에 없으면 신규 추가 + resize_token_embeddings 로 폴백 (v3 동작).
        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            print(
                f"[model] '{IMAGE_TOKEN}' vocab 에 없음 "
                "→ 신규 추가 + resize_token_embeddings"
            )
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_TOKEN]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
        else:
            print(f"[model] '{IMAGE_TOKEN}' 내장 토큰 재사용 — embedding resize 없음")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        vision_hidden = self.vision.config.hidden_size
        llm_hidden = self.llm.config.hidden_size
        self.projector = MultiModalProjector(vision_hidden, llm_hidden)
        # projector 는 fp32 로 유지한다 (torch_dtype 으로 캐스팅하지 않음).
        # projector 는 학습 대상 — AdamW 의 master weight / moment(exp_avg 등) 를
        # fp32 로 둬야 파라미터 업데이트 정밀도가 보장된다 (QLoRA 정석: 4-bit base +
        # fp32 학습 파라미터). forward 시 image_embeds 는 _merge 에서 LLM 의 bf16
        # 임베딩으로 자연 다운캐스트되고, encode_image 가 projector dtype 에 맞춰
        # 입력을 정렬하므로 dtype 충돌은 없다.

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()

        # QLoRA: prepare_model_for_kbit_training 가 (1) base 전 파라미터 freeze,
        # (2) LayerNorm 을 fp32 로 upcast, (3) gradient checkpointing 활성화 +
        # input require-grad hook 등록 — 4-bit 학습 안정화를 한 번에 처리.
        # use_reentrant=False: 일부 입력만 grad 를 요구하는 경우에도 견고한
        # non-reentrant checkpointing 사용.
        if self._is_qlora:
            from peft import prepare_model_for_kbit_training

            self.llm = prepare_model_for_kbit_training(
                self.llm,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            print(
                "[qlora] prepare_model_for_kbit_training 완료 "
                f"(gradient_checkpointing={gradient_checkpointing})"
            )
        elif gradient_checkpointing:
            # 비-QLoRA 경로: prepare_model_for_kbit_training 를 거치지 않으므로
            # gradient checkpointing 을 직접 활성화. 이 시점의 self.llm 은 항상
            # raw 모델 (PEFT wrap 은 train.py 에서 이후 단계) 이라 직접 호출 가능.
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.llm.enable_input_require_grads()
            print("[init] gradient_checkpointing 활성화 (use_reentrant=False)")

        # prepare_model_for_kbit_training 가 이미 base 를 freeze 하므로 QLoRA 경로에선
        # 아래가 사실상 중복이지만, 비-QLoRA Stage 1 (freeze_llm=True) 을 위해 유지.
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    # ──────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, N_patches, D_llm]. CLS 토큰 제외.

        bf16 학습 시 dataloader 의 fp32 pixel_values + vision/projector 의 bf16
        가중치가 충돌하지 않도록 양쪽 모두 dtype 정렬.
        """
        # 1) vision encoder dtype 에 맞춰 pixel_values 변환
        vision_dtype = next(self.vision.parameters()).dtype
        if pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(vision_dtype)
        outputs = self.vision(pixel_values=pixel_values)
        patch_features = outputs.last_hidden_state[:, 1:, :]
        # 2) projector dtype 에 맞춰 patch_features 변환 (CLIP 이 fp32 로 promote 하는 경우 대비)
        proj_dtype = next(self.projector.parameters()).dtype
        if patch_features.dtype != proj_dtype:
            patch_features = patch_features.to(proj_dtype)
        return self.projector(patch_features)

    # ──────────────────────────────────────────────────────────────────
    # Embedding fusion: <image> 위치를 patch tokens로 splice
    # ──────────────────────────────────────────────────────────────────
    def _merge(
        self,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """input_ids에서 <image> 위치를 image_embeds(N개 patch)로 교체.

        - 모든 샘플은 정확히 1개의 <image> 토큰을 가진다고 가정
        - text/mask/label을 모두 일관되게 재정렬
        """
        B, L, D = text_embeds.shape
        N = image_embeds.shape[1]
        new_L = L - 1 + N

        device = text_embeds.device
        merged_embeds = torch.zeros(B, new_L, D, dtype=text_embeds.dtype, device=device)
        merged_mask = torch.zeros(B, new_L, dtype=attention_mask.dtype, device=device)
        merged_labels = (
            torch.full((B, new_L), IGNORE_INDEX, dtype=torch.long, device=device)
            if labels is not None
            else None
        )

        for b in range(B):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) != 1:
                raise ValueError(
                    f"sample {b}는 <image> 토큰이 {len(img_pos)}개 — 정확히 1개여야 합니다."
                )
            p = img_pos.item()

            # 앞 / 이미지 / 뒤 순으로 splice
            merged_embeds[b, :p] = text_embeds[b, :p]
            merged_embeds[b, p : p + N] = image_embeds[b]
            merged_embeds[b, p + N :] = text_embeds[b, p + 1 :]

            merged_mask[b, :p] = attention_mask[b, :p]
            merged_mask[b, p : p + N] = 1
            merged_mask[b, p + N :] = attention_mask[b, p + 1 :]

            if labels is not None:
                merged_labels[b, :p] = labels[b, :p]
                # 이미지 patch 위치는 IGNORE_INDEX 유지 (이미 채워둠)
                merged_labels[b, p + N :] = labels[b, p + 1 :]

        return merged_embeds, merged_mask, merged_labels

    # ──────────────────────────────────────────────────────────────────
    # Forward (학습)
    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)

        merged_embeds, merged_mask, merged_labels = self._merge(
            text_embeds, attention_mask, image_embeds, input_ids, labels
        )

        return self.llm(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            return_dict=True,
        )

    # ──────────────────────────────────────────────────────────────────
    # Generation (추론)
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)
        merged_embeds, merged_mask, _ = self._merge(
            text_embeds, attention_mask, image_embeds, input_ids, labels=None
        )

        # do_sample=False (greedy) 인데 temperature/top_p 를 넘기면 transformers 가
        # "temperature is set but do_sample is False" 경고를 매 호출 띄운다 — 게이트
        # 평가는 greedy 라 경고가 수백 번 반복된다. sampling 일 때만 인자를 전달한다.
        gen_kwargs = dict(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        return self.llm.generate(**gen_kwargs)

    # ──────────────────────────────────────────────────────────────────
    # OOD 신호: 답변 첫 토큰 logits (생성 루프 없이 forward 1회)
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def first_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """프롬프트까지 forward → 답변 첫 토큰 분포 logits [vocab_size].

        generate 는 output_scores 를 노출하지 않으므로, generate 와 동일한 splice
        경로(_merge)를 거친 뒤 마지막 위치 logits 를 직접 취한다. 입력이
        encode_for_inference (add_generation_prompt=True) 산출물이면 마지막 위치의
        next-token = 답변 첫 토큰. OOD entropy 신호(src/ood_detection.py)와 OOD
        ROC 보정(scripts/ood_roc_analysis.py)이 공유하는 단일 경로 — 보정 임계값이
        데모에서도 유효하려면 양쪽이 같은 코드로 entropy 를 계산해야 한다.
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)
        merged_embeds, merged_mask, _ = self._merge(
            text_embeds, attention_mask, image_embeds, input_ids, labels=None
        )
        out = self.llm(
            inputs_embeds=merged_embeds, attention_mask=merged_mask, return_dict=True
        )
        return out.logits[0, -1, :]

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint I/O — projector만 저장 (LLM/CLIP은 HF에서 다시 로드)
    # ──────────────────────────────────────────────────────────────────
    def save_projector(self, path: str) -> None:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.projector.load_state_dict(state)

    def load_lora_adapter(self, adapter_path: str) -> None:
        """학습된 LoRA adapter를 frozen LLM 위에 부착."""
        from peft import PeftModel

        self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
        self.llm.eval()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
