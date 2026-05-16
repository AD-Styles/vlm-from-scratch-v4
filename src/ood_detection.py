"""OOD (Out-of-Distribution) Detection — v3 신규 모듈.

학습된 MiniLLaVA 가 본 적 없는 도메인의 이미지를 받았을 때, 자신있게 틀리는 대신
"잘 모르겠다" 라고 응답할 수 있도록 두 가지 신호를 결합:

  1. CLIP image-text similarity
     - 학습 분포(COCO + 일상 객체) 카테고리들과의 코사인 유사도
     - 최대 유사도가 낮으면 in-distribution 후보가 없음 → OOD 가능성 ↑

  2. LLM first-token entropy
     - 답변 첫 토큰 분포의 엔트로피
     - 모델이 다음 토큰을 자신있게 결정 못 할수록 entropy ↑ → OOD 가능성 ↑

  ood_score = w_clip * (1 - clip_max_sim) + w_entropy * normalized_entropy
  is_ood    = ood_score > threshold

설계 결정:
  - MiniLLaVA 의 CLIPVisionModel 과 별개로 CLIPModel (텍스트 인코더 포함) 을 따로 로드.
    같은 ViT 가중치를 공유하지만, OOD 검사는 image-text similarity 가 필요하므로
    text encoder 가 있는 full CLIPModel 이 필요함.
  - 결합 가중치(weight_clip)와 판정 임계값(threshold)은 calibration set 으로 함께
    보정한다. v4 는 100 케이스 OOD 벤치마크 (in-dist 40 + 만화 30 + 추상화 30)
    위에서 ROC 분석을 수행했다 (scripts/ood_roc_analysis.py):
      · v3 기본 가중치(CLIP 0.6)의 결합 ROC AUC = 0.848
      · w_clip 스윕 → 최적 0.35 에서 AUC = 0.897 (5-fold CV 0.892 — 과적합 아님)
      · 그 가중치에서 Youden's J 최적 임계값 = 0.35 (정확도 86%, TPR 92%)
    CLIP·entropy 는 상보적이라 어느 한쪽 단독(0.66 / 0.855)보다 0.35:0.65 결합이
    낫다. v3 의 초기값(CLIP 0.6 / threshold 0.5)은 N=2 검증뿐이라 미보정 상태였고
    같은 벤치마크 정확도 53% 로 OOD 대부분을 놓침이 실측됐다 → 기본값 재보정.
    (v3 회고록 숙제 #3 완수.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# In-distribution 카테고리 — COCO 80 classes + 자주 등장하는 일상 객체
# 학습 데이터 (COCO + VQAv2 + LocalizedNarratives + AOKVQA + KoLLaVA) 분포에 맞춤
DEFAULT_CATEGORIES = [
    # COCO classes (people, animals)
    "a person", "a man", "a woman", "a child", "a baby",
    "a dog", "a cat", "a horse", "a cow", "a sheep", "a bird", "an elephant",
    "a bear", "a zebra", "a giraffe",
    # COCO classes (vehicles)
    "a car", "a truck", "a bus", "a train", "a motorcycle", "a bicycle",
    "an airplane", "a boat",
    # COCO classes (indoor objects)
    "a chair", "a table", "a couch", "a bed", "a tv", "a laptop",
    "a book", "a cup", "a bottle", "a bowl", "a fork", "a knife",
    "a clock", "a vase",
    # COCO classes (food)
    "a pizza", "a sandwich", "a cake", "a banana", "an apple", "an orange",
    # Outdoor / scene
    "a building", "a tree", "a mountain", "a beach", "a road", "a sky",
    "a kitchen", "a living room", "a bathroom", "an office",
    # Misc everyday
    "a phone", "a keyboard", "a backpack", "an umbrella",
]


@dataclass
class OODResult:
    """OOD detection 단일 결과."""

    clip_max_sim: float
    """가장 비슷한 카테고리와의 cosine similarity (0~1)."""

    clip_match: str
    """clip_max_sim 에 해당하는 카테고리 텍스트."""

    llm_entropy: Optional[float]
    """LLM 첫 토큰 분포의 entropy (nats). first_logits 미제공 시 None."""

    ood_score: float
    """0~1 범위의 OOD 점수 (높을수록 OOD 가능성 ↑)."""

    is_ood: bool
    """ood_score > threshold 여부."""

    def to_dict(self) -> dict:
        return {
            "clip_max_sim": round(self.clip_max_sim, 4),
            "clip_match": self.clip_match,
            "llm_entropy": (
                round(self.llm_entropy, 4) if self.llm_entropy is not None else None
            ),
            "ood_score": round(self.ood_score, 4),
            "is_ood": self.is_ood,
        }


class OODDetector:
    """CLIP similarity + LLM entropy 기반 OOD 검출기.

    사용 예:
        detector = OODDetector(device="cuda")
        result = detector.score(image, first_logits=outputs.scores[0][0])
        if result.is_ood:
            answer = "이 이미지는 학습 도메인 밖입니다 (자신 없음)."

    GPU 필요시 device="cuda" 명시. 별도 인자 없으면 CPU 로 동작 (안전 기본값).
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        categories: Optional[list[str]] = None,
        threshold: float = 0.35,   # v4 ROC 재보정 (scripts/ood_roc_analysis.py)
        weight_clip: float = 0.35,    # v4 가중치 최적화 (이전 0.6 — CLIP 과대평가)
        weight_entropy: float = 0.65,
        device: Optional[str] = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold는 0~1 범위여야 함: {threshold}")
        if not math.isclose(weight_clip + weight_entropy, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"weight_clip + weight_entropy 는 1.0 이어야 함: "
                f"{weight_clip} + {weight_entropy} = {weight_clip + weight_entropy}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        self.w_clip = weight_clip
        self.w_entropy = weight_entropy

        self.categories = categories or DEFAULT_CATEGORIES
        if not self.categories:
            raise ValueError("categories 가 비어 있음 — 최소 1개 필요")

        print(f"[ood] loading CLIP ({clip_model_name}) on {self.device} ...")
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip.eval()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # 텍스트 임베딩은 한 번만 계산해 캐시 (카테고리는 고정)
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.categories, return_tensors="pt", padding=True
            ).to(self.device)
            text_result = self.clip.get_text_features(**text_inputs)
            # transformers 5.x 호환: BaseModelOutputWithPooling 반환 → .pooler_output
            # transformers 4.x 호환: Tensor 직접 반환
            text_emb = text_result.pooler_output if hasattr(text_result, "pooler_output") else text_result
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        self.text_embeddings = text_emb  # [n_categories, D]

        print(f"[ood] {len(self.categories)} 카테고리 임베딩 캐시 완료")

    # ──────────────────────────────────────────────────────────────────
    # 신호 1: CLIP image-text similarity
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def clip_similarity(self, image: Image.Image) -> tuple[float, str]:
        """Image 와 카테고리들 간 max cosine similarity → (score, best_category)."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        img_result = self.clip.get_image_features(**img_inputs)
        img_emb = img_result.pooler_output if hasattr(img_result, "pooler_output") else img_result
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb @ self.text_embeddings.T).squeeze(0)  # [n_categories]
        best_idx = int(sims.argmax().item())
        return float(sims[best_idx].item()), self.categories[best_idx]

    # ──────────────────────────────────────────────────────────────────
    # 신호 2: LLM first-token entropy
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def llm_entropy(first_logits: torch.Tensor) -> float:
        """답변 첫 토큰의 entropy (nats).

        Args:
            first_logits: [vocab_size] 또는 [1, vocab_size] — generate 의 scores[0]
        """
        if first_logits.dim() == 2:
            first_logits = first_logits.squeeze(0)
        if first_logits.dim() != 1:
            raise ValueError(
                f"first_logits 는 1D vocab logits 이어야 함, got shape {tuple(first_logits.shape)}"
            )
        log_probs = torch.log_softmax(first_logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()
        return float(entropy)

    # ──────────────────────────────────────────────────────────────────
    # 통합 score
    # ──────────────────────────────────────────────────────────────────
    def score(
        self,
        image: Image.Image,
        first_logits: Optional[torch.Tensor] = None,
        max_entropy_nats: float = 8.0,
    ) -> OODResult:
        """이미지 (+ 선택적 LLM logits) → OODResult.

        Args:
            image: 검사할 PIL 이미지
            first_logits: 답변 첫 토큰 logits ([vocab_size]). 없으면 LLM 신호 미사용.
            max_entropy_nats: entropy 정규화 스케일. Qwen2.5 vocab(151,936) 의
                              uniform entropy ≈ ln(151936) ≈ 11.93. 보수적으로 8.0
                              ( "꽤 불확실" 수준) 을 1.0 으로 매핑.
        """
        clip_sim, clip_match = self.clip_similarity(image)

        # CLIP 신호: 유사도가 낮을수록 OOD ↑
        # 일반적으로 similarity 가 0.2~0.35 범위이므로 (1 - sim) 그대로면 너무 큼.
        # 0.15 (꽤 비슷) ~ 0.30 (전혀 안비슷) 을 0~1 로 매핑.
        clip_signal = max(0.0, min(1.0, (0.30 - clip_sim) / 0.15))

        if first_logits is not None:
            entropy = self.llm_entropy(first_logits)
            entropy_signal = max(0.0, min(1.0, entropy / max_entropy_nats))
            ood_score = self.w_clip * clip_signal + self.w_entropy * entropy_signal
        else:
            entropy = None
            # LLM 신호 없으면 CLIP 만으로 판단 (가중치 재정규화)
            ood_score = clip_signal

        return OODResult(
            clip_max_sim=clip_sim,
            clip_match=clip_match,
            llm_entropy=entropy,
            ood_score=ood_score,
            is_ood=ood_score > self.threshold,
        )
