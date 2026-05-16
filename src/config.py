from dataclasses import dataclass


# Vision encoder
VISION_MODEL = "openai/clip-vit-base-patch32"
"""v1/v2/v3/v4 — 49 patches (7×7), 768 hidden, 224×224 input.

v3 Step 2 에서 ViT-L/14 를 시도했으나 0.5B LLM 한계로 효과 없음을 확인.
v4 는 "LLM 크기" 단일 변수만 키우는 비교이므로 ViT-B/32 를 그대로 유지한다."""


# v3 → v4 의 핵심 변경: 0.5B → 1.5B (3x params)
# v3 회고록에서 "0.5B LLM 의 시각적 추론" 을 진짜 병목으로 진단 → v4 가 직접 해결
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
"""v4 — Qwen2.5-1.5B-Instruct.

v3 의 Qwen2.5-0.5B 가 시각 detail 약하고 wrapper router 의존도 8/12 였던 한계 해소.
8GB VRAM 에 fp16 으로 로드 시 OOM 위험 → QLoRA 4-bit (bitsandbytes NF4) 필수."""

LLM_MODEL_V3 = "Qwen/Qwen2.5-0.5B-Instruct"
"""v3 baseline — v4 와 head-to-head 비교 시 참조용."""


# v4: Qwen2.5 tokenizer 에 이미 내장된 <|image_pad|> 토큰을 재사용한다.
# v3 는 새 토큰 "<image>" 를 add_special_tokens 로 추가 → resize_token_embeddings
# 호출 → PEFT 가 embed_tokens 를 modules_to_save 로 자동 등록 → adapter 1GB
# (v3 Step 4·5 의 slim adapter saga). v4 는 기존 토큰을 그대로 써서 resize 자체를
# 없앤다 → 4-bit 모델 resize 리스크 + 1GB adapter 문제를 원천 차단.
# splice 로직은 token id 로 위치만 찾으므로 토큰의 사전학습 임베딩 품질은 무관
# (해당 위치는 어차피 49개 image patch 로 교체됨).
IMAGE_TOKEN = "<|image_pad|>"
IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "You receive an image and a question, and answer concisely and accurately."
)

# v4 — QLoRA 정석 LoRA 대상: attention(q/k/v/o) + MLP(gate/up/down) 전 linear layer.
# QLoRA 논문 (Dettmers et al., 2023): "LoRA on all linear transformer block layers
# is required to match full finetuning performance." v3 는 attention(q/k/v/o) 만
# 적용해 raw 모델이 약했고 wrapper router 의존도가 컸음 (11/12 중 8/12).
# v4 는 MLP 까지 확장해 raw 모델 자체를 강화 → router 의존도 자동 해소를 노림.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]


@dataclass
class TrainConfig:
    data_path: str = "data/coco_subset/manifest.json"
    output_dir: str = "checkpoints/v4_stage1"

    # 1.5B + QLoRA + LoRA activations 가 8GB 에 빠듯
    # → batch_size 1 + grad_accum 으로 보완 (v3 의 0.5B 는 batch 2 + grad_accum 4)
    batch_size: int = 1
    grad_accum_steps: int = 8

    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_text_length: int = 512

    log_every: int = 20
    save_every: int = 500
    seed: int = 42

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Stage 2 → Stage 1 projector 이어받기용 (선택)
    init_projector: str | None = None

    # Vision encoder (v4 는 ViT-B/32 고정)
    vision_model: str = VISION_MODEL

    # tie_word_embeddings 분리 — v3 slim adapter 실험 계승
    untie_embeddings: bool = False

    # bf16 활성화 — 1.5B 에서는 LoRA 학습 안정성 + 메모리 양쪽 모두를 위해 권장
    # v4 default True (v3 는 ViT-L/14 시에만 활성)
    bf16: bool = True

    # ──────────────────────────────────────────────────────────────────
    # v4 신규: QLoRA 4-bit 양자화 (bitsandbytes NF4)
    # ──────────────────────────────────────────────────────────────────
    # 4-bit NF4 양자화로 1.5B LLM 을 8GB VRAM 에 fit.
    # 동작: bitsandbytes 가 LLM 가중치를 4-bit NF4 로 양자화 (≈0.9GB) → LoRA
    # adapter 만 bf16 으로 학습. v3 의 fp16/bf16 base 대비 ~4x 메모리 절감.
    # portfolio 10 (unsloth-qlora-finetuning) 에서 검증된 기법 재사용.
    # use_qlora=True 시 use_lora 도 자동 활성화 (4-bit base 는 LoRA 가 유일 학습 경로).
    use_qlora: bool = False

    # 4-bit 양자화 base 위에서 LoRA 학습 시 compute dtype.
    # bf16 (Ampere+ 권장) / fp16 (구형 GPU) 중 선택.
    qlora_compute_dtype: str = "bf16"

    # bnb nested quantization — quantization constant 자체를 한 번 더 양자화.
    # 추가 ~0.4 bit/param 절감, 정확도 영향 거의 없음 (Dettmers et al., QLoRA paper).
    qlora_use_double_quant: bool = True

    # gradient checkpointing — 1.5B + QLoRA 학습 시 activation memory 절감 필수.
    # 학습 속도 ~25% 느려지지만 VRAM ~50% 절감.
    gradient_checkpointing: bool = True


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
