# Mini-LLaVA v4 — 8GB 노트북 GPU 한 장에서 조립·학습한 비전-언어 모델

CLIP 비전 인코더와 Qwen2.5-1.5B 언어 모델을 직접 이어 붙여 만든 소형 비전-언어 모델(VLM)입니다. HuggingFace 의 `LlavaForConditionalGeneration` 같은 통합 클래스를 쓰지 않고, 이미지 임베딩을 텍스트 시퀀스에 끼워 넣는 융합 로직을 저수준에서 직접 구현했습니다. 학습은 RTX 4060 Laptop 8GB 한 장에서 QLoRA 4-bit 로 진행했습니다.

LLaVA-1.5 구조를 **소비자용 GPU 한 장**이라는 제약 안에서 재현해 보는 학습용 프로젝트입니다. 8GB VRAM 과 약 9만 개의 학습 샘플(LLaVA-1.5 는 120만+)로 어디까지 되는지 확인하는 것이 목표이고, SOTA 성능이 목표는 아닙니다. v1→v4 로 이어진 시리즈의 마지막 버전이며, v3 대비 가장 큰 변경은 LLM 을 0.5B 에서 1.5B 로 키운 것입니다.

## 구조

```
  이미지 224×224                     텍스트 + <image> placeholder
       │                                     │
  CLIP-ViT-B/32 (frozen)            Qwen2.5 tokenizer + embedding
       │  49 patch × 768-d                   │
  ★ 2-layer MLP Projector                    │
       │  49 × 1536                          │
       └──────────────┬──────────────────────┘
                       │  <image> 위치를 49개 patch 임베딩으로 교체 (splice)
                       ▼
        Qwen2.5-1.5B-Instruct  (4-bit NF4 frozen + ★ LoRA r=16)
                       ▼
                   텍스트 응답

  ★ = 학습 대상 (projector fp32 / LoRA bf16). CLIP·LLM base 는 frozen.
```

- **비전** — CLIP-ViT-B/32, frozen. 224px 입력을 49개 patch × 768-d 로 인코딩.
- **Projector** — 2-layer MLP (768 → 1536, GELU). LLaVA-1.5 의 `mlp2x_gelu` 와 동일.
- **LLM** — Qwen2.5-1.5B-Instruct. 4-bit NF4 로 frozen 하고 LoRA(r=16) 만 학습.

## 직접 구현한 부분

VLM 의 핵심인 이미지-텍스트 융합을 고수준 라이브러리에 맡기지 않고 직접 다뤘습니다.

- **임베딩 splice (`src/model.py`)** — 입력 시퀀스에서 `<image>` 토큰 위치를 찾아 그 자리를 projector 가 낸 49개 patch 임베딩으로 교체합니다. text·attention_mask·label 을 모두 일관되게 재정렬하는 `_merge` 를 직접 구현했습니다.
- **QLoRA 로 8GB fit** — Qwen2.5-1.5B 를 fp32 로 올리면 가중치만 6GB 라 학습 자체가 불가능합니다. bitsandbytes 4-bit NF4 + double quantization 으로 base 를 압축하고 gradient checkpointing 으로 activation 메모리를 줄여 batch_size 1 로 학습했습니다. 학습 대상인 projector·LoRA 는 fp32 master weight 로 둬 업데이트 정밀도를 유지했습니다.
- **`<image>` 토큰 재사용** — 새 토큰을 추가하는 대신 Qwen2.5 에 내장된 `<|image_pad|>` 를 그대로 썼습니다. `resize_token_embeddings` 를 호출하지 않으므로, v3 에서 겪었던 "embedding resize → PEFT 가 `embed_tokens` 를 통째로 저장 → adapter 1GB" 문제가 생기지 않습니다. splice 는 토큰의 *위치*만 쓰므로 임베딩 품질과 무관합니다.
- **instruction-only label masking (`src/dataset.py`)** — system·user 토큰은 `IGNORE_INDEX` 로 가리고 assistant 응답 토큰에만 loss 를 줍니다.

## 학습

LLaVA-1.5 의 2단계 학습을 따랐습니다 — 먼저 projector 만 정렬하고, 그 위에서 LoRA instruction tuning.

| | Stage 1 — 정렬 | Stage 2 — instruction |
|---|---|---|
| 학습 대상 | projector 3.5M | projector + LoRA = 22.0M |
| 데이터 | 40K (이미지 2만 장 × 캡션 2) | 46K 믹스 |
| step / 시간 | 5,000 / ≈3.3h | 5,750 / ≈7.4h |
| loss | 5.0 → 1.98 | 3.65 → ≈1.01 |

- batch_size 1 + grad_accum 8 (effective batch 8), gradient checkpointing ON. Stage 1·2 모두 OOM 없이 완주했습니다.
- Stage 2 믹스는 VQAv2 18K(짧은 사실) + LocalizedNarratives 10K(긴 묘사) + A-OKVQA 6K(추론) + KoLLaVA 12K(한국어). 한 능력에 치우치지 않도록 의도적으로 섞었습니다.
- 학습 루프 `src/train.py` — cosine LR + warmup, grad clipping, 중간 체크포인트 저장.

## 결과

`scripts/eval_gate.py` 로 학습된 raw 모델(추론 wrapper 없음)을 평가했습니다 — VQAv2 val / POPE test 각 n=400, greedy decoding.

- **VQAv2 56.8%** — 유형별로 yes/no 76%, 개방형(other) 46%, 숫자 43%. 짧은 사실형 질문에 비교적 강하고 개방형·숫자에 약합니다.
- **POPE 71.8%** (yes-F1 0.74) — 객체 존재 여부 판별. 예측 분포는 yes 227 / no 173 으로 양쪽을 실제로 가립니다.

같은 시리즈의 v3 raw 는 각각 36.7% / 50.0% 였습니다 (v3 는 POPE 에서 전부 "yes" 만 답해 사실상 랜덤). v4 는 분명히 올랐지만, **절대 수치로는 공개된 소형 VLM(보통 VQAv2 70%대+, POPE 85%+)에 못 미칩니다.** 8GB·약 9만 학습 샘플이라는 제약을 그대로 반영한 결과입니다. 평가 하네스가 달라 외부 모델과의 직접 비교는 주의가 필요합니다.

이 평가는 배포 *전에* 통과 기준을 미리 정해놓고 진행했습니다 — v3 에서 검증 없이 배포했다가 성능 미달을 뒤늦게 발견한 경험 때문입니다. 기준값(VQAv2 ≥0.45, POPE ≥0.65 등 5개)은 SOTA 가 아니라 "v3·랜덤보다 분명히 위" 라는 최소선이고, 사후가 아니라 사전에 고정했다는 점이 핵심입니다. raw 모델이 5개 기준을 모두 통과해 배포했습니다.

**라이브 데모 확인** — 배포된 [HF Space](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) 에 영어/한국어 10케이스를 직접 입력했습니다 (`scripts/smoke_space_demo.py`, 무료 CPU 티어 · 샘플링 디코딩이라 실행마다 답이 달라짐).

- 잘 되는 것 — 장면 분류("무슨 방?" → "Kitchen"), yes/no("헤드폰 썼나?" → "Yes"), 단순 행동("뭐 하나?" → "Typing").
- 안 되는 것 — 고양이를 안은 여성 사진에 "What animal?" → **"Dog"** (세밀한 구분 실패). 긴 묘사를 시키면 형식은 유창하나 없는 디테일을 지어냅니다 (같은 사진을 "한 남성이 주방에서…" 로 서술). 추상화에는 "Birds" 처럼 자신있게 틀립니다.

정리하면 v4 는 짧은 사실형 VQA 에는 쓸 만하고, 세밀한 인식과 장문 생성에는 약합니다 — 1.5B 규모의 분명한 천장입니다.

## OOD 검출 (오프라인 분석)

분포 밖 입력(만화·추상화)에 모델이 자신있게 틀리는 문제를 별도로 분석했습니다. CLIP image-text 유사도와 LLM 첫 토큰 entropy 두 신호를 100케이스(in-dist 40 + 만화 30 + 추상화 30)로 ROC 비교한 결과, entropy 단독이 AUC **0.971** 로 가장 강했고(CLIP 단독 0.66), 5-fold 교차검증 0.969 로 과적합이 아님을 확인했습니다 (`scripts/ood_roc_analysis.py`). 최적 임계값에서 판정 정확도는 91% 입니다.

이 검출기는 오프라인 분석용입니다 — 배포된 데모에는 적용하지 않았고, 데모는 raw 모델만 서빙하므로 위 추상화 사례처럼 OOD 입력에 그대로 답합니다.

## 한계

- **모델 성능** — 절대 성능은 소형 VLM 기준에도 못 미칩니다. 1.5B 용량과 약 9만 샘플이라는 학습 규모의 천장으로, 더 큰 모델이나 수십만 규모 데이터 없이는 넘기 어렵습니다.
- **한국어 정량 평가 없음** — 한국어 학습 데이터는 4K→12K 로 늘렸으나, 신뢰할 만한 한국어 VLM 벤치마크가 없어 정량 평가셋은 만들지 못했습니다 (라이브 데모 정성 확인으로 갈음).
- **배포 정밀도 갭** — 학습은 4-bit base 로 했으나 무료 CPU 데모는 fp32 base 로 추론합니다 (QLoRA 의 표준 배포 절충). `scripts/diag_deploy_gap.py` 비교상 답변 차이는 거의 없었습니다.

## 재현

환경: Windows 11 · RTX 4060 Laptop 8GB · Python 3.11. 데이터셋은 HuggingFace 에서 받습니다.

```powershell
pip install -r requirements.txt

# Stage 1 — projector 정렬
python -m src.train --data-path data/coco_subset/manifest.json `
  --output-dir checkpoints/v4_stage1 `
  --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 1e-3 --use-qlora --bf16

# Stage 2 — QLoRA instruction tuning
python -m src.train --data-path data/v4_stage2_mix/manifest.json `
  --output-dir checkpoints/v4_stage2_qlora `
  --init-projector checkpoints/v4_stage1/projector.pt `
  --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 2e-4 `
  --use-qlora --use-lora --lora-r 16 --lora-alpha 32 --bf16

# 배포 게이트 — raw 모델 평가
python scripts/eval_gate.py --n-vqav2 400 --n-pope 400 `
  --projector checkpoints/v4_stage2_qlora/projector.pt `
  --lora-adapter checkpoints/v4_stage2_qlora/lora_adapter
```

데이터 다운로드·믹스, POPE 분할, OOD 분석 등 나머지 스크립트는 `scripts/` 에 있습니다.

## 링크

- 데모 — [HF Space: mini-llava-v4-demo](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) (무료 CPU 티어, 응답에 수십 초)
- 가중치 — [HF Hub: mini-llava-v4](https://huggingface.co/AD-Styles/mini-llava-v4)
- 이전 버전 — [v3](https://github.com/AD-Styles/vlm-from-scratch-v3) · [v2](https://github.com/AD-Styles/vlm-from-scratch)
- 참고 — Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS 2023)
