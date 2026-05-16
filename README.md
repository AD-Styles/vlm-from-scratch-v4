# Mini-LLaVA v4 — 1.5B LLM 업그레이드로 v3 병목 돌파 (raw VQAv2 +20%p · POPE +22%p)

> ✅ **학습·평가 완료 — 배포 게이트 5/5 GO.** v3 는 병목을 *"0.5B LLM"* 으로 지목했지만, 재검토해 보니 절반은 학습 데이터 문제였습니다 (Stage 1 정렬 5K — LLaVA 의 ~1%). v4 는 LLM 을 1.5B 로 키우고, 정렬 데이터를 8배(5K→40K) 늘리고, Stage 2 instruction 을 균형 믹스로 재구성한 뒤 배포 전 평가 게이트로 검증했습니다. 결과 (RTX 4060 Laptop 8GB 한 장, wrapper 없는 raw 모델): **VQAv2 36.67%→56.83%, POPE 50.00%→71.75%**. v4 는 Mini-LLaVA 시리즈(v1→v4)의 마지막 버전입니다.

[← v3: vlm-from-scratch-v3](https://github.com/AD-Styles/vlm-from-scratch-v3) | [← v2: vlm-from-scratch](https://github.com/AD-Styles/vlm-from-scratch)

---

## 🎯 v3 의 한계 6가지 — v4 의 대응

v3 [회고록의 한계 표](https://github.com/AD-Styles/vlm-from-scratch-v3#%EF%B8%8F-한계와-v4-계획-limitations--next-steps)가 남긴 항목과 v4 의 대응입니다. #1~#5 는 v3 가 표로 적어둔 숙제, #6 은 v4 가 재검토하며 추가로 식별한 항목입니다.

| # | v3 의 한계 | 진단 | v4 의 대응 |
|---|---|---|---|
| 1 | ⭐ **0.5B LLM 의 시각적 추론** | 시각 detail 약함 (cartoon case 등) | **Qwen2.5-1.5B + QLoRA 4-bit** — LLM 키우기 |
| 2 | POPE threshold 가 test set 으로 튜닝됨 | 70% 일반화 보장 X (untuned 53%) | POPE train/test 분리 후 untuned 재측정 |
| 3 | OOD 검증 N=2 | 임계값 0.5 일반화 부족 | 만화·추상화·in-dist 100 케이스로 ROC AUC 재보정 (N=2 → 100) |
| 4 | ⭐ wrapper 11/12 중 8/12 가 router 기여 | "VLM 능력" 보다 "ensemble routing" | raw 모델이 게이트 5/5 통과 → router wrapper 불필요 |
| 5 | 한국어 데이터 4K 만 | 환각 잔존 + 한국어 평가셋 없음 | KoLLaVA 한국어 4K→12K (3배) 확대. 표준 벤치마크 부재로 평가셋은 미도입 |
| 6 | ⭐ **학습 데이터 규모** | Stage 1 정렬 5K·Stage 2 13K = LLaVA 의 1~2%. projector 가 5K 로만 학습돼 비전-언어 정렬 자체가 약함 | Stage 1 정렬 데이터 5K→40K (8배) 확대 — projector-only 라 학습 비용 저렴 |

---

## 🏗️ 아키텍처 — v3 와 같은 LLaVA-1.5 mini 구조, LLM 만 교체

```
   이미지 (224×224)              텍스트 + <image> 자리표시
        │                                  │
        ▼                                  ▼
   CLIP-ViT-B/32 (frozen)         Tokenizer + Embed
        │ [49, 768]                        │
        ▼                                  │
   ★ MLP Projector                        │
        │ [49, 1536]   ← v3 의 896 → 1536 (Qwen2.5-1.5B hidden)
        └────────┬─────────────────────────┘
                 ▼
   <image> 자리에 patch 49개 splice
                 │
                 ▼
   Qwen2.5-1.5B (★ 4-bit NF4 frozen + ★ LoRA on 전 linear layer)
                 │
                 ▼
       "응답 ..."
```

★ 학습 대상 (projector fp32 + LoRA). 4-bit base 는 frozen.

**v3 대비 두 가지 구현 결정:**
- **LoRA 를 전 linear layer 에** — attention(q/k/v/o) 뿐 아니라 MLP(gate/up/down)까지.
  QLoRA 논문이 *"LoRA on all linear transformer block layers is required to match
  full finetuning performance"* 라고 명시합니다. v3 는 attention 만 적용했습니다.
- **`<image>` 신규 토큰 대신 Qwen2.5 내장 `<|image_pad|>` 재사용** — `resize_token_embeddings`
  를 호출하지 않음 → v3 의 "embedding resize → PEFT 가 embed_tokens 자동 저장 →
  adapter 1GB" 문제를 원천 차단. splice 는 token id 로 위치만 찾으므로
  토큰 임베딩 품질과 무관.

### 왜 4-bit QLoRA 인가

LLM 가중치만 따진 산술 메모리 (1.5B params × dtype 크기):

| 모델 dtype | 가중치 메모리 (산술) |
|---|---|
| fp32 (4 bytes) | 1.5B × 4 ≈ 6 GB |
| bf16 (2 bytes) | 1.5B × 2 ≈ 3 GB |
| **4-bit NF4 (0.5 byte)** | **1.5B × 0.5 ≈ 0.75 GB** |

가중치 외에 activations · gradient · optimizer state 가 더 쌓이므로, 8GB 한 장에서
1.5B 를 학습하려면 가중치를 최대한 압축하는 4-bit NF4 가 현실적 선택입니다.
`scripts/vram_test_v4.py` 의 smoke test 로 batch_size / grad_accum / max_text_length
를 정한 뒤 학습에 들어갔고, Stage 1·2 모두 OOM 없이 완주했습니다.

[portfolio 10 (unsloth-qlora-finetuning)](https://github.com/AD-Styles/unsloth-qlora-finetuning) 에서 검증된 기법 재사용. v4 는 같은 NF4 + double quantization + bf16 compute 조합.

---

## 🧪 학습 셋업 (RTX 4060 Laptop, 8GB VRAM)

| 항목 | v3 | **v4** |
|---|---|---|
| LLM | Qwen2.5-0.5B-Instruct | **Qwen2.5-1.5B-Instruct** |
| LLM dtype | fp16 | **4-bit NF4 (bnb) + bf16 compute** |
| Vision encoder | CLIP-ViT-B/32 (frozen) | CLIP-ViT-B/32 (frozen, 변경 없음) |
| LoRA target | q/k/v/o (attention) | **q/k/v/o + gate/up/down (전 linear, QLoRA 정석)** |
| `<image>` 토큰 | 신규 추가 + embedding resize | **Qwen2.5 내장 `<|image_pad|>` 재사용 (resize 없음)** |
| 학습 파라미터 dtype | fp16 | **projector·LoRA fp32 (master weight) / base 4-bit** |
| batch_size | 2 | **1** |
| grad_accum | 4 | **8** (effective batch 동일) |
| gradient checkpointing | OFF | **ON** (activation memory 절감) |
| Korean 데이터 | 4K (KoLLaVA) | **12K (Stage 2 mix 46K 의 26%)** |
| Stage 1 trainable params | projector 1.49M | **projector 3.54M** (LLM hidden 896→1536) |
| Stage 2 trainable params | 3.66M | **22.0M** (projector 3.54M + LoRA 전 linear) |
| Stage 1 / Stage 2 학습 시간 | — | **≈ 3.3h / ≈ 7.4h** (RTX 4060 8GB · Stage 1 40K·5,000 step / Stage 2 46K·5,750 step) |
| LoRA adapter 크기 | 1GB→8MB (slim 작업) | **70MB** (전부 LoRA — embedding 군더더기 0) |

> 8GB 한 장에 1.5B 를 올린 핵심은 4-bit NF4 + gradient checkpointing + batch_size 1. Stage 1·2 모두 OOM 없이 완주했고, 위 수치는 전부 실측값입니다.

---

## 📈 데이터 규모 — v3 가 놓친 lever

v3 의 projector 는 v1 이 **5K 캡션** 으로 학습한 것을 그대로 물려받았습니다 (LLaVA-1.5 는 558K). 비전-언어를 잇는 다리가 약하면 LLM 만 키워도 한계가 분명하므로, v4 는 Stage 1 정렬 데이터를 먼저 키웠습니다.

**Stage 1 확대는 비용이 쌉니다:**
- projector 만 학습 (LLM frozen) → step 이 가볍고 빠름
- 이미지 1장당 캡션이 여러 개 (Flickr30k·COCO 각 5개) → 한 번의 다운로드로 데이터가 N배. 이미지는 1번만 저장하고 manifest 만 캡션 수만큼 늘림 ([`scripts/download_alignment_data.py`](scripts/download_alignment_data.py))

| 단계 | v3 | **v4 (실측)** |
|---|---|---|
| Stage 1 정렬 | 5K (v1 projector 계승) | **40K** — COCO+Flickr 이미지 2만 장 × 캡션 2개. v3 의 **8배** |
| Stage 2 instruction | 13K | **46K** — vqav2 18K + localized_narratives 10K + aokvqa 6K + KoLLaVA 한국어 12K |

> Stage 2 믹스는 짧은 사실(vqav2)·긴 묘사(localized_narratives)·추론(aokvqa)·한국어를 의도적으로 균형 배합했습니다 — 한 능력에 치우치지 않도록. 풀 LLaVA 규모(558K/665K)는 노트북 한 장에서 비현실적이므로, 주어진 예산 안에서 믹스 비율로 균형을 잡는 데 집중했습니다.

---

## 🚦 배포 게이트 — 검증을 통과해야 배포

v4 는 성능 검증을 배포 절차 안에 넣었습니다. v3 는 검증 없이 배포부터 한 뒤 수습했지만, v4 는 학습이 끝난 모델이 표준 benchmark 게이트를 통과해야만 배포합니다.

[`scripts/eval_gate.py`](scripts/eval_gate.py) 가 학습된 **raw 모델** (추론 wrapper 없이) 을 표준 benchmark 로 측정하고, 사전에 정한 bar 를 통과할 때만 배포를 허용합니다 (exit code 0/1 로 배포 스크립트를 게이팅).

| 게이트 기준 | bar | 근거 (사전 결정 기준) |
|---|---|---|
| VQAv2 정답률 | ≥ 0.45 | v3 raw 36.67% 대비 +8%p 이상 |
| POPE 정답률 | ≥ 0.65 | v3 raw 50%(전부 yes=랜덤) 대비 +15%p 이상 |
| POPE yes-F1 | ≥ 0.55 | yes/no 한쪽 쏠림 방지 |
| POPE 예측 쏠림 | ≤ 0.85 | v3 의 "전부 yes" 퇴행 직접 탐지 |
| yes/no 미응답률 | ≤ 0.15 | yes/no 로 못 답한(?) 비율 상한 |

---

## 🚀 Quickstart — 직접 재현

> 아래는 v4 가 **실제로 실행한** 명령입니다. 데이터셋은 HuggingFace streaming 으로
> 받으므로 샘플 구성이 시드/버전에 따라 약간 달라질 수 있습니다.

```powershell
# (1) 환경 셋업
cd "34. vlm-from-scratch-v4"
pip install -r requirements.txt

# (2) VRAM smoke test — 8GB fit 여부 + step time 추정
python scripts/vram_test_v4.py --use-lora

# (3) Stage 1 정렬 데이터 — 40K (이미지 2만 장 × 캡션 2개)
python scripts/download_alignment_data.py --num-samples 40000 --out data/coco_subset

# (4) Stage 2 instruction 데이터 — the_cauldron 3종 + KoLLaVA 한국어 → mix 46K
python scripts/download_instruct_data.py --configs vqav2 --num-samples 18000 --out data/instruct_vqav2
python scripts/download_instruct_data.py --configs localized_narratives --num-samples 10000 --out data/instruct_narr
python scripts/download_instruct_data.py --configs aokvqa --num-samples 6000 --out data/instruct_aok
python scripts/download_korean_data.py --num-samples 12000 --out data/korean_subset
python scripts/mix_manifests.py `
  --inputs data/instruct_vqav2/manifest.json data/instruct_narr/manifest.json data/instruct_aok/manifest.json data/korean_subset/manifest.json `
  --output data/v4_stage2_mix/manifest.json

# (5) 평가셋 prep — POPE train/test 분리 + OOD benchmark 100 케이스
python scripts/prepare_pope_split.py
python scripts/prepare_ood_benchmark.py --indist-source data/coco_subset/manifest.json

# (6) Stage 1 학습 — projector alignment (≈ 3.3h)
python -m src.train `
  --data-path data/coco_subset/manifest.json `
  --output-dir checkpoints/v4_stage1 `
  --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 1e-3 `
  --use-qlora --bf16

# (7) Stage 2 학습 — QLoRA + projector 동시 (≈ 7.4h)
python -m src.train `
  --data-path data/v4_stage2_mix/manifest.json `
  --output-dir checkpoints/v4_stage2_qlora `
  --init-projector checkpoints/v4_stage1/projector.pt `
  --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 2e-4 `
  --use-qlora --use-lora --lora-r 16 --lora-alpha 32 --bf16

# (8) 배포 게이트 — raw 모델이 bar 를 통과해야만 배포 (n=400)
python scripts/eval_gate.py --n-vqav2 400 --n-pope 400 `
  --projector checkpoints/v4_stage2_qlora/projector.pt `
  --lora-adapter checkpoints/v4_stage2_qlora/lora_adapter
# → exit 0 = GO (배포 진행) / exit 1 = NO-GO (배포 차단)

# (9) OOD ROC 분석 — 결합 가중치·임계값 재보정 (homework #3)
python scripts/ood_roc_analysis.py `
  --projector checkpoints/v4_stage2_qlora/projector.pt `
  --lora-adapter checkpoints/v4_stage2_qlora/lora_adapter
```

---

## 📊 결과 (Results)

> 측정 환경: RTX 4060 Laptop 8GB · 모든 평가는 **raw 모델** (추론 wrapper 없음) · greedy decoding.

### 학습 — Stage 1 → Stage 2

| | Stage 1 (projector 정렬) | Stage 2 (QLoRA instruction) |
|---|---|---|
| 데이터 | 40K (이미지 2만 × 캡션 2) | 46K (vqav2 18K + narratives 10K + aokvqa 6K + 한국어 12K) |
| 학습 대상 | projector 3.54M | projector 3.54M + LoRA 18.5M = **22.0M** |
| epochs / steps | 1 ep / 5,000 optim steps | 1 ep / 5,750 optim steps |
| loss | 5.0 → **1.98** | 3.65 → **≈ 1.01** |
| 학습 시간 | ≈ 3.3 시간 | ≈ 7.4 시간 |
| 산출물 | projector.pt (13.5 MB) | projector.pt + lora_adapter (**70 MB**) |

**LoRA adapter 70 MB** — 전부 LoRA 가중치입니다 (전 linear layer, r=16). `<|image_pad|>` 재사용으로 embedding resize 를 하지 않아, v3 처럼 `embed_tokens`(≈1 GB) 가 adapter 에 딸려 들어가 별도 슬림화가 필요했던 문제가 없습니다.

> Stage 2 loss 가 v3(≈0.75)보다 높게 끝난 건 퇴행이 아닙니다 — 균형 믹스(긴 묘사·추론·한국어)가 vqav2 초단답보다 본질적으로 어렵기 때문입니다. loss 의 절대값이 아니라 게이트·데모에서의 실제 능력으로 판단합니다.

### 정량 평가 — 표준 benchmark

[`scripts/eval_gate.py`](scripts/eval_gate.py), VQAv2 val + POPE test. **v4 는 n=400 으로 측정**했습니다. v2/v3 수치는 v3 README 의 소표본 측정값이라 표본 오차가 있지만, 20%p 안팎의 격차는 그 오차를 크게 상회합니다.

| | v2 | v3 raw | v3 + wrapper | **v4 raw** |
|---|---|---|---|---|
| VQAv2 정답률 | 34.67% | 36.67% | 36.67% | **56.83%** |
| POPE 정답률 | 50.00% | 50.00% | 53.33% | **71.75%** |
| POPE yes-F1 | — | — | — | **0.735** |
| POPE 예측 쏠림 | — | 1.00 (전부 yes) | — | 0.57 |

- **VQAv2 +20.2%p** (36.67 → 56.83). 유형별 — yes/no 76.0%, other 45.9%, number 42.9%.
- **POPE +21.8%p** (50.00 → 71.75). v3 raw 는 *전부 "yes"* 로 사실상 랜덤 (skew 1.00) 이었으나, v4 raw 는 yes 227 / no 173 으로 양쪽을 실제 판별 (skew 0.57, yes-F1 0.74).

**"v4 + wrapper" 열이 없는 이유 (homework #4).** v3 는 raw 모델이 약해 추론 wrapper(ensemble router 등)로 점수를 끌어올렸습니다. v4 는 raw 모델 자체가 게이트 5/5 를 통과했으므로 wrapper 를 만들지 않았습니다 — raw 점수의 도약 자체가 router 의존도 해소의 증거입니다.

### OOD ROC 분석 + 가중치 최적화 (homework #3 의 답)

v3 의 OOD 검출기는 in-dist 1 + OOD 1 = **N=2** 로만 검증돼 결합 가중치도 임계값도 미보정 상태였습니다. v4 는 100 케이스 벤치마크 (in-dist COCO 40 + 만화 30 + 추상화 30) 로 ROC 를 측정하고, **CLIP·entropy 결합 가중치와 판정 임계값을 함께 재보정**했습니다 (`scripts/ood_roc_analysis.py` → `eval_results/v4_ood_roc.json`).

| 지표 | v3 기본 설정 | **v4 재보정** |
|---|---|---|
| 결합 가중치 w_clip | 0.6 (미보정 추측) | **0.0** (ROC 최적 — entropy 단독) |
| 결합 ROC AUC | 0.884 † | **0.971** |
| 5-fold 교차검증 AUC | — | **0.969** |
| 판정 정확도 · TPR | — | **91% · 90%** (임계값 0.4582, FPR 7.5%) |

† v3 의 w_clip 0.6 을 v4 모델·벤치마크에 적용했을 때의 결합 AUC.

- **신호 단독 AUC** — entropy-only **0.971**, CLIP-only 0.660. LLM 첫 토큰 entropy 가 압도적으로 강한 신호입니다 — CLIP-ViT-B/32 는 만화 얼굴조차 "a person" 으로 매칭해 분포 밖임을 못 잡습니다. w_clip 을 0→1 로 스윕하면 AUC 가 0.971(entropy 단독)에서 0.660(CLIP 단독)으로 단조 감소했고, ROC 최적값은 entropy 단독이었습니다.
- **가중치가 모델 성능을 따라 움직였습니다** — v3 의 미보정 추측 0.6 → v4 의 ROC 보정 **0.0**. 모델이 좋아질수록 자기 불확실성(entropy)이 OOD 의 충분 신호가 됐다는 뜻입니다.
- **과적합 검증** — 가중치·임계값은 이 100-케이스 셋으로 *보정*한 값이라 낙관 편향 우려가 있어 **5-fold 교차검증** 으로 확인했습니다. 튜닝에 안 쓴 fold 들의 평균 AUC 0.969 로 보정셋 0.971 과 거의 같고, fold 별 최적 w_clip 이 모두 0.0 으로 일관 — 노이즈가 아닌 구조적 최적값입니다. 카테고리별 AUC 는 추상화 0.978 / 만화 0.965 (만화는 애니 얼굴이 사람 사진과 가까워 더 어려움).

### 배포 게이트 판정 — ✅ GO

[`scripts/eval_gate.py`](scripts/eval_gate.py) 5개 기준 **5/5 통과** (v4 raw 모델, n=400).

| 기준 | bar | 측정 (n=400) | 판정 |
|---|---|---|---|
| VQAv2 정답률 | ≥ 0.45 | 0.5683 | ✅ |
| POPE 정답률 | ≥ 0.65 | 0.7175 | ✅ |
| POPE yes-F1 | ≥ 0.55 | 0.7354 | ✅ |
| POPE 예측 쏠림 | ≤ 0.85 | 0.5675 | ✅ |
| yes/no 미응답률 | ≤ 0.15 | 0.0000 | ✅ |

5개 기준 전부 통과 → 게이트 GO → HF Spaces 배포 진행.

### 라이브 데모 검증 — 배포된 Space 직접 점검

벤치마크 점수와 별개로, 배포된 데모가 실제로 어떻게 답하는지 [HF Space](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) 에 영어/한국어 · 사실/묘사/yes-no · in-dist/OOD 10 케이스를 직접 입력해 점검했습니다 (`scripts/smoke_space_demo.py` → `eval_results/v4_space_smoke.json`). 데모는 샘플링(T=0.7) 이라 응답이 실행마다 달라집니다 — 아래는 한 번의 대표 실행입니다.

**잘 되는 것** — 장면 분류 ("무슨 방?" → "Kitchen" ✓), yes/no ("헤드폰 쓰고 있나?" → "Yes" ✓), 객체·행동 ("소년이 뭐 하나?" → "Typing" ✓) 는 반복 실행해도 일관됩니다. 묘사 요청에는 *"In this picture we can see the kitchen. In front of it there is a stove, in which there are pots and pans ..."* 처럼 여러 문장의 구조적 서술을 내놓고, 한국어 질문에는 유창한 한국어 문장으로 답합니다.

**안 되는 것 (정직하게)**:
- **세밀한 구분 실패** — 고양이를 안은 여성 사진에 "What animal?" → **"Dog"**. 고양이↔개 수준의 fine-grained 구분이 약합니다 (데이터 믹스가 아니라 CLIP-ViT-B/32 + 1.5B 용량의 한계).
- **장문 묘사 = 환각** — 형식은 유창하나 없는 디테일을 지어냅니다. *고양이 안은 여성* 사진을 *"한 남성이 주방에서 …"* 로 서술하는 식으로 성별·객체·배경이 틀립니다. 짧은 답에 강하고 길어질수록 환각이 느는, 1.5B 규모의 분명한 천장입니다.
- **추상화** — 추상 회화에 "What do you see?" → **"Birds"**. raw 모델은 OOD 입력을 거르지 않고 자신있게 답합니다 (배포 Space 는 raw 모델만 서빙) — OOD 검출기(위 절)가 이런 입력을 가려내야 하는 이유입니다.

요약하면 v4 는 **장면·yes/no·객체·행동 같은 큰 단위의 VQA 와 묘사 형식 생성에 강하고, 세밀한 구분과 장문의 사실 정확도에 약한** 모델입니다 — 1.5B + 8GB 한 장이라는 제약 안에서 v3 를 분명히 넘어섰습니다.

### 한국어

Stage 2 mix 46K 의 26% (12K) 가 KoLLaVA 한국어 — v3 의 4K 대비 **3배**. 라이브 데모 검증에서 한국어 질문에는 **문법적으로 유창한 한국어 서술**로, 영어 질문에는 영어로 답해 **언어 정렬이 유지**되는 것을 확인했습니다. 한국어 *정량* 평가셋은 도입하지 못했는데, VQAv2·POPE 급의 신뢰할 만한 한국어 VLM 벤치마크가 없어서입니다 — 무리하게 만든 숫자보다 정직한 공백으로 남겼습니다 (homework #5 의 절반, 아래 한계 절).

---

## 🏁 시리즈를 마치며 — v4 가 닫은 것, 남긴 것

v4 는 Mini-LLaVA 시리즈 (v1 → v4) 의 마지막 버전입니다. v3 가 남긴 6개 숙제 중 다섯 개를 닫고, 한 개(한국어 정량 평가셋)는 절반까지 진행했습니다.

**v4 가 닫은 것**
- **raw 성능** — VQAv2 36.67→56.83 (+20.2%p), POPE 50.00→71.75 (+21.8%p), 배포 게이트 5개 기준 5/5 GO.
- **1.5B on 8GB** — Qwen2.5-1.5B 를 QLoRA 4-bit 로 노트북 GPU 한 장에 올려 Stage 1·2 를 OOM 없이 완주.
- **wrapper 제거 (homework #4)** — raw 모델이 게이트를 통과해 v3 식 router wrapper 가 불필요해졌습니다.
- **OOD 재보정 (homework #3)** — 100 케이스 ROC 로 entropy 단독 AUC 0.971, 5-fold CV 0.969, 판정 정확도 91%.
- **한국어 3배 (homework #5 의 절반)** — 학습 데이터 4K→12K, 라이브 데모에서 유창한 한국어 응답 확인.

**해결하지 못한 한계**
- **세밀한 구분 · 장문 생성** — 라이브 데모 검증에서 드러난 대로, 거친 VQA 는 준수하나 고양이↔개 같은 fine-grained 구분과 긴 묘사형 생성(환각)에서 약합니다. 1.5B + 약 9만 샘플 학습 규모의 천장으로, 모델을 키우거나(8GB 메모리 한계) 데이터를 수십만 규모로 늘리지 않는 한 남는 한계입니다.
- **한국어 정량 평가셋** — homework #5 의 미완 절반. 학습 데이터는 4K→12K 로 늘렸으나, 신뢰할 만한 한국어 VLM 표준 벤치마크 부재로 객관적 정량 평가셋을 끝내 만들지 못했습니다 (한국어 능력은 라이브 데모 정성 평가로 갈음).
- **배포 정밀도 갭** — LoRA 는 4-bit base 에서 학습됐으나 무료 CPU 데모는 fp32 base 에 얹어 추론합니다 (QLoRA 의 표준 배포 절충). `scripts/diag_deploy_gap.py` 로 4-bit/fp32 답변을 직접 비교한 결과 차이는 거의 없었지만 — 원리상 미세한 분포 차이는 존재하며, GPU Space 였다면 4-bit 그대로 서빙해 해소됐을 항목입니다.

**의도적으로 다루지 않은 확장 방향** (이 시리즈의 범위 밖)
- **ViT-L/14 비전 인코더** — v3 가 0.5B 한계로 효과 없다고 결론냈고, v4 는 "LLM 크기" 단일 변수 비교를 위해 ViT-B/32 를 유지. 1.5B 에서 ViT-L/14 가 다시 효과 있는지는 검증하지 않았습니다.
- **3B/7B LLM · vLLM/Triton 서빙** — RTX 4060 8GB 로는 QLoRA 로도 3B 가 한계이고, 프로덕션 서빙 최적화는 [nlp-triton-deployment](https://github.com/AD-Styles/nlp-triton-deployment) 등 별도 포트폴리오의 주제입니다.

---

## 🔗 참고

- v2/v3 README — Mini-LLaVA 시리즈 전체 narrative
- Dettmers et al., **"QLoRA: Efficient Finetuning of Quantized LLMs"** (NeurIPS 2023) — [arxiv:2305.14314](https://arxiv.org/abs/2305.14314)
- portfolio 10 [unsloth-qlora-finetuning](https://github.com/AD-Styles/unsloth-qlora-finetuning) — 같은 QLoRA 기법으로 Llama-3 8B 학습한 선행 작업
