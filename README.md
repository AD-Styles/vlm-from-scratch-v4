# Mini-LLaVA v4 — LLM 1.5B 업그레이드로 v3 의 진짜 병목 해결

> ✅ **학습·평가 완료 — 배포 게이트 GO.** v3 는 병목을 *"0.5B LLM"* 으로 지목했지만, 재검토 결과 절반은 **학습 데이터** 였습니다 (Stage 1 정렬 5K — LLaVA 의 ~1%). v4 는 **LLM 1.5B 확대 + 정렬 데이터 8배 확대 (5K→40K) + Stage 2 instruction 균형 믹스** 를 함께 다루고, **배포 전 평가 게이트** 로 검증했습니다. 결과 (RTX 4060 Laptop 8GB 한 장, 모두 wrapper 없는 raw 모델): **VQAv2 36.67%→56.83%, POPE 50.00%→71.75%** — 게이트 5/5 통과. 상세는 아래 **📊 결과** 절 참고. v4 는 Mini-LLaVA 시리즈(v1→v4)의 **마지막 버전**입니다.

[← v3: vlm-from-scratch-v3](https://github.com/AD-Styles/vlm-from-scratch-v3) | [← v2: vlm-from-scratch](https://github.com/AD-Styles/vlm-from-scratch)

---

## 🎯 v3 가 v4 에 남긴 6가지 숙제 (v3 회고록 인용)

v3 의 [⚠️ 한계와 v4 계획](https://github.com/AD-Styles/vlm-from-scratch-v3#%EF%B8%8F-한계와-v4-계획-limitations--next-steps) 표 6개 항목을 그대로 가져왔습니다. v4 는 이 6개를 다른 자리에서 새로 짜는 게 아니라 v3 가 글로 약속한 것을 그대로 수행합니다.

| # | v3 의 한계 | v3 의 진단 | v4 의 대응 |
|---|---|---|---|
| 1 | ⭐ **0.5B LLM 의 시각적 추론** | 시각 detail 약함 (cartoon case 등) | **Qwen2.5-1.5B + QLoRA 4-bit** — LLM 키우기 |
| 2 | POPE threshold 가 test set 으로 튜닝됨 | 70% 일반화 보장 X (untuned 53%) | POPE train/test 분리 후 untuned 재측정 |
| 3 | OOD 검증 N=2 | 임계값 0.5 일반화 부족 | 만화 + 추상화 + in-distribution 케이스로 ROC AUC 재보정 (ImageNet-O 의존 제거 — HF 공식 미러 없음) |
| 4 | ⭐ wrapper 11/12 중 8/12 가 router 기여 | "VLM 능력" 보다 "ensemble routing" | LLM 1.5B 가 #1 과 연동되어 자동 해소 검증 |
| 5 | 한국어 데이터 4K 만 | 환각 잔존 + 한국어 평가셋 없음 | KoLLaVA 를 노트북 한 장에서 끝나는 subset 으로 확장 + 한국어 평가셋 도입 |
| 6 | ⭐ **학습 데이터 규모** | Stage 1 정렬 5K·Stage 2 13K = LLaVA 의 1~2%. projector 가 5K 로만 학습돼 비전-언어 정렬 자체가 약함 | Stage 1 정렬 데이터 **대폭 확대** (projector-only 라 학습 비용 저렴) + **배포 전 평가 게이트** 도입 |

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
  full finetuning performance"* 라고 명시. v3 의 attention-only 가 wrapper router
  의존도(11/12 중 8/12)를 키운 원인 중 하나 → v4 는 raw 모델 자체를 강화.
- **`<image>` 신규 토큰 대신 Qwen2.5 내장 `<|image_pad|>` 재사용** — `resize_token_embeddings`
  를 호출하지 않음 → v3 의 "embedding resize → PEFT 가 embed_tokens 자동 저장 →
  adapter 1GB" (v3 Step 4·5) 문제를 원천 차단. splice 는 token id 로 위치만 찾으므로
  토큰 임베딩 품질과 무관.

### 왜 4-bit QLoRA 가 필수인가

LLM 가중치만 따진 산술 메모리 (1.5B params × dtype 크기):

| 모델 dtype | 가중치 메모리 (산술) |
|---|---|
| fp32 (4 bytes) | 1.5B × 4 ≈ 6 GB |
| bf16 (2 bytes) | 1.5B × 2 ≈ 3 GB |
| **4-bit NF4 (0.5 byte)** | **1.5B × 0.5 ≈ 0.9 GB** |

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

> 8GB 한 장에 1.5B 를 fit 시킨 핵심: 4-bit NF4 + gradient checkpointing + batch_size 1.
> 학습은 OOM 없이 완주했습니다. 위 수치는 모두 실측 — 평가 상세는 아래 **📊 결과** 절.

---

## 📈 데이터 규모 — v3 가 놓친 lever

v3 회고록은 병목을 "0.5B LLM" 으로만 지목했습니다. 하지만 절반은 **데이터** 였습니다 — Stage 1 비전-언어 정렬을 **5K 캡션** 으로만 학습했고 (LLaVA-1.5 는 558K), v3 의 projector 는 v1 의 그 5K-학습 projector 를 그대로 계승했습니다. 다리(projector)가 약하면 LLM 만 키워도 한계가 명확합니다.

**Stage 1 확대는 비용이 싸다** — 이게 핵심입니다:
- projector 만 학습 (LLM frozen) → step 이 가볍고 빠름
- 이미지 1장당 캡션이 여러 개 (Flickr30k·COCO 각 5개) → 한 번의 다운로드로 데이터가 N배. 이미지는 1번만 저장하고 manifest 만 캡션 수만큼 늘림 ([`scripts/download_alignment_data.py`](scripts/download_alignment_data.py))

| 단계 | v3 | **v4 (실측)** |
|---|---|---|
| Stage 1 정렬 | 5K (v1 projector 계승) | **40K** — COCO+Flickr 이미지 2만 장 × 캡션 2개. v3 의 **8배** |
| Stage 2 instruction | 13K | **46K** — vqav2 18K + localized_narratives 10K + aokvqa 6K + KoLLaVA 한국어 12K |

> Stage 2 믹스는 짧은 사실(vqav2)·긴 묘사(localized_narratives)·추론(aokvqa)·한국어를 의도적으로 균형 배합한 것입니다 — 한 능력에 치우치지 않도록. 풀 LLaVA 규모(558K/665K)는 노트북 한 장에서 비현실적입니다 — v4 의 현실 목표는 *"v3 보다 확실히 낫고 1.5B 급에선 준수"* 이지 GPT-4V 가 아닙니다.

---

## 🚦 배포 게이트 — v3 실수 반복 방지

v3 의 실수: 성능 검증 없이 HF Spaces 에 먼저 배포 → 라이브에서 성능 미달을 발견하고 수습. v4 는 이 패턴을 *절차* 로 차단합니다.

[`scripts/eval_gate.py`](scripts/eval_gate.py) 가 학습된 **raw 모델** (추론 wrapper 없이) 을 표준 benchmark 로 측정하고, 사전에 정한 bar 를 통과할 때만 배포를 허용합니다 (exit code 0/1 로 배포 스크립트를 게이팅).

| 게이트 기준 | bar | 근거 (결정 임계값 — 결과 예측 아님) |
|---|---|---|
| VQAv2 정답률 | ≥ 0.45 | v3 raw 36.67% 대비 +8%p 이상 |
| POPE 정답률 | ≥ 0.65 | v3 raw 50%(전부 yes=랜덤) 대비 +15%p 이상 |
| POPE yes-F1 | ≥ 0.55 | yes/no 한쪽 쏠림 방지 |
| POPE 예측 쏠림 | ≤ 0.85 | v3 의 "전부 yes" 퇴행 직접 탐지 |
| yes/no 미응답률 | ≤ 0.15 | yes/no 로 못 답한(?) 비율 상한 |

**미달 시 → 배포 차단.** v3 처럼 "한계 분석" 포트폴리오로 정직하게 프레이밍하는 게 맞습니다. raw 모델은 wrapper 없이 평가합니다 — v3 에서 wrapper 가 raw 의 약함을 가렸기 때문입니다.

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

# (4) Stage 2 instruction 데이터 — the_cauldron 3종(불균등) + KoLLaVA 한국어 → mix 46K
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
# → exit 0 (GO): 배포 진행 / exit 1 (NO-GO): 한계 분석으로 프레이밍

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

**LoRA adapter 70 MB 의 의미** — v3 는 `resize_token_embeddings` 때문에 PEFT 가 `embed_tokens`(≈1 GB) 를 adapter 에 자동 포함해, slim 작업으로 8 MB 까지 깎아내야 했습니다 (v3 Step 4·5). v4 는 Qwen2.5 내장 `<|image_pad|>` 를 재사용해 resize 자체를 없앴습니다 → 70 MB 는 **전부 LoRA 가중치** (전 linear layer, r=16), embedding 군더더기 0. "1 GB 중 99% 가 사고로 들어간 embedding" 이던 v3 와 달리, v4 의 70 MB 는 전부 실제 학습된 어댑터입니다.

> Stage 2 loss 가 v3(≈0.75)보다 높게 끝난 건 퇴행이 아닙니다 — 균형 믹스(긴 묘사·추론·한국어)가 vqav2 초단답보다 본질적으로 어렵기 때문입니다. loss 의 절대값이 아니라 게이트·데모에서의 실제 능력으로 판단합니다.

### 정량 평가 — 표준 benchmark

[`scripts/eval_gate.py`](scripts/eval_gate.py), VQAv2 val + POPE test. **v4 는 n=400** (v2/v3 는 v3 README 의 n=50 측정값 — 20%p 격차는 표본 오차를 크게 상회).

| | v2 | v3 raw | v3 + wrapper | **v4 raw** |
|---|---|---|---|---|
| VQAv2 정답률 | 34.67% | 36.67% | 36.67% | **56.83%** |
| POPE 정답률 | 50.00% | 50.00% | 53.33% | **71.75%** |
| POPE yes-F1 | — | — | — | **0.735** |
| POPE 예측 쏠림 | — | 1.00 (전부 yes) | — | 0.57 |

- **VQAv2 +20.2%p** (36.67 → 56.83). 유형별 — yes/no 76.0%, other 45.9%, number 42.9%.
- **POPE +21.8%p** (50.00 → 71.75). v3 raw 는 *전부 "yes"* 로 사실상 랜덤 (skew 1.00) 이었으나, v4 raw 는 yes 227 / no 173 으로 양쪽을 실제 판별 (skew 0.57, yes-F1 0.74).

**"v4 + wrapper" 열이 없는 이유 (homework #4 의 답).** v3 의 12-케이스 정성 테스트는 raw 1/12 → wrapper 11/12 였고, 그중 8/12 는 "VLM 능력" 이 아니라 wrapper 의 **ensemble router** 기여였습니다 — wrapper 가 raw 의 약함을 가린 분장이었던 셈입니다. v4 의 목표는 분장이 아니라 raw 모델 자체를 배포 가능 수준으로 끌어올리는 것. raw 가 게이트 5/5 를 통과했으므로 v4 는 **의도적으로 wrapper 를 만들지 않았습니다.** "raw 36.67→56.83, raw 50→71.75" 자체가 router 의존도 해소의 증거입니다.

### OOD ROC 분석 + 가중치 최적화 (homework #3 의 답)

v3 의 OOD 검출기는 in-dist 1 + OOD 1 = **N=2** 로만 검증돼 결합 가중치도 임계값도 미보정 상태였습니다. v4 는 100 케이스 벤치마크 (in-dist COCO 40 + 만화 30 + 추상화 30) 로 ROC 를 측정하고, **CLIP·entropy 결합 가중치와 판정 임계값을 함께 재보정**했습니다 (`scripts/ood_roc_analysis.py` → `eval_results/v4_ood_roc.json`).

| 지표 | v3 기본 설정 | **v4 재보정** |
|---|---|---|
| 결합 가중치 w_clip | 0.6 (미보정 추측) | **0.0** (ROC 최적 — entropy 단독) |
| 결합 ROC AUC | 0.884 † | **0.971** |
| 정직한 일반화 (5-fold CV) | — | **0.969** |
| 판정 정확도 · TPR | — | **91% · 90%** (임계값 0.4582, FPR 7.5%) |

† v3 의 w_clip 0.6 을 v4 모델·벤치마크에 적용했을 때의 결합 AUC.

- **신호 단독 AUC** — CLIP-only 0.660, entropy-only **0.971**. LLM 첫 토큰 entropy 가 압도적으로 강한 신호입니다 (CLIP-ViT-B/32 는 만화 얼굴조차 "a person" 으로 무난히 매칭해 분포 밖임을 못 잡음). w_clip 을 0→1 로 스윕하면 AUC 가 **0.0(entropy 단독) 0.971 에서 1.0(CLIP 단독) 0.660 까지 단조 감소** — v4 모델은 자기 entropy 만으로 OOD 를 거의 가르고, CLIP 을 섞을수록 신호가 흐려집니다.
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

게이트 GO → HF Spaces 배포 진행. (NO-GO 였다면 v3 처럼 "한계 분석" 포트폴리오로 정직하게 프레이밍할 계획이었습니다.)

### 라이브 데모 검증 — 배포된 Space 직접 점검

게이트(VQAv2/POPE)는 표준 benchmark 점수일 뿐, *배포된 데모가 실제로 어떻게 답하는지* 는 별개입니다. 배포 후 [HF Space](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) 에 영어/한국어 · 사실/묘사/yes-no · in-dist/OOD 를 섞은 10 케이스를 직접 입력했습니다 (`scripts/smoke_space_demo.py` → `eval_results/v4_space_smoke.json`). 데모는 `do_sample=True` (T=0.7) 라 응답이 실행마다 달라집니다 — 아래 인용은 한 번의 대표 실행이며, 짧은 사실 답변은 실행 간 안정적이나 장문 묘사는 변동이 큽니다.

**잘 되는 것** — 장면 분류 ("무슨 방?" → "Kitchen" ✓), yes/no ("헤드폰 쓰고 있나?" → "Yes" ✓), 객체·행동 ("소년이 뭐 하나?" → "Typing" ✓) 는 반복 실행해도 일관됩니다. 결정적으로 — **묘사 요청이 고쳐졌습니다**: "Describe this image" 에 *"In this picture we can see the kitchen. In front of it there is a stove, in which there are pots and pans. On the right side we can see the cupboard ..."* 처럼 **여러 문장의 구조적 영어 서술**을 내놓고, 한국어 질문에는 **문법적으로 유창한 한국어 문장**으로 답합니다.

**안 되는 것 (정직하게)**:
- **세밀한 구분 실패** — 고양이를 안은 여성 사진에 "What animal?" → **"Dog"**. 고양이↔개 수준의 fine-grained 구분이 약합니다 (데이터 믹스가 아니라 CLIP-ViT-B/32 + 1.5B 용량의 한계).
- **장문 묘사 = 환각** — 형식은 유창하나 없는 디테일을 지어냅니다. *고양이를 안은 여성* 사진에 한국어 묘사를 시키면 *"한 남성이 주방에서 포장지나 식기를 들고 있는 …"* — 성별·객체·배경이 전부 틀립니다. 짧은 사실 질문엔 강하고, 생성이 길어질수록 환각이 늘어나는 1.5B + 8만 샘플 학습 규모의 분명한 천장입니다.
- **추상화** — 추상 회화에 "What do you see?" → **"Birds"**. raw 모델은 OOD 입력을 거르지 않고 자신있게 답합니다 (배포 Space 는 raw 모델만 서빙) — OOD 검출기(위 절)가 이런 입력을 가려내야 하는 이유입니다.

요약하면 v4 는 **거친 입도의 VQA(장면·yes/no·객체·행동)와 묘사 형식 생성에서 준수하고, 세밀한 구분과 장문의 사실 정확도에서 약한** 모델입니다 — 1.5B + 8GB 노트북 한 장이라는 제약 안에서 v3 를 분명히 넘어선 결과이자, GPT-4V 는 아니라는 점도 데모가 정직하게 보여줍니다.

### 한국어

Stage 2 mix 46K 의 26% (12K) 가 KoLLaVA 한국어 — v3 의 4K 대비 3배. 위 라이브 데모 검증에서 한국어 질문에 문법적으로 유창한 한국어 서술로 답하는 것을 확인했습니다. 다만 **한국어 정량 평가셋은 끝내 만들지 못했습니다**: VQAv2·POPE 급의 신뢰할 만한 한국어 VLM 벤치마크가 없어, 자체 큐레이션으로는 객관적인 숫자를 담보하기 어려웠습니다. 무리하게 만든 숫자보다 정직한 공백이 낫다고 판단 — homework #5 의 "평가셋 도입" 절반은 미완으로 기록합니다 (아래 한계 절).

---

## 💡 회고 — v3 의 6가지 숙제, v4 의 결산

### Step 0 — v3 가 진단했지만 v3 가 못 푼 것

v3 의 표는 여섯 가지를 말해뒀습니다. 정작 v3 가 푼 건 (2)·(5) 의 일부까지였고, 1.5B 키우기 (1) · OOD 케이스 확장 (3) · router 의존도 자동 해소 (4) · KoLLaVA subset 확장 (5) · **학습 데이터 규모 확대 (6)** 는 v4 의 출발선에 그대로 남았습니다. 특히 (6) — Stage 1 정렬 5K — 은 v3 가 "LLM 크기" 에 가려 표에서 명시하지 못했던 항목으로, v4 검토 과정에서 추가했습니다.

### Step 1 — QLoRA 4-bit 로 8GB 에 1.5B fit

4-bit NF4 + double quantization + gradient checkpointing + `batch_size 1` / `grad_accum 8` (effective batch 8) 조합으로 1.5B 를 8GB 한 장에 올렸습니다. 가중치만 보면 4-bit 는 0.9 GB 지만 activation·optimizer state 가 더 쌓입니다 — checkpointing 으로 activation 메모리를 절반으로 깎고, 학습 대상인 projector·LoRA 만 fp32 master weight 로 둬 업데이트 정밀도를 지켰습니다 (4-bit base 는 frozen). Stage 1·2 모두 OOM 없이 완주했습니다.

### Step 2 — POPE train/test 분리: 막상 raw 모델엔 튜닝할 게 없었다

v3 의 잘못은 wrapper 의 yes/no 결정 규칙을 POPE *test set* 으로 튜닝한 데이터 누수였습니다. v4 는 `prepare_pope_split.py` 로 480 케이스를 train 240 / test 240 으로 갈랐습니다. 그런데 v4 raw 모델은 yes/no 를 **직접 생성** — 튜닝할 임계값이 애초에 없습니다. 게이트의 POPE 71.75% 가 그대로 untuned 측정값이고, v3 식 누수는 v4 에서 구조적으로 불가능합니다. 분리해 둔 split 은 향후 wrapper 가 생길 때를 위한 안전장치로 남겼습니다 — v4 는 wrapper 를 안 만들었으므로 결국 미사용.

### Step 3 — OOD: 100 케이스 ROC 로 가중치·임계값 재보정

N=2 → 100 케이스. v3 의 OOD 검출기는 임계값(0.5)과 결합 가중치(CLIP 0.6) 둘 다 미보정이었습니다. v4 는 100 케이스 벤치마크(in-dist 40 + 만화 30 + 추상화 30) ROC 로 w_clip 을 0→1 스윕해 **entropy 단독(w_clip 0.0)이 정점** 임을 확인했습니다 — entropy-only AUC 0.971 vs CLIP-only 0.660, 결합할수록 단조 감소. w_clip 0.0 + 임계값 0.4582 로 재보정 → 정확도 91%, 5-fold CV(AUC 0.969)로 과적합이 아님을 확인했습니다. 최적 가중치가 모델 성능을 따라 움직인 점이 흥미롭습니다 — v3 의 미보정 추측 0.6 에서 v4 의 ROC 보정 0.0 으로. 모델이 좋아질수록 자기 entropy 가 OOD 판정의 충분 신호가 됩니다.

### Step 4 — wrapper 의존도: raw 가 게이트를 통과해 wrapper 가 사라졌다

v3 는 raw 1/12 를 wrapper 로 11/12 까지 끌어올렸고 그중 8/12 가 router 기여였습니다. v4 는 raw 자체를 VQAv2 36.67→56.83, POPE 50→71.75 로 올려 게이트 5/5 를 통과 → **wrapper 를 만들 이유가 없어졌습니다.** homework #4 가 예고한 "1.5B 가 #1 과 연동되어 router 의존도를 자동 해소" 가 그대로 실현된 셈입니다.

### Step 5 — 한국어: 데이터는 3배 늘렸지만 평가셋은 못 만들었다

KoLLaVA 한국어를 4K → 12K 로 확대했고, 라이브 데모에서 한국어 질문에 유창한 한국어로 답하는 것을 확인했습니다. 하지만 homework #5 의 나머지 절반 — "한국어 평가셋 도입" — 은 끝내 못 했습니다. VQAv2·POPE 급의 신뢰할 만한 한국어 VLM 벤치마크가 없어, 자체 큐레이션으로는 객관적인 숫자를 담보하기 어려웠습니다. v4 가 시리즈의 마지막이라 "v5 로 이월" 도 불가능 — 무리하게 만든 숫자보다 **정직한 공백**으로 남기는 쪽을 택했습니다.

### Step 6 — 학습 데이터 규모: v3 가 놓친 절반

raw 성능 도약의 절반이 LLM 1.5B 확대였다면, 나머지 절반은 **학습 데이터 규모** 입니다. v4 는 Stage 1 정렬을 **5K → 40K (8배)** 로 키웠습니다 — projector 만 학습하므로 step 이 가볍고, 이미지 1장당 캡션이 여러 개라 다운로드 한 번에 데이터가 N배가 되는 *값싼 lever* 입니다. Stage 2 instruction 은 vqav2(짧은 사실) 18K + localized_narratives(긴 묘사) 10K + aokvqa(추론) 6K + 한국어 12K = **46K 균형 믹스** 로, 짧은 단답과 긴 묘사·추론·한국어를 한 모델이 고르게 다루도록 의도적으로 배합했습니다. LLM 1.5B 확대와 이 데이터 축이 함께 raw 모델의 VQAv2 36.67→56.83 · POPE 50→71.75 도약을 만들었습니다.

> 정량 게이트(VQAv2/POPE)는 *측정하는 능력* 만 본증합니다 — 둘 다 짧은-답 benchmark 라 묘사·장문 능력은 게이트가 보지 못합니다. v4 가 게이트 통과 뒤 라이브 데모 정성 점검(위 절)을 한 단계 더 두는 이유입니다: 게이트와 데모는 **상호 보완** 이지 어느 하나로 갈음되지 않습니다.

---

## ⚠️ 시리즈를 마치며 — 남은 한계

v4 는 Mini-LLaVA 시리즈 (v1 → v4) 의 **마지막 버전**입니다 — v5 는 만들지 않습니다. v3 가 남긴 6개 숙제 중 OOD 가중치처럼 닫을 수 있는 항목은 이번에 끝냈고 (v5 로 미루지 않음), 끝내 닫지 못한 한계와 의도적으로 다루지 않은 확장 방향을 정직하게 남깁니다.

**해결하지 못한 한계**
- **세밀한 구분 · 장문 생성** — 라이브 데모 검증에서 드러난 대로, 거친 VQA 는 준수하나 고양이↔개 같은 fine-grained 구분과 긴 묘사형 생성(환각)에서 약합니다. 1.5B + 8만 샘플 학습 규모의 천장으로, 모델을 키우거나(8GB 메모리 한계) 데이터를 수십만 규모로 늘리지 않는 한 남는 한계입니다.
- **한국어 정량 평가셋** — homework #5 의 미완 절반. 학습 데이터는 4K→12K 로 늘렸으나, 신뢰할 만한 한국어 VLM 표준 벤치마크 부재로 객관적 정량 평가셋을 끝내 만들지 못했습니다 (한국어 능력은 라이브 데모 정성 평가로 갈음).
- **배포 정밀도 갭** — LoRA 는 4-bit base 에서 학습됐으나 무료 CPU 데모는 fp32 base 에 얹어 추론합니다 (QLoRA 의 표준 배포 절충). `diag_deploy_gap.py` 로 4-bit/fp32 답변을 직접 비교한 결과 차이는 거의 없었지만 — 원리상 미세한 분포 차이는 존재하며, GPU Space 였다면 4-bit 그대로 서빙해 해소됐을 항목입니다.

**의도적으로 다루지 않은 확장 방향** (이 시리즈의 범위 밖)
- **ViT-L/14 비전 인코더** — v3 가 0.5B 한계로 효과 없다고 결론냈고, v4 는 "LLM 크기" 단일 변수 비교를 위해 ViT-B/32 를 유지. 1.5B 에서 ViT-L/14 가 다시 효과 있는지는 검증하지 않았습니다.
- **3B/7B LLM · vLLM/Triton 서빙** — RTX 4060 8GB 로는 QLoRA 로도 3B 가 한계이고, 프로덕션 서빙 최적화는 [nlp-triton-deployment](https://github.com/AD-Styles/nlp-triton-deployment) 등 별도 포트폴리오의 주제입니다.

---

## 🔗 참고

- v2/v3 README — Mini-LLaVA 시리즈 전체 narrative
- Dettmers et al., **"QLoRA: Efficient Finetuning of Quantized LLMs"** (NeurIPS 2023) — [arxiv:2305.14314](https://arxiv.org/abs/2305.14314)
- portfolio 10 [unsloth-qlora-finetuning](https://github.com/AD-Styles/unsloth-qlora-finetuning) — 같은 QLoRA 기법으로 Llama-3 8B 학습한 선행 작업
