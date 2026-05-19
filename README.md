# Mini-LLaVA v4 — 8GB 노트북 GPU 한 장에서 조립·학습한 비전-언어 모델

> v1 → v4 로 이어진 from-scratch 비전-언어 모델(VLM) 시리즈의 **마지막 버전**. CLIP 비전 인코더와 Qwen2.5-1.5B 언어 모델을 `LlavaForConditionalGeneration` 같은 통합 클래스 없이 직접 이어 붙였습니다. [v3](https://github.com/AD-Styles/vlm-from-scratch-v3) 회고록이 진짜 병목으로 지목한 **0.5B LLM 을 1.5B 로 키우고**, 8GB VRAM 한계는 **QLoRA 4-bit** 로, 배포는 **사전 게이트** 로, 분포 밖 입력 환각은 **OOD abstention 배너** 로 닫았습니다 — 학습 / 평가 / 배포를 각각의 게이트로 검증한 마무리 버전. SOTA 가 아니라 *소비자용 GPU 한 장* 이라는 제약 안에서 LLaVA-1.5 구조가 어디까지 되는지 확인하는 것이 목표입니다.

---

## 🏗️ 모델 구조 (Architecture)

기본 골격은 v1~v3 와 같은 LLaVA-1.5 mini 입니다. 이미지를 CLIP 으로 인코딩하고, projector 로 LLM 임베딩 공간에 맞춘 뒤, 텍스트 시퀀스의 이미지 토큰 자리에 patch 임베딩을 끼워 넣어 Qwen2.5 가 함께 처리합니다.

```
   이미지 (224×224)              텍스트 + <|image_pad|> 자리표시
        │                                  │
        ▼                                  ▼
   CLIP-ViT-B/32 (가중치 고정)    Tokenizer + Embed
        │ [49, 768]                        │
        ▼                                  │
   ★ 2-layer MLP Projector                │
        │ [49, 1536]                       │
        └────────┬─────────────────────────┘
                 ▼
   <|image_pad|> 자리에 patch 49개 splice   ← src/model.py: _merge 직접 구현
                 │
                 ▼
   Qwen2.5-1.5B-Instruct
   (4-bit NF4 frozen  +  ★ LoRA r=16 on all-linear)
                 │
                 ▼
       "Kitchen."   +   ⚠️ OOD abstention 배너 (첫 토큰 entropy)
```

> ★ = 학습 대상 (projector + LoRA). CLIP 과 Qwen2.5 base 가중치는 frozen.

### ▸ 구성 요소

| 블록 | 모델 | 상태 | 역할 |
|---|---|---|---|
| **비전** | CLIP-ViT-B/32 | frozen | 224px → 49 patch × 768-d 인코딩 |
| **Projector** | 2-layer MLP (768→1536, GELU) | **학습** (3.5M) | patch 를 LLM 임베딩 공간으로 사상. LLaVA-1.5 `mlp2x_gelu` 와 동일 |
| **LLM** | Qwen2.5-1.5B-Instruct | 4-bit NF4 frozen | 융합된 시퀀스를 받아 답변 생성 |
| **LoRA** | r=16, α=32, all-linear | **학습** | q/k/v/o + gate/up/down 전 linear layer |

### ▸ 왜 1.5B 인가

v3 는 0.5B 를 썼고, 회고에서 시각 detail 약함의 진짜 병목을 0.5B LLM 의 추론 능력으로 진단했습니다. v4 는 그 진단을 직접 검증하려고 **"LLM 크기" 단일 변수만** 키웠습니다 — vision encoder 는 ViT-B/32 그대로입니다 (v3 Step 2 의 ViT-L/14 ablation 이 0.5B 환경에서 효과 없음을 이미 확인). 8GB VRAM 에 1.5B 를 fp16 으로 올리면 가중치만 ~3GB + activation 으로 OOM 위험이라, 4-bit 양자화가 사실상 필수였습니다.

---

## 🔧 직접 구현한 부분 (Implementation)

VLM 의 핵심인 이미지-텍스트 융합을 고수준 라이브러리에 맡기지 않고 저수준에서 직접 다뤘습니다.

### ▸ 임베딩 splice — `src/model.py: _merge`

입력 시퀀스에서 `<|image_pad|>` 토큰 위치를 찾아, 그 한 자리를 projector 가 낸 **49개 patch 임베딩으로 펼쳐 교체**합니다. 토큰 1개 → 임베딩 49개로 시퀀스 길이가 바뀌므로 `input_embeds`·`attention_mask`·`labels` 를 모두 일관되게 재정렬해야 합니다. 이 `_merge` 를 배치 단위로 직접 구현했습니다.

### ▸ QLoRA 로 8GB fit

Qwen2.5-1.5B 를 fp32 로 올리면 가중치만 6GB 라 학습 자체가 불가능합니다. bitsandbytes **4-bit NF4 + double quantization** 으로 base 를 ~0.9GB 로 압축하고, **gradient checkpointing** 으로 activation 메모리를 줄여 batch_size 1 로 학습했습니다. 학습 대상인 projector·LoRA 는 fp32 master weight 로 둬 업데이트 정밀도를 유지했습니다 (base 만 4-bit).

### ▸ `<|image_pad|>` 토큰 재사용 — v3 의 1GB adapter 문제 원천 차단

새 토큰을 추가하는 대신 Qwen2.5 에 내장된 `<|image_pad|>` (id 151655) 를 그대로 썼습니다. `resize_token_embeddings` 를 호출하지 않으므로, v3 에서 겪었던 "토큰 추가 → embedding resize → PEFT 가 `embed_tokens` 를 통째로 저장 → adapter 1GB" 연쇄가 처음부터 생기지 않습니다. splice 는 토큰의 *위치* 만 쓰므로(그 자리는 어차피 patch 로 교체됨) 사전학습 임베딩 품질과 무관합니다.

### ▸ instruction-only label masking — `src/dataset.py`

system·user 토큰은 `IGNORE_INDEX(-100)` 로 가리고 assistant 응답 토큰에만 loss 를 줍니다. 모델이 질문을 따라 외우지 않고 답변 생성만 학습하도록 했습니다.

---

## 🧪 학습 (Training)

LLaVA-1.5 의 2단계 학습을 따랐습니다 — 먼저 projector 만 정렬하고(Stage 1), 그 위에서 LoRA instruction tuning(Stage 2).

| | Stage 1 — 정렬 | Stage 2 — instruction |
|---|---|---|
| 학습 대상 | projector 3.5M | projector + LoRA = 22.0M |
| 데이터 | 40K (이미지 2만 장 × 캡션 2) | 46K 믹스 |
| step / 시간 | 5,000 / ≈3.3h | 5,750 / ≈7.4h |
| loss (콘솔 관측값) | 5.0 → 1.98 | 3.65 → ≈1.01 |

### ▸ Stage 2 데이터 믹스

한 능력에 치우치지 않도록 네 종류를 의도적으로 섞었습니다.

| 출처 | 샘플 | 능력 |
|---|---|---|
| VQAv2 | 18K | 짧은 사실형 QA |
| LocalizedNarratives | 10K | 긴 묘사 |
| A-OKVQA | 6K | 추론형 QA |
| KoLLaVA | 12K | 한국어 |
| **합계** | **46K** | |

- batch_size 1 + grad_accum 8 (effective batch 8), gradient checkpointing ON — Stage 1·2 모두 OOM 없이 8GB 한 장에서 완주했습니다.
- 학습 루프 `src/train.py` — cosine LR + warmup, gradient clipping, 중간 체크포인트 저장.
- 위 loss·step·시간은 v4 학습 당시 **콘솔 관측값** 입니다 — 본 학습은 step별 로그를 파일로 남기지 않았습니다. `src/train.py` 는 이후 학습부터 step별 loss 를 `eval_results/train_log_*.csv` 로 기록합니다.

---

## 🚦 정량 평가와 배포 게이트 (Benchmarks & Deployment Gate)

### ▸ 검증을 배포 앞에 둔다

v3 는 학습 직후 배포하고 성능 미달을 라이브에서 발견했습니다 — 순서가 틀렸습니다. v4 는 통과 기준을 배포 *전에* 고정하고, raw 모델이 그 기준을 넘어야만 HF Spaces 에 배포되도록 게이트(`scripts/eval_gate.py`)를 절차에 넣었습니다. 기준은 SOTA 가 아니라 **"v3·랜덤보다 분명히 위"** 라는 최소선입니다.

| 게이트 기준 | bar | v4 raw 측정 | 통과 |
|---|---|---|---|
| VQAv2 정답률 | ≥ 0.45 | **0.568** | ✅ |
| POPE 정답률 | ≥ 0.65 | **0.718** | ✅ |
| POPE yes-F1 | ≥ 0.55 | **0.735** | ✅ |
| POPE 예측 쏠림 | ≤ 0.85 | **0.568** | ✅ |
| 거부율 | ≤ 0.15 | **0.000** | ✅ |

→ 5/5 통과, verdict **GO**. 측정 산출물은 [`eval_results/v4_gate_report.json`](eval_results/v4_gate_report.json).

### ▸ VQAv2 · POPE 점수

`scripts/eval_gate.py` 로 학습된 raw 모델(추론 wrapper 없이, OOD 배너 등 데모 레이어 미포함)을 평가했습니다 — VQAv2 val / POPE test 각 n=400, greedy decoding.

| 지표 | v3 raw | **v4 raw** |
|---|---|---|
| VQAv2 정답률 | 36.7% | **56.8%** |
| POPE 정답률 | 50.0% | **71.8%** |
| POPE yes-F1 | — | **0.74** |

v3 는 POPE 에서 전부 "yes" 만 답해 사실상 랜덤(50%)이었지만, v4 는 yes 227 / no 173 으로 양쪽을 실제로 가립니다. VQAv2 를 유형별로 보면 **yes/no 76% · 개방형 46% · 숫자 43%** (`v4_gate_report.json` 의 `vqav2_by_type`) — 짧은 사실형에 강하고 개방형·숫자에 약합니다.

> 절대 수치로는 공개 소형 VLM(보통 VQAv2 70%대+, POPE 85%+)에 못 미칩니다. 1.5B·8GB·약 9만 학습 샘플(LLaVA-1.5 는 120만+)의 제약을 그대로 반영한 결과이고, 평가 하네스가 달라 외부 모델과 직접 비교하기는 어렵습니다 (v3 수치도 이전 저장소의 소표본 측정값).

---

## ▶️ 직접 확인하기 (How to Verify)

### ▸ Live Demo

웹에서 바로 써 볼 수 있는 라이브 데모: **[Mini-LLaVA v4 Demo](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo)**. 무료 CPU 티어라 응답에 수십 초가 걸립니다. greedy 디코딩이라 같은 입력은 항상 같은 답을 냅니다.

### ▸ 번들 입력 이미지

`assets/` 의 두 이미지를 데모에 올려 바로 시험해 볼 수 있습니다.

| `source_dog.jpg` (학습 분포 안) | `source_pikachu.png` (학습 분포 밖, 만화) |
|:---:|:---:|
| <img src="assets/source_dog.jpg" width="220" alt="강아지 + 헬로키티 모자" /> | <img src="assets/source_pikachu.png" width="220" alt="피카츄 + 선장 모자" /> |
| 일반 객체 VQA | OOD 입력 — 분포 밖 만화 캐릭터 |

### ▸ 배포 Space 스모크 점검

배포된 Space 에 영어/한국어 · in-dist/OOD 10케이스를 입력해 실제 동작을 확인했습니다 (`scripts/smoke_space_demo.py` → [`eval_results/v4_space_smoke.json`](eval_results/v4_space_smoke.json), greedy decoding).

| 입력 | 질문 | 응답 (greedy) | 판정 |
|---|---|---|---|
| 사실 (EN) | What is in this image? | Woman. | ✅ 객체 정답 · ⚠️ OOD 배너 (score 0.48, 경계) |
| 사실 (KO) | 이 사진에 무엇이 보이나요? | "이미지에는 한 여성이 휴대폰으로 캡처한 사진을…" | △ 장황 · ⚠️ OOD 배너 (0.48) |
| 행동 (EN) | What is the boy doing? | Typing. | ✅ |
| 특정 객체 (EN) | What animal is the woman holding? | Dog. | ❌ 실제 고양이 |
| 묘사 (KO) | 이미지를 한국어로 설명해 주세요. | "…빨간색과 검은색의 두 마리의 개를 들고…" | ❌ 환각 |
| 장면 (EN) | What room is this? | Kitchen. | ✅ |
| 묘사 (EN) | Describe this image. | "In this image we can see a kitchen with some appliances…" | △ 형식은 유창 |
| Yes/No (EN) | Is the boy wearing headphones? | Yes. | ✅ |
| OOD · 만화 | What is in this image? | Girl. | ⚠️ 분포 밖 — OOD 배너 정상 (0.56) |
| OOD · 추상화 | What do you see in this picture? | Birds. | ⚠️ 분포 밖 오답 — OOD 배너 정상 (0.61) |

정리하면 — 장면·yes/no·단순 객체 같은 **짧은 영어 사실형은 안정적**이고, 세밀한 구분(고양이↔개)과 장문 묘사는 약하며(형식은 유창하나 없는 디테일을 지어냄), 한국어 장문은 환각이 잦습니다. OOD abstention 배너는 만화·추상화 두 OOD 입력에서 정상 동작했습니다. 표의 1·2행은 같은 in-dist 사진(질문만 영어/한국어로 다름)이라 OOD score 가 0.48 로 동일합니다 — OOD 점수는 질문이 아니라 이미지 속성이라 같은 값이고, 임계값 0.46 을 살짝 넘어 배너가 떴는데 보정 FPR 7.5% 와 일치하는 경계값 오탐입니다. 배너는 답변 *내용* 을 바꾸지 않고 신뢰도만 표시합니다.

---

## 🔍 OOD 검출 (OOD Detection)

분포 밖 입력(만화·추상화)에 모델이 자신있게 틀리는 문제를 정량 분석하고, 그 결과를 데모에 abstention 레이어로 연동했습니다.

### ▸ 두 신호를 100케이스로 ROC 비교

CLIP image-text 유사도와 LLM 첫 토큰 entropy 두 신호를 100케이스(in-dist 40 + 만화 30 + 추상화 30)로 비교했습니다 (`scripts/ood_roc_analysis.py` → [`eval_results/v4_ood_roc.json`](eval_results/v4_ood_roc.json)).

![OOD 검출 — entropy/CLIP 가중치 스윕에 따른 ROC AUC](assets/ood_roc_sweep.png)

| 신호 | ROC AUC |
|---|---|
| **entropy 단독** (w_clip = 0) | **0.971** |
| 가중 합 (w_clip = 0.6) | 0.884 |
| CLIP 단독 (w_clip = 1.0) | 0.660 |

entropy 단독이 가장 강했고, CLIP 신호를 섞을수록 단조 감소했습니다. **5-fold 교차검증 평균 0.969** (fold당 test 20개, fold별 0.95~0.99)로 과적합이 아님을 확인했습니다. 최적 가중치에서 Youden's J 임계값은 **0.4582**, 그 100케이스 위 판정 정확도 91% (TPR 0.90 / FPR 0.075) — 임계값을 같은 셋에서 골랐으므로 정확도는 in-sample 참고치이고, 교차검증한 지표는 AUC 입니다. 카테고리별 AUC 는 만화 0.965 · 추상화 0.978.

### ▸ 데모 연동 — abstention 배너

이 검출기를 v4 데모에 연결했습니다 (`src/ood_detection.py` → `src/infer.py` → `space/app.py`). 데모는 답변마다 고정 프롬프트로 첫 토큰 entropy 를 한 번 더 재고, OOD score 가 0.4582 임계값을 넘으면 답변 위에 ⚠️ 저신뢰 경고를 띄웁니다 — **답변 내용은 그대로 두고 신뢰도만 표시** 합니다. 위 스모크 점검의 만화(0.56)·추상화(0.61) 두 건에서 배너가 정상적으로 떴습니다.

### ▸ 한계 — entropy 는 *불확실성* 신호다

entropy 검출은 모델이 분포 밖 입력을 *자신있게* 틀리면 놓칩니다. 번들 샘플 `assets/source_pikachu.png` 가 그 예입니다 — 모델이 "Teddy bear" 라 확신해 entropy 가 낮고(score 0.35 < 0.4582) 배너가 뜨지 않습니다. 검출 대상은 *불확실한 OOD* 이지 *자신있게 틀린 OOD* 가 아니라는 구조적 한계입니다. 반대 방향으로는, 경계값 in-dist 입력이 임계값을 살짝 넘어 오탐이 날 수 있습니다(보정 FPR 7.5%).

> 보정은 4-bit 학습 모델, 무료 CPU 데모는 fp32 라 entropy 가 미세하게 다를 수 있어 `scripts/diag_deploy_gap.py` 로 그 차이를 정량화했습니다 (아래 회고).

---

## 🔄 v4 가 v3 대비 무엇이 바뀌었나 (What Changed from v3)

| | v3 | **v4** |
|---|---|---|
| **LLM** | Qwen2.5-0.5B | **Qwen2.5-1.5B-Instruct** (3× params) |
| **학습 방식** | fp16/bf16 base + LoRA | **QLoRA 4-bit NF4** — 8GB 에 1.5B fit |
| **LoRA 대상** | attention q/k/v/o | **all-linear** (+ gate/up/down MLP) — QLoRA 논문 정석 |
| **이미지 토큰** | 새 토큰 추가 → adapter 1GB | **내장 `<\|image_pad\|>` 재사용** → resize 자체를 제거 |
| **raw VQAv2 / POPE** | 36.7% / 50.0% | **56.8% / 71.8%** |
| **배포 절차** | 학습 직후 배포 (성능 미달을 라이브에서 발견) | **사전 게이트** — 기준 통과 시에만 배포 |
| **분포 밖 입력** | 추론 wrapper 가 점수에 개입 | **OOD abstention 배너** — 답변 불변, 신뢰도만 표시 |
| **성능 보강 방식** | CLIP grounding·OOD router 등 추론 wrapper | **raw 모델 자체 강화** (wrapper 없음) |

v3 는 raw 모델의 약함을 추론 wrapper(CLIP grounding 등)로 가렸고, 그 wrapper 가 올린 점수의 상당 부분은 모델 능력이 아니라 routing 이었습니다. v4 는 wrapper 를 걷어내고 raw 모델 자체를 키워, 게이트가 측정하는 56.8% / 71.8% 가 모델의 실제 능력입니다.

---

## 💡 회고록 (Retrospective)

v1 부터 네 번에 걸쳐 만들면서 배운 것과, 시리즈를 마치며 남기는 솔직한 평가입니다.

### ▸ 시리즈가 가르쳐 준 것

v1·v2 에서 CLIP·projector·LLM 을 잇는 기본 구조를 만들었고, v3 에서는 모델이 시각 detail 에 약한 원인을 진단하고 추론 wrapper 로 점수를 끌어올려 배포했습니다 — 그런데 그 점수의 상당 부분은 모델 능력이 아니라 routing 이었고, raw 모델의 약함을 가리고 있었습니다. v4 는 두 가지를 바로잡았습니다. **LLM 을 1.5B 로 키워 raw 모델 자체를 강화** 했고, v3 의 진단을 다시 따져 병목의 절반은 LLM 크기가 아니라 **학습 데이터 규모** (Stage 1 정렬이 5K — LLaVA 의 1%)였음을 확인했습니다. 시행착오는 v1~v3 의 몫이었고, v4 는 그 교훈을 모아 적용한 버전입니다.

### ▸ 구체적으로 배운 세 가지

| 교훈 | v3 에서 무엇이 틀렸나 | v4 에서 어떻게 바꿨나 |
|---|---|---|
| **검증은 배포 앞에 와야 한다** | 학습 직후 배포 → 성능 문제를 라이브에서 발견 | 통과 기준을 배포 *전에* 고정, 게이트를 절차에 삽입 |
| **도구가 왜 그렇게 동작하는지 알아야 한다** | 새 토큰 추가가 `embed_tokens` 저장까지 연쇄됨을 모름 → adapter 1GB | 내장 토큰 재사용 — 버그를 *고치는* 게 아니라 *안 만드는* 방향 |
| **평가에서 데이터가 새면 숫자가 거짓말한다** | POPE 임계값을 test set 으로 튜닝 → 70% 가 일반화 보장 못 함 | train/test 분리로 누수 제거 |

### ▸ v4 를 마치며

v4 는 목표한 것을 해냈지만 성능 자체는 제한적입니다. VQAv2 56.8% / POPE 71.8% 는 공개 소형 VLM 에 못 미치고, 1.5B·8GB·약 9만 샘플이라는 천장은 분명합니다 — 다만 이 시리즈의 목표는 SOTA 가 아니라 *제약 안에서 끝까지 검증된 구현* 이었습니다. v3 가 *분석* 으로만 끝낸 OOD 숙제는 이번엔 데모의 abstention 레이어로 닫았습니다. 이건 v3 가 거부했던 "성능 wrapper" 와 다릅니다 — v3 wrapper 는 답변을 바꿔 벤치마크 점수를 부풀렸지만(그래서 거부), abstention 레이어는 답변을 바꾸지 않고 신뢰도만 표시하므로 게이트의 raw 측정을 건드리지 않습니다.

학습(4-bit)과 무료 CPU 데모(fp32) 사이의 정밀도 절충은 `scripts/diag_deploy_gap.py` 로 정량화했습니다 — 7개 진단 케이스에서 답변은 **7/7 동일**, OOD-probe entropy 차이도 평균 **0.17 nats** 로 작아 OOD 판정이 두 config 에서 모두 일치했습니다 ([`eval_results/v4_deploy_gap.json`](eval_results/v4_deploy_gap.json)). 4-bit 로 보정한 OOD 임계값을 fp32 데모에 그대로 써도 무방하다는 근거입니다.

한국어는 학습 데이터를 4K(v3) → 12K 로 늘려 출력 언어는 안정적으로 복원했지만, 신뢰할 표준 한국어 VQA 벤치마크가 없어 정량 평가셋은 만들지 못했습니다 — 짧은 영어 사실형이 가장 안정적이고 한국어 장문은 환각이 잦다는 것이 정성 점검의 결론입니다. 네 번의 반복으로 from-scratch VLM 의 구조·학습·평가·배포를 한 번씩 직접 통과한 것이 이 시리즈가 남긴 것입니다.

---

## 📚 참고 자료 (References)

- **LLaVA-1.5** — Liu et al. (2023), [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)
- **QLoRA** — Dettmers et al. (2023), [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **LoRA** — Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Qwen2.5** — Yang et al. (2024), [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- **CLIP** — Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **POPE** — Li et al. (2023), [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)
- **KoLLaVA** — tabtoyou (2024), [KoLLaVA-Instruct-150k Dataset](https://huggingface.co/datasets/tabtoyou/KoLLaVA-Instruct-150k) (DeepL 번역, CC-BY-NC-4.0)

### ▸ 시리즈 repo · 산출물

- **v3** — [vlm-from-scratch-v3](https://github.com/AD-Styles/vlm-from-scratch-v3) — 한국어 mix · OOD 분석 · slim adapter
- **v2** — [vlm-from-scratch](https://github.com/AD-Styles/vlm-from-scratch) — 2-Stage 학습 baseline
- **모델 가중치** — [HF Hub: mini-llava-v4](https://huggingface.co/AD-Styles/mini-llava-v4)
- **라이브 데모** — [HF Space: mini-llava-v4-demo](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo)
