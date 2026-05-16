---
title: Mini-LLaVA v4 Demo
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Mini-LLaVA v4 — Vision-Language Demo

처음부터 조립한 멀티모달 LLM 의 **raw 모델** 데모입니다.

- **구조** — CLIP-ViT-B/32 (frozen) + 2-layer MLP Projector + Qwen2.5-1.5B-Instruct
- **학습** — QLoRA 4-bit (NF4) Stage 1 정렬 → Stage 2 instruction 46K (영문 + 한국어 균형 믹스), RTX 4060 8GB 한 장
- **평가** — 배포 게이트 5/5 GO · VQAv2 36.7%→56.8%, POPE 50.0%→71.8% (v3→v4, raw 모델 · wrapper 없음)

v3 와 달리 추론 wrapper (CLIP grounding / 번역 / OOD router) 가 **없습니다** — raw
모델 자체가 VQAv2 + POPE 배포 게이트를 통과했기 때문입니다.

## ⚙️ 실행 메모

이 Space 는 무료 **CPU** 티어에서 동작합니다. 학습은 QLoRA 4-bit (CUDA) 였으나,
CPU 추론에서는 base Qwen2.5-1.5B 를 fp32 로 로드하고 학습된 projector·LoRA 를
얹습니다. 1.5B 모델을 GPU 없이 돌리므로 **응답에 수십 초가 걸립니다** — 정성 확인용
데모입니다.

가중치는 부팅 시 HF Hub `AD-Styles/mini-llava-v4` 에서 자동 다운로드됩니다
(`MODEL_REPO` 환경변수로 변경 가능).

## 🔗 링크

- 소스 / 학습 회고: [github.com/AD-Styles/vlm-from-scratch-v4](https://github.com/AD-Styles/vlm-from-scratch-v4)
- 이전 버전: [v3](https://github.com/AD-Styles/vlm-from-scratch-v3) · [v2](https://github.com/AD-Styles/vlm-from-scratch)
