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

CLIP-ViT-B/32 + 2-layer MLP Projector + Qwen2.5-1.5B-Instruct 를 직접 이어 붙여 만든
소형 비전-언어 모델입니다. RTX 4060 Laptop 8GB 한 장에서 QLoRA 4-bit 로 학습했습니다.

**제약된 환경의 학습용 모델입니다.** 장면·yes/no·단순 객체 같은 짧은 사실형 질문에는
어느 정도 답하지만, 세밀한 구분(고양이↔개)과 긴 묘사에서는 자주 틀리고 환각합니다 —
1.5B 규모와 약 9만 학습 샘플의 한계입니다. 정량 평가는 raw 모델 기준 VQAv2 56.8% /
POPE 71.8% (n=400).

## 실행 메모

무료 **CPU** 티어에서 동작합니다. 학습은 QLoRA 4-bit (CUDA) 였으나, CPU 추론에서는
base Qwen2.5-1.5B 를 fp32 로 로드하고 학습된 projector·LoRA 를 얹습니다. 1.5B 모델을
GPU 없이 돌리므로 **응답에 수십 초가 걸립니다.**

가중치는 부팅 시 HF Hub `AD-Styles/mini-llava-v4` 에서 자동 다운로드됩니다
(`MODEL_REPO` 환경변수로 변경 가능).

## 링크

- 소스 / 구현 설명: [github.com/AD-Styles/vlm-from-scratch-v4](https://github.com/AD-Styles/vlm-from-scratch-v4)
- 이전 버전: [v3](https://github.com/AD-Styles/vlm-from-scratch-v3) · [v2](https://github.com/AD-Styles/vlm-from-scratch)
