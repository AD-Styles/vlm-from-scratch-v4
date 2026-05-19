"""Mini-LLaVA v4 — Hugging Face Spaces 데모.

v4 raw 모델 (CLIP-ViT-B/32 + MultiModalProjector + Qwen2.5-1.5B + LoRA) 을 서빙한다.
답변 *내용* 을 바꾸는 성능(capability) wrapper — v3 의 CLIP grounding / OOD router —
는 없다. 그것이 raw 의 약함을 벤치마크 점수에서 가렸다는 회고 때문이고, raw 모델은
배포 게이트(`scripts/eval_gate.py`) 5/5 를 자력으로 통과했다. 대신 답변을 바꾸지
않는 abstention 레이어만 얹는다: 첫 토큰 entropy 로 학습 분포 밖 입력을 감지해
저신뢰 경고 배너를 띄운다 (src/ood_detection.py, OOD entropy AUC 0.971 로 보정).
성능 wrapper(거부) ≠ abstention 레이어(채택) — 후자는 게이트 raw 측정을 건드리지 않는다.

⚙️ 실행 환경 메모
  - 학습은 QLoRA 4-bit (CUDA) 였으나 HF Spaces 무료 티어는 CPU.
    → base Qwen2.5-1.5B 를 fp32 로 로드하고 학습된 projector·LoRA 를 얹어 추론한다.
    4-bit 로 학습한 LoRA 를 fp32 base 에 적용하는 것은 표준 QLoRA 배포 절충이며,
    미세한 분포 차이가 있을 수 있다 (README "배포 정밀도 갭" 참고).
  - CPU 에서 1.5B 추론은 느리다 — 응답에 수십 초가 걸린다. 정성 확인용 데모.

가중치는 Space 부팅 시 HF Hub (기본 AD-Styles/mini-llava-v4) 에서 자동 다운로드.
"""
from __future__ import annotations

import os
import traceback

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from src.config import GenerationConfig
from src.infer import VLMInference

# 가중치 저장소 — Space 설정에서 MODEL_REPO 환경변수로 덮어쓸 수 있음.
MODEL_REPO = os.environ.get("MODEL_REPO", "AD-Styles/mini-llava-v4")


HEADER_MD = """
# 🖼️ Mini-LLaVA v4 — Vision-Language Demo

**CLIP-ViT-B/32 + MultiModalProjector + Qwen2.5-1.5B-Instruct (QLoRA)** 를 처음부터
조립한 멀티모달 LLM. 짧은 영어 사실형 질문(객체·yes/no)에 가장 안정적이고, 한국어도
되지만 답이 길어지고 환각이 늘어납니다. 학습 분포 밖(만화·추상화 등) 입력에는 답변
위에 ⚠️ 저신뢰 경고가 표시됩니다.

> ⏳ **무료 CPU Space 입니다.** 1.5B 모델을 GPU 없이 돌리는 비용으로 — 짧은 답변은
> 20~40초, 긴 묘사형 답변은 수 분까지 걸립니다. 정성 확인용 데모입니다.
> 학습·평가 상세는 [vlm-from-scratch-v4](https://github.com/AD-Styles/vlm-from-scratch-v4) 참고.
"""

FOOTER_MD = """
---
> 🛠️ `vlm-from-scratch-v4` — raw 모델 서빙 + entropy 기반 OOD abstention 배너.
> 답변 내용을 바꾸는 성능 wrapper 는 없습니다 (v3 의 실수). 단, 학습 분포 밖
> 입력에는 첫 토큰 entropy(AUC 0.971 로 보정)로 저신뢰 경고를 띄웁니다.
"""

# 데모 추천 질문 — 검증상 모델이 안정적인 짧은 영어 사실형(객체 식별·yes/no) 위주.
# 긴 묘사·계수·추론, 한국어 질문은 환각·장황 답변이 잦아 예시에서 뺐다 (README 참고).
EXAMPLES = [
    ["What is in this image?"],
    ["What animal is in this image?"],
    ["Is there an animal in this picture?"],
    ["Is the dog wearing a hat?"],
]


def load_engine() -> tuple[VLMInference | None, str]:
    """HF Hub 에서 v4 가중치를 받아 추론 엔진 빌드 → (engine, status)."""
    try:
        print(f"[space] downloading v4 weights from {MODEL_REPO} ...")
        local = snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns=["projector.pt", "lora_adapter/*"],
        )
        projector = os.path.join(local, "projector.pt")
        lora = os.path.join(local, "lora_adapter")
        print(f"[space] projector = {projector}")
        print(f"[space] lora      = {lora}")

        # HF Spaces 무료 티어 = CPU → fp32 base. (학습은 QLoRA 4-bit / CUDA)
        engine = VLMInference(
            checkpoint_path=projector,
            lora_adapter_path=lora,
            device="cpu" if not torch.cuda.is_available() else None,
            torch_dtype=torch.float32,
        )
        return engine, "ok"
    except Exception as e:  # noqa: BLE001 — 부팅 실패를 UI 로 노출
        traceback.print_exc()
        return None, f"{type(e).__name__}: {e}"


ENGINE, ENGINE_STATUS = load_engine()


def predict(
    image: Image.Image | None,
    question: str,
    max_new_tokens: int,
):
    if ENGINE is None:
        return (
            f"⚠️ 모델 로드 실패 — 가중치 저장소 `{MODEL_REPO}` 를 확인하세요.\n"
            f"({ENGINE_STATUS})",
            "",
        )
    if image is None:
        return "⚠️ 이미지를 먼저 업로드해 주세요.", ""
    if not question or not question.strip():
        return "⚠️ 질문을 입력해 주세요.", ""

    # greedy 디코딩 — 약한 모델에서 sampling 보다 짧은 사실형 답이 안정적이고
    # 실행마다 동일하다. temperature/top_p 는 쓰지 않는다.
    cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    result = ENGINE(image, question.strip(), gen_cfg=cfg)
    answer, ood = result["answer"], result.get("ood")

    # OOD abstention — 학습 분포 밖 입력이면 답변 위에 저신뢰 경고 배너.
    # 답변 자체는 그대로 노출한다 (검출기 동작을 보여주는 투명한 데모).
    if ood and ood["is_ood"]:
        thr = ENGINE.detector.threshold
        answer = (
            f"⚠️ 학습 분포 밖(OOD)으로 보이는 입력입니다 — 아래 답변의 신뢰도가 "
            f"낮습니다 (OOD score {ood['ood_score']:.2f} > 임계값 {thr:.2f}).\n"
            f"{'─' * 38}\n{answer}"
        )

    meta = f"⏱️ {result['elapsed']:.1f}s · max_new={cfg.max_new_tokens} · greedy"
    if ood:
        tag = "⚠️ OOD" if ood["is_ood"] else "✅ in-dist"
        meta += f" · {tag} (OOD score {ood['ood_score']:.2f})"
    return answer, meta


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Mini-LLaVA v4 Demo") as demo:
        gr.Markdown(HEADER_MD)

        if ENGINE is None:
            gr.Markdown(
                f"### 🔴 모델을 불러오지 못했습니다\n"
                f"`{MODEL_REPO}` 에서 가중치를 받지 못했습니다 — `{ENGINE_STATUS}`.\n"
                f"`scripts/push_v4_weights.py` 로 projector·LoRA 를 먼저 업로드하세요."
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="🖼️ 이미지 업로드", height=380)
                question_in = gr.Textbox(
                    label="❓ 질문",
                    placeholder="예: 이 이미지에 무엇이 보이나요?",
                    lines=2,
                )
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_in],
                    label="💡 예시 질문 (이미지를 올린 뒤 클릭)",
                )
                with gr.Accordion("⚙️ 생성 옵션 (고급)", open=False):
                    max_new_tokens = gr.Slider(
                        16, 160, value=48, step=16,
                        label="max_new_tokens (CPU — 클수록 느림)",
                    )
                submit_btn = gr.Button("🚀 응답 생성", variant="primary")

            with gr.Column(scale=1):
                answer_out = gr.Textbox(label="🤖 모델 응답", lines=12, interactive=False)
                meta_out = gr.Markdown("")

        inputs = [image_in, question_in, max_new_tokens]
        submit_btn.click(fn=predict, inputs=inputs, outputs=[answer_out, meta_out])
        question_in.submit(fn=predict, inputs=inputs, outputs=[answer_out, meta_out])

        gr.Markdown(FOOTER_MD)

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
