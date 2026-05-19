"""Mini-LLaVA v4 — 로컬 Gradio 런처 (raw 모델 sanity-check 용).

배포용 데모는 `space/app.py` (HF Spaces). 이 파일은 학습한 v4 raw 모델을 로컬에서
바로 띄워 출력을 확인하는 단순 launcher 다 — v4 는 답변 내용을 바꾸는 추론 wrapper
가 없고, 학습 분포 밖 입력에 OOD entropy 경고 배너만 얹는다.

사용:
  python app.py \\
    --checkpoint checkpoints/v4_stage2_qlora/projector.pt \\
    --lora-adapter checkpoints/v4_stage2_qlora/lora_adapter
  python app.py --share                                   # 공개 링크 생성
"""
from __future__ import annotations

import argparse
import os

import gradio as gr
from PIL import Image

from src.config import GenerationConfig
from src.infer import VLMInference


HEADER_MD = """
# 🖼️ Mini-LLaVA v4 — Vision-Language Demo

**CLIP-ViT-B/32 + MultiModalProjector + Qwen2.5-1.5B-Instruct** 를 처음부터 조립한
멀티모달 LLM. 이미지를 올리고 질문해 보세요 — 짧은 영어 사실형 질문(객체·yes/no)에
가장 안정적이고, 한국어도 되지만 답이 길어지고 환각이 늘어납니다.
"""

FOOTER_MD = """
---
> 🛠️ Powered by `vlm-from-scratch-v4` — CLIP-ViT + Qwen2.5-1.5B + QLoRA 직접 구현.
> 이 launcher 는 raw 모델 출력 확인용. 배포 데모는 [HF Space](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) 또는 `python space/app.py` 참조.
"""

# 데모 추천 질문 — 검증상 모델이 안정적인 짧은 영어 사실형(객체 식별·yes/no) 위주.
# 긴 묘사·계수·추론, 한국어 질문은 환각·장황 답변이 잦아 예시에서 뺐다 (README 참고).
EXAMPLES = [
    ["What is in this image?"],
    ["What animal is in this image?"],
    ["Is there an animal in this picture?"],
    ["Is the dog wearing a hat?"],
]


def build_engine(checkpoint: str | None, lora_adapter: str | None) -> VLMInference:
    if checkpoint and not os.path.exists(checkpoint):
        print(f"[warn] checkpoint not found: {checkpoint} — random init projector 사용")
        checkpoint = None
    if lora_adapter and not os.path.exists(lora_adapter):
        print(f"[warn] LoRA adapter not found: {lora_adapter} — base LLM 사용")
        lora_adapter = None
    return VLMInference(
        checkpoint_path=checkpoint, lora_adapter_path=lora_adapter
    )


def make_predict_fn(engine: VLMInference):
    def predict(
        image: Image.Image | None,
        question: str,
        max_new_tokens: int,
    ):
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
        result = engine(image, question.strip(), gen_cfg=cfg)
        answer, ood = result["answer"], result.get("ood")

        # OOD abstention — 분포 밖 입력이면 답변 위에 저신뢰 경고 배너 (답변은 유지).
        if ood and ood["is_ood"]:
            thr = engine.detector.threshold
            answer = (
                f"⚠️ 학습 분포 밖(OOD)으로 보이는 입력 — 아래 답변의 신뢰도가 낮습니다 "
                f"(OOD score {ood['ood_score']:.2f} > 임계값 {thr:.2f}).\n"
                f"{'─' * 38}\n{answer}"
            )

        meta = f"⏱️ {result['elapsed']:.2f}s · max_new={cfg.max_new_tokens} · greedy"
        if ood:
            tag = "⚠️ OOD" if ood["is_ood"] else "✅ in-dist"
            meta += f" · {tag} (OOD score {ood['ood_score']:.2f})"
        return answer, meta

    return predict


def build_ui(engine: VLMInference) -> gr.Blocks:
    predict = make_predict_fn(engine)

    with gr.Blocks(title="Mini-LLaVA Demo") as demo:
        gr.Markdown(HEADER_MD)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="pil", label="🖼️ 이미지 업로드", height=380
                )
                question_in = gr.Textbox(
                    label="❓ 질문",
                    placeholder="예: 이 이미지에 무엇이 보이나요?",
                    lines=2,
                )
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_in],
                    label="💡 예시 질문",
                )

                with gr.Accordion("⚙️ 생성 옵션 (고급)", open=False):
                    max_new_tokens = gr.Slider(
                        16, 512, value=128, step=16, label="max_new_tokens"
                    )

                submit_btn = gr.Button("🚀 응답 생성", variant="primary")

            with gr.Column(scale=1):
                answer_out = gr.Textbox(
                    label="🤖 모델 응답", lines=12, interactive=False
                )
                meta_out = gr.Markdown("")

        submit_btn.click(
            fn=predict,
            inputs=[image_in, question_in, max_new_tokens],
            outputs=[answer_out, meta_out],
        )
        question_in.submit(
            fn=predict,
            inputs=[image_in, question_in, max_new_tokens],
            outputs=[answer_out, meta_out],
        )

        gr.Markdown(FOOTER_MD)

    return demo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v4_stage2_qlora/projector.pt",
        help="학습된 projector 가중치 경로",
    )
    p.add_argument(
        "--lora-adapter",
        type=str,
        default="checkpoints/v4_stage2_qlora/lora_adapter",
        help="Stage 2 LoRA adapter 디렉터리",
    )
    p.add_argument("--server-name", type=str, default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Gradio 공개 링크 생성")
    return p.parse_args()


def main():
    args = parse_args()
    engine = build_engine(args.checkpoint, args.lora_adapter)
    demo = build_ui(engine)
    # Gradio 6.0+: theme은 launch()로 전달 (이전엔 Blocks 생성자에 있었음)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
