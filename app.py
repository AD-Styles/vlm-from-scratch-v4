"""Mini-LLaVA v4 — 로컬 Gradio 런처 (raw 모델 sanity-check 용).

배포용 데모는 `space/app.py` (HF Spaces). 이 파일은 학습한 v4 raw 모델을 로컬에서
바로 띄워 출력을 확인하는 단순 launcher 다 — v4 는 추론 wrapper 가 없다.

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
멀티모달 LLM. 이미지를 업로드하고 자연어로 질문해보세요. 한국어 / 영어 모두 가능합니다.
"""

FOOTER_MD = """
---
> 🛠️ Powered by `vlm-from-scratch-v4` — CLIP-ViT + Qwen2.5-1.5B + QLoRA 직접 구현.
> 이 launcher 는 raw 모델 출력 확인용. 배포 데모는 [HF Space](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) 또는 `python space/app.py` 참조.
"""

EXAMPLES = [
    ["이 이미지에 무엇이 보이나요? 자세히 묘사해 주세요."],
    ["What objects are in this image?"],
    ["사진 속 분위기는 어떤가요?"],
    ["Count the number of people in this image."],
    ["What might happen next in this scene?"],
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
        temperature: float,
        top_p: float,
    ):
        if image is None:
            return "⚠️ 이미지를 먼저 업로드해 주세요.", ""
        if not question or not question.strip():
            return "⚠️ 질문을 입력해 주세요.", ""

        cfg = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
        )
        result = engine(image, question.strip(), gen_cfg=cfg)
        meta = f"⏱️ {result['elapsed']:.2f}s · max_new={cfg.max_new_tokens} · T={cfg.temperature} · top_p={cfg.top_p}"
        return result["answer"], meta

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
                    temperature = gr.Slider(
                        0.1, 1.5, value=0.7, step=0.05, label="temperature"
                    )
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")

                submit_btn = gr.Button("🚀 응답 생성", variant="primary")

            with gr.Column(scale=1):
                answer_out = gr.Textbox(
                    label="🤖 모델 응답", lines=12, interactive=False
                )
                meta_out = gr.Markdown("")

        submit_btn.click(
            fn=predict,
            inputs=[image_in, question_in, max_new_tokens, temperature, top_p],
            outputs=[answer_out, meta_out],
        )
        question_in.submit(
            fn=predict,
            inputs=[image_in, question_in, max_new_tokens, temperature, top_p],
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
