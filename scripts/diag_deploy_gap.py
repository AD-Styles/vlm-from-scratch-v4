"""배포 정밀도 갭 진단 — 4-bit base (학습·게이트 config) vs fp32 base (CPU 데모 config).

projector·LoRA·이미지·질문·디코딩(greedy)을 모두 고정하고 base 양자화만 바꿔
답변을 비교한다. 4-bit 가 fp32 보다 확연히 낫다면 HF Space(fp32) 가 v4 의 실제
성능을 못 보여주고 있다는 뜻 → 배포 config 가 문제.

사용:
  python scripts/diag_deploy_gap.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402

PROJECTOR = "checkpoints/v4_stage2_qlora/projector.pt"
LORA = "checkpoints/v4_stage2_qlora/lora_adapter"

# (이미지, 질문, 실제 정답/내용)
TESTS = [
    ("data/coco_subset/images/000000.jpg", "What is in this image?", "사람이 케이크 자름"),
    ("data/coco_subset/images/000001.jpg", "What is the boy doing?", "컴퓨터 사용"),
    ("data/coco_subset/images/000004.jpg", "What animal is the woman holding?", "고양이"),
    ("data/coco_subset/images/000003.jpg", "What room is this?", "주방"),
    ("data/coco_subset/images/000005.jpg", "Describe this image.", "상업용 주방"),
    ("data/coco_subset/images/000001.jpg", "Is the boy wearing headphones?", "yes"),
    ("data/v4_ood_benchmark/images/cartoon_000.jpg", "What is in this image?", "애니 소녀"),
]


def gen(model, image, question, max_new=32):
    dev = model.llm.device
    pv = model.image_processor(image, return_tensors="pt")["pixel_values"].to(dev)
    ids, attn = encode_for_inference(model.tokenizer, question)
    ids = ids.unsqueeze(0).to(dev)
    attn = attn.unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=attn, pixel_values=pv,
                             max_new_tokens=max_new, do_sample=False)
    return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def load_4bit():
    m = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.bfloat16,
                  use_qlora=True, qlora_compute_dtype="bf16", gradient_checkpointing=False)
    m.load_projector(PROJECTOR, map_location="cpu")
    m.load_lora_adapter(LORA)
    m.vision.to("cuda")
    m.projector.to("cuda")
    m.eval()
    return m


def load_fp32(device):
    m = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32,
                  use_qlora=False, gradient_checkpointing=False)
    m.load_projector(PROJECTOR, map_location="cpu")
    m.load_lora_adapter(LORA)
    m.to(device)
    m.eval()
    return m


def run(tag, model):
    print(f"\n{'=' * 64}\n  [{tag}] 추론\n{'=' * 64}")
    rows = []
    for img, q, gt in TESTS:
        t0 = time.time()
        ans = gen(model, Image.open(img).convert("RGB"), q)
        rows.append((Path(img).name, q, gt, ans))
        print(f"  {Path(img).name:18s} | {q[:34]:34s} | GT={gt:12s} | {ans!r} ({time.time()-t0:.0f}s)")
    return rows


def main():
    if not torch.cuda.is_available():
        print("CUDA 필요 (4-bit config 진단)")
        sys.exit(1)

    print("[1] 4-bit base (학습·게이트와 동일 config) 로드 ...")
    m4 = load_4bit()
    r4 = run("4-bit base", m4)
    del m4
    torch.cuda.empty_cache()

    print("\n[2] fp32 base (HF Space CPU 데모와 동일 config) 로드 ...")
    # fp32 1.5B ≈ 6GB — 8GB GPU 에 fit. OOM 시 CPU 폴백.
    try:
        m32 = load_fp32("cuda")
        dev = "cuda"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("  GPU OOM → CPU 로 폴백 (느림)")
        m32 = load_fp32("cpu")
        dev = "cpu"
    r32 = run(f"fp32 base / {dev}", m32)

    print(f"\n{'=' * 64}\n  나란히 비교 — 4-bit (게이트 config) vs fp32 (데모 config)\n{'=' * 64}")
    for (name, q, gt, a4), (_, _, _, a32) in zip(r4, r32):
        flag = "  <<< 다름" if a4.strip().lower() != a32.strip().lower() else ""
        print(f"\n{name} | {q}")
        print(f"  GT    : {gt}")
        print(f"  4-bit : {a4!r}")
        print(f"  fp32  : {a32!r}{flag}")


if __name__ == "__main__":
    main()
