"""배포 정밀도 갭 진단 — 4-bit base (학습·게이트 config) vs fp32 base (CPU 데모 config).

projector·LoRA·이미지·질문·디코딩(greedy)을 모두 고정하고 base 양자화만 바꿔
답변을 비교한다. 4-bit 가 fp32 보다 확연히 낫다면 HF Space(fp32) 가 v4 의 실제
성능을 못 보여주고 있다는 뜻 → 배포 config 가 문제.

사용:
  python scripts/diag_deploy_gap.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import OOD_PROBE_PROMPT  # noqa: E402
from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402
from src.ood_detection import OODDetector  # noqa: E402

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


def probe_entropy(model, image):
    """OOD 프롬프트로 답변 첫 토큰 entropy — 4-bit↔fp32 OOD 신호 전이 점검용.

    OOD 임계값(0.4582)은 4-bit 모델 entropy 로 보정됐다. 데모는 fp32 라,
    같은 이미지의 entropy 가 두 config 에서 얼마나 다른지가 임계값 전이의 근거다.
    """
    dev = model.llm.device
    pv = model.image_processor(image, return_tensors="pt")["pixel_values"].to(dev)
    ids, attn = encode_for_inference(model.tokenizer, OOD_PROBE_PROMPT)
    ids = ids.unsqueeze(0).to(dev)
    attn = attn.unsqueeze(0).to(dev)
    logits = model.first_token_logits(ids, attn, pv)
    return OODDetector.llm_entropy(logits)


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
        image = Image.open(img).convert("RGB")
        ans = gen(model, image, q)
        ent = probe_entropy(model, image)
        rows.append((Path(img).name, q, gt, ans, ent))
        print(f"  {Path(img).name:18s} | {q[:30]:30s} | GT={gt:12s} | {ans!r} "
              f"(ent={ent:.2f}, {time.time()-t0:.0f}s)")
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
    cases = []
    n_same = 0
    ent_deltas = []
    for (name, q, gt, a4, e4), (_, _, _, a32, e32) in zip(r4, r32):
        identical = a4.strip().lower() == a32.strip().lower()
        n_same += int(identical)
        ent_deltas.append(abs(e4 - e32))
        flag = "" if identical else "  <<< 다름"
        print(f"\n{name} | {q}")
        print(f"  GT    : {gt}")
        print(f"  4-bit : {a4!r}  (OOD-probe entropy {e4:.3f})")
        print(f"  fp32  : {a32!r}{flag}  (OOD-probe entropy {e32:.3f})")
        cases.append({
            "image": name, "question": q, "gt": gt,
            "ans_4bit": a4, "ans_fp32": a32, "identical": identical,
            "ood_probe_entropy_4bit": round(e4, 4),
            "ood_probe_entropy_fp32": round(e32, 4),
            "ood_probe_entropy_abs_delta": round(abs(e4 - e32), 4),
        })

    n = len(cases)
    mean_delta = sum(ent_deltas) / max(1, n)
    report = {
        "n_cases": n,
        "fp32_device": dev,
        "answer_agreement": f"{n_same}/{n}",
        "answer_agreement_rate": round(n_same / max(1, n), 4),
        "ood_probe_prompt": OOD_PROBE_PROMPT,
        "ood_probe_entropy_mean_abs_delta": round(mean_delta, 4),
        "cases": cases,
    }
    out_path = Path("eval_results/v4_deploy_gap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{'=' * 64}")
    print(f"  답변 일치: {n_same}/{n}  ·  OOD-probe entropy 평균 차이: "
          f"{mean_delta:.3f} nats")
    print(f"  → 4-bit 보정 OOD 임계값(0.4582)이 fp32 데모로 전이되는지의 근거.")
    print(f"  [저장] {out_path}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
