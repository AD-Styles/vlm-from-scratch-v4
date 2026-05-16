"""v4 Stage 1 — 대규모 비전-언어 정렬(alignment) 데이터 다운로드.

v3 의 한계 진단: Stage 1 정렬을 5K 캡션으로만 학습 → projector(비전-언어 다리)가
구조적으로 약했음. LLaVA-1.5 는 558K 를 썼다. v4 는 이 격차를 좁힌다.

핵심 아이디어 — Stage 1 확대는 *싸다*:
  - projector 만 학습 (LLM frozen) → step 이 가볍고 빠름
  - 이미지 1장당 캡션이 여러 개 (Flickr30k 5개, COCO 5개) → 같은 다운로드에서
    데이터가 N배. 이미지는 1번만 저장하고 manifest 만 캡션 수만큼 늘린다.

사용:
  # 검사 — 어떤 소스가 가능한지 + 스키마 확인
  python scripts/download_alignment_data.py --inspect

  # 본 다운로드 (규모는 학습 시간 예산에 맞춰 --num-samples 로 조정)
  python scripts/download_alignment_data.py --num-samples 100000 --out data/coco_subset
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import random
import sys
from pathlib import Path

# Windows cp949 콘솔에서 한글 / em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset
from tqdm import tqdm

# 이미지당 여러 instruction 프롬프트를 회전 — 캡션 task 표현 다양화.
CAPTION_PROMPTS = [
    "Describe this image in detail.",
    "What is in this image?",
    "Provide a caption for this image.",
    "이 이미지를 자세히 설명해 주세요.",
    "사진 속에는 무엇이 있나요?",
]

# 정렬 데이터 소스 — 순서대로 시도. COCO 를 먼저 두는 이유: Stage 2 instruction
# (LLaVA-Instruct / KoLLaVA) 이 COCO 이미지 기반이라 도메인이 일치.
DATASET_FALLBACKS = [
    ("Multimodal-Fatima/COCO_captions_train", "train"),
    ("lmms-lab/flickr30k", "test"),
]

IMAGE_KEYS = ("image", "img", "pixel_values", "jpg", "photo")
CAPTION_KEYS = ("caption", "captions", "sentences", "text",
                "description", "sentences_raw")


def detect_columns(keys):
    image_col = next((c for c in IMAGE_KEYS if c in keys), None)
    caption_col = next((c for c in CAPTION_KEYS if c in keys), None)
    return image_col, caption_col


def extract_all_captions(value) -> list[str]:
    """캡션 필드에서 가능한 모든 캡션을 리스트로 추출.

    str / list[str] / list[dict] 등 다양한 스키마를 일반화 — 이미지당 5개의
    캡션을 모두 살려 정렬 데이터를 N배로 확장하는 게 핵심.
    """
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for k in ("raw", "caption", "text", "sentence"):
                    if k in item and isinstance(item[k], str) and item[k].strip():
                        out.append(item[k].strip())
                        break
        return out
    return []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None,
                   help="HF dataset id (parquet). 미지정 시 fallback 순회.")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=100000,
                   help="총 (이미지,캡션) 쌍 상한. 학습 시간 예산에 맞춰 조정.")
    p.add_argument("--captions-per-image", type=int, default=5,
                   help="이미지당 사용할 캡션 수 (데이터셋이 더 적으면 있는 만큼).")
    p.add_argument("--out", type=str, default="data/coco_subset")
    p.add_argument("--inspect", action="store_true",
                   help="저장 없이 소스 가용성 + 스키마만 확인")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def inspect_mode():
    print("[inspect] 정렬 데이터 소스 가용성 + 스키마 확인\n")
    for ds_id, split in DATASET_FALLBACKS:
        print(f"--- {ds_id} (split={split})")
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
            first = next(iter(itertools.islice(ds, 1)))
            keys = list(first.keys())
            img_col, cap_col = detect_columns(keys)
            print(f"    ✅ 로드 가능 — keys: {keys}")
            print(f"    image_col={img_col}, caption_col={cap_col}")
            if cap_col:
                caps = extract_all_captions(first.get(cap_col))
                print(f"    추출된 캡션 {len(caps)}개 (예: {caps[:1]})")
        except Exception as e:
            print(f"    ❌ {type(e).__name__}: {str(e)[:120]}")
        print()


def main():
    args = parse_args()

    if args.inspect:
        inspect_mode()
        return

    candidates = (
        [(args.dataset, args.split or "train")]
        if args.dataset else DATASET_FALLBACKS
    )

    ds_iter = None
    chosen = None
    for ds_id, split in candidates:
        print(f"[try] {ds_id} (split={split}, streaming=True)")
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
            ds_iter = iter(ds)
            chosen = ds_id
            print(f"[ok] using {ds_id}")
            break
        except Exception as e:
            print(f"[fail] {type(e).__name__}: {str(e)[:160]}")

    if ds_iter is None:
        raise RuntimeError("모든 fallback 데이터셋 로드 실패. --dataset 으로 직접 지정.")

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest: list[dict] = []
    image_col = caption_col = None
    n_images = 0

    print(f"[plan] 목표 {args.num_samples} 쌍 "
          f"(이미지당 캡션 최대 {args.captions_per_image}개)")
    pbar = tqdm(total=args.num_samples, desc="export")
    for ex in ds_iter:
        if len(manifest) >= args.num_samples:
            break
        if image_col is None:
            image_col, caption_col = detect_columns(list(ex.keys()))
            if image_col is None or caption_col is None:
                raise RuntimeError(
                    f"image/caption 컬럼 자동 감지 실패. keys={list(ex.keys())}"
                )
            print(f"[info] image_col={image_col}, caption_col={caption_col}")

        image = ex.get(image_col)
        captions = extract_all_captions(ex.get(caption_col))
        if image is None or not captions:
            continue
        captions = captions[: args.captions_per_image]

        # 이미지는 1번만 저장 → 디스크 절약. manifest 만 캡션 수만큼 늘림.
        try:
            img_path = img_dir / f"{n_images:06d}.jpg"
            image.convert("RGB").save(img_path, "JPEG", quality=90)
        except Exception as e:
            print(f"[skip] {e}")
            continue
        n_images += 1

        for cap in captions:
            if len(manifest) >= args.num_samples:
                break
            manifest.append({
                "image": str(img_path).replace("\\", "/"),
                "question": rng.choice(CAPTION_PROMPTS),
                "answer": cap,
            })
            pbar.update(1)
    pbar.close()

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    avg_cap = len(manifest) / max(1, n_images)
    print(f"\n[done] {len(manifest)} 쌍 (이미지 {n_images}장, "
          f"이미지당 평균 {avg_cap:.1f} 캡션) — source: {chosen}")
    print(f"  manifest: {manifest_path}")
    print(f"  images:   {img_dir}")
    if len(manifest) < args.num_samples:
        print(f"  ⚠️ 목표 {args.num_samples} 미달 — 데이터셋 소진. "
              "다른 --dataset 추가 또는 captions-per-image 확인.")


if __name__ == "__main__":
    main()
