"""POPE benchmark 를 train/test 로 분리 — v3 의 test-set self-tuning 한계 해소.

v3 의 POPE 결과 (53.33% untuned / 70.00% tuned) 에서 tuned 점수는 60개 평가 샘플
*안에서* best threshold 를 찾은 결과로, 일반화 보장이 없었습니다. v4 는 train 셋에서
threshold 를 튜닝하고 test 셋에서 측정합니다.

POPE 데이터 소스: lmms-lab/POPE (Hugging Face), category 3종 (random, popular, adversarial).
각 category 별로 train/test 분리 (예: train 80, test 80) → 동일 분포 검증.

사용:
  # 검사 모드 — 데이터 구조 확인 (저장 X)
  python scripts/prepare_pope_split.py --inspect

  # 본 분리 — category 별 train 80 + test 80 (=160) × 3 category = 총 480
  python scripts/prepare_pope_split.py --train-per-category 80 --test-per-category 80

  # 작은 셋 (빠른 반복용)
  python scripts/prepare_pope_split.py --train-per-category 40 --test-per-category 40
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
from PIL import Image
from tqdm import tqdm

POPE_DATASET = "lmms-lab/POPE"
POPE_CONFIGS = ["Default"]  # lmms-lab/POPE 는 단일 config 안에 category 필드로 random/popular/adversarial 구분
POPE_CATEGORIES = ["random", "popular", "adversarial"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/v4_pope")
    p.add_argument("--train-per-category", type=int, default=80)
    p.add_argument("--test-per-category", type=int, default=80)
    p.add_argument("--inspect", action="store_true", help="데이터 구조만 확인 (저장 X)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def inspect_mode():
    print(f"[inspect] {POPE_DATASET} 데이터 구조 확인\n")
    ds = load_dataset(POPE_DATASET, split="test", streaming=True)
    for i, sample in enumerate(itertools.islice(ds, 5)):
        print(f"--- sample {i} ---")
        for k, v in sample.items():
            if isinstance(v, Image.Image):
                print(f"  {k}: <PIL.Image size={v.size}>")
            elif isinstance(v, str) and len(v) > 80:
                print(f"  {k}: {v[:80]}...")
            else:
                print(f"  {k}: {v}")
        print()


def main():
    args = parse_args()

    if args.inspect:
        inspect_mode()
        return

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plan] {POPE_DATASET}")
    print(f"       category {POPE_CATEGORIES}")
    print(f"       train {args.train_per_category} + test {args.test_per_category} per category")
    total = (args.train_per_category + args.test_per_category) * len(POPE_CATEGORIES)
    print(f"       total {total} samples\n")

    print(f"[1/3] POPE streaming + category 분리")
    by_category: dict[str, list[dict]] = {c: [] for c in POPE_CATEGORIES}
    target_per_cat = (args.train_per_category + args.test_per_category) * 2  # 2배 수집 (이미지 누락 대비 여유분)

    ds = load_dataset(POPE_DATASET, split="test", streaming=True)
    pbar = tqdm(total=target_per_cat * len(POPE_CATEGORIES), desc="loading")
    for sample in ds:
        cat = sample.get("category", "").lower()
        if cat not in POPE_CATEGORIES:
            continue
        if len(by_category[cat]) >= target_per_cat:
            continue
        by_category[cat].append(sample)
        pbar.update(1)
        if all(len(v) >= target_per_cat for v in by_category.values()):
            break
    pbar.close()

    for cat, samples in by_category.items():
        print(f"  {cat}: {len(samples)} 수집")
    print()

    print(f"[2/3] train/test 분리 + 이미지 저장")
    rng = random.Random(args.seed)
    train_records: list[dict] = []
    test_records: list[dict] = []

    for cat in POPE_CATEGORIES:
        samples = by_category[cat]
        rng.shuffle(samples)
        train_samples = samples[: args.train_per_category]
        test_samples = samples[
            args.train_per_category : args.train_per_category + args.test_per_category
        ]

        for split_name, split_samples, target_list in [
            ("train", train_samples, train_records),
            ("test", test_samples, test_records),
        ]:
            for i, sample in enumerate(tqdm(split_samples, desc=f"{cat}/{split_name}")):
                img = sample.get("image")
                if img is None:
                    continue
                img_filename = f"{cat}_{split_name}_{i:04d}.jpg"
                img_path = img_dir / img_filename
                if not img_path.exists():
                    img.convert("RGB").save(img_path, "JPEG", quality=90)

                target_list.append({
                    "image": str(img_path).replace("\\", "/"),
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", "").lower(),
                    "category": cat,
                    "split": split_name,
                    "source": "pope",
                })
    print()

    print(f"[3/3] manifest 저장")
    train_path = out_dir / "train.json"
    test_path = out_dir / "test.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    print(f"  train: {len(train_records)} samples → {train_path}")
    print(f"  test:  {len(test_records)} samples → {test_path}")

    # answer 분포 (yes/no 균형 확인)
    for name, records in [("train", train_records), ("test", test_records)]:
        yes = sum(1 for r in records if r["answer"] == "yes")
        no = len(records) - yes
        print(f"  {name} 분포: yes={yes} ({100*yes/max(1,len(records)):.1f}%) "
              f"no={no} ({100*no/max(1,len(records)):.1f}%)")


if __name__ == "__main__":
    main()
