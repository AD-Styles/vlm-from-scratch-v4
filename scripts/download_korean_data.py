"""KoLLaVA-Instruct-150k 한국어 데이터 다운로드 (v4 Stage 2 학습용).

Source: tabtoyou/KoLLaVA-Instruct-150k (DeepL 한국어 번역, CC-BY-NC-4.0)
Format: v2 의 manifest 형식과 동일 ({image, question, answer, source})

특이사항:
  - 데이터셋에 이미지 없음 (filename 만) → COCO 2014 train/val 에서 별도 다운로드
  - <image> 토큰이 conversations 에 이미 포함 → strip 필요 (학습 파이프라인이 자체 추가)
  - 다중 턴 데이터는 첫 턴만 추출

사용:
  # 검사 모드 — 5 샘플 print, 저장 X
  python scripts/download_korean_data.py --inspect

  # 소규모 검증 — 100 샘플로 파이프라인 동작 확인
  python scripts/download_korean_data.py --num-samples 100 --out data/korean_subset_test

  # 전체 다운로드 (4K)
  python scripts/download_korean_data.py --num-samples 4000 --out data/korean_subset
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import random
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Windows cp949 콘솔/파이프에서 em-dash(—) 등 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

DATASET_NAME = "tabtoyou/KoLLaVA-Instruct-150k"

# COCO 2014 공식 URL 패턴 (train2014 우선, val2014 폴백)
COCO_URL_TEMPLATES = [
    "http://images.cocodataset.org/train2014/COCO_train2014_{}.jpg",
    "http://images.cocodataset.org/val2014/COCO_val2014_{}.jpg",
]

# <image> 토큰 + 주변 개행/공백 제거
IMAGE_TOKEN_RE = re.compile(r"\s*<image>\s*\n?|\n?\s*<image>\s*")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=4000)
    p.add_argument("--out", type=str, default="data/korean_subset")
    p.add_argument(
        "--inspect", action="store_true", help="저장 없이 5 샘플만 print"
    )
    p.add_argument("--min-answer-len", type=int, default=4)
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="이미지 동시 다운로드 수 (기본 8)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clean_question(value: str) -> str:
    """human turn value 에서 <image> 토큰과 주변 공백 제거."""
    return IMAGE_TOKEN_RE.sub(" ", value).strip()


def download_image(image_id: str, save_path: Path) -> bool:
    """COCO 2014 train → val 폴백. PIL 로 검증, 성공 시 True."""
    if save_path.exists():
        try:
            with Image.open(save_path) as im:
                im.verify()
            return True
        except Exception:
            save_path.unlink(missing_ok=True)

    img_id = image_id.replace(".jpg", "")
    for tmpl in COCO_URL_TEMPLATES:
        url = tmpl.format(img_id)
        try:
            urllib.request.urlretrieve(url, str(save_path))
            with Image.open(save_path) as im:
                im.verify()
            return True
        except Exception:
            save_path.unlink(missing_ok=True)
            continue
    return False


def parse_sample(sample: dict, min_answer_len: int):
    """KoLLaVA sample → (image_id, question, answer) or None (filtered)."""
    image_id = (sample.get("image") or "").strip()
    convs = sample.get("conversations") or []
    if not image_id or len(convs) < 2:
        return None

    human, gpt = convs[0], convs[1]
    if human.get("from") != "human" or gpt.get("from") != "gpt":
        return None

    question = clean_question(human.get("value", ""))
    answer = (gpt.get("value") or "").strip()

    if not question or not answer or len(answer) < min_answer_len:
        return None
    return image_id, question, answer


def korean_ratio(text: str) -> float:
    """텍스트의 한글 (Hangul) 문자 비율."""
    if not text:
        return 0.0
    hangul = sum(1 for c in text if "가" <= c <= "힣")
    return hangul / len(text)


def inspect_mode():
    print(f"[inspect] {DATASET_NAME} 첫 5 샘플 검사 (저장 X)\n")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    for i, sample in enumerate(itertools.islice(ds, 5)):
        parsed = parse_sample(sample, min_answer_len=4)
        print(f"--- sample {i} ---")
        if parsed is None:
            print("  (FILTERED — 빈 question/answer 또는 짧은 답변)\n")
            continue
        image_id, question, answer = parsed
        print(f"  image: {image_id}")
        print(f"  question ({len(question)}자, 한글 {100*korean_ratio(question):.0f}%):")
        print(f"    {question[:120]}{'...' if len(question) > 120 else ''}")
        print(f"  answer ({len(answer)}자, 한글 {100*korean_ratio(answer):.0f}%):")
        print(f"    {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print()


def main():
    args = parse_args()

    if args.inspect:
        inspect_mode()
        return

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plan] {DATASET_NAME} → {args.num_samples} 샘플")
    print(f"       이미지 → {img_dir} (COCO 2014 train/val 에서 다운로드)")
    print(f"       동시 다운로드: {args.max_workers}\n")

    # 1) KoLLaVA streaming + parsing — 30% 여유로 후보 수집 (다운 실패 대비)
    target_buffer = int(args.num_samples * 1.3)
    print(f"[1/3] KoLLaVA streaming + 파싱 ({target_buffer} 후보 수집)")
    candidates: list[tuple[str, str, str]] = []
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    pbar = tqdm(total=target_buffer, desc="parsing")
    for sample in ds:
        if len(candidates) >= target_buffer:
            break
        parsed = parse_sample(sample, args.min_answer_len)
        if parsed is None:
            continue
        candidates.append(parsed)
        pbar.update(1)
    pbar.close()
    print(f"  → 후보: {len(candidates)}\n")

    # 2) 병렬 이미지 다운로드
    print(f"[2/3] COCO 이미지 다운로드 ({len(candidates)} 개)")
    successful: list[dict] = []

    def task(item: tuple[str, str, str]):
        image_id, question, answer = item
        save_path = img_dir / image_id
        ok = download_image(image_id, save_path)
        return ok, image_id, question, answer, str(save_path)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(task, c) for c in candidates]
        pbar = tqdm(total=len(futures), desc="downloading")
        for f in as_completed(futures):
            ok, image_id, question, answer, path = f.result()
            if ok and len(successful) < args.num_samples:
                successful.append(
                    {
                        "image": path.replace("\\", "/"),
                        "question": question,
                        "answer": answer,
                        "source": "kollava",
                    }
                )
            pbar.update(1)
        pbar.close()

    success_rate = 100 * len(successful) / max(1, len(candidates))
    print(f"  → 성공: {len(successful)}/{len(candidates)} ({success_rate:.1f}%)\n")

    if len(successful) < args.num_samples * 0.8:
        print(f"  ⚠️ 목표의 80% 미달 — 추가 다운로드 또는 다른 소스 검토 필요")

    # 3) Manifest 저장
    print(f"[3/3] Manifest 저장")
    random.Random(args.seed).shuffle(successful)
    successful = successful[: args.num_samples]

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)

    # 통계 검증
    avg_q = sum(len(e["question"]) for e in successful) / max(1, len(successful))
    avg_a = sum(len(e["answer"]) for e in successful) / max(1, len(successful))
    avg_kor_q = sum(korean_ratio(e["question"]) for e in successful) / max(1, len(successful))
    avg_kor_a = sum(korean_ratio(e["answer"]) for e in successful) / max(1, len(successful))

    print(f"  manifest: {manifest_path} ({len(successful)} samples)")
    print(f"  images dir: {img_dir}")
    print(f"  평균 길이 — question: {avg_q:.1f}자, answer: {avg_a:.1f}자")
    print(f"  Korean (Hangul) 비율 — question: {100*avg_kor_q:.1f}%, answer: {100*avg_kor_a:.1f}%")
    print()

    # 검증 경고
    if avg_kor_q < 0.5 or avg_kor_a < 0.5:
        print("  ⚠️ Korean 비율 < 50% — 데이터 품질 검토 필요")
    if avg_a < 20:
        print("  ⚠️ 답변 평균 길이 < 20자 — Yes/No 편향 가능성")


if __name__ == "__main__":
    main()
