"""Stage 2 LoRA 학습용 Instruction 데이터 다운로드.

HuggingFaceM4/the_cauldron — Stage 2 instruction tuning에 적합한 50종 VQA 데이터셋
모음. parquet 기반이라 신버전 datasets 라이브러리와 완전 호환.

다중 config을 섞어 균형 잡힌 데이터셋 생성:
  - localized_narratives  : 긴 묘사 캡션 (캡셔닝 능력)
  - aokvqa                : 추론 필요한 개방형 답변 (이해 능력)
  - vqav2                 : 짧은 사실 질문 (정밀 답변)

사용:
  # 기본 — 3개 config 균등 분배 10K
  python scripts/download_instruct_data.py --num-samples 10000 --out data/instruct_subset

  # 단일 config 만
  python scripts/download_instruct_data.py --configs localized_narratives --num-samples 5000

  # 커스텀 mix
  python scripts/download_instruct_data.py --configs aokvqa,okvqa,cocoqa --num-samples 8000
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

# Windows cp949 콘솔 / 파이프에서 한글·em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402


DEFAULT_CONFIGS = ["localized_narratives", "aokvqa", "vqav2"]
# 단일 config 폴백 순서 (모든 config 실패 시)
SINGLE_FALLBACKS = ["vqav2", "okvqa", "cocoqa", "visual7w", "aokvqa"]

# the_cauldron 는 VQA 질문 끝에 "Concise answer only." 류의 답변형식 지시문을
# 한 줄 덧붙인다. 평가셋(lmms-lab/VQAv2·POPE)은 지시문 없는 raw 질문을 쓰므로,
# train/eval 질문 분포를 맞추려면 마지막 줄이 지시문이면 제거해야 한다.
_HINT_KEYWORDS = ("answer", "response", "concise", "brief", "short",
                  "word or phrase", "single word", "sentence")


def strip_format_hint(question: str) -> str:
    """질문 끝에 붙은 답변형식 지시문 한 줄을 제거 (the_cauldron 포맷).

    마지막 줄이 짧고(<60자) 지시문 키워드를 포함하면 그 줄을 떼어낸다.
    그렇지 않으면 원본을 그대로 반환 (멀티라인 실제 질문 보호).
    """
    if "\n" not in question:
        return question
    head, _, tail = question.rpartition("\n")
    t = tail.strip().lower()
    if t and len(t) < 60 and any(k in t for k in _HINT_KEYWORDS):
        return head.strip()
    return question


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="comma-separated config 목록. 균등 분배.",
    )
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--out", type=str, default="data/instruct_subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=1)
    p.add_argument(
        "--min-answer-len",
        type=int,
        default=1,
        help="최소 답변 글자수 (빈 답변 제외용). v4: yes/no·짧은 정답을 버리면 "
             "POPE·VQAv2 가 무너지므로 기본 1 (사실상 빈 답변만 필터).",
    )
    return p.parse_args()


def stream_config(config: str, target: int, min_len: int, max_turns: int, img_dir: Path, start_idx: int):
    """단일 config을 streaming으로 받아 (image, manifest_entry) 페어 생성."""
    print(f"\n[config] {config} — target {target} samples")
    try:
        ds = load_dataset(
            "HuggingFaceM4/the_cauldron",
            config,
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"[fail] {config}: {type(e).__name__}: {str(e)[:200]}")
        return [], 0

    entries = []
    skipped_short = 0
    pbar = tqdm(total=target, desc=f"  {config}")
    for sample in ds:
        if len(entries) >= target:
            break

        images = sample.get("images") or []
        texts = sample.get("texts") or []
        if not images or not texts:
            continue

        image = images[0]
        global_idx = start_idx + len(entries)
        try:
            img_path = img_dir / f"{global_idx:06d}.jpg"
            image.convert("RGB").save(img_path, "JPEG", quality=90)
        except Exception as e:
            continue

        added = 0
        for turn in texts[:max_turns]:
            if len(entries) >= target:
                break
            question = strip_format_hint((turn.get("user") or "").strip())
            answer = (turn.get("assistant") or "").strip()
            if not question or not answer:
                continue
            if len(answer) < min_len:
                skipped_short += 1
                continue
            entries.append(
                {
                    "image": str(img_path).replace("\\", "/"),
                    "question": question,
                    "answer": answer,
                    "source": config,
                }
            )
            added += 1
            pbar.update(1)

        if added == 0:
            try:
                img_path.unlink(missing_ok=True)
            except Exception:
                pass
    pbar.close()

    print(f"  → 수집 {len(entries)}, 짧은 답변 필터 {skipped_short}")
    return entries, len(entries)


def main():
    args = parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    per_config = args.num_samples // len(configs)
    print(f"[plan] {len(configs)} configs × {per_config} samples = {per_config * len(configs)} 총합")

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_manifest = []

    for config in configs:
        entries, _ = stream_config(
            config, per_config, args.min_answer_len, args.max_turns, img_dir,
            start_idx=len(all_manifest),
        )
        all_manifest.extend(entries)

    # config 실패 등으로 부족하면 폴백 시도
    if len(all_manifest) < args.num_samples * 0.5:
        print(f"\n[warn] 수집량 부족({len(all_manifest)}). 폴백 시도...")
        for config in SINGLE_FALLBACKS:
            if config in configs:
                continue
            need = args.num_samples - len(all_manifest)
            if need <= 0:
                break
            entries, _ = stream_config(
                config, need, args.min_answer_len, args.max_turns, img_dir,
                start_idx=len(all_manifest),
            )
            all_manifest.extend(entries)

    # 셔플 — config 별로 모여있지 않도록
    rng.shuffle(all_manifest)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, ensure_ascii=False, indent=2)

    # 통계 요약
    from collections import Counter
    src_counter = Counter(e["source"] for e in all_manifest)
    avg_ans_len = sum(len(e["answer"]) for e in all_manifest) / max(1, len(all_manifest))
    print(f"\n[done] 총 {len(all_manifest)} samples → {manifest_path}")
    print(f"        분포: {dict(src_counter)}")
    print(f"        평균 답변 길이: {avg_ans_len:.1f} 글자")
    print(f"        images → {img_dir}")


if __name__ == "__main__":
    main()
