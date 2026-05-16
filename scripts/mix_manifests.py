"""여러 manifest 를 하나로 합치는 헬퍼 — v4 Stage 2 학습용 mix 생성.

v4 Stage 2 사용 예 (영문 instruct manifest + 한국어 manifest 결합):
  python scripts/mix_manifests.py \\
    --inputs data/instruct_subset/manifest.json data/korean_subset/manifest.json \\
    --output data/v4_stage2_mix/manifest.json

옵션:
  --inputs              합칠 manifest 경로들 (공백으로 구분, 1개 이상)
  --output              결과 manifest 저장 경로
  --weights             각 input 의 비중 (선택, 미지정 시 전체 사용)
                        예: --weights 1.0 0.5  → 두 번째 manifest 의 50% 만 사용
  --seed                셔플 시드 (기본 42)
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Windows cp949 콘솔/파이프에서 em-dash(—) 등 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="합칠 manifest.json 경로들 (공백으로 구분)",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="각 input 비중 (0.0~1.0). 미지정 시 모두 1.0 (전체 사용)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def korean_ratio(text: str) -> float:
    if not text:
        return 0.0
    hangul = sum(1 for c in text if "가" <= c <= "힣")
    return hangul / len(text)


def main():
    args = parse_args()

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"manifest 없음: {p}")

    weights = args.weights or [1.0] * len(inputs)
    if len(weights) != len(inputs):
        raise ValueError(
            f"--weights 길이 ({len(weights)}) ≠ --inputs 길이 ({len(inputs)})"
        )

    rng = random.Random(args.seed)

    print(f"[mix] {len(inputs)} manifests 합치기:")
    combined: list[dict] = []
    per_input_stats = []

    for path, weight in zip(inputs, weights):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        target_count = int(len(data) * weight)
        rng.shuffle(data)  # 같은 input 내에서도 셔플
        sampled = data[:target_count]
        combined.extend(sampled)
        per_input_stats.append((path.name, len(data), len(sampled), weight))
        print(f"  {path}: {len(data)} samples × weight {weight} = {len(sampled)} 사용")

    # 전체 셔플 (source 가 섞이도록)
    rng.shuffle(combined)

    # 저장
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # 통계
    print()
    print(f"[done] 총 {len(combined)} samples → {out}")
    src_counter = Counter(e.get("source", "unknown") for e in combined)
    print(f"  source 분포: {dict(src_counter)}")

    avg_q = sum(len(e["question"]) for e in combined) / max(1, len(combined))
    avg_a = sum(len(e["answer"]) for e in combined) / max(1, len(combined))
    print(f"  평균 길이 — Q: {avg_q:.1f}자, A: {avg_a:.1f}자")

    avg_kor_q = sum(korean_ratio(e["question"]) for e in combined) / max(1, len(combined))
    avg_kor_a = sum(korean_ratio(e["answer"]) for e in combined) / max(1, len(combined))
    print(f"  Korean (Hangul) 비율 — Q: {100*avg_kor_q:.1f}%, A: {100*avg_kor_a:.1f}%")


if __name__ == "__main__":
    main()
