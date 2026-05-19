"""OOD detection benchmark prep — v3 의 검증 N=2 한계 해소.

v3 의 OODDetector 는 in-dist 1 + OOD 1 = 2 케이스로만 검증되어 임계값 0.5 의
일반화 보장이 부족했습니다. v4 는 케이스 수를 늘려 ROC AUC 로 재보정합니다.

OOD 구성 — v3 의 Pikachu (만화) narrative 와 직결되는 소스 위주:
  - in-distribution : v3/v4 학습·평가용 COCO 이미지 재사용 (label 0)
  - OOD 만화         : 학습 분포 밖 만화/애니메이션 이미지 (label 1)
  - OOD 추상화       : 추상 회화 (label 1)

ImageNet-O 는 의도적으로 제외 — HuggingFace datasets 에 공식 미러가 없어 zip 직접
다운로드 마찰이 크고, v3 의 한계가 "만화/추상화 같은 학습 분포 외" 였으므로 자연
이미지 OOD 인 ImageNet-O 보다 만화/추상화가 narrative 와 더 정합합니다.

데이터 소스는 고정하지 않습니다. 본 스크립트는 후보 HF 데이터셋을 순서대로 시도하고,
모두 실패하면 placeholder manifest 를 만들어 수동 수집 단계로 안내합니다 (라이선스
동의가 필요한 데이터셋을 자동 우회).

사용:
  # 어떤 OOD 소스가 즉시 가능한지 점검
  python scripts/prepare_ood_benchmark.py --inspect

  # 본 prep — 케이스 수는 CLI 인자로 조정 (기본값은 예시)
  python scripts/prepare_ood_benchmark.py \\
    --indist-source data/coco_subset/manifest.json
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

from PIL import Image
from tqdm import tqdm

# OOD 만화 후보 — 순서대로 시도, 먼저 로드되는 것 사용.
CARTOON_DATASET_CANDIDATES = [
    "huggan/anime-faces",
    "jlbaker361/anime",
    "Norod78/cartoon-blip-captions",
]
# OOD 추상화 후보.
ABSTRACT_DATASET_CANDIDATES = [
    "huggan/wikiart",          # style 필드에 abstract 포함
    "Artificio/WikiArt",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/v4_ood_benchmark")
    p.add_argument(
        "--indist-source",
        type=str,
        default=None,
        help="기존 COCO manifest 경로 (이미지 path 재사용). 미지정 시 placeholder.",
    )
    # 케이스 수 — 노트북 부담이 아니라 (추론만) 큐레이션 편의 기준으로 조정.
    # 기본값은 예시일 뿐, ROC AUC 가 안정적으로 나오는 선에서 자유롭게 조정.
    p.add_argument("--n-indist", type=int, default=40)
    p.add_argument("--n-ood-cartoon", type=int, default=30)
    p.add_argument("--n-ood-abstract", type=int, default=30)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _try_load_streaming(candidates: list[str]):
    """후보 데이터셋을 순서대로 streaming 로드 시도 → (name, dataset) or (None, None)."""
    from datasets import load_dataset

    for name in candidates:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            next(iter(itertools.islice(ds, 1)))  # 첫 샘플 접근으로 실제 로드 확인
            return name, load_dataset(name, split="train", streaming=True)
        except Exception as e:
            print(f"    - {name}: 로드 실패 ({type(e).__name__})")
    return None, None


def inspect_mode():
    """어떤 OOD 데이터 소스가 즉시 가능한지 점검."""
    print("[inspect] OOD 후보 데이터셋 로드 가능 여부\n")
    print("만화 후보:")
    name, _ = _try_load_streaming(CARTOON_DATASET_CANDIDATES)
    print(f"  → 사용 가능: {name or '없음 (placeholder + 수동 수집 필요)'}\n")
    print("추상화 후보:")
    name, _ = _try_load_streaming(ABSTRACT_DATASET_CANDIDATES)
    print(f"  → 사용 가능: {name or '없음 (placeholder + 수동 수집 필요)'}")


def _placeholder(prefix: str, n: int, category: str, ood_label: int) -> list[dict]:
    return [
        {
            "image": f"PLACEHOLDER_{prefix}_{i:03d}.jpg",
            "ood_label": ood_label,
            "category": category,
            "source": "placeholder",
        }
        for i in range(n)
    ]


def collect_indist(args) -> list[dict]:
    """기존 COCO manifest 에서 in-dist 케이스 추출 (이미지 복사 X, path 참조)."""
    if not args.indist_source or not Path(args.indist_source).exists():
        print("  ⚠️ in-dist source 없음 — placeholder 생성")
        return _placeholder("indist", args.n_indist, "indist_coco", 0)

    with open(args.indist_source, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(manifest)

    seen: set[str] = set()
    records: list[dict] = []
    for entry in manifest:
        img_path = entry.get("image", "")
        if img_path in seen or not Path(img_path).exists():
            continue
        seen.add(img_path)
        records.append({
            "image": img_path,
            "ood_label": 0,
            "category": "indist_coco",
            "source": "v3_v4_train_subset",
        })
        if len(records) >= args.n_indist:
            break
    return records


def _label_text(ds, sample, *fields) -> str:
    """ClassLabel(정수)·문자열 필드들을 소문자 문자열로 합쳐 반환.

    huggan/wikiart 의 style·genre 는 ClassLabel — sample[field] 가 정수라
    ds.features 로 int→name 복원해야 'abstract' 매칭이 된다.
    """
    feats = getattr(ds, "features", None) or {}
    parts: list[str] = []
    for f in fields:
        v = sample.get(f)
        if v is None:
            continue
        feat = feats.get(f)
        if feat is not None and hasattr(feat, "int2str") and isinstance(v, int):
            try:
                parts.append(str(feat.int2str(v)))
            except Exception:
                parts.append(str(v))
        else:
            parts.append(str(v))
    return " ".join(parts).lower()


def collect_ood(args, candidates: list[str], n: int, category: str,
                prefix: str, img_dir: Path, style_filter: str | None = None):
    """후보 데이터셋에서 OOD 이미지 n 장 수집. 실패 시 placeholder.

    style_filter 가 주어지면 sample 의 style·genre 필드에 해당 문자열이 포함된
    것만 채택 (추상화 필터링용 — wikiart 등은 genre 필드로 라벨).
    """
    if n <= 0:
        return []
    print(f"  [{category}] {n} 케이스 시도 ...")
    name, ds = _try_load_streaming(candidates)
    if ds is None:
        print(f"    → 모든 후보 실패 — placeholder {n} 개 생성 (수동 수집 안내)")
        return _placeholder(prefix, n, category, 1)

    img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for sample in ds:
        if style_filter is not None:
            # style·genre 둘 다 검사 — wikiart 는 ClassLabel(정수)이라 복원 필요.
            label = _label_text(ds, sample, "style", "genre")
            if style_filter not in label:
                continue
        img = sample.get("image")
        if not isinstance(img, Image.Image):
            continue
        img_path = img_dir / f"{prefix}_{len(records):03d}.jpg"
        img.convert("RGB").save(img_path, "JPEG", quality=90)
        records.append({
            "image": str(img_path).replace("\\", "/"),
            "ood_label": 1,
            "category": category,
            "source": name,
        })
        if len(records) >= n:
            break
    print(f"    → {len(records)}/{n} 수집 (source: {name})")
    return records


def main():
    args = parse_args()

    if args.inspect:
        inspect_mode()
        return

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("[plan] OOD benchmark — in-dist + 만화 + 추상화 (ImageNet-O 미사용)\n")

    records: list[dict] = []
    records.extend(collect_indist(args))
    records.extend(collect_ood(
        args, CARTOON_DATASET_CANDIDATES, args.n_ood_cartoon,
        "ood_cartoon", "cartoon", img_dir,
    ))
    records.extend(collect_ood(
        args, ABSTRACT_DATASET_CANDIDATES, args.n_ood_abstract,
        "ood_abstract", "abstract", img_dir, style_filter="abstract",
    ))

    random.Random(args.seed).shuffle(records)

    out_path = out_dir / "manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_indist = sum(1 for r in records if r["ood_label"] == 0)
    n_ood = sum(1 for r in records if r["ood_label"] == 1)
    n_placeholder = sum(1 for r in records if r["source"] == "placeholder")

    print(f"\n[done] {len(records)} 케이스 → {out_path}")
    print(f"  in-dist (label=0): {n_indist}")
    print(f"  OOD (label=1):     {n_ood}")
    if n_placeholder > 0:
        print(f"  ⚠️ placeholder: {n_placeholder} — 해당 소스는 수동 수집 후 manifest 갱신 필요")
    print()
    print("  다음 단계: ROC AUC 분석 스크립트로 OODDetector 임계값 재보정")


if __name__ == "__main__":
    main()
