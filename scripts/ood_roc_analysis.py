"""v4 OOD ROC 분석 + 가중치 최적화 — v3 회고록 숙제 #3 완수.

v3 의 OODDetector 는 in-distribution 1 장 + OOD 1 장 = N=2 케이스로만 검증되어
임계값 0.5 의 일반화 보장이 없었다. v4 는 100 케이스 벤치마크
(in-dist COCO 40 + 만화 30 + 추상화 30) 위에서 ROC 를 측정하고,
(1) CLIP·entropy 결합 가중치와 (2) 판정 임계값을 함께 재보정한다.

OOD score (src/ood_detection.py 와 동일):
  ood_score = w_clip · clip_signal + (1 - w_clip) · entropy_signal
    - clip_signal    : 학습 분포 카테고리와의 CLIP image-text 유사도가 낮을수록 ↑
    - entropy_signal : 답변 첫 토큰 분포의 entropy 가 높을수록 ↑

산출:
  - clip-only / entropy-only / 결합 score 의 ROC AUC (signal ablation)
  - 가중치 w_clip 스윕 → 결합 AUC 최대화 지점
  - 5-fold stratified CV — 보정셋에 과적합되지 않은 정직한 일반화 AUC 추정
  - 최적 가중치에서 Youden's J 임계값
  → eval_results/v4_ood_roc.json

사용:
  python scripts/ood_roc_analysis.py \\
    --projector checkpoints/v4_stage2_qlora/projector.pt \\
    --lora-adapter checkpoints/v4_stage2_qlora/lora_adapter
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

# Windows cp949 콘솔에서 한글 / em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402
from src.ood_detection import OODDetector  # noqa: E402

# OODDetector 의 내부 신호 변환 상수 — src/ood_detection.py 와 일치해야 함.
CLIP_SIM_HI = 0.30   # 이 이상이면 "분포 안" → clip_signal 0
CLIP_SIM_LO = 0.15   # (0.30 - sim) / 0.15 로 0~1 매핑
MAX_ENTROPY_NATS = 8.0
W_SWEEP_STEP = 0.05  # 가중치 스윕 간격


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str,
                   default="data/v4_ood_benchmark/manifest.json")
    p.add_argument("--projector", type=str,
                   default="checkpoints/v4_stage2_qlora/projector.pt")
    p.add_argument("--lora-adapter", type=str,
                   default="checkpoints/v4_stage2_qlora/lora_adapter")
    p.add_argument("--out", type=str, default="eval_results/v4_ood_roc.json")
    p.add_argument(
        "--prompt",
        type=str,
        default="What is in this image?",
        help="entropy 신호용 질문. OOD 벤치마크는 질문이 없어 고정 프롬프트 사용.",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_v4_model(projector_path: str, lora_path: str) -> MiniLLaVA:
    """eval_gate.py 와 동일한 v4 raw 모델 로드 (QLoRA 4-bit + projector + LoRA)."""
    print("[load] v4 모델 (QLoRA 4-bit + projector + LoRA) ...")
    model = MiniLLaVA(
        freeze_vision=True,
        freeze_llm=True,
        torch_dtype=torch.bfloat16,
        use_qlora=True,
        qlora_compute_dtype="bf16",
        gradient_checkpointing=False,  # 추론 — checkpointing 불필요
    )
    model.load_projector(projector_path, map_location="cpu")
    model.load_lora_adapter(lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.vision.to(device)       # 4-bit llm 은 이미 cuda (device_map)
    model.projector.to(device)
    model.eval()
    return model


@torch.no_grad()
def first_token_logits(model: MiniLLaVA, image: Image.Image, question: str) -> torch.Tensor:
    """이미지+질문 → 답변 첫 토큰 분포 logits [vocab]. 전처리 후 모델에 위임.

    splice·forward 본체는 MiniLLaVA.first_token_logits 한 곳에만 둔다 — 데모
    (src/infer.py)와 이 보정 스크립트가 *문자 그대로 같은 코드* 로 entropy 를
    계산해야 보정 임계값이 양쪽에서 유효하다.
    """
    device = model.llm.device
    pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"].to(device)
    input_ids, attn = encode_for_inference(model.tokenizer, question)
    input_ids = input_ids.unsqueeze(0).to(device)
    attn = attn.unsqueeze(0).to(device)
    return model.first_token_logits(input_ids, attn, pixel_values)


# ──────────────────────────────────────────────────────────────────
# ROC / AUC — sklearn 의존 없이 직접 구현 (n=100, O(n²) 충분)
# ──────────────────────────────────────────────────────────────────
def roc_auc(scores: list[float], labels: list[int]) -> float:
    """rank 기반 AUC (Mann-Whitney U). label 1 = OOD(positive), 0 = in-dist.

    AUC = P(무작위 OOD score > 무작위 in-dist score), tie 는 0.5.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def combine(clip: list[float], entropy: list[float], w: float) -> list[float]:
    """결합 score = w·clip_signal + (1-w)·entropy_signal."""
    return [w * c + (1.0 - w) * e for c, e in zip(clip, entropy)]


def weight_sweep(clip: list[float], entropy: list[float], labels: list[int]):
    """w_clip 을 0~1 스윕하며 결합 AUC 측정 → (rows, best)."""
    rows, w = [], 0.0
    while w <= 1.0 + 1e-9:
        rows.append({"w_clip": round(w, 2),
                     "auc": round(roc_auc(combine(clip, entropy, w), labels), 4)})
        w += W_SWEEP_STEP
    best = max(rows, key=lambda r: r["auc"])
    return rows, best


def stratified_folds(labels: list[int], k: int, seed: int) -> list[list[int]]:
    """label 비율을 유지한 채 인덱스를 k 개 fold 로 분할."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        by_label.setdefault(l, []).append(i)
    folds: list[list[int]] = [[] for _ in range(k)]
    for idxs in by_label.values():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        for j, i in enumerate(shuffled):
            folds[j % k].append(i)
    return folds


def cross_val_auc(clip: list[float], entropy: list[float], labels: list[int],
                  k: int, seed: int):
    """k-fold CV — 각 fold 마다 train 으로 w 튜닝, held-out test 로 AUC 측정.

    보정셋(전체 100) 에 w 를 튜닝한 뒤 같은 100 에서 AUC 를 읽으면 낙관 편향이
    생긴다. CV 는 "튜닝에 안 쓴" fold 에서만 AUC 를 재 → 정직한 일반화 추정.
    """
    folds = stratified_folds(labels, k, seed)
    per_fold = []
    for f in range(k):
        test_idx = set(folds[f])
        train_idx = [i for i in range(len(labels)) if i not in test_idx]
        tr_clip = [clip[i] for i in train_idx]
        tr_ent = [entropy[i] for i in train_idx]
        tr_lbl = [labels[i] for i in train_idx]
        _, best = weight_sweep(tr_clip, tr_ent, tr_lbl)  # train 에서 w 튜닝
        w = best["w_clip"]
        te_score = combine([clip[i] for i in folds[f]],
                            [entropy[i] for i in folds[f]], w)
        te_auc = roc_auc(te_score, [labels[i] for i in folds[f]])
        per_fold.append({"fold": f, "tuned_w_clip": w,
                         "test_auc": round(te_auc, 4)})
    mean_auc = round(sum(p["test_auc"] for p in per_fold) / k, 4)
    return mean_auc, per_fold


def metrics_at(scores: list[float], labels: list[int], t: float) -> dict:
    """임계값 t (is_ood = score > t) 에서 정확도 / TPR / FPR / precision / F1."""
    pos_total = sum(1 for l in labels if l == 1)
    neg_total = sum(1 for l in labels if l == 0)
    tp = sum(1 for s, l in zip(scores, labels) if s > t and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s > t and l == 0)
    tn = neg_total - fp
    tpr = tp / max(1, pos_total)
    fpr = fp / max(1, neg_total)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * tpr / max(1e-9, precision + tpr)
    return {
        "threshold": round(t, 4),
        "accuracy": round((tp + tn) / max(1, pos_total + neg_total), 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "youden_j": round(tpr - fpr, 4),
    }


def best_threshold(scores: list[float], labels: list[int]) -> dict:
    """모든 후보 임계값 중 Youden's J 최대 지점."""
    cand = sorted(set(scores))
    mids = [(cand[i] + cand[i + 1]) / 2 for i in range(len(cand) - 1)]
    rows = [metrics_at(scores, labels, t) for t in [-1e-9] + mids + [1.0]]
    return max(rows, key=lambda r: (r["youden_j"], r["accuracy"]))


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("[ood-roc] CUDA 없음 — QLoRA 4-bit 추론은 GPU 필요.")
        sys.exit(1)

    proj_path, lora_path = Path(args.projector), Path(args.lora_adapter)
    if not proj_path.exists() or not lora_path.exists():
        print("[ood-roc] 체크포인트 없음 — Stage 2 학습을 먼저 완료하세요.")
        sys.exit(1)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    n_indist = sum(1 for r in manifest if r["ood_label"] == 0)
    n_ood = sum(1 for r in manifest if r["ood_label"] == 1)

    print("=" * 72)
    print(f"  v4 OOD ROC 분석 — {len(manifest)} 케이스 "
          f"(in-dist {n_indist} + OOD {n_ood})")
    print("=" * 72)

    model = load_v4_model(args.projector, args.lora_adapter)
    print("[load] OODDetector (CLIP ViT-B/32) ...")
    detector = OODDetector(device="cuda")

    print(f"\n[score] {len(manifest)} 케이스 채점 (prompt={args.prompt!r}) ...")
    cases: list[dict] = []
    t0 = time.time()
    for i, rec in enumerate(manifest):
        image = Image.open(rec["image"]).convert("RGB")
        logits = first_token_logits(model, image, args.prompt)

        clip_sim, clip_match = detector.clip_similarity(image)
        entropy = detector.llm_entropy(logits)
        clip_signal = max(0.0, min(1.0, (CLIP_SIM_HI - clip_sim) / CLIP_SIM_LO))
        entropy_signal = max(0.0, min(1.0, entropy / MAX_ENTROPY_NATS))

        cases.append({
            "image": rec["image"],
            "category": rec["category"],
            "ood_label": rec["ood_label"],
            "clip_max_sim": round(clip_sim, 4),
            "clip_match": clip_match,
            "clip_signal": round(clip_signal, 4),
            "llm_entropy": round(entropy, 4),
            "entropy_signal": round(entropy_signal, 4),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(manifest)} ({time.time() - t0:.0f}s)")

    labels = [c["ood_label"] for c in cases]
    clip = [c["clip_signal"] for c in cases]
    entropy = [c["entropy_signal"] for c in cases]

    # ── signal ablation ──────────────────────────────────────────────
    auc_clip = round(roc_auc(clip, labels), 4)
    auc_entropy = round(roc_auc(entropy, labels), 4)
    auc_w06 = round(roc_auc(combine(clip, entropy, 0.6), labels), 4)  # v3 기본 가중치

    # ── 가중치 스윕 → 최적 w_clip ───────────────────────────────────
    sweep, best_w = weight_sweep(clip, entropy, labels)
    w_star = best_w["w_clip"]

    # ── 5-fold CV — 정직한 일반화 추정 ──────────────────────────────
    cv_auc, cv_folds = cross_val_auc(clip, entropy, labels, args.cv_folds, args.seed)

    # ── 최적 가중치에서 결합 score + 임계값 ─────────────────────────
    combined_star = combine(clip, entropy, w_star)
    for c, s in zip(cases, combined_star):
        c["ood_score"] = round(s, 4)
    opt_thr = best_threshold(combined_star, labels)

    # 카테고리별 AUC (최적 가중치 기준, vs in-dist)
    auc_by_cat = {}
    for cat in ("ood_cartoon", "ood_abstract"):
        sub = [(c["ood_score"], c["category"]) for c in cases
               if c["category"] in (cat, "indist_coco")]
        auc_by_cat[cat] = round(
            roc_auc([s for s, _ in sub], [1 if g == cat else 0 for _, g in sub]), 4)

    final_cfg = {
        "weight_clip": w_star,
        "weight_entropy": round(1.0 - w_star, 2),
        "threshold": opt_thr["threshold"],
    }
    report = {
        "n_cases": len(cases), "n_indist": n_indist, "n_ood": n_ood,
        "prompt": args.prompt,
        "auc_single": {"clip_only": auc_clip, "entropy_only": auc_entropy},
        "auc_combined_default_w0.6": auc_w06,
        "weight_sweep": sweep,
        "optimal_weight": best_w,
        "cross_val": {"k": args.cv_folds, "mean_test_auc": cv_auc, "folds": cv_folds},
        "auc_by_category_at_optimal": auc_by_cat,
        "final_config": final_cfg,
        "threshold_at_optimal_weight": opt_thr,
        "cases": cases,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ─────── 콘솔 요약 ───────
    print("\n" + "=" * 72)
    print("  결과")
    print("=" * 72)
    print("\n[1] signal ablation — ROC AUC (1.0=완벽 분리, 0.5=랜덤)")
    print(f"  CLIP-only            : {auc_clip:.3f}")
    print(f"  entropy-only         : {auc_entropy:.3f}")
    print(f"  결합 (v3 가중치 0.6) : {auc_w06:.3f}")

    print("\n[2] 가중치 재보정 — w_clip 스윕")
    for r in sweep:
        mark = "  ← 최적" if r["w_clip"] == w_star else ""
        print(f"  w_clip={r['w_clip']:.2f}  AUC={r['auc']:.4f}{mark}")
    print(f"  → 최적 w_clip = {w_star:.2f}, 결합 AUC = {best_w['auc']:.4f}")

    print(f"\n[3] {args.cv_folds}-fold CV — 정직한 일반화 추정")
    for p in cv_folds:
        print(f"  fold {p['fold']}: w*={p['tuned_w_clip']:.2f}  test AUC={p['test_auc']:.4f}")
    print(f"  → CV 평균 test AUC = {cv_auc:.4f}")

    print("\n[4] 카테고리별 AUC (최적 가중치, vs in-dist)")
    print(f"  만화   : {auc_by_cat['ood_cartoon']:.3f}")
    print(f"  추상화 : {auc_by_cat['ood_abstract']:.3f}")

    print("\n[5] 최적 가중치에서 임계값 (Youden's J)")
    print(f"  threshold={opt_thr['threshold']:.3f}  acc={opt_thr['accuracy']*100:.1f}%  "
          f"TPR={opt_thr['tpr']*100:.1f}%  FPR={opt_thr['fpr']*100:.1f}%")

    print("\n[권장 src/ood_detection.py 기본값]")
    print(f"  weight_clip={final_cfg['weight_clip']}, "
          f"weight_entropy={final_cfg['weight_entropy']}, "
          f"threshold={final_cfg['threshold']}")
    print(f"\n[저장] {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
