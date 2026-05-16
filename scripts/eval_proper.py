"""표준 benchmark 평가 엔진 — VQAv2 + POPE.

v4 배포 게이트(scripts/eval_gate.py)가 이 모듈의 데이터 로더·평가 루프·metric 을
그대로 재사용한다. 수치 metric 만 쓴다 (eyeballing 0).

평가 데이터셋:
  1. VQAv2 val (lmms-lab/VQAv2, streaming)
     metric: 공식 VQA accuracy = mean( min(matches_in_GT / 3, 1.0) )
  2. POPE test (lmms-lab/POPE, streaming)
     metric: yes/no accuracy, yes-recall/precision/F1, refusal rate
     (POPE = Polling-based Object Probing Evaluation — hallucination 측정 표준)

두 benchmark 모두 학습 데이터와 분리돼 있다:
  - VQAv2: train ⊥ val (공식 분리)
  - POPE : COCO val2014 기반 — 학습 이미지(train2014 계열)와 분리
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Windows cp949 콘솔/파이프에서 한글·em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# IMPORTANT: datasets MUST be imported BEFORE torch on Windows.
# torch 먼저 import 시 pyarrow/datasets 의 C 확장과 DLL 충돌로 segfault 발생.
from datasets import load_dataset  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# 데이터셋 로드
# ──────────────────────────────────────────────────────────────────
def load_vqav2(n: int) -> list[dict]:
    """VQAv2 val streaming → n 개 샘플."""
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    out = []
    for s in ds:
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(
            {
                "image": img,
                "question_id": s["question_id"],
                "question": s["question"],
                "mc_answer": s["multiple_choice_answer"],
                "all_answers": [a["answer"] for a in s["answers"]],
                "answer_type": s["answer_type"],  # 'yes/no', 'number', 'other'
            }
        )
        if len(out) >= n:
            break
    return out


def load_pope(n: int) -> list[dict]:
    """POPE test streaming → n 개 샘플."""
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    out = []
    for s in ds:
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(
            {
                "image": img,
                "id": s["id"],
                "question": s["question"],
                "gt_answer": s["answer"].lower().strip(),
                "category": s["category"],  # 'random', 'popular', 'adversarial'
            }
        )
        if len(out) >= n:
            break
    return out


# ──────────────────────────────────────────────────────────────────
# 추론 헬퍼
# ──────────────────────────────────────────────────────────────────
def generate(model: MiniLLaVA, image: Image.Image, question: str, max_new: int = 20) -> str:
    pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"].to(model.llm.device)
    input_ids, attn = encode_for_inference(model.tokenizer, question)
    input_ids = input_ids.unsqueeze(0).to(model.llm.device)
    attn = attn.unsqueeze(0).to(model.llm.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            max_new_tokens=max_new,
            do_sample=False,  # greedy → deterministic
        )
    return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────
# Metric 계산
# ──────────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    """VQA 공식 normalize: 소문자, 양끝 공백/구두점 제거."""
    s = s.lower().strip()
    # 양끝 따옴표/구두점 제거
    while s and s[-1] in ".,;:!?'\"`":
        s = s[:-1]
    while s and s[0] in "'\"`":
        s = s[1:]
    return s.strip()


def vqa_accuracy(pred: str, all_answers: list[str]) -> float:
    """공식 VQA accuracy = min(# matching / 3, 1.0)."""
    p = normalize_answer(pred)
    matches = sum(1 for a in all_answers if normalize_answer(a) == p)
    return min(matches / 3.0, 1.0)


def pope_predict_yn(pred: str) -> str:
    """POPE 응답에서 첫 yes/no 토큰 추출. 없으면 '?'."""
    p = pred.lower().strip()
    # 처음 30자 안에서 찾기
    head = p[:30]
    if "yes" in head and "no" not in head[: head.find("yes") + 3]:
        return "yes"
    if "no" in head and "yes" not in head[: head.find("no") + 2]:
        return "no"
    if head.startswith("yes"):
        return "yes"
    if head.startswith("no"):
        return "no"
    return "?"


# ──────────────────────────────────────────────────────────────────
# 평가 실행 (모델당 1회)
# ──────────────────────────────────────────────────────────────────
def run_eval(model: MiniLLaVA, label: str, vqav2: list[dict], pope: list[dict]) -> dict:
    print(f"\n{'=' * 72}")
    print(f"  {label} 평가 시작 (VQAv2 {len(vqav2)} + POPE {len(pope)})")
    print(f"{'=' * 72}")

    # VQAv2 (max_new=10, VQA 응답은 보통 단답)
    print(f"\n[{label}] VQAv2 inference ...")
    vqa_results = []
    t0 = time.time()
    for i, s in enumerate(vqav2):
        pred = generate(model, s["image"], s["question"], max_new=10)
        acc = vqa_accuracy(pred, s["all_answers"])
        vqa_results.append(
            {
                "question_id": s["question_id"],
                "question": s["question"],
                "answer_type": s["answer_type"],
                "mc_answer": s["mc_answer"],
                "all_answers": s["all_answers"],
                "pred": pred,
                "vqa_acc": acc,
            }
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] VQAv2 {i + 1}/{len(vqav2)} ({elapsed:.0f}s)")

    # POPE (max_new=5, yes/no 단답)
    print(f"\n[{label}] POPE inference ...")
    pope_results = []
    t0 = time.time()
    for i, s in enumerate(pope):
        pred = generate(model, s["image"], s["question"], max_new=5)
        pred_yn = pope_predict_yn(pred)
        correct = pred_yn == s["gt_answer"]
        pope_results.append(
            {
                "id": s["id"],
                "question": s["question"],
                "gt_answer": s["gt_answer"],
                "category": s["category"],
                "pred": pred,
                "pred_yn": pred_yn,
                "correct": correct,
            }
        )
        if (i + 1) % 15 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] POPE {i + 1}/{len(pope)} ({elapsed:.0f}s)")

    # 집계
    vqa_acc_overall = sum(r["vqa_acc"] for r in vqa_results) / max(1, len(vqa_results))

    # answer_type 별
    by_type = {}
    for r in vqa_results:
        t = r["answer_type"]
        by_type.setdefault(t, []).append(r["vqa_acc"])
    vqa_by_type = {t: sum(xs) / len(xs) for t, xs in by_type.items()}

    # POPE 집계
    pope_acc = sum(1 for r in pope_results if r["correct"]) / max(1, len(pope_results))

    # POPE yes/no recall + precision (binary classification)
    tp = sum(1 for r in pope_results if r["correct"] and r["gt_answer"] == "yes")
    fn = sum(1 for r in pope_results if not r["correct"] and r["gt_answer"] == "yes")
    fp = sum(1 for r in pope_results if not r["correct"] and r["gt_answer"] == "no")
    tn = sum(1 for r in pope_results if r["correct"] and r["gt_answer"] == "no")
    yes_recall = tp / max(1, tp + fn)
    yes_precision = tp / max(1, tp + fp)
    yes_f1 = 2 * yes_precision * yes_recall / max(1e-6, yes_precision + yes_recall)
    refusal_rate = sum(1 for r in pope_results if r["pred_yn"] == "?") / max(1, len(pope_results))

    # POPE category 별
    by_cat = {}
    for r in pope_results:
        c = r["category"]
        by_cat.setdefault(c, []).append(int(r["correct"]))
    pope_by_cat = {c: sum(xs) / len(xs) for c, xs in by_cat.items()}

    summary = {
        "label": label,
        "n_vqav2": len(vqa_results),
        "n_pope": len(pope_results),
        "vqav2_accuracy": vqa_acc_overall,
        "vqav2_by_type": vqa_by_type,
        "pope_accuracy": pope_acc,
        "pope_yes_recall": yes_recall,
        "pope_yes_precision": yes_precision,
        "pope_yes_f1": yes_f1,
        "pope_refusal_rate": refusal_rate,
        "pope_by_category": pope_by_cat,
        "pope_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    print(f"\n[{label}] 집계 완료:")
    print(f"  VQAv2 accuracy:     {summary['vqav2_accuracy'] * 100:.2f}%")
    for t, a in vqa_by_type.items():
        print(f"    by '{t}':          {a * 100:.2f}%")
    print(f"  POPE accuracy:      {summary['pope_accuracy'] * 100:.2f}%")
    print(f"  POPE yes-F1:        {summary['pope_yes_f1']:.3f}")
    print(f"  POPE refusal rate:  {summary['pope_refusal_rate'] * 100:.2f}%")

    return {
        "summary": summary,
        "vqa_results": vqa_results,
        "pope_results": pope_results,
    }
