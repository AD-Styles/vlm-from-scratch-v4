"""v4 배포 게이트 — raw 모델이 기준을 통과해야만 HF Spaces 배포.

v3 의 실수: 성능 검증 없이 배포부터 → 라이브에서 성능 미달 발견 → 수습.
v4 는 이 패턴을 *절차* 로 차단한다. 학습이 끝난 raw 모델(추론 wrapper 없이)을
표준 benchmark 로 측정하고, 사전에 정한 bar 를 통과할 때만 배포한다.

평가는 wrapper 를 끼우지 않은 **raw 모델** 로만 한다 — v3 에서 wrapper 가 raw 의
약함(11/12 중 8/12 가 router 기여)을 가렸기 때문. 게이트는 모델 자체를 본다.

판정:
  - 통과 → exit code 0 → 배포 스크립트 진행 허용
  - 미달 → exit code 1 → 배포 차단. "한계 분석" 포트폴리오로 정직하게 프레이밍.

사용 (Stage 2 학습 완료 후):
  python scripts/eval_gate.py \\
    --projector checkpoints/v4_stage2_qlora/projector.pt \\
    --lora-adapter checkpoints/v4_stage2_qlora/lora_adapter

배포 스크립트에서 게이트로 사용:
  python scripts/eval_gate.py ... ; if ($?) { python scripts/deploy_to_hf_space.py }
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Windows cp949 콘솔/파이프에서 em-dash(—) 등 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src
sys.path.insert(0, str(Path(__file__).resolve().parent))         # eval_proper

# eval_proper 의 데이터 로더 / 평가 루프 재사용 (import 시 datasets→torch 순서 처리됨).
from eval_proper import load_pope, load_vqav2, run_eval  # noqa: E402

import torch  # noqa: E402

from src.model import MiniLLaVA  # noqa: E402

# ──────────────────────────────────────────────────────────────────
# 배포 게이트 기준 (deployment bars)
# ──────────────────────────────────────────────────────────────────
# 아래 수치는 "결과 예측" 이 아니라 "배포를 정당화하는 최소 결정 기준" 이다.
# 게이트에는 명시적 임계값이 반드시 필요하다. 근거:
#   - v3 raw baseline: VQAv2 36.67%, POPE 50.00% (전부 "yes" → 사실상 랜덤)
#   - bar 는 v3 raw 보다, 그리고 랜덤보다 "분명히 위" 로 잡는다.
# 미달 = v3 대비 의미있는 개선이 아님 → 배포 대신 한계 분석으로 프레이밍.
VQAV2_MIN = 0.45      # v3 raw 36.67% 대비 +8%p 이상
POPE_ACC_MIN = 0.65   # v3 raw 50%(랜덤) 대비 +15%p 이상
POPE_F1_MIN = 0.55    # yes/no 한쪽 쏠림이면 F1 가 낮게 나옴
POPE_SKEW_MAX = 0.85  # 예측의 한 클래스 비율 상한 — v3 의 "전부 yes" 퇴행 탐지
REFUSAL_MAX = 0.15    # yes/no 로 못 답한(?) 비율 상한


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--projector", type=str,
                   default="checkpoints/v4_stage2_qlora/projector.pt")
    p.add_argument("--lora-adapter", type=str,
                   default="checkpoints/v4_stage2_qlora/lora_adapter")
    p.add_argument("--n-vqav2", type=int, default=400,
                   help="VQAv2 평가 샘플 수. 기본 400 — 표본 오차를 줄여 "
                        "게이트 GO/NO-GO 판정을 안정화하는 값.")
    p.add_argument("--n-pope", type=int, default=400)
    p.add_argument("--out", type=str, default="eval_results/v4_gate_report.json")
    return p.parse_args()


def load_v4_raw(projector_path: str, lora_path: str) -> MiniLLaVA:
    """v4 raw 모델 로드 — QLoRA 4-bit base + 학습된 projector + LoRA adapter.

    학습과 동일하게 4-bit base 로 로드해 추론 충실도(fidelity)를 맞춘다.
    """
    print("[gate] v4 raw 모델 로드 (QLoRA 4-bit + projector + LoRA) ...")
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


def check(name: str, value: float, bar: float, mode: str) -> tuple[bool, str]:
    """mode='min': value>=bar 통과 / mode='max': value<=bar 통과."""
    ok = value >= bar if mode == "min" else value <= bar
    op = "≥" if mode == "min" else "≤"
    tag = "PASS" if ok else "FAIL"
    return ok, f"  [{tag}] {name}: {value:.4f} (기준 {op} {bar})"


def main():
    args = parse_args()

    proj_path = Path(args.projector)
    lora_path = Path(args.lora_adapter)
    if not proj_path.exists() or not lora_path.exists():
        print("[gate] ❌ 체크포인트 없음 — Stage 2 학습을 먼저 완료하세요.")
        print(f"       projector: {proj_path} (exists={proj_path.exists()})")
        print(f"       lora:      {lora_path} (exists={lora_path.exists()})")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[gate] ⚠️  CUDA 없음 — QLoRA 4-bit 추론은 GPU 필요.")
        sys.exit(1)

    print("=" * 72)
    print("  v4 배포 게이트 — raw 모델 성능 검증")
    print("=" * 72)

    t0 = time.time()
    print(f"\n[1/3] benchmark 데이터 로드 (VQAv2 {args.n_vqav2} + POPE {args.n_pope}) ...")
    vqav2 = load_vqav2(args.n_vqav2)
    pope = load_pope(args.n_pope)
    print(f"  loaded ({time.time() - t0:.0f}s)")

    print("\n[2/3] v4 raw 모델 평가 ...")
    model = load_v4_raw(args.projector, args.lora_adapter)
    results = run_eval(model, "v4-raw", vqav2, pope)
    summary = results["summary"]

    # POPE 예측 분포 — "전부 yes" 퇴행 직접 탐지
    yn = [r["pred_yn"] for r in results["pope_results"]]
    n = max(1, len(yn))
    yes_ct, no_ct, q_ct = yn.count("yes"), yn.count("no"), yn.count("?")
    skew = max(yes_ct, no_ct) / n
    refusal = q_ct / n

    # ─────── [3/3] 게이트 판정 ───────
    print("\n[3/3] 게이트 판정")
    print("-" * 72)
    checks = [
        check("VQAv2 정답률", summary["vqav2_accuracy"], VQAV2_MIN, "min"),
        check("POPE 정답률", summary["pope_accuracy"], POPE_ACC_MIN, "min"),
        check("POPE yes-F1", summary["pope_yes_f1"], POPE_F1_MIN, "min"),
        check("POPE 예측 쏠림 (퇴행 탐지)", skew, POPE_SKEW_MAX, "max"),
        check("yes/no 미응답(?) 비율", refusal, REFUSAL_MAX, "max"),
    ]
    for ok, line in checks:
        print(line)
    print("-" * 72)

    all_pass = all(ok for ok, _ in checks)
    verdict = "GO — 배포 허용" if all_pass else "NO-GO — 배포 차단"
    print(f"\n  최종 판정: {'✅' if all_pass else '🔴'} {verdict}")
    if not all_pass:
        print("  → raw 모델이 기준 미달. HF Spaces 배포 대신 v3 처럼")
        print("     '한계 분석' 포트폴리오로 정직하게 프레이밍할 것.")
    print("=" * 72)

    # 리포트 저장
    report = {
        "verdict": "GO" if all_pass else "NO-GO",
        "bars": {
            "vqav2_min": VQAV2_MIN, "pope_acc_min": POPE_ACC_MIN,
            "pope_f1_min": POPE_F1_MIN, "pope_skew_max": POPE_SKEW_MAX,
            "refusal_max": REFUSAL_MAX,
        },
        "measured": {
            "vqav2_accuracy": summary["vqav2_accuracy"],
            "vqav2_by_type": summary["vqav2_by_type"],
            "pope_accuracy": summary["pope_accuracy"],
            "pope_yes_f1": summary["pope_yes_f1"],
            "pope_pred_skew": skew,
            "pope_refusal_rate": refusal,
            "pope_pred_dist": {"yes": yes_ct, "no": no_ct, "?": q_ct},
            "pope_by_category": summary["pope_by_category"],
        },
        "n_vqav2": args.n_vqav2,
        "n_pope": args.n_pope,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  리포트 저장: {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
