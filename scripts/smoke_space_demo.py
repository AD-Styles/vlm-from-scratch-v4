"""배포된 HF Space (Mini-LLaVA v4 데모) 라이브 스모크 테스트.

학습/평가와 별개로, 실제 배포된 데모가 다양한 입력에 어떻게 응답하는지
gradio_client 로 점검한다. 영어/한국어 · 사실질문/묘사/yes-no · in-dist/OOD 를 섞어
배포 모델이 실제로 동작하는지 정성 확인한다.

사용:
  python scripts/smoke_space_demo.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Windows cp949 콘솔/파이프에서 한글·em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from gradio_client import Client, handle_file  # noqa: E402

SPACE_ID = "AD-Styles/mini-llava-v4-demo"

# (이미지, 질문, 테스트 의도)
TESTS = [
    ("data/coco_subset/images/000000.jpg", "What is in this image?", "EN 사실"),
    ("data/coco_subset/images/000000.jpg", "이 사진에 무엇이 보이나요?", "KO 사실"),
    ("data/coco_subset/images/000001.jpg", "What is the boy doing?", "EN 행동"),
    ("data/coco_subset/images/000004.jpg", "What animal is the woman holding?", "EN 특정객체"),
    ("data/coco_subset/images/000004.jpg", "이미지를 한국어로 설명해 주세요.", "KO 묘사"),
    ("data/coco_subset/images/000003.jpg", "What room is this?", "EN 장면"),
    ("data/coco_subset/images/000005.jpg", "Describe this image.", "EN 묘사"),
    ("data/coco_subset/images/000001.jpg", "Is the boy wearing headphones?", "yes/no"),
    ("data/v4_ood_benchmark/images/cartoon_000.jpg", "What is in this image?", "OOD 만화"),
    ("data/v4_ood_benchmark/images/abstract_000.jpg", "What do you see in this picture?", "OOD 추상화"),
]


def main():
    print(f"[smoke] connecting to {SPACE_ID} ...")
    c = Client(SPACE_ID)
    results = []
    for i, (img, q, intent) in enumerate(TESTS):
        t0 = time.time()
        try:
            ans, _meta = c.predict(
                handle_file(img), q, 48, 0.7, 0.9, api_name="/predict"
            )
        except Exception as e:  # noqa: BLE001
            ans = f"ERROR: {type(e).__name__}: {e}"
        dt = time.time() - t0
        print(f"\n[{i + 1}/{len(TESTS)}] ({intent}) {Path(img).name}")
        print(f"  Q: {q}")
        print(f"  A: {ans}")
        print(f"  ({dt:.0f}s)")
        results.append({
            "image": img, "question": q, "intent": intent,
            "answer": ans, "elapsed_s": round(dt, 1),
        })
    out = Path("eval_results/v4_space_smoke.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[smoke] saved → {out}")


if __name__ == "__main__":
    main()
