"""HF Hub 에 v4 가중치 push — HF Space 데모가 부팅 시 받아갈 저장소.

업로드 대상 (~85 MB):
  projector.pt                            13.5 MB
  lora_adapter/adapter_config.json        ~1 KB
  lora_adapter/adapter_model.safetensors  ~70 MB
  lora_adapter/README.md                  (PEFT auto)
  README.md                               (간단 model card — 인라인 생성)

저장소: AD-Styles/mini-llava-v4 — space/app.py 의 MODEL_REPO 기본값과 일치.

토큰: HF Hub 에 쓰기 권한이 있는 토큰이 필요하다 (write).
  python scripts/push_v4_weights.py                  # 캐시된 huggingface-cli login 사용
  python scripts/push_v4_weights.py --token hf_xxx   # 토큰 직접 지정
  python scripts/push_v4_weights.py --repo-id me/my-repo
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

# Windows cp949 콘솔/파이프에서 em-dash(—) 등 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from huggingface_hub import HfApi  # noqa: E402

MODEL_CARD = """\
---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - vision-language
  - multimodal
  - llava
  - qlora
---

# Mini-LLaVA v4 — weights

처음부터 조립한 멀티모달 LLM (`vlm-from-scratch-v4`) 의 학습된 가중치.

- **구조**: CLIP-ViT-B/32 (frozen) + 2-layer MLP Projector + Qwen2.5-1.5B-Instruct + LoRA
- **학습**: QLoRA 4-bit NF4 · Stage 1 정렬 → Stage 2 instruction 46K (영문 + 한국어 균형 믹스) · RTX 4060 8GB
- **평가**: 배포 게이트 5/5 GO — VQAv2 36.7%→**56.8%**, POPE 50.0%→**71.8%** (v3→v4, n=400, raw 모델 · wrapper 없음)

## 파일

| 파일 | 설명 |
|---|---|
| `projector.pt` | MultiModalProjector (CLIP 768 → LLM 1536) state_dict |
| `lora_adapter/` | Qwen2.5-1.5B 전 linear layer LoRA 어댑터 (r=16) |

`<image>` 토큰으로 Qwen2.5 내장 `<|image_pad|>` 를 재사용하므로 adapter 에
embedding 군더더기가 없다 (70 MB 전부 LoRA).

## 사용

추론 코드는 [github.com/AD-Styles/vlm-from-scratch-v4](https://github.com/AD-Styles/vlm-from-scratch-v4)
의 `src/` 참고. 데모: HF Space `AD-Styles/mini-llava-v4-demo`.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="AD-Styles/mini-llava-v4")
    p.add_argument("--token", default=None,
                   help="HF write 토큰. 미지정 시 HF_TOKEN env / 캐시된 login.")
    p.add_argument("--projector", default="checkpoints/v4_stage2_qlora/projector.pt")
    p.add_argument("--lora-adapter", default="checkpoints/v4_stage2_qlora/lora_adapter")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    projector = Path(args.projector)
    lora = Path(args.lora_adapter)
    if not projector.exists() or not lora.exists():
        print("[push] 체크포인트 없음 — Stage 2 학습을 먼저 완료하세요.")
        print(f"  projector: {projector} (exists={projector.exists()})")
        print(f"  lora:      {lora} (exists={lora.exists()})")
        sys.exit(1)

    token = (args.token or os.environ.get("HF_TOKEN", "")).strip() or None
    api = HfApi(token=token)
    try:
        me = api.whoami()
        print(f"[push] logged in as: {me.get('name', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"[push] 인증 실패 — --token 또는 `huggingface-cli login` 필요 ({e})")
        sys.exit(1)

    print(f"\n[push] repo 준비: {args.repo_id} (private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    # 1) model card
    print("[push] model card 업로드 ...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add v4 model card",
    )

    # 2) projector.pt
    size_mb = projector.stat().st_size / 1024 / 1024
    print(f"[push] projector.pt 업로드 ({size_mb:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj=str(projector),
        path_in_repo="projector.pt",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add v4 projector (Stage 2)",
    )

    # 3) lora_adapter/
    total_mb = sum(p.stat().st_size for p in lora.glob("*")) / 1024 / 1024
    print(f"[push] lora_adapter/ 업로드 ({total_mb:.1f} MB) ...")
    api.upload_folder(
        folder_path=str(lora),
        path_in_repo="lora_adapter",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add v4 LoRA adapter (all-linear r16)",
    )

    files = sorted(api.list_repo_files(repo_id=args.repo_id, repo_type="model"))
    print(f"\n[push] repo 파일 {len(files)}개:")
    for f in files:
        print(f"  {f}")
    print(f"\n[OK] v4 weights → https://huggingface.co/{args.repo_id}")
    print("  다음: python scripts/deploy_to_hf_space.py --space-id <user>/mini-llava-v4-demo")


if __name__ == "__main__":
    main()
