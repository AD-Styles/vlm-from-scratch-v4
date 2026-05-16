"""Hugging Face Spaces 데모 배포 자동화.

배포되는 구조:
  Space root/
  ├── app.py              ← space/app.py
  ├── requirements.txt    ← space/requirements.txt
  ├── README.md           ← space/README.md (Spaces 메타데이터 포함)
  └── src/                ← src/ (모델 코드)

가중치 (~85 MB) 는 Space 첫 부팅 시 HF Hub (AD-Styles/mini-llava-v4) 에서
자동 다운로드되므로 여기서 업로드하지 않습니다.

사용:
  python scripts/deploy_to_hf_space.py --space-id AD-Styles/mini-llava-v4-demo

옵션:
  --token <hf_xxx>       토큰 직접 지정 (기본: HF_TOKEN env / 캐시된 login)
  --private              비공개 Space
  --space-folder <path>  업로드할 Space 파일 폴더 (기본: space)
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Windows cp949 콘솔에서 한글 / em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from huggingface_hub import HfApi  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--space-id",
        required=True,
        help="HuggingFace Space ID (예: AD-Styles/mini-llava-v4-demo)",
    )
    p.add_argument("--token", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--space-folder", default="space")
    return p.parse_args()


def main():
    args = parse_args()

    space_folder = Path(args.space_folder)
    if not space_folder.exists():
        raise FileNotFoundError(f"Space 폴더 없음: {space_folder}")
    if not (space_folder / "app.py").exists():
        raise FileNotFoundError(f"{space_folder}/app.py 없음")
    if not (space_folder / "README.md").exists():
        raise FileNotFoundError(f"{space_folder}/README.md 없음 (Spaces 메타데이터 필요)")

    src_folder = Path("src")
    if not src_folder.exists():
        raise FileNotFoundError("src/ 폴더 없음 (모델 코드 필수)")

    # 토큰 우선순위: --token > HF_TOKEN env > 캐시 login
    raw_token = args.token or os.environ.get("HF_TOKEN", "")
    token = raw_token.strip() if raw_token else None

    if token:
        source = "--token 인자" if args.token else "HF_TOKEN 환경변수"
        print(f"[init] 토큰 사용처: {source} (길이={len(token)}, prefix={token[:3]!r})")
        if not token.startswith("hf_"):
            print(f"[warn] 토큰이 'hf_' 로 시작하지 않음 — 형식 오류일 수 있음")
        api = HfApi(token=token)
    else:
        print("[init] 토큰 없음 — 캐시된 로그인 시도")
        api = HfApi()

    # 1) Space 생성/확인
    print(f"\n[1/3] Space 생성/확인: {args.space_id} (private={args.private})")
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )

    # 2) 임시 디렉터리에 space/ + src/ 합치기 (Space 는 단일 루트 필요)
    print(f"\n[2/3] 업로드 폴더 준비 (임시)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # space/* → tmp/
        for item in space_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, tmp_path / item.name)

        # src/ → tmp/src/  (단, __pycache__ 제외)
        shutil.copytree(
            "src",
            tmp_path / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        print(f"  업로드할 파일:")
        for f in sorted(tmp_path.rglob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"    {f.relative_to(tmp_path)} ({size_kb:.1f} KB)")

        # 3) 업로드
        print(f"\n[3/3] Space 업로드 중...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.space_id,
            repo_type="space",
            token=token,
        )

    url = f"https://huggingface.co/spaces/{args.space_id}"
    print(f"\n[완료] {url}")
    print(f"\n빌드 진행 상황:")
    print(f"  - Logs 탭: {url}/logs/build")
    print(f"  - 첫 빌드는 5-10분 소요 (pip install + 모델 가중치 ~85MB 다운로드)")
    print(f"  - 'Running' 상태가 되면 데모 사용 가능")


if __name__ == "__main__":
    main()
