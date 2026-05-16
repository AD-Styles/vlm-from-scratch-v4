"""v4 — Qwen2.5-1.5B + QLoRA 4-bit + LoRA 가 RTX 4060 8GB 에 fit 하는지 측정.

LLM 1.5B + 4-bit NF4 양자화 + bf16 compute + gradient checkpointing 조합이
8GB VRAM 안에 들어가는지, 1 step forward+backward 가 얼마나 걸리는지 측정한다.

실제 학습 시작 전에 반드시 실행 — 학습 도중 OOM 으로 수 시간을 날리는 것보다
수 분짜리 smoke test 가 훨씬 저렴하다.

데이터 의존성 없음: 외부 manifest/이미지 대신 합성 배치(random tensor)를 쓴다.
메모리·step time 측정에는 토큰/픽셀 내용이 무관하고, 길이만 실제 학습 상한
(max_text_length)에 맞추면 worst-case 를 보수적으로 잡을 수 있다.

사용 (Stage 2 시뮬레이션 — LoRA + projector 동시 학습):
  python scripts/vram_test_v4.py --use-lora

사용 (Stage 1 시뮬레이션 — projector only, 4-bit LLM frozen):
  python scripts/vram_test_v4.py

성공 기준:
  peak VRAM < 7.5 GB (다른 process 여유 ≥ 0.5 GB)
"""
from __future__ import annotations

import argparse
import gc
import io
import sys
import time
from pathlib import Path

# Windows cp949 콘솔에서 한글 / em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from src.config import IGNORE_INDEX, LORA_TARGET_MODULES, VISION_MODEL  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="실제 학습 effective batch 계산용. test 자체는 step 단위 측정.")
    p.add_argument("--max-text-length", type=int, default=512,
                   help="합성 배치의 텍스트 길이 — 실제 학습 상한에 맞춰 worst-case 측정.")
    p.add_argument("--n-steps", type=int, default=3,
                   help="warmup 1 + 측정 N-1")
    p.add_argument("--use-lora", action="store_true",
                   help="Stage 2 시뮬레이션 — LoRA + projector 동시 학습 (없으면 Stage 1)")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="gradient checkpointing 비활성 — 메모리 효과 비교용")
    return p.parse_args()


def gb(bytes_):
    return bytes_ / (1024 ** 3)


def make_synthetic_batch(model, batch_size, seq_len, device):
    """외부 데이터 없이 worst-case 길이의 합성 배치 생성.

    - input_ids: 무작위 토큰 + 샘플당 정확히 1개의 <image> 토큰 (위치 3)
    - pixel_values: 무작위 [3, 224, 224] (CLIP-ViT-B/32 입력 규격)
    메모리/step time 측정용 — 내용은 무관, 길이만 max_text_length 에 맞춘다.
    """
    # 작은 id 범위(0~999)면 어떤 vocab 에서도 유효한 임베딩 인덱스.
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    input_ids[:, 3] = model.image_token_id  # 샘플당 <image> 토큰 1개
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :10] = IGNORE_INDEX  # 앞부분(프롬프트) 마스킹 흉내
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "pixel_values": pixel_values.to(device),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 사용 불가 — QLoRA 는 GPU 필수")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()

    use_gc = not args.no_gradient_checkpointing
    stage = "Stage 2 (LoRA + projector)" if args.use_lora else "Stage 1 (projector only)"
    print("=" * 70)
    print("v4 — VRAM smoke test")
    print(f"  vision   : {VISION_MODEL}")
    print("  llm      : Qwen/Qwen2.5-1.5B-Instruct (4-bit NF4)")
    print("  dtype    : bfloat16 compute")
    print(f"  seq_len  : {args.max_text_length} (+ 49 image patches)")
    print(f"  bsz      : {args.batch_size} (grad_accum={args.grad_accum} "
          f"→ effective={args.batch_size * args.grad_accum})")
    print(f"  ckpt     : {'ON' if use_gc else 'OFF'}")
    print(f"  stage    : {stage}")
    print("=" * 70)

    # ─────── [1/3] 모델 로드 (QLoRA 4-bit) ───────
    t0 = time.time()
    print("\n[1/3] MiniLLaVA + Qwen2.5-1.5B 4-bit NF4 로드 ...")
    model = MiniLLaVA(
        vision_model_name=VISION_MODEL,
        freeze_vision=True,
        freeze_llm=not args.use_lora,
        torch_dtype=torch.bfloat16,
        use_qlora=True,
        qlora_compute_dtype="bf16",
        qlora_use_double_quant=True,
        gradient_checkpointing=use_gc,
    )
    print(f"   load 완료 ({time.time() - t0:.1f}s)")
    after_4bit_gb = gb(torch.cuda.memory_allocated())
    print(f"   after 4-bit base 로드 → memory_allocated = {after_4bit_gb:.2f} GB")

    if args.use_lora:
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, lora_cfg)

    print("\n[2/3] vision/projector → GPU ...")
    model.vision.to(device)
    model.projector.to(device)
    after_load_gb = gb(torch.cuda.memory_allocated())
    print(f"   trainable params: {model.num_trainable():,}")
    print(f"   after full setup → memory_allocated = {after_load_gb:.2f} GB")

    optimizer = AdamW(model.trainable_parameters(), lr=2e-4)

    # ─────── [3/3] forward+backward 측정 (합성 배치) ───────
    print(f"\n[3/3] {args.n_steps} step 측정 (1 warmup + {args.n_steps - 1} 측정) ...")
    model.train()
    model.vision.eval()

    step_times = []
    for i in range(args.n_steps):
        batch = make_synthetic_batch(
            model, args.batch_size, args.max_text_length, device
        )
        torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_time = time.time() - t0
        peak = gb(torch.cuda.max_memory_allocated())
        cur = gb(torch.cuda.memory_allocated())
        tag = "[warmup] " if i == 0 else "[measure]"
        print(f"   {tag} step {i + 1}: loss={loss.item():.4f}  "
              f"time={step_time:.2f}s  cur={cur:.2f}GB  peak={peak:.2f}GB")
        if i > 0:
            step_times.append(step_time)

    avg_step = sum(step_times) / max(1, len(step_times))
    peak_final = gb(torch.cuda.max_memory_allocated())

    # ─────── 결과 분석 ───────
    print()
    print("=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"  4-bit base 로드 후    : {after_4bit_gb:.2f} GB")
    print(f"  setup 완료 후         : {after_load_gb:.2f} GB")
    print(f"  학습 중 peak VRAM     : {peak_final:.2f} / 8.0 GB "
          f"({100 * peak_final / 8:.1f}%)")
    print(f"  평균 step time        : {avg_step:.2f}s "
          f"(batch={args.batch_size} × forward+backward+opt)")
    print()

    if peak_final > 7.8:
        print("  🔴 OOM 위험 — 다음 중 하나 시도:")
        print("     1. gradient_checkpointing ON 확인 (--no-gradient-checkpointing 빼기)")
        print("     2. max_text_length 줄이기")
        print("     3. LoRA rank 줄이기 (r=16 → 8)")
    elif peak_final > 7.0:
        print("  ⚠️  빡빡 — batch=1 가능하나 다른 GPU process 가 잡으면 OOM 위험")
    else:
        print(f"  ✅ 안전 — {8 - peak_final:.2f}GB 여유. "
              f"batch={args.batch_size} grad_accum={args.grad_accum} 학습 가능")
        print()
        print("  학습 시간 추정 (effective batch = "
              f"{args.batch_size * args.grad_accum}):")
        opt_step_time = avg_step * args.grad_accum
        for n_samples, name in [
            (5_000, "Stage 1 (5K alignment, 예시)"),
            (13_000, "Stage 2 (13K mix, 예시)"),
            (25_000, "Stage 2 (25K mix, 예시)"),
        ]:
            n_opt = n_samples // (args.batch_size * args.grad_accum)
            for n_epoch in [1, 2]:
                total_min = n_opt * n_epoch * opt_step_time / 60
                print(f"    {name}: {n_samples} samples × {n_epoch}ep "
                      f"≈ {total_min:.0f}분 ({total_min / 60:.1f}h)")
        print()
        print("  ※ 위 sample 수는 예시 — 실제 규모는 이 step time 을 보고 "
              "학습 시간 예산 안에서 결정.")

    print("=" * 70)


if __name__ == "__main__":
    main()
