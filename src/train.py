"""Stage 1 / Stage 2 학습 — projector (+ optional LoRA / QLoRA) 학습.

v4 default: Qwen2.5-1.5B + QLoRA 4-bit NF4 + bf16 + gradient checkpointing
→ 8GB VRAM 에 fit.

사용 예 (v4 Stage 1 — projector alignment, 4-bit LLM frozen):
  python -m src.train \\
    --data-path data/coco_subset/manifest.json \\
    --output-dir checkpoints/v4_stage1 \\
    --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 1e-3 \\
    --use-qlora --bf16

사용 예 (v4 Stage 2 — QLoRA + projector 동시 학습):
  python -m src.train \\
    --data-path data/v4_stage2_mix/manifest.json \\
    --output-dir checkpoints/v4_stage2_qlora \\
    --init-projector checkpoints/v4_stage1/projector.pt \\
    --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 2e-4 \\
    --use-qlora --use-lora --lora-r 16 --lora-alpha 32 --bf16
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys

# Windows cp949 콘솔/파이프에서 한글·em-dash(—) 출력 시 UnicodeEncodeError 방지.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import LORA_TARGET_MODULES, VISION_MODEL, TrainConfig
from .dataset import VQACollator, VQADataset
from .model import MiniLLaVA


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr_lambda(total_steps: int, warmup_steps: int):
    def fn(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def maybe_apply_lora(model: MiniLLaVA, cfg: TrainConfig):
    """Stage 2: 기존 projector는 그대로 학습 가능 + LLM에 LoRA 어댑터 추가.

    QLoRA: 4-bit 양자화 base 에 LoRA 부착. LoRA adapter 만 bf16 으로 학습됨.
    """
    if not cfg.use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    # PEFT 가 base LLM을 자동 freeze. projector는 외부라 trainable 유지.
    return model


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/v4_stage1")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-text-length", type=int, default=512)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-lora", action="store_true",
                   help="Stage 2: LoRA adapter on LLM + projector 동시 학습")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--init-projector", type=str, default=None,
                   help="기존 projector ckpt에서 시작 (Stage 1 → Stage 2 이어 학습)")
    p.add_argument(
        "--vision-model",
        type=str,
        default=VISION_MODEL,
        help="CLIP vision encoder. v1/v2/v3/v4: openai/clip-vit-base-patch32 (기본).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="bfloat16 학습 — v4 의 QLoRA 1.5B 학습 시 compute dtype 표준.",
    )
    # ── v4 신규: QLoRA 4-bit NF4 양자화 옵션 ──────────────────────────
    p.add_argument(
        "--use-qlora",
        action="store_true",
        help="bitsandbytes 4-bit NF4 양자화. 1.5B LLM 을 8GB VRAM 에 fit 시키는 필수 옵션.",
    )
    p.add_argument(
        "--no-double-quant",
        action="store_true",
        help="QLoRA double quantization 끄기 (기본 ON — 메모리 추가 절감).",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="gradient checkpointing 끄기 (기본 ON — 활성화 메모리 ~50%% 절감).",
    )
    args = p.parse_args()

    # argparse 의 store_true 결과를 TrainConfig 의 양/음 dataclass 필드로 매핑.
    cfg_kwargs = vars(args).copy()
    cfg_kwargs["qlora_use_double_quant"] = not cfg_kwargs.pop("no_double_quant")
    cfg_kwargs["gradient_checkpointing"] = not cfg_kwargs.pop("no_gradient_checkpointing")
    return TrainConfig(**cfg_kwargs)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    print(
        f"[init] loading MiniLLaVA (vision={cfg.vision_model}, "
        f"dtype={dtype}, qlora={cfg.use_qlora}) ..."
    )
    model = MiniLLaVA(
        vision_model_name=cfg.vision_model,
        freeze_vision=True,
        freeze_llm=not cfg.use_lora,
        torch_dtype=dtype,
        use_qlora=cfg.use_qlora,
        qlora_compute_dtype=cfg.qlora_compute_dtype,
        qlora_use_double_quant=cfg.qlora_use_double_quant,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    if cfg.init_projector and os.path.exists(cfg.init_projector):
        print(f"[init] loading existing projector → {cfg.init_projector}")
        model.load_projector(cfg.init_projector, map_location="cpu")
    model = maybe_apply_lora(model, cfg)

    # QLoRA: LLM 은 이미 device_map={"":0} 로 cuda 위에 로드됨.
    # vision/projector 만 cuda 로 이동 (model.to 가 4-bit 가중치 건드리지 않음).
    # gradient checkpointing 은 model.py __init__ 에서 이미 처리 — 여기서 재호출 X.
    if cfg.use_qlora:
        model.vision.to(device)
        model.projector.to(device)
    else:
        model.to(device)

    print(f"[init] trainable params: {model.num_trainable():,}")

    print(f"[data] loading {cfg.data_path}")
    dataset = VQADataset(
        cfg.data_path, model.tokenizer, model.image_processor, cfg.max_text_length
    )
    collator = VQACollator(pad_token_id=model.tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    print(f"[data] {len(dataset)} samples, {len(loader)} batches/epoch")

    optimizer = AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    total_steps = (len(loader) // cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = LambdaLR(optimizer, cosine_lr_lambda(total_steps, warmup_steps))

    # step별 loss/lr 를 CSV 로 기록 — 학습 곡선의 검증 가능한 근거.
    # checkpoints/ 는 .gitignore 대상이라 커밋되는 eval_results/ 에 남긴다.
    run_name = os.path.basename(cfg.output_dir.rstrip("/\\")) or "run"
    os.makedirs("eval_results", exist_ok=True)
    log_path = os.path.join("eval_results", f"train_log_{run_name}.csv")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("step,loss,lr\n")
    print(f"[init] step별 loss 로그 → {log_path}")

    global_step = 0
    model.train()
    if hasattr(model, "vision"):
        model.vision.eval()

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running_loss / (cfg.log_every * cfg.grad_accum_steps)
                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
                    log_file.write(f"{global_step},{avg:.6f},{lr_now:.3e}\n")
                    log_file.flush()
                    running_loss = 0.0

                if global_step % cfg.save_every == 0:
                    ckpt = os.path.join(
                        cfg.output_dir, f"projector_step{global_step}.pt"
                    )
                    model.save_projector(ckpt)
                    # 장시간 Stage 2 학습 중단 대비 — projector 와 짝이 되는
                    # LoRA adapter 도 함께 저장 (projector 만 있으면 복구 불가).
                    if cfg.use_lora:
                        model.llm.save_pretrained(
                            os.path.join(cfg.output_dir, f"lora_step{global_step}")
                        )

    log_file.close()
    print(f"[done] step별 loss 로그 저장 → {log_path}")

    final_path = os.path.join(cfg.output_dir, "projector.pt")
    model.save_projector(final_path)
    print(f"[done] saved → {final_path}")

    if cfg.use_lora:
        lora_dir = os.path.join(cfg.output_dir, "lora_adapter")
        model.llm.save_pretrained(lora_dir)
        print(f"[done] saved LoRA → {lora_dir}")


if __name__ == "__main__":
    main()
