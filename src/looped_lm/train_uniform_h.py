"""
Uniform-H trainer for pointer-chasing.

Stratified per-batch H sampling: each training batch contains exactly
samples_per_H samples for every H in {H_min..H_max}. No curriculum, no
threshold, no patience. Fixed step budget. Per-hop accuracy logged to
trajectory.jsonl every eval_interval steps.

Usage:
    uv run python -m looped_lm.train_uniform_h \\
        --config_name ouro-2x4 --total_steps 200000 \\
        --save_dir runs/ouro-2x4_s42 --seed 42
"""

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from looped_lm.model import LoopedTransformer
from looped_lm.model_configs import create_model
from looped_lm.utils import (
    CosineWarmupSchedule,
    TokenMappingTokenizer,
    count_parameters,
    format_params,
    seed_all,
    unwrap_model,
)
from looped_lm.datasets.pointer_chasing import generate_pointer_chasing_sample
from looped_lm.evaluate import evaluate_per_hop


def make_stratified_batch(
    rng: random.Random,
    N: int,
    H_min: int,
    H_max_current: int,
    total_batch: int,
    M: int,
    tokenizer,
    max_len: int,
):
    """Build one stratified training batch over H ∈ [H_min, H_max_current].

    Splits total_batch as evenly as possible across the H values; remainder
    (if total_batch % H_range != 0) is distributed to the lowest H values.
    This keeps batch size constant while letting H_max_current ramp up.
    """
    H_range = list(range(H_min, H_max_current + 1))
    n_H = len(H_range)
    base = total_batch // n_H
    rem = total_batch - base * n_H

    H_list = []
    texts = []
    for i, H in enumerate(H_range):
        n_samples = base + (1 if i < rem else 0)
        for _ in range(n_samples):
            text, _, _ = generate_pointer_chasing_sample(
                rng=rng, N=N, H=H, M=M, shuffle_edges=True,
            )
            texts.append(text)
            H_list.append(H)

    # Shuffle within batch so H ordering is not positional
    perm = list(range(len(texts)))
    rng.shuffle(perm)
    texts = [texts[i] for i in perm]
    H_list = [H_list[i] for i in perm]

    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {
        "tokens": out["input_ids"].long(),
        "attention_mask": out["attention_mask"].to(torch.bool),
        "H_list": H_list,
    }


def append_trajectory(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Uniform-H trainer for h1_clean_sweep")

    # Model / task
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--H_min", type=int, default=1)
    parser.add_argument("--H_max", type=int, default=15,
                        help="Inclusive upper bound; default 15 to avoid H=N wraparound")
    parser.add_argument("--total_batch", type=int, default=120,
                        help="Total batch size. Stratified across active H range.")
    parser.add_argument("--curriculum_floor_steps", type=int, default=5000,
                        help="Steps to stay at H_max=1 before any ramping begins. Lets H=1 lock in.")
    parser.add_argument("--curriculum_ramp_steps", type=int, default=25000,
                        help="Steps over which H_max grows linearly from 1 to H_max, AFTER floor.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max grad norm. 0 disables.")

    # Training schedule
    parser.add_argument("--total_steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--no_answer_mask", action="store_true", default=True,
                        help="Train on all tokens (matches h1_depth_gating). Default True.")

    # Eval
    parser.add_argument("--eval_interval", type=int, default=4000)
    parser.add_argument("--eval_n_samples", type=int, default=8000,
                        help="Total per evaluation; split evenly across hops")

    # Logging / checkpointing
    parser.add_argument("--log_interval", type=int, default=200,
                        help="Print/log loss every N steps")
    parser.add_argument("--ckpt_interval", type=int, default=20000)

    # Infra
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--use_nope", action="store_true", default=False,
                        help="Use NoPE (no positional embeddings). Default: positional embeddings on.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str,
                        default="hf-internal-testing/llama-tokenizer")

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    seed_all(args.seed)
    device = args.device

    # --- Tokenizer ---
    print("Loading tokenizer...")
    base_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if base_tok.pad_token_id is None:
        base_tok.pad_token_id = base_tok.eos_token_id

    print("Estimating token map...")
    map_rng = random.Random(999)
    map_texts = [
        generate_pointer_chasing_sample(
            map_rng, args.N, H=H, M=args.M, shuffle_edges=True
        )[0]
        for H in range(args.H_min, args.H_max + 1)
        for _ in range(50)
    ]
    tokenizer = TokenMappingTokenizer(base_tok)
    tokenizer.estimate_token_map_from_dataset(map_texts)
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    print(f"Vocab size after mapping: {vocab_size}")

    # --- Model ---
    print(f"Creating model: {args.config_name}")
    model, config = create_model(
        args.config_name, vocab_size=vocab_size,
        max_sequence_len=args.max_len, pad_token_idx=pad_id,
        use_nope=args.use_nope, zero_init=True,
    )
    model = model.to(device)
    is_looped = config["model_type"] == "looped"
    K_max = config.get("K_max", None)
    n_params = count_parameters(model)
    print(f"  type={config['model_type']}  params={format_params(n_params)}"
          + (f"  K_max={K_max}  effective_depth={model.effective_depth()}" if is_looped else ""))

    with open(save_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.wd,
        betas=(0.9, 0.95), eps=1e-8,
    )
    schedule_fn = CosineWarmupSchedule(
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)

    # --- Eval helper ---
    H_values = list(range(args.H_min, args.H_max + 1))
    n_per_hop = max(1, args.eval_n_samples // len(H_values))

    def run_eval(step: int) -> dict:
        model.eval()
        per_hop = evaluate_per_hop(
            model, tokenizer, device, config,
            N=args.N, M=args.M, H_values=H_values,
            K=K_max if is_looped else None,
            n_samples_per_hop=n_per_hop,
            eval_seed=99999 + step,  # vary across evals
        )
        model.train()
        return {str(h): float(per_hop[h]) for h in H_values}

    # --- Training loop ---
    rng = random.Random(args.seed)
    trajectory_path = save_dir / "trajectory.jsonl"
    train_log_path = save_dir / "train.log"
    trajectory_path.unlink(missing_ok=True)

    def current_H_max(step: int) -> int:
        floor = args.curriculum_floor_steps
        ramp = args.curriculum_ramp_steps
        if step <= floor:
            return 1
        if ramp <= 0 or step >= floor + ramp:
            return args.H_max
        progress = (step - floor) / ramp
        h = max(args.H_min, int(1 + (args.H_max - 1) * progress + 0.5))
        return min(h, args.H_max)

    print(f"Total batch size: {args.total_batch}")
    print(f"Curriculum: floor={args.curriculum_floor_steps} steps at H=1, then ramp to {args.H_max} over {args.curriculum_ramp_steps} steps, then stratified-uniform")
    print(f"Gradient clipping: max_norm={args.grad_clip}")

    # Sanity: stratification at full H range
    sanity_batch = make_stratified_batch(
        rng=random.Random(args.seed + 1000),
        N=args.N, H_min=args.H_min, H_max_current=args.H_max,
        total_batch=args.total_batch, M=args.M,
        tokenizer=tokenizer, max_len=args.max_len,
    )
    counter = Counter(sanity_batch["H_list"])
    print(f"Stratification check (full range): {dict(sorted(counter.items()))}")

    log_f = open(train_log_path, "w")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== Training {args.config_name} seed={args.seed} ===")
    log(f"  total_steps={args.total_steps}  eval_interval={args.eval_interval}")
    log(f"  lr={args.lr} wd={args.wd} warmup={args.warmup_steps}  cosine to {args.lr * 0.1:.1e}")
    log(f"  H ∈ [{args.H_min}, {args.H_max}]  N={args.N}  M={args.M}")

    model.train()
    start_time = time.time()
    last_log_time = start_time

    for step in range(1, args.total_steps + 1):
        h_max_now = current_H_max(step)
        batch = make_stratified_batch(
            rng=rng,
            N=args.N, H_min=args.H_min, H_max_current=h_max_now,
            total_batch=args.total_batch, M=args.M,
            tokenizer=tokenizer, max_len=args.max_len,
        )
        tokens = batch["tokens"].to(device)
        mask = batch["attention_mask"].to(device)

        if is_looped:
            logits = model(tokens, mask, K=K_max)
        else:
            logits = model(tokens, mask)

        # Next-token CE loss on all non-pad tokens (matches h1_depth_gating)
        logits_shifted = logits[:, :-1].reshape(-1, config["vocab_size"])
        targets = tokens[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits_shifted, targets, ignore_index=pad_id)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0 or step == 1:
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6) if step > 1 else 0.0
            last_log_time = now
            lr_now = scheduler.get_last_lr()[0]
            log(f"step={step:>7d}  loss={loss.item():.4f}  lr={lr_now:.2e}  "
                f"H_max={h_max_now:>2}  sps={steps_per_sec:.1f}  elapsed={(now-start_time)/60:.1f}min")

        if step % args.eval_interval == 0 or step == args.total_steps:
            per_hop = run_eval(step)
            row = {
                "step": step,
                "loss": float(loss.item()),
                "per_hop_acc": per_hop,
            }
            append_trajectory(trajectory_path, row)
            # Monotone-from-bottom MLRD: largest H with all h<=H meeting threshold
            sorted_hs = sorted(int(k) for k in per_hop)
            frontier_h = 0
            for h in sorted_hs:
                if per_hop[str(h)] >= 0.8:
                    frontier_h = h
                else:
                    break
            mid_h = max(1, args.H_max // 2)
            log(f"  EVAL step={step}  H=1: {per_hop['1']:.3f}  "
                f"H={mid_h}: {per_hop[str(mid_h)]:.3f}  "
                f"H={args.H_max}: {per_hop[str(args.H_max)]:.3f}  "
                f"MLRD@0.8={frontier_h}")

        if step % args.ckpt_interval == 0 or step == args.total_steps:
            ckpt_path = save_dir / "checkpoint.pt"
            torch.save({
                "step": step,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)

    log(f"Done. Total time: {(time.time() - start_time)/60:.1f}min")
    log_f.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
