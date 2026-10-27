"""
Trainer for every run in the paper (directed permutation traversal, dePo).

Each training batch is stratified over the hop lengths the curriculum allows at
that step: H_max is held at 1 for curriculum_floor_steps, then grows linearly to
H_max over curriculum_ramp_steps, after which every batch covers all hop lengths
evenly. Fixed step budget. Per-hop accuracy of the evaluation during training is
appended to trajectory.jsonl every eval_interval steps.

Usage (the full paper recipe is in recipes/common.sh):
    uv run python src/train_uniform_h.py --config_name ouro-4x8 --seed 42 \
        --save_dir runs/h1_clean_sweep/ouro-4x8_s42
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
torch.backends.cuda.matmul.allow_tf32 = True  # [local 2026-10-21] TF32 for 12GB laptop GPU
torch.backends.cudnn.allow_tf32 = True
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from src.model import LoopedTransformer
from src.model_configs import create_model
from src.utils import (
    CosineWarmupSchedule,
    TokenMappingTokenizer,
    count_parameters,
    format_params,
    seed_all,
    unwrap_model,
)
from src.datasets.pointer_chasing import generate_pointer_chasing_sample
from src.evaluate import evaluate_per_hop
from src.metrics import fold_note, frontier_three, parse_eval_hops


def sample_stratified_texts(
    rng: random.Random,
    N: int,
    H_min: int,
    H_max_current: int,
    total_batch: int,
    M: int,
    node_format: str = "plain",
):
    """Draw one stratified batch of texts over H ∈ [H_min, H_max_current], untokenized.

    Splits total_batch as evenly as possible across the H values; remainder
    (if total_batch % H_range != 0) is distributed to the lowest H values.
    This keeps batch size constant while letting H_max_current ramp up.
    Every draw from rng for a training batch happens here, so calling it again
    with the same arguments replays the batch (see replay_data_rng).
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
                rng=rng, N=N, H=H, M=M, shuffle_edges=True, node_format=node_format,
            )
            texts.append(text)
            H_list.append(H)

    # Shuffle within batch so H ordering is not positional
    perm = list(range(len(texts)))
    rng.shuffle(perm)
    texts = [texts[i] for i in perm]
    H_list = [H_list[i] for i in perm]
    return texts, H_list


def replay_data_rng(rng: random.Random, steps: int, H_max_at, repeats: int = 1,
                   **sample_kwargs) -> None:
    """Advance rng past the batches of training steps 1..steps.

    The data rng is not checkpointed, so a resumed run calls this to see the
    batches an uninterrupted run would have seen from step steps+1 on.
    H_max_at(step) is the curriculum ceiling used at that step.
    repeats = micro-batches per step (gradient accumulation); each micro-batch
    draws its own samples, so the replay has to draw them as well.
    """
    for step in range(1, steps + 1):
        for _ in range(max(1, repeats)):
            sample_stratified_texts(rng, H_max_current=H_max_at(step), **sample_kwargs)


def make_stratified_batch(
    rng: random.Random,
    N: int,
    H_min: int,
    H_max_current: int,
    total_batch: int,
    M: int,
    tokenizer,
    max_len: int,
    node_format: str = "plain",
):
    """Build one tokenized stratified training batch (see sample_stratified_texts)."""
    texts, H_list = sample_stratified_texts(
        rng, N=N, H_min=H_min, H_max_current=H_max_current, total_batch=total_batch, M=M,
        node_format=node_format,
    )

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


def build_parser() -> argparse.ArgumentParser:
    """The trainer's command line. Defaults are the paper's N=16 recipe."""
    parser = argparse.ArgumentParser(description="Trainer for looped and untied models on dePo")

    # Model / task
    parser.add_argument("--config_name", type=str, required=True,
                        help="ouro-LxT (looped) or deep-N (untied); see src/model_configs.py")
    parser.add_argument("--N", type=int, default=16, help="Nodes in the permutation cycle")
    parser.add_argument("--M", type=int, default=5, help="Queries per prompt")
    parser.add_argument("--H_min", type=int, default=1)
    parser.add_argument("--H_max", type=int, default=15,
                        help="Inclusive upper bound; default 15 to avoid H=N wraparound")
    parser.add_argument("--total_batch", type=int, default=120,
                        help="Prompts per step, stratified across the active H range.")
    parser.add_argument("--curriculum_floor_steps", type=int, default=7000,
                        help="Steps to stay at H_max=1 before any ramping begins.")
    parser.add_argument("--curriculum_ramp_steps", type=int, default=23000,
                        help="Steps over which H_max grows linearly from 1 to H_max, after the floor.")
    parser.add_argument("--H_sample_max", type=int, default=0,
                        help="Optional hard ceiling on the H actually sampled, independent of "
                             "--H_max (0 = off, the default, which reproduces the old "
                             "behaviour exactly). Needed to resume a checkpoint whose token map "
                             "was built on a wider H range: --H_max keeps the vocabulary "
                             "(and the fold notes) at the parent's setting while "
                             "--H_sample_max restricts what the optimizer sees. Added "
                             "2026-10-27 for the ISO experiment (resume the letters run on "
                             "H<=4 data); eval hops must then be set explicitly with "
                             "--eval_hops, e.g. --eval_hops 1,2,3,4.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max grad norm. 0 disables.")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Micro-batches accumulated before one optimizer step "
                             "(default 1 = the original behaviour). Added 2026-10-26 so "
                             "ouro-4x10 can keep the effective batch at 120 samples while "
                             "only 60 fit in 12 GB: --total_batch 60 --grad_accum_steps 2. "
                             "The logged loss is the sum of the micro-batch losses (the "
                             "mean over the effective batch, exact except for the unequal "
                             "non-pad token counts of the halves).")

    # Training schedule
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--no_answer_mask", action="store_true", default=True,
                        help="Train on all non-pad tokens (always on; kept for args.json compatibility).")

    # Eval
    parser.add_argument("--eval_interval", type=int, default=4000)
    parser.add_argument("--eval_n_samples", type=int, default=8000,
                        help="Prompts per evaluation, split evenly across hop lengths")
    parser.add_argument("--eval_hops", type=str, default="",
                        help="Hops to evaluate (P2). '' = every hop in [H_min, H_max], the "
                             "paper's 15-hop grid; 'half' = 1..N//2, one representative per "
                             "effective hop (saves 46.7 percent of eval wall-clock); or an "
                             "explicit list such as '1,2,3'. Reporting must then use h_eff.")

    # Logging / checkpointing
    parser.add_argument("--log_interval", type=int, default=200,
                        help="Print/log loss every N steps")
    parser.add_argument("--ckpt_interval", type=int, default=20000,
                        help="Overwrite save_dir/checkpoint.pt every N steps and at the end")

    # Infra
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len", type=int, default=200,
                        help="Context length; prompts are left-padded to it")
    parser.add_argument("--use_nope", action="store_true", default=False,
                        help="Use NoPE (no positional embeddings). Default: rotary embeddings.")
    parser.add_argument("--no_timestep_embedding", action="store_true", default=False,
                        help="Ablate the learned iteration embedding tau_k (looped models only).")
    parser.add_argument("--aux_deep_supervision", action="store_true", default=False,
                        help="Untied models only: add the next-token loss at intermediate layers, "
                             "read out through the shared final norm and unembedding.")
    parser.add_argument("--aux_layers", type=str, default="8,16,24",
                        help="Comma-separated 1-indexed layer positions for the auxiliary loss.")
    parser.add_argument("--aux_weight", type=float, default=0.3,
                        help="Weight on the mean auxiliary intermediate-layer loss.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="If set and save_dir/checkpoint.pt exists, resume training from that step.")
    parser.add_argument("--tokenizer", type=str,
                        default="hf-internal-testing/llama-tokenizer",
                        help="Base tokenizer, downloaded from the Hugging Face Hub on first use")
    parser.add_argument("--node_format", type=str, default="plain",
                        choices=("plain", "padded", "letters"),
                        help="How node ids are written (P5): 'plain' = 0..15, the paper's "
                             "setting; 'padded' = 00..15 (measured worse); 'letters' = A..P, "
                             "which removes the 7/16 first-token floor but is a NEW benchmark "
                             "and cannot be compared with existing runs.")
    # The paper's looped runs zero-initialize the attention and MLP output projections and its
    # untied runs do not. Without either flag the family's paper setting is used; the resolved
    # choice is written to args.json.
    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument("--zero_init", dest="zero_init", action="store_true",
                            help="Zero attention o_proj and FFN w3 output projections "
                                 "(default for ouro-*).")
    zero_group.add_argument("--no_zero_init", dest="zero_init", action="store_false",
                            help="Leave output projections at ordinary initialization "
                                 "(default for deep-*).")
    parser.set_defaults(zero_init=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.zero_init is None:
        args.zero_init = args.config_name.startswith("ouro-")

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
            map_rng, args.N, H=H, M=args.M, shuffle_edges=True,
            node_format=args.node_format,
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
        use_nope=args.use_nope, zero_init=args.zero_init,
        use_timestep_embedding=not args.no_timestep_embedding,
    )
    model = model.to(device)
    zeroed = sum(1 for name, parameter in model.named_parameters()
                 if name.endswith(("attention.o_proj.weight", "feed_forward.w3.weight"))
                 and float(parameter.detach().abs().sum()) == 0.0)
    print(f"  zero_init: requested={args.zero_init}  "
          f"output projections at exactly zero={zeroed}")
    is_looped = config["model_type"] == "looped"
    K_max = config.get("K_max", None)

    # Auxiliary deep supervision: capture intermediate layer outputs via
    # forward hooks so an extra next-token loss can flow gradient into early layers.
    aux_captured = []
    if args.aux_deep_supervision:
        assert not is_looped, "aux_deep_supervision is for unlooped deep-N models"
        aux_layer_idx = [int(x) - 1 for x in args.aux_layers.split(",")]  # 1-indexed -> 0
        n_layers = config["n_layers"]
        assert all(0 <= i < n_layers for i in aux_layer_idx), \
            f"aux_layers {args.aux_layers} out of range for {n_layers} layers"
        for i in aux_layer_idx:
            model.layers[i].register_forward_hook(
                lambda mod, inp, out: aux_captured.append(out)
            )
        print(f"  aux deep supervision at layers {[i+1 for i in aux_layer_idx]} "
              f"(weight {args.aux_weight})")
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
    # P2: the evaluated hop list is configurable. Default = the paper's full grid;
    # 'half' keeps one representative per effective hop (H and 16-H are the same
    # question because pi^15 = pi^-1).
    H_values = parse_eval_hops(args.eval_hops, args.H_min, args.H_max, args.N)
    n_per_hop = max(1, args.eval_n_samples // len(H_values))

    def run_eval(step: int) -> dict:
        model.eval()
        per_hop = evaluate_per_hop(
            model, tokenizer, device, config,
            N=args.N, M=args.M, H_values=H_values,
            K=K_max if is_looped else None,
            n_samples_per_hop=n_per_hop,
            eval_seed=99999 + step,  # vary across evals
            node_format=args.node_format,
        )
        model.train()
        return {str(h): float(per_hop[h]) for h in H_values}

    # --- Training loop ---
    rng = random.Random(args.seed)
    trajectory_path = save_dir / "trajectory.jsonl"
    train_log_path = save_dir / "train.log"

    # Optional resume from an existing checkpoint in save_dir.
    # Loads model+optimizer+scheduler state and continues from saved step.
    # Use this to extend training beyond the original total_steps.
    resume_step = 0
    if args.resume and (save_dir / "checkpoint.pt").exists():
        ckpt = torch.load(save_dir / "checkpoint.pt", map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        resume_step = int(ckpt["step"])
        print(f"[resume] loaded checkpoint at step={resume_step}")
    else:
        trajectory_path.unlink(missing_ok=True)

    def current_H_max(step: int) -> int:
        floor = args.curriculum_floor_steps
        ramp = args.curriculum_ramp_steps
        if step <= floor:
            h = 1
        elif ramp <= 0 or step >= floor + ramp:
            h = args.H_max
        else:
            progress = (step - floor) / ramp
            h = max(args.H_min, int(1 + (args.H_max - 1) * progress + 0.5))
            h = min(h, args.H_max)
        # [ISO] Optional sampling ceiling, independent of --H_max. Set --H_max to
        # the value the checkpoint's vocabulary was built with and --H_sample_max
        # to the hop range actually fed to the optimizer; that is the only way to
        # resume an H_max=15 letters checkpoint on H<=4 data without shrinking the
        # token map (H<=4 never emits the digits 5..9, so the mapped vocab drops
        # from 34 to 28 and load_state_dict fails on the embeddings).
        # Default 0 = off, byte-identical behaviour for every existing run.
        if args.H_sample_max:
            h = min(h, args.H_sample_max)
        return h

    print(f"Total batch size: {args.total_batch}")
    print(f"Eval hops ({len(H_values)}): {H_values}   node_format={args.node_format}")
    note = fold_note(args.H_max, args.N)
    if note:
        print(f"[B2/P3] {note}")
    # P1 depends on the reverse hop N-H being evaluated. On the 'half' grid it is
    # not, so the rev frontier is 0 by construction - say so instead of letting a
    # reader conclude the model "only solves forward".
    missing_rev = [args.N - h for h in H_values if (args.N - h) not in H_values]
    if missing_rev:
        print(f"[P1] eval grid is not closed under H -> N-H (missing {missing_rev}): "
              f"frontier_rev is 0 by construction and fwd/rev/max collapses to fwd. "
              f"Use --eval_hops '' to read the reverse side.")
    print(f"Curriculum: floor={args.curriculum_floor_steps} steps at H=1, then ramp to {args.H_max} over {args.curriculum_ramp_steps} steps, then stratified-uniform")
    print(f"Gradient clipping: max_norm={args.grad_clip}")

    # Sanity: stratification at full H range
    sanity_batch = make_stratified_batch(
        rng=random.Random(args.seed + 1000),
        N=args.N, H_min=args.H_min, H_max_current=args.H_max,
        total_batch=args.total_batch, M=args.M,
        tokenizer=tokenizer, max_len=args.max_len,
        node_format=args.node_format,
    )
    counter = Counter(sanity_batch["H_list"])
    print(f"Stratification check (full range): {dict(sorted(counter.items()))}")

    log_f = open(train_log_path, "a" if resume_step > 0 else "w")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== Training {args.config_name} seed={args.seed} ===")
    log(f"  total_steps={args.total_steps}  eval_interval={args.eval_interval}")
    log(f"  lr={args.lr} wd={args.wd} warmup={args.warmup_steps}  cosine to {args.lr * 0.1:.1e}")
    log(f"  H ∈ [{args.H_min}, {args.H_max}]  N={args.N}  M={args.M}")

    if resume_step > 0:
        t0 = time.time()
        replay_data_rng(rng, resume_step, current_H_max, N=args.N, H_min=args.H_min,
                        total_batch=args.total_batch, M=args.M,
                        repeats=args.grad_accum_steps)
        log(f"[resume] replayed the data rng through step {resume_step} in {time.time() - t0:.0f}s")

    model.train()
    start_time = time.time()
    last_log_time = start_time

    accum = max(1, args.grad_accum_steps)
    for step in range(resume_step + 1, args.total_steps + 1):
        h_max_now = current_H_max(step)
        optimizer.zero_grad()
        loss_sum = 0.0
        for _micro in range(accum):
            batch = make_stratified_batch(
                rng=rng,
                N=args.N, H_min=args.H_min, H_max_current=h_max_now,
                total_batch=args.total_batch, M=args.M,
                tokenizer=tokenizer, max_len=args.max_len,
                node_format=args.node_format,
            )
            tokens = batch["tokens"].to(device)
            mask = batch["attention_mask"].to(device)

            if args.aux_deep_supervision:
                aux_captured.clear()

            # [local 2026-10-21] bf16 autocast for 12GB laptop GPU (paper used H100;
            # deviation recorded in notes/deviations_4.md)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                if is_looped:
                    logits = model(tokens, mask, K=K_max)
                else:
                    logits = model(tokens, mask)
            logits = logits.float()

            # Next-token CE loss on all non-pad tokens
            logits_shifted = logits[:, :-1].reshape(-1, config["vocab_size"])
            targets = tokens[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits_shifted, targets, ignore_index=pad_id)

            # Auxiliary deep supervision: same next-token loss read out from each
            # captured intermediate layer through the shared norm + unembedding.
            if args.aux_deep_supervision and aux_captured:
                base = unwrap_model(model)
                aux_loss = 0.0
                for state in aux_captured:
                    aux_logits = base.out_proj(base.norm(state))
                    aux_logits = aux_logits[:, :-1].reshape(-1, config["vocab_size"])
                    aux_loss = aux_loss + F.cross_entropy(aux_logits, targets, ignore_index=pad_id)
                loss = loss + args.aux_weight * aux_loss / len(aux_captured)

            loss_sum += loss.item()
            (loss / accum).backward()
            del logits, logits_shifted, targets, tokens, mask

        loss_rep = loss_sum / accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0 or step == 1:
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6) if step > 1 else 0.0
            last_log_time = now
            lr_now = scheduler.get_last_lr()[0]
            log(f"step={step:>7d}  loss={loss_rep:.4f}  lr={lr_now:.2e}  "
                f"H_max={h_max_now:>2}  sps={steps_per_sec:.1f}  elapsed={(now-start_time)/60:.1f}min")

        if step % args.eval_interval == 0 or step == args.total_steps:
            per_hop = run_eval(step)
            # P1: report fwd / rev / max, all counted in EFFECTIVE hops (h_eff =
            # min(H, N-H)). fwd is the paper's H_sol@0.8 unchanged; rev counts the
            # same curve from H=N-1 downward, because pi^(N-1) = pi^-1.
            front = frontier_three(per_hop, thr=0.8, N=args.N)
            row = {
                "step": step,
                "loss": float(loss_rep),
                "per_hop_acc": per_hop,
                "frontier_thr": front["thr"],
                "frontier_fwd": front["fwd"],
                "frontier_rev": front["rev"],
                "frontier_max": front["max"],
            }
            append_trajectory(trajectory_path, row)

            def _acc(h: int) -> str:
                value = per_hop.get(str(h))
                return f"{value:.3f}" if value is not None else "n/a"

            mid_h = H_values[len(H_values) // 2]
            log(f"  EVAL step={step}  H=1: {_acc(1)}  "
                f"H={mid_h}: {_acc(mid_h)}  "
                f"H={H_values[-1]}: {_acc(H_values[-1])}  "
                f"H_sol@0.8={front['fwd']}  "
                f"fwd/rev/max={front['fwd']}/{front['rev']}/{front['max']}")

        if step % args.ckpt_interval == 0 or step == args.total_steps:
            ckpt_path = save_dir / "checkpoint.pt"
            ckpt = {
                "step": step,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(ckpt, ckpt_path)

    log(f"Done. Total time: {(time.time() - start_time)/60:.1f}min")
    log_f.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
