"""
Forward-walk probe: which forward distances along the cycle are decodable?

The logit lens is low-amplitude, so it cannot cleanly separate forward from
backward node decoding. The linear probe decodes node identity (class 0..N-1)
directly from the residual stream, with no tokenization ambiguity, and is the
instrument the paper's staircase is built on. Here we train probes to
decode BOTH the forward node f_k = pi^k(s) AND the backward node
b_k = pi^{-k}(s) = order[(start_idx - k) % N] at each iteration k, and compare
their selectivity (acc_real - acc_shuffled, Hewitt-Liang control).

Claim tested: on the looped 2x16 model (K=16) at a high hop (H_eval=12, backward shortcut
= 16-12 = 4), the FORWARD diagonal f_k is decodable in order (staircase) while
the BACKWARD diagonal b_k stays at chance. That is direct evidence the model
walks forward, not along the cheaper backward path.

Outputs (in --out_dir):
    directional_probe.npz      fwd_real/fwd_shuf/bwd_real/bwd_shuf diagonals
    directional_probe_summary.json  forward vs backward diagonal selectivity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.model import LoopedTransformer
from src.model_configs import create_model
from src.probe_h2 import (
    build_tokenizer,
    extract_states_looped,
    find_query_positions,
    generate_eval_set,
    get_q_token_ids,
    load_train_args,
    tokenize_batch,
    train_probe,
)


@torch.no_grad()
def extract_states_and_targets(
    eval_set, tokenizer, model, device, K, max_len, batch_size, N,
):
    """Return states (n, K+1, D) and fwd_targets (n, N) where
    fwd_targets[:, d] = pi^d(s) = order[(start_idx+d)%N] for EVERY forward
    distance d in 0..N-1 (not just up to the queried hop). This lets us ask
    which forward distances are represented anywhere in the loop: a forward-H
    walk represents f_1..f_H, a backward-(N-H) shortcut would instead represent
    f_{N-1}, f_{N-2}, ..., f_{H}."""
    q_token_ids = get_q_token_ids(tokenizer)
    all_states, all_fwd = [], []

    for start in range(0, len(eval_set), batch_size):
        chunk = eval_set[start : start + batch_size]
        texts = [s["text"] for s in chunk]
        tok = tokenize_batch(texts, tokenizer, max_len=max_len)
        states = extract_states_looped(
            model, tok["tokens"], tok["attention_mask"], K=K, device=device,
        )  # (K+1, B, T, D)
        positions = find_query_positions(tok["tokens"], q_token_ids)

        for b, sample in enumerate(chunk):
            meta = sample["meta"]
            order = meta["order"]
            qmeta = meta["queries"]
            qpos = positions[b]
            if len(qpos) < len(qmeta):
                continue
            for q_idx, q in enumerate(qmeta):
                pos = qpos[q_idx]
                si = q["start_idx"]
                fwd = np.array([order[(si + d) % N] for d in range(N)],
                               dtype=np.int64)
                hidden = states[:, b, pos, :].numpy()  # (K+1, D)
                all_states.append(hidden)
                all_fwd.append(fwd)

    return np.stack(all_states), np.stack(all_fwd)


def grid_selectivity(states, targets, N, train_frac, seed, device):
    """Train a linear probe at EVERY level to decode each forward distance d.
    Return selectivity grid (n_levels, N) = acc_real - acc_shuffled, plus the
    per-distance peak over levels and the level at which it peaks."""
    n_total, n_levels, dim = states.shape
    n_dist = targets.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_train = int(train_frac * n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    sel = np.full((n_levels, n_dist), np.nan, dtype=np.float32)
    for level in range(n_levels):
        Xl = states[:, level, :].astype(np.float32)
        for d in range(n_dist):
            yd = targets[:, d]
            real = train_probe(
                Xl[train_idx], yd[train_idx], Xl[test_idx], yd[test_idx],
                n_classes=N, probe_type="linear", device=device,
            )
            sp = rng.permutation(yd.shape[0])
            shuf = train_probe(
                Xl[train_idx], yd[sp][train_idx], Xl[test_idx], yd[test_idx],
                n_classes=N, probe_type="linear", device=device,
            )
            sel[level, d] = real - shuf
        print(f"[dirprobe]  level {level}/{n_levels-1} done")
    peak = np.nanmax(sel, axis=0)              # (n_dist,)
    peak_level = np.nanargmax(sel, axis=0)     # (n_dist,)
    return sel, peak, peak_level


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--H_eval", type=int, default=12)
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_nope", default=None, action=argparse.BooleanOptionalAction)
    p.add_argument("--zero_init", default=None, action=argparse.BooleanOptionalAction)
    p.add_argument("--no_timestep_embedding", default=None,
                   action=argparse.BooleanOptionalAction)
    args = p.parse_args()

    train_args = load_train_args(args.checkpoint)
    if args.use_nope is None:
        args.use_nope = bool(train_args.get("use_nope", True))
    if args.zero_init is None:
        args.zero_init = bool(train_args.get("zero_init", True))
    if args.no_timestep_embedding is None:
        args.no_timestep_embedding = bool(train_args.get("no_timestep_embedding", False))
    if args.max_len is None:
        args.max_len = int(train_args.get("max_len", 200))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.N)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    model, config = create_model(
        args.config_name, vocab_size=vocab_size, max_sequence_len=args.max_len,
        pad_token_idx=pad_id, use_nope=args.use_nope, zero_init=args.zero_init,
        use_timestep_embedding=not args.no_timestep_embedding,
    )
    model = model.to(args.device)
    assert isinstance(model, LoopedTransformer)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    K = config["K_max"]

    eval_set = generate_eval_set(
        N=args.N, H_eval=args.H_eval, M=args.M, n_samples=args.n_samples,
        seed=args.eval_seed,
    )
    print(f"[dirprobe] extracting states K={K} query-hop={args.H_eval} ...")
    t0 = time.time()
    states, fwd_t = extract_states_and_targets(
        eval_set, tokenizer, model, args.device, K, args.max_len,
        args.batch_size, args.N,
    )
    print(f"[dirprobe] states={states.shape} ({time.time()-t0:.0f}s); "
          f"training probes over all {args.N} forward distances ...")

    sel, peak, peak_level = grid_selectivity(
        states, fwd_t, args.N, args.train_frac, seed=0, device=args.device)

    fwd_dist = args.H_eval               # forward walk length for this query hop
    bwd_dist = args.N - args.H_eval      # backward shortcut length
    print(f"[dirprobe] query hop H={fwd_dist}; forward walk visits f_1..f_{fwd_dist}; "
          f"a backward-{bwd_dist} shortcut would instead visit "
          f"f_{args.N-1}..f_{fwd_dist}")
    print("[dirprobe]  d | peak_sel  peak_level   (forward distance d, pi^d(s))")
    for d in range(args.N):
        tag = ""
        if 1 <= d <= fwd_dist:
            tag = "forward-intermediate" if d < fwd_dist else "ANSWER"
        elif d > fwd_dist:
            tag = "backward-shortcut node" if d >= args.N - bwd_dist else ""
        print(f"[dirprobe] {d:2d} |  {peak[d]:+.3f}     L{peak_level[d]:<2d}   {tag}")

    np.savez_compressed(
        out_dir / "directional_probe.npz",
        selectivity_grid=sel, peak=peak, peak_level=peak_level,
    )
    summary = {
        "config_name": args.config_name, "checkpoint": str(args.checkpoint),
        "N": args.N, "H_eval": args.H_eval, "K": K,
        "forward_walk_length": fwd_dist, "backward_shortcut_length": bwd_dist,
        "chance": 1.0 / args.N, "n_queries": int(states.shape[0]),
        "forward_distance_d": list(range(args.N)),
        "peak_selectivity_per_distance": peak.tolist(),
        "peak_level_per_distance": peak_level.tolist(),
    }
    with open(out_dir / "directional_probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dirprobe] wrote {out_dir}/directional_probe_summary.json")


if __name__ == "__main__":
    main()
