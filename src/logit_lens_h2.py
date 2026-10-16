"""
First-token logit lens on trained dePo checkpoints.

Per-iteration distribution over node identities, read directly through the
model's own LM head (Coconut Sec 4.3 style; Anthropic 2412.06769). For each
'?' position in each dePo input, we extract the post-iteration-k hidden state
(or post-layer-l for unlooped baselines), apply the model's readout
(final norm + out_proj), restrict the vocab logits to the entity-token
subset {token(0), ..., token(N-1)}, and softmax. The result is a tensor

    P[query, level, node]

that we plot against the ground-truth chain (n_0, ..., n_H) to visualise
"how the model commits to each hop with depth".

Sanity checks (both must pass):
    1. lm_head(states[K]) at the '?+1' position equals model.forward
       logits at the same position to within fp32 tolerance.
    2. argmax over {0..N-1} of the lensed distribution at level=K matches the
       model's actual answer prediction (i.e. its full forward-pass argmax
       restricted to entity tokens) on >= 99% of queries.

Outputs (in --out_dir):
    logit_lens.npz           probs (n_queries, n_levels, N), chains (n_queries, H+1),
                             solved (n_queries, bool)
    logit_lens_summary.json  config + sanity check results + a hero example
                             (single query: probs along the correct chain)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.model import LoopedTransformer, Transformer
from src.model_configs import create_model
from src.probe_h2 import (
    build_tokenizer,
    extract_states_deep,
    extract_states_looped,
    find_query_positions,
    generate_eval_set,
    get_q_token_ids,
    load_train_args,
    tokenize_batch,
)


# ---------------------------------------------------------------------------
# Entity-token mapping
# ---------------------------------------------------------------------------

def build_node_token_map(tokenizer, N: int) -> tuple[np.ndarray, list[int]]:
    """Return (first_token_per_node, ambiguous_nodes).

    LLaMA splits ` {v}` as `[▁, d_0, ...]`. We strip the leading whitespace
    token and read the FIRST digit token. The lens is taken at position
    '?+1' (where the model commits to the first answer token), so this is
    the natural readout. For N <= 10 every node has a unique first token.
    For larger N (e.g. N=16), two-digit nodes share their first digit token
    (1, 10, 11, ..., 15 all start with "1"); these collisions are returned
    in `ambiguous_nodes` so plots can flag or filter them. For the hero
    figure we explicitly pick chains whose nodes are all unambiguous.
    """
    first_tok = np.full(N, -1, dtype=np.int64)
    for v in range(N):
        out = tokenizer(f" {v}", add_special_tokens=False)
        ids = out["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) == 0:
            raise RuntimeError(f"tokenizer produced 0 tokens for ' {v}'")
        first_tok[v] = ids[1] if len(ids) > 1 else ids[0]

    # Find groups that share the same first token.
    from collections import defaultdict
    groups = defaultdict(list)
    for v, t in enumerate(first_tok.tolist()):
        groups[t].append(v)
    ambiguous = sorted([v for vs in groups.values() if len(vs) > 1 for v in vs])
    if ambiguous:
        print(f"[lens] {len(ambiguous)} ambiguous nodes (multi-digit, share "
              f"first token): {ambiguous}. Hero plot will avoid these.")
    return first_tok, ambiguous


# ---------------------------------------------------------------------------
# Apply the model's readout to an arbitrary intermediate state
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_readout(
    model, x: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Apply the model's own final norm + out_proj to hidden state x.

    For a looped model at the last iteration this equals the forward pass. For
    an untied model the lens at layer l is norm + out_proj applied directly to
    that layer's output (the canonical logit lens).
    """
    x = model.norm(x)
    return model.out_proj(x)


# ---------------------------------------------------------------------------
# Core extraction: per-(query, level, node) probabilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_logit_lens(
    eval_set: list[dict],
    tokenizer,
    model,
    device: str,
    K_or_layers: int,
    is_looped: bool,
    max_len: int,
    batch_size: int,
    H_eval: int,
    node_to_tok: np.ndarray,
    ambiguous: list[int],
    per_layer: bool = False,
) -> dict:
    """Iterate over eval_set, for each '?' query collect the (n_levels, N) lensed
    probability matrix, the ground-truth chain (H+1,), and whether the model's
    final-K argmax at '?+1' equals the ground-truth answer (solved bool).

    Also returns sanity-check tensors:
        forward_logits_K: (Q, vocab) full forward-pass logits at the '?+1' position
                          for the final iteration (or final layer for deep models)
        lens_logits_K:    (Q, vocab) the lensed logits at the same position from
                          intermediate states. These should match exactly.
    """
    q_token_ids = get_q_token_ids(tokenizer)
    N = node_to_tok.shape[0]
    node_to_tok_t = torch.from_numpy(node_to_tok).long().to(device)

    # index 0 is post-embedding baseline; with per_layer=True a looped model
    # yields K*L+1 levels (one per unrolled layer), matching the probe axis.
    n_levels = None

    all_probs = []        # list of (n_levels, N) np.ndarray
    all_chains = []       # list of (H+1,) np.ndarray
    all_solved = []       # list of bool

    sanity_lens_logits_K = []     # list of (vocab,) tensors
    sanity_forward_logits_K = []  # list of (vocab,) tensors

    for start in range(0, len(eval_set), batch_size):
        chunk = eval_set[start : start + batch_size]
        texts = [s["text"] for s in chunk]
        tok = tokenize_batch(texts, tokenizer, max_len=max_len)
        tokens_dev = tok["tokens"].to(device)
        mask_dev = tok["attention_mask"].to(device)

        # 1) Hidden states at every level (CPU, fp32)
        if is_looped:
            states = extract_states_looped(
                model, tok["tokens"], tok["attention_mask"],
                K=K_or_layers, device=device, per_layer=per_layer,
            )
        else:
            states = extract_states_deep(
                model, tok["tokens"], tok["attention_mask"], device=device,
            )
        # states: (n_levels, B, T, D) on CPU
        if n_levels is None:
            n_levels = int(states.shape[0])
        elif int(states.shape[0]) != n_levels:
            raise RuntimeError(f"level count changed: {states.shape[0]} vs {n_levels}")

        # 2) Forward-pass logits (for sanity check at level=K_max / final layer)
        if is_looped:
            fwd_logits = model(tokens_dev, mask_dev, K=K_or_layers)
        else:
            fwd_logits = model(tokens_dev, mask_dev)
        # fwd_logits: (B, T, vocab)

        positions = find_query_positions(tok["tokens"], q_token_ids)

        # 3) Apply readout (norm + out_proj) level-by-level, one at a time to limit memory.
        # Full-vocab softmax at every level, then the node first-token entries
        # are kept (B, T, N); the full logits are retained only at the final
        # level, for the sanity comparison against the forward pass. With 33
        # per-layer levels the full (B, T, vocab) tensors no longer fit.
        per_level_node_probs = []  # list of (B, T, N) tensors on CPU
        final_logits = None        # (B, T, vocab) on CPU
        for lvl in range(n_levels):
            x_lvl = states[lvl].to(device)
            logits_lvl = apply_readout(model, x_lvl, mask_dev)
            probs_lvl = torch.softmax(logits_lvl.float(), dim=-1)[..., node_to_tok_t]
            per_level_node_probs.append(probs_lvl.cpu())
            if lvl == n_levels - 1:
                final_logits = logits_lvl.cpu()
            del x_lvl, logits_lvl, probs_lvl

        for b, sample in enumerate(chunk):
            qmeta = sample["meta"]["queries"]
            qpos = positions[b]
            if len(qpos) < len(qmeta):
                continue
            for q_idx, q in enumerate(qmeta):
                pos = qpos[q_idx]
                chain = np.asarray(q["intermediates"], dtype=np.int64)
                if chain.shape[0] != H_eval + 1:
                    continue

                # Per-level node probabilities at '?+1' for this query:
                # (n_levels, N). Full-vocab softmax was taken above, so these
                # are proper probabilities; ambiguous nodes (multi-digit
                # sharing first tokens) get IDENTICAL values across the group,
                # which the plotter will flag / filter.
                probs = torch.stack(
                    [per_level_node_probs[lvl][b, pos] for lvl in range(n_levels)],
                    dim=0,
                ).numpy()  # (n_levels, N)
                all_probs.append(probs)
                all_chains.append(chain)

                # Solved: forward-pass argmax at '?+1' over full vocab equals
                # the ground-truth answer token.
                fwd_argmax = int(fwd_logits[b, pos].argmax(-1).item())
                gt_tok = int(node_to_tok[chain[-1]])
                all_solved.append(bool(fwd_argmax == gt_tok))

                # Sanity buffers (final level only)
                sanity_lens_logits_K.append(final_logits[b, pos].clone())
                sanity_forward_logits_K.append(fwd_logits[b, pos].cpu().clone())

        del per_level_node_probs, final_logits, fwd_logits, states

    probs = np.stack(all_probs, axis=0)        # (Q, n_levels, N)
    chains = np.stack(all_chains, axis=0)      # (Q, H+1)
    solved = np.asarray(all_solved, dtype=bool)
    lens_K = torch.stack(sanity_lens_logits_K, dim=0)
    fwd_K = torch.stack(sanity_forward_logits_K, dim=0)

    return {
        "probs": probs,
        "chains": chains,
        "solved": solved,
        "sanity_lens_logits_K": lens_K,
        "sanity_forward_logits_K": fwd_K,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    out: dict, node_to_tok: np.ndarray, atol: float = 1e-3
) -> dict:
    """Two checks:
    1. Lensed final-level logits == forward-pass logits at '?+1'.
    2. Lensed argmax over entity tokens at level=K agrees with forward
       full-vocab argmax (when forward predicts an entity token at all).
    """
    lens_K = out["sanity_lens_logits_K"]    # (Q, vocab)
    fwd_K = out["sanity_forward_logits_K"]  # (Q, vocab)
    probs = out["probs"]                    # (Q, n_levels, N)
    chains = out["chains"]
    solved = out["solved"]

    diff = (lens_K - fwd_K).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    pass1 = max_abs < atol

    node_to_tok_t = torch.from_numpy(node_to_tok).long()
    fwd_argmax_full = fwd_K.argmax(-1)                  # (Q,)
    fwd_in_entities = torch.isin(fwd_argmax_full, node_to_tok_t)
    lens_argmax_K = torch.from_numpy(probs[:, -1, :].argmax(-1))  # node id
    lens_argmax_K_tok = node_to_tok_t[lens_argmax_K]
    matches = (lens_argmax_K_tok == fwd_argmax_full) & fwd_in_entities
    if fwd_in_entities.any():
        agree_rate = float(matches.float().sum() / fwd_in_entities.float().sum())
    else:
        agree_rate = float("nan")
    pass2 = agree_rate >= 0.99

    # Solved-rate sanity: looped model that solves hops up to H_eval should solve
    # most queries.
    solved_rate = float(solved.mean())

    return {
        "logits_max_abs_diff": max_abs,
        "logits_mean_abs_diff": mean_abs,
        "logits_match_atol": atol,
        "logits_match_pass": pass1,
        "lens_vs_forward_argmax_agree_rate": agree_rate,
        "argmax_agree_pass": pass2,
        "solved_rate": solved_rate,
    }


# ---------------------------------------------------------------------------
# Hero example: a single solved query, full per-level distribution along chain
# ---------------------------------------------------------------------------

def pick_hero_example(
    probs: np.ndarray, chains: np.ndarray, solved: np.ndarray,
    node_to_tok: np.ndarray,
) -> dict:
    """Pick a solved query whose chain has all-distinct nodes AND all-distinct
    first tokens (so the lens probability of each chain node is unambiguous).
    Return per-level probability over each chain node + top-3 narrative.
    """
    Q, n_levels, N = probs.shape
    H_plus_1 = chains.shape[1]

    def chain_unambiguous(chain):
        toks = node_to_tok[chain]
        return len(set(toks.tolist())) == H_plus_1

    candidates = []
    for q in range(Q):
        if not solved[q]:
            continue
        if len(set(chains[q].tolist())) != H_plus_1:
            continue
        if not chain_unambiguous(chains[q]):
            continue
        candidates.append(q)
    if not candidates:
        # fallback: any solved query, ignoring ambiguity (plot may show
        # overlapping curves for chain nodes that share first tokens)
        candidates = [q for q in range(Q) if solved[q]] or [0]
    q_pick = int(candidates[0])

    chain = chains[q_pick]                   # (H+1,)
    p = probs[q_pick]                        # (n_levels, N)

    chain_probs = p[:, chain]                # (n_levels, H+1)
    top3_idx = np.argsort(-p, axis=-1)[:, :3]   # (n_levels, 3)
    top3_prob = np.take_along_axis(p, top3_idx, axis=-1)

    return {
        "query_index": q_pick,
        "chain_nodes": chain.tolist(),
        "n_levels": int(n_levels),
        "chain_probs_per_level": chain_probs.tolist(),  # row=level, col=hop
        "top3_node_per_level": top3_idx.tolist(),
        "top3_prob_per_level": top3_prob.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--H_eval", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--max_len", type=int, default=None,
                   help="Override training max_len. Default: read from training args.json.")
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--random_init", action="store_true",
                   help="Skip checkpoint load (random-init control)")
    p.add_argument("--use_nope", default=None, action=argparse.BooleanOptionalAction,
                   help="Override training use_nope. Default: read from training args.json.")
    p.add_argument("--zero_init", default=None, action=argparse.BooleanOptionalAction,
                   help="Override training zero_init. Default: read from training args.json.")
    p.add_argument("--per_layer", action="store_true",
                   help="Looped models: one level after every unrolled layer "
                        "(K*L+1 levels, block boundaries every L) instead of one "
                        "per iteration, matching probe_h2.py --per_layer. "
                        "Untied models are per-layer either way.")
    p.add_argument("--sanity_only", action="store_true",
                   help="Run on just --n_samples (e.g. 8) and only report sanity "
                        "checks. Used to validate the lens before a full sweep.")
    p.add_argument("--sanity_min_solved", type=float, default=0.6,
                   help="Abort if solved_rate at K=K_max is below this. Catches "
                        "PE mismatch (fallback baseline ~0.4375). 0.0 to skip; "
                        "--random_init implicitly skips.")
    args = p.parse_args()

    if args.sanity_only:
        args.n_samples = max(8, min(args.n_samples, 64))

    # Resolve runtime config from training args.
    train_args = {} if args.random_init else load_train_args(args.checkpoint)
    if args.use_nope is None:
        args.use_nope = bool(train_args.get("use_nope", True))
    if args.zero_init is None:
        args.zero_init = bool(train_args.get("zero_init", True))
    if args.max_len is None:
        args.max_len = int(train_args.get("max_len", 512))
    print(f"[lens] runtime: use_nope={args.use_nope} zero_init={args.zero_init} "
          f"max_len={args.max_len}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lens] config={args.config_name} ckpt={args.checkpoint}")
    print(f"[lens] N={args.N} H_eval={args.H_eval} n_samples={args.n_samples}")

    # Tokenizer + model
    print("[lens] building tokenizer...")
    tokenizer = build_tokenizer(args.N)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    print("[lens] creating model...")
    model, config = create_model(
        args.config_name, vocab_size=vocab_size,
        max_sequence_len=args.max_len, pad_token_idx=pad_id,
        use_nope=args.use_nope, zero_init=args.zero_init,
    )
    model = model.to(args.device)
    is_looped = isinstance(model, LoopedTransformer)
    if not args.random_init:
        print(f"[lens] loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[lens] RANDOM-INIT control")
    model.eval()

    K_or_layers = config["K_max"] if is_looped else config["n_layers"]
    print(f"[lens] is_looped={is_looped}, levels=1..{K_or_layers} (+ level 0 baseline)")

    # Entity-token map
    node_to_tok, ambiguous = build_node_token_map(tokenizer, args.N)
    print(f"[lens] first-token map: nodes 0..{args.N-1} -> {node_to_tok.tolist()[:8]}...")

    # Eval set
    eval_set = generate_eval_set(
        N=args.N, H_eval=args.H_eval, M=args.M,
        n_samples=args.n_samples, seed=args.eval_seed,
    )

    # Run lens
    print("[lens] running logit lens...")
    t0 = time.time()
    out = run_logit_lens(
        eval_set=eval_set,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        K_or_layers=K_or_layers,
        is_looped=is_looped,
        max_len=args.max_len,
        batch_size=args.batch_size,
        H_eval=args.H_eval,
        node_to_tok=node_to_tok,
        ambiguous=ambiguous,
        per_layer=args.per_layer,
    )
    print(f"[lens] lens done: probs={out['probs'].shape} "
          f"({time.time()-t0:.0f}s)")

    # Sanity
    sanity = run_sanity_checks(out, node_to_tok)
    print(f"[lens] SANITY: max|lens-fwd|={sanity['logits_max_abs_diff']:.2e} "
          f"(pass={sanity['logits_match_pass']}), "
          f"argmax-agree={sanity['lens_vs_forward_argmax_agree_rate']:.3f} "
          f"(pass={sanity['argmax_agree_pass']}), "
          f"solved-rate={sanity['solved_rate']:.3f}")

    # Sanity guard: catch PE / init mismatches. Solved-rate at K=K_max is the
    # model's actual accuracy on H_eval queries. ~0.4375 fallback baseline
    # means the model is just predicting "1" because the runtime PE config
    # disagrees with training.
    if (not args.random_init) and sanity['solved_rate'] < args.sanity_min_solved:
        msg = (
            f"[lens] LENS SANITY FAILED: solved_rate at K=K_max = "
            f"{sanity['solved_rate']:.3f} < {args.sanity_min_solved} at H_eval={args.H_eval}.\n"
            f"  Likely causes:\n"
            f"    (a) PE mismatch: lens use_nope={args.use_nope} vs "
            f"train use_nope={train_args.get('use_nope')}\n"
            f"    (b) init mismatch: lens zero_init={args.zero_init} vs "
            f"train zero_init={train_args.get('zero_init')}\n"
            f"    (c) checkpoint not yet saturated H_eval={args.H_eval}\n"
            f"  Override with --sanity_min_solved 0 if intentional."
        )
        raise RuntimeError(msg)

    # Persist
    np.savez_compressed(
        out_dir / "logit_lens.npz",
        probs=out["probs"].astype(np.float32),
        chains=out["chains"],
        solved=out["solved"],
        node_to_tok=node_to_tok,
        ambiguous=np.asarray(ambiguous, dtype=np.int64),
    )

    summary = {
        "ambiguous_nodes": ambiguous,
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint),
        "random_init": bool(args.random_init),
        "is_looped": bool(is_looped),
        "n_levels": int(out["probs"].shape[1]),
        "per_layer": bool(args.per_layer),
        "K_or_layers": int(K_or_layers),
        "N": int(args.N),
        "H_eval": int(args.H_eval),
        "n_queries": int(out["probs"].shape[0]),
        "sanity": sanity,
    }
    if not args.sanity_only:
        summary["hero_example"] = pick_hero_example(
            out["probs"], out["chains"], out["solved"], node_to_tok,
        )

    with open(out_dir / "logit_lens_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[lens] wrote {out_dir}/logit_lens.npz and logit_lens_summary.json")

    if args.sanity_only:
        if not (sanity["logits_match_pass"] and sanity["argmax_agree_pass"]):
            print("[lens] SANITY FAILED - investigate before running full sweep")
            raise SystemExit(2)
        print("[lens] SANITY PASSED")


if __name__ == "__main__":
    main()
