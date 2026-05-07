"""
H2 linear probing on trained dePo checkpoints.

Tests the hypothesis that each loop iteration of a looped transformer corresponds
to one (or a constant fraction of) algorithmic hop step. We extract per-iteration
hidden states at the '?' position of each query, then train linear and MLP probes
to decode the ground-truth intermediate node n_h (the node reached after h hops).

Methodology follows Alain & Bengio (2016) for linear probes and Hewitt & Liang
(2019) for the shuffled-label control task. We report both raw probe accuracy
and selectivity (real - control).

Usage:
    uv run python -m looped_lm.probe_h2 \
        --checkpoint runs/ouro-8x4_s42/checkpoint_latest.pt \
        --config_name ouro-8x4 \
        --out_dir results/probes/ouro-8x4 \
        --H_eval 10 --n_samples 5000

Outputs (in --out_dir):
    probe_results.json   summary metrics + config
    probe_accuracies.npz  full (probe_type, k, h) accuracy tensors
    trajectory.npz        L2 / cosine state-change vs k, by H
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from looped_lm.datasets.pointer_chasing import generate_pointer_chasing_sample
from looped_lm.model import LoopedTransformer, Transformer
from looped_lm.model_configs import create_model
from looped_lm.utils import TokenMappingTokenizer


# ---------------------------------------------------------------------------
# Training-args resolution
# ---------------------------------------------------------------------------
# The probe pipeline must match the training-time positional-encoding scheme
# (use_nope) and architecture init (zero_init) or the model is evaluated against
# a different position regime than it was trained on, collapsing accuracy to the
# fallback baseline. Resolve from the run's args.json next to the checkpoint.

def load_train_args(checkpoint_path) -> dict:
    ckpt = Path(checkpoint_path).resolve()
    for d in [ckpt.parent, ckpt.parent.parent]:
        cand = d / "args.json"
        if cand.exists():
            with open(cand) as f:
                cfg = json.load(f)
            print(f"[probe] resolved training args from {cand}: "
                  f"use_nope={cfg.get('use_nope')} zero_init={cfg.get('zero_init')} "
                  f"max_len={cfg.get('max_len')}")
            return cfg
    print(f"[probe] WARN: no args.json found near {ckpt}; using CLI defaults only")
    return {}


# ---------------------------------------------------------------------------
# Tokenizer reconstruction (must match training-time vocab map exactly)
# ---------------------------------------------------------------------------

def build_tokenizer(N: int, base_name: str = "hf-internal-testing/llama-tokenizer"):
    """Recreate the TokenMappingTokenizer using the same procedure as training.

    The training script (`looped_lm.train_uniform_h`) builds the token map
    from 50 samples per H in {1..15} at seed=999. We replicate that exactly so
    vocab IDs match the trained model.
    """
    base = AutoTokenizer.from_pretrained(base_name)
    if base.pad_token_id is None:
        base.pad_token_id = base.eos_token_id

    rng = random.Random(999)
    map_texts = [
        generate_pointer_chasing_sample(rng, N, H=h, M=5)[0]
        for h in range(1, 16)
        for _ in range(50)
    ]

    tokenizer = TokenMappingTokenizer(base)
    tokenizer.estimate_token_map_from_dataset(map_texts)
    return tokenizer


def find_query_positions(tokens: torch.Tensor, q_token_ids: set[int]) -> list[list[int]]:
    """For each row in `tokens`, return positions of the token AFTER each '?' token.

    LLaMA tokenizes the format `... ? <digit> |` as `[▁?, ▁, <digit>, ▁|]`. So
    the '?' itself is followed by a `▁` space token, then the answer digit. We
    probe (and sanity-check) at position `?+1` (the `▁`), since that is where
    the model commits to predicting the answer digit at `?+2`.
    """
    out = []
    for row in tokens.tolist():
        positions = [i + 1 for i, t in enumerate(row) if t in q_token_ids]
        out.append(positions)
    return out


def get_q_token_ids(tokenizer) -> set[int]:
    """Return the set of token IDs that decode to a string containing '?'."""
    if hasattr(tokenizer, "old_tokenizer"):
        old_tok = tokenizer.old_tokenizer
        vocab = old_tok.get_vocab()
        raw = [v for k, v in vocab.items() if "?" in k]
        if tokenizer.map_old_to_new is not None:
            return {tokenizer.map_old_to_new[t] for t in raw if t in tokenizer.map_old_to_new}
        return set(raw)
    vocab = tokenizer.get_vocab()
    return {v for k, v in vocab.items() if "?" in k}


# ---------------------------------------------------------------------------
# Eval-set generation and tokenization
# ---------------------------------------------------------------------------

def generate_eval_set(
    N: int, H_eval: int, M: int, n_samples: int, seed: int
) -> list[dict]:
    """Generate dePo samples with full chain metadata. Seed must differ from training."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        text, _, meta = generate_pointer_chasing_sample(rng, N=N, H=H_eval, M=M)
        out.append({"text": text, "meta": meta})
    return out


def tokenize_batch(texts: list[str], tokenizer, max_len: int) -> dict:
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {
        "tokens": enc["input_ids"].long(),
        "attention_mask": enc["attention_mask"].to(torch.bool),
    }


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_states_looped(
    model: LoopedTransformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    K: int,
    device: str,
) -> torch.Tensor:
    """Return tensor of shape (K+1, batch, seq_len, dim).

    Index 0 is the post-embedding state (iteration-0 baseline), indices 1..K
    are post-iteration-k states. For ouro models with n_prelude=0 this matches
    "before any compute" vs "after k iterations".
    """
    model.eval()
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)

    # Iteration-0 baseline: post-embedding (with prelude applied if any)
    x0 = model.token_embeddings(tokens)
    for layer in model.prelude_layers:
        x0 = layer(x0, attention_mask)

    # K post-iteration states
    states = model.get_intermediate_states(tokens, attention_mask=attention_mask, K=K)

    out = torch.stack([x0.detach()] + states, dim=0)  # (K+1, B, T, D)
    return out.cpu()


@torch.no_grad()
def extract_states_deep(
    model: Transformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Return tensor of shape (n_layers+1, batch, seq_len, dim) via forward hooks.

    Index 0 is the post-embedding state, indices 1..n_layers are post-block states.
    """
    model.eval()
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)

    captured: list[torch.Tensor] = []

    # Pre-embedding (matches index 0 of looped extractor for symmetry)
    x0 = model.token_embeddings(tokens)
    captured.append(x0.detach())

    handles = []
    for layer in model.layers:
        handles.append(
            layer.register_forward_hook(
                lambda mod, inp, out: captured.append(out.detach())
            )
        )

    try:
        _ = model(tokens, attention_mask)
    finally:
        for h in handles:
            h.remove()

    out = torch.stack(captured, dim=0)  # (n_layers+1, B, T, D)
    return out.cpu()


# ---------------------------------------------------------------------------
# Probe-input construction (states at '?' positions, paired with chain targets)
# ---------------------------------------------------------------------------

def model_predict_at_q(
    eval_set: list[dict], tokenizer, model, device: str, max_len: int,
    batch_size: int, K: int = None,
) -> dict:
    """Sanity check: at the probe position (the `▁` after each `?`), compare the
    model's next-token argmax to the actual next token in the input. If the
    model was trained to MLRD=H and we eval at H_eval<=H, this should be near
    100% (it is just greedy decoding of the trained behaviour).
    """
    q_token_ids = get_q_token_ids(tokenizer)
    correct, total = 0, 0
    is_looped = isinstance(model, LoopedTransformer)

    with torch.no_grad():
        for start in range(0, len(eval_set), batch_size):
            chunk = eval_set[start : start + batch_size]
            texts = [s["text"] for s in chunk]
            tok = tokenize_batch(texts, tokenizer, max_len=max_len)
            tokens_dev = tok["tokens"].to(device)
            mask_dev = tok["attention_mask"].to(device)
            if is_looped and K is not None:
                logits = model(tokens_dev, mask_dev, K=K)
            else:
                logits = model(tokens_dev, mask_dev)
            preds = logits.argmax(-1).cpu()
            tokens = tok["tokens"]
            # find_query_positions returns position of the `▁` immediately after
            # each `?`. The model's prediction at that position should match the
            # actual next token in the sequence (the answer digit).
            positions = find_query_positions(tokens, q_token_ids)
            for b, sample in enumerate(chunk):
                qmeta = sample["meta"]["queries"]
                qpos = positions[b]
                for q_idx, q in enumerate(qmeta):
                    if q_idx >= len(qpos):
                        continue
                    pos = qpos[q_idx]
                    if pos + 1 >= tokens.size(1):
                        continue
                    pred_tok = preds[b, pos].item()
                    actual_next = tokens[b, pos + 1].item()
                    if pred_tok == actual_next:
                        correct += 1
                    total += 1
    return {
        "next_token_accuracy_at_probe_pos": correct / max(total, 1),
        "n_evaluated": total,
    }


def build_probe_inputs(
    eval_set: list[dict],
    tokenizer,
    model,
    device: str,
    K_or_layers: int,
    is_looped: bool,
    max_len: int,
    batch_size: int,
    M: int,
    H_eval: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model forward on eval set and extract per-(query, iteration) probe inputs.

    Returns:
        states: float32 array (n_total, n_levels, dim) where
                n_total = sum over samples of (number of '?' positions found),
                n_levels = K+1 for looped, n_layers+1 for deep.
        targets: int64 array (n_total, H_eval+1) of node IDs (n_0, ..., n_H).
    """
    q_token_ids = get_q_token_ids(tokenizer)

    all_states = []
    all_targets = []

    for start in range(0, len(eval_set), batch_size):
        chunk = eval_set[start : start + batch_size]
        texts = [s["text"] for s in chunk]
        tok = tokenize_batch(texts, tokenizer, max_len=max_len)

        if is_looped:
            states = extract_states_looped(
                model, tok["tokens"], tok["attention_mask"],
                K=K_or_layers, device=device,
            )
        else:
            states = extract_states_deep(
                model, tok["tokens"], tok["attention_mask"], device=device,
            )
        # states: (n_levels, B, T, D)

        positions = find_query_positions(tok["tokens"], q_token_ids)

        for b, sample in enumerate(chunk):
            qmeta = sample["meta"]["queries"]
            qpos = positions[b]
            # If tokenization split a node into multiple tokens, we may have
            # extra '?' or fewer; truncate / skip mismatched samples.
            if len(qpos) < len(qmeta):
                continue
            for q_idx, q in enumerate(qmeta):
                pos = qpos[q_idx]
                hidden = states[:, b, pos, :].numpy()  # (n_levels, D)
                target = np.array(q["intermediates"], dtype=np.int64)  # (H+1,)
                if target.shape[0] != H_eval + 1:
                    continue
                all_states.append(hidden)
                all_targets.append(target)

    return np.stack(all_states), np.stack(all_targets)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, dim_in: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, dim_in: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    probe_type: str = "linear",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    wd: float = 1e-4,
    device: str = "cpu",
) -> float:
    """Train a probe and return test accuracy."""
    dim = X_train.shape[1]
    if probe_type == "linear":
        model = LinearProbe(dim, n_classes).to(device)
    elif probe_type == "mlp":
        model = MLPProbe(dim, n_classes).to(device)
    else:
        raise ValueError(probe_type)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr = torch.from_numpy(X_train).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(X_test).float().to(device)
    yte = torch.from_numpy(y_test).long().to(device)

    n = Xtr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(-1)
        acc = (pred == yte).float().mean().item()
    return acc


def run_probe_grid(
    states: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    train_frac: float,
    seed: int,
    device: str,
    probe_types: tuple[str, ...] = ("linear", "mlp"),
) -> dict:
    """Train probes for every (probe_type, level k, hop h) and (real, shuffled-control).

    Returns a dict of np.ndarrays indexed by f"{probe_type}_{kind}" where kind in
    {"real", "shuffled"}, each of shape (n_levels, H+1).
    """
    n_total, n_levels, dim = states.shape
    H_plus_1 = targets.shape[1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_train = int(train_frac * n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    out = {}
    for probe_type in probe_types:
        for kind in ("real", "shuffled"):
            out[f"{probe_type}_{kind}"] = np.full(
                (n_levels, H_plus_1), np.nan, dtype=np.float32
            )

    t_start = time.time()
    for k in range(n_levels):
        Xk = states[:, k, :].astype(np.float32)
        for h in range(H_plus_1):
            yh = targets[:, h]

            for probe_type in probe_types:
                # Real probe
                acc_real = train_probe(
                    Xk[train_idx], yh[train_idx],
                    Xk[test_idx], yh[test_idx],
                    n_classes=n_classes, probe_type=probe_type, device=device,
                )

                # Hewitt-Liang shuffled-label control: same hidden states,
                # labels randomly permuted on the train set. We then evaluate
                # on the original test labels (which are NOT decodable from the
                # randomly-mapped train signal), so chance accuracy = 1/n_classes.
                shuffle_perm = rng.permutation(yh.shape[0])
                yh_shuf = yh[shuffle_perm]
                acc_ctrl = train_probe(
                    Xk[train_idx], yh_shuf[train_idx],
                    Xk[test_idx], yh[test_idx],
                    n_classes=n_classes, probe_type=probe_type, device=device,
                )

                out[f"{probe_type}_real"][k, h] = acc_real
                out[f"{probe_type}_shuffled"][k, h] = acc_ctrl

        elapsed = time.time() - t_start
        print(f"  level {k}/{n_levels-1} done ({elapsed:.0f}s elapsed)")

    return out


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def compute_accumulation_score(mat: np.ndarray, threshold: float = 0.5) -> dict:
    """Per-hop onset k* (earliest level where probe acc >= threshold) and
    persistence (mean acc at levels k >= k*). Mirrors compute_persistence in
    plot_probes.py but returns JSON-serialisable types so we can save per run.

    Args:
        mat: (n_levels, n_hops) probe accuracy. Hop 0 should already be excluded.
        threshold: probe-acc cutoff defining "decoded".

    Returns dict with int onsets, float persistence per hop, and a global
    persistence scalar averaged over hops that ever crossed.
    """
    n_levels, n_hops = mat.shape
    onsets = []
    persistence = []
    for h in range(n_hops):
        crossings = np.where(mat[:, h] >= threshold)[0]
        if crossings.size:
            k_star = int(crossings[0])
            onsets.append(k_star)
            persistence.append(float(mat[k_star:, h].mean()))
        else:
            onsets.append(-1)
            persistence.append(float("nan"))
    crossed = [p for p in persistence if not np.isnan(p)]
    global_p = float(np.mean(crossed)) if crossed else float("nan")
    return {
        "threshold": threshold,
        "onset_per_hop": onsets,
        "persistence_per_hop": persistence,
        "global_persistence": global_p,
        "n_hops_crossed": len(crossed),
        "n_hops_total": n_hops,
    }


def compute_trajectory(states: np.ndarray) -> dict:
    """Compute L2 distance and cosine similarity between consecutive levels."""
    n_total, n_levels, dim = states.shape
    if n_levels < 2:
        return {"l2": np.array([]), "cos": np.array([])}

    diff = states[:, 1:, :] - states[:, :-1, :]
    l2 = np.linalg.norm(diff, axis=-1).mean(axis=0)  # (n_levels-1,)

    a = states[:, :-1, :]
    b = states[:, 1:, :]
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    cos = (a_n * b_n).sum(axis=-1).mean(axis=0)  # (n_levels-1,)

    return {"l2": l2.astype(np.float32), "cos": cos.astype(np.float32)}


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
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--max_len", type=int, default=None,
                   help="Override training max_len. Default: read from training args.json.")
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--random_init", action="store_true",
                   help="Skip checkpoint load (random-init control)")
    p.add_argument("--use_nope", default=None, action=argparse.BooleanOptionalAction,
                   help="Override training use_nope. Default: read from training args.json.")
    p.add_argument("--zero_init", default=None, action=argparse.BooleanOptionalAction,
                   help="Override training zero_init. Default: read from training args.json.")
    p.add_argument("--probe_types", default="linear,mlp")
    p.add_argument("--k_sweep", action="store_true",
                   help="Run forward at K_eval in {1..K_max} and record model-head "
                        "accuracy per K_eval (looped models only). Used by H2 "
                        "strengthening pass to quantify fixed-K commitment deferral.")
    p.add_argument("--sanity_min_acc", type=float, default=0.6,
                   help="Abort if model next-token-acc at probe pos is below this. "
                        "Catches PE mismatch (fallback baseline ~0.4375). "
                        "Set to 0.0 to skip; --random_init implicitly skips.")
    args = p.parse_args()

    # Resolve runtime config from training args (use_nope / zero_init / max_len)
    train_args = {} if args.random_init else load_train_args(args.checkpoint)
    if args.use_nope is None:
        args.use_nope = bool(train_args.get("use_nope", True))
    if args.zero_init is None:
        args.zero_init = bool(train_args.get("zero_init", True))
    if args.max_len is None:
        args.max_len = int(train_args.get("max_len", 512))
    print(f"[probe] runtime: use_nope={args.use_nope} zero_init={args.zero_init} "
          f"max_len={args.max_len}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] config={args.config_name} ckpt={args.checkpoint}")
    print(f"[probe] N={args.N} M={args.M} H_eval={args.H_eval} n_samples={args.n_samples}")

    # Tokenizer (must match training)
    print("[probe] building tokenizer...")
    tokenizer = build_tokenizer(args.N)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    print(f"[probe] vocab_size={vocab_size} pad_id={pad_id}")

    # Model
    print("[probe] creating model...")
    model, config = create_model(
        args.config_name, vocab_size=vocab_size,
        max_sequence_len=args.max_len, pad_token_idx=pad_id,
        use_nope=args.use_nope, zero_init=args.zero_init,
    )
    model = model.to(args.device)

    is_looped = isinstance(model, LoopedTransformer)
    if not args.random_init:
        print(f"[probe] loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[probe] RANDOM-INIT control: skipping checkpoint load.")
    model.eval()

    if is_looped:
        n_levels_label = "iteration"
        K = config["K_max"]
    else:
        n_levels_label = "layer"
        K = config["n_layers"]
    print(f"[probe] is_looped={is_looped}, {n_levels_label}s={K}")

    # Eval set
    print(f"[probe] generating {args.n_samples} eval samples (seed={args.eval_seed})...")
    eval_set = generate_eval_set(
        N=args.N, H_eval=args.H_eval, M=args.M,
        n_samples=args.n_samples, seed=args.eval_seed,
    )

    # Sanity check: model's own next-token accuracy at '?' position
    print("[probe] running model-prediction sanity check at '?' positions...")
    sanity = model_predict_at_q(
        eval_set=eval_set[: min(500, len(eval_set))], tokenizer=tokenizer,
        model=model, device=args.device, max_len=args.max_len,
        batch_size=args.batch_size, K=K if is_looped else None,
    )
    print(f"[probe] model next-token-at-probe-pos accuracy = "
          f"{sanity['next_token_accuracy_at_probe_pos']:.3f} on {sanity['n_evaluated']} queries")

    # Sanity guard: catch PE / init mismatches before wasting compute on probes.
    # Fallback baseline is ~0.4375 for N=16 (predict-"1" default). If we see that
    # for a non-random-init model, the runtime config almost certainly disagrees
    # with training. Skip with --sanity_min_acc 0.
    if (not args.random_init) and sanity['next_token_accuracy_at_probe_pos'] < args.sanity_min_acc:
        msg = (
            f"[probe] PROBE SANITY FAILED: next-token-acc = "
            f"{sanity['next_token_accuracy_at_probe_pos']:.3f} < "
            f"{args.sanity_min_acc} at H_eval={args.H_eval}.\n"
            f"  Likely causes:\n"
            f"    (a) PE mismatch: probe use_nope={args.use_nope} vs "
            f"train use_nope={train_args.get('use_nope')}\n"
            f"    (b) init mismatch: probe zero_init={args.zero_init} vs "
            f"train zero_init={train_args.get('zero_init')}\n"
            f"    (c) checkpoint not yet saturated H_eval={args.H_eval}\n"
            f"  Aborting before probe training. Override with --sanity_min_acc 0 if intentional."
        )
        raise RuntimeError(msg)

    # Hidden states + targets
    print("[probe] extracting hidden states...")
    t0 = time.time()
    states, targets = build_probe_inputs(
        eval_set=eval_set,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        K_or_layers=K,
        is_looped=is_looped,
        max_len=args.max_len,
        batch_size=args.batch_size,
        M=args.M,
        H_eval=args.H_eval,
    )
    print(f"[probe] states shape={states.shape}, targets shape={targets.shape} "
          f"({time.time()-t0:.0f}s)")

    if states.shape[0] < 1000:
        print(f"WARNING: only {states.shape[0]} probe inputs. Tokenization may "
              f"be dropping samples (see build_probe_inputs).")

    # Probes
    probe_types = tuple(args.probe_types.split(","))
    print(f"[probe] training probes ({probe_types})...")
    t0 = time.time()
    accs = run_probe_grid(
        states=states, targets=targets,
        n_classes=args.N, train_frac=args.train_frac,
        seed=args.probe_seed, device=args.device,
        probe_types=probe_types,
    )
    print(f"[probe] probe grid done ({time.time()-t0:.0f}s)")

    # Trajectory
    print("[probe] computing trajectory...")
    traj = compute_trajectory(states)

    # K-sweep: model-head accuracy at K_eval in {1..K_max} (looped models only).
    # The probe staircase is recorded *during* a single K=K_max forward pass
    # (intermediate states), so probe(iter k, hop h) tells us the *linear
    # decodability* of hop h at iteration k. Running the model with K_eval=k and
    # reading the answer through the unembedding tells us whether the model
    # itself commits to that answer at k iterations. The gap quantifies the
    # "early iterations as scratch" prediction of fixed-K training.
    k_sweep_head = None
    if args.k_sweep and is_looped:
        print(f"[probe] running K-sweep model-head accuracy for K_eval in 1..{K}...")
        k_sweep_head = np.zeros(K, dtype=np.float32)
        for k_eval in range(1, K + 1):
            r = model_predict_at_q(
                eval_set=eval_set, tokenizer=tokenizer,
                model=model, device=args.device, max_len=args.max_len,
                batch_size=args.batch_size, K=k_eval,
            )
            acc = r["next_token_accuracy_at_probe_pos"]
            k_sweep_head[k_eval - 1] = acc
            print(f"  K_eval={k_eval}: head_acc={acc:.3f} (n={r['n_evaluated']})")
    elif args.k_sweep and not is_looped:
        print("[probe] --k_sweep ignored on unlooped model")

    # Accumulation score: per-hop onset (earliest level k* where linear probe
    # acc >= 0.5) and persistence (mean acc at k >= k*). Captures the
    # "once decoded, info is retained" claim. Computed on the linear-real
    # probe matrix excluding hop 0 (the start node, trivially decodable).
    real_for_score = accs["linear_real"][:, 1:]
    accumulation = compute_accumulation_score(real_for_score, threshold=0.5)
    print(f"[probe] accumulation: global_persistence={accumulation['global_persistence']:.3f}")

    # Save
    np.savez(out_dir / "probe_accuracies.npz", **accs)
    np.savez(out_dir / "trajectory.npz", **traj)
    np.savez(out_dir / "states_targets.npz", states=states, targets=targets)
    if k_sweep_head is not None:
        np.savez(out_dir / "k_sweep_head_accuracy.npz", head_acc=k_sweep_head)
    with open(out_dir / "accumulation_score.json", "w") as f:
        json.dump(accumulation, f, indent=2)

    summary = {
        "config_name": args.config_name,
        "checkpoint": args.checkpoint,
        "random_init": args.random_init,
        "is_looped": is_looped,
        "n_levels": int(states.shape[1]),
        "n_levels_label": n_levels_label,
        "n_classes": args.N,
        "n_probe_examples": int(states.shape[0]),
        "H_eval": args.H_eval,
        "n_samples": args.n_samples,
        "M": args.M,
        "model_config": config,
        "args": vars(args),
        "accuracy_summary": {
            kind: {
                "max": float(np.nanmax(v)),
                "argmax": [int(x) for x in np.unravel_index(np.nanargmax(v), v.shape)],
                "mean": float(np.nanmean(v)),
            }
            for kind, v in accs.items()
        },
        "model_prediction_sanity": sanity,
        "accumulation_score": accumulation,
        "k_sweep_head_accuracy": (
            [float(x) for x in k_sweep_head] if k_sweep_head is not None else None
        ),
    }
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[probe] DONE. Artifacts in {out_dir}")
    for k, v in summary["accuracy_summary"].items():
        print(f"  {k}: max={v['max']:.3f} at {v['argmax']}, mean={v['mean']:.3f}")


if __name__ == "__main__":
    main()
