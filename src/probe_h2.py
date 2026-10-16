"""
Linear probes on trained dePo checkpoints.

We extract the hidden state at the readout position of each query (the space
token after '?', which predicts the answer's first digit) after every iteration
(or every layer, --per_layer), then train linear (and optionally MLP) probes to
decode the ground-truth intermediate node z_h = pi^h(s), the node reached after
h hops.

Methodology follows Alain & Bengio (2016) for linear probes and Hewitt & Liang
(2019) for the shuffled-label control task. We report both raw probe accuracy
and selectivity (real - control).

Usage (the paper's settings are in recipes/probe_*.sh):
    uv run python src/probe_h2.py \
        --checkpoint runs/h1_clean_sweep/ouro-4x8_s42/checkpoint.pt \
        --config_name ouro-4x8 --out_dir probe_outputs/ouro-4x8_s42_H4 \
        --H_eval 4 --n_samples 5000 --split_by_prompt --per_layer --probe_types linear

Outputs (in --out_dir):
    probe_results.json   summary metrics + config
    probe_accuracies.npz  full (probe_type, k, h) accuracy tensors
    trajectory.npz        L2 / cosine state-change vs k, by H
"""

from __future__ import annotations

import argparse
import hashlib
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

from src.datasets.pointer_chasing import generate_pointer_chasing_sample
from src.model import LoopedTransformer, Transformer
from src.model_configs import create_model
from src.utils import TokenMappingTokenizer


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

def build_tokenizer(
    N: int,
    base_name: str = "hf-internal-testing/llama-tokenizer",
    M: int = 5,
    H_min: int = 1,
    H_max: int = 15,
):
    """Recreate the TokenMappingTokenizer using the same procedure as training.

    The trainer (src/train_uniform_h.py) builds the token map from 50 samples
    per H in [H_min, H_max] at seed=999. We replicate that exactly so vocab IDs
    match the trained model.
    A mismatch here does not error, it silently permutes token IDs and the
    sanity gate then fires, so keep the two paths in step.
    """
    base = AutoTokenizer.from_pretrained(base_name)
    if base.pad_token_id is None:
        base.pad_token_id = base.eos_token_id

    rng = random.Random(999)
    map_texts = [
        generate_pointer_chasing_sample(rng, N, H=h, M=M)[0]
        for h in range(H_min, H_max + 1)
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

def generate_eval_set(N: int, H_eval: int, M: int, n_samples: int, seed: int) -> list[dict]:
    """Generate dePo samples with full per-hop metadata. Seed must differ from training."""
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
    per_layer: bool = False,
) -> torch.Tensor:
    """Return tensor of shape (K+1, batch, seq_len, dim).

    Index 0 is the post-embedding state (iteration-0 baseline), indices 1..K
    are post-iteration-k states ("before any compute" vs "after k iterations").
    With per_layer=True the shape is (K*L+1, ...), one state after every
    layer inside every iteration (block boundaries every L entries), matching
    the untied stack's per-layer probe axis.
    """
    model.eval()
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)

    # Iteration-0 baseline: post-embedding
    x0 = model.token_embeddings(tokens)

    # K post-iteration states (or K*L per-layer states)
    states = model.get_intermediate_states(
        tokens, attention_mask=attention_mask, K=K, per_layer=per_layer)

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
    model has solved hops up to H and we eval at H_eval<=H, this should be near
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


def pointer_query_record(meta: dict, prompt_id: int, query_index: int) -> dict:
    """Describe a retained query relative to all earlier queries in its prompt."""
    query = meta["queries"][query_index]
    repeated = any(q["start"] == query["start"]
                   for q in meta["queries"][:query_index])
    return {
        "prompt_ids": prompt_id,
        "query_indices": query_index,
        "start_nodes": query["start"],
        "answer_nodes": query["answer"],
        "is_first_query": query_index == 0,
        "is_fresh_start": not repeated,
        "is_repeated_start": repeated,
    }


def query_stratum_masks(metadata: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """First queries are a subset of fresh starts; fresh/repeated partition rows."""
    return {
        "first_query": metadata["is_first_query"],
        "fresh_start": metadata["is_fresh_start"],
        "repeated_start": metadata["is_repeated_start"],
    }


def summarize_query_accuracy(
    correct: np.ndarray, metadata: dict[str, np.ndarray],
) -> dict:
    strata = {"all": np.ones(correct.size, dtype=bool),
              **query_stratum_masks(metadata)}
    summary = {}
    for name, mask in strata.items():
        count = int(mask.sum())
        n_correct = int(correct[mask].sum())
        summary[name] = {
            "n_evaluated": count, "n_correct": n_correct,
            "accuracy": n_correct / count if count else None,
        }
    return summary


def hash_evaluation_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(json.dumps(text, ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def complete_answer_spans(
    tokens: list[int], valid: list[bool], q_token_ids: set[int],
    separator_token_ids: set[int], tokenizer, answers: list[int],
) -> list[tuple[int, int]]:
    """Return [answer start, separator] indices, failing on truncated spans.

    The answer span includes whitespace after '?' and every node digit. Its
    decoded gold string must equal the metadata node. The separator is a real
    '|' token, never a padding or EOS target.
    """
    questions = [i for i, token in enumerate(tokens) if valid[i] and token in q_token_ids]
    if len(questions) != len(answers):
        raise ValueError("Complete-answer scoring requires every query to survive tokenization")
    spans = []
    for index, (question, answer) in enumerate(zip(questions, answers)):
        limit = questions[index + 1] if index + 1 < len(questions) else len(tokens)
        boundary = next((i for i in range(question + 1, limit)
                         if valid[i] and tokens[i] in separator_token_ids), None)
        if boundary is None:
            raise ValueError("Missing answer separator: scoring prompt may be truncated")
        start = question + 1
        if not all(valid[start:boundary + 1]):
            raise ValueError("Padding inside an answer span")
        if tokenizer.decode(tokens[start:boundary]).strip() != str(answer):
            raise ValueError("Tokenized answer span disagrees with the metadata node")
        spans.append((start, boundary))
    return spans


def score_complete_answer_span(
    predictions: list[int], tokens: list[int], start: int, boundary: int,
) -> tuple[bool, bool, bool]:
    """Check every answer token and its boundary along the gold causal prefix.

    Matching all of these argmaxes is also greedy exact sequence correctness
    through this delimiter; a correct first digit alone is insufficient.
    """
    answer_correct = predictions[start - 1:boundary - 1] == tokens[start:boundary]
    boundary_correct = predictions[boundary - 1] == tokens[boundary]
    return answer_correct, boundary_correct, answer_correct and boundary_correct


def model_complete_answer_at_q(
    eval_set: list[dict], tokenizer, model, device: str, max_len: int,
    batch_size: int, query_metadata: dict[str, np.ndarray], K: int = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Score complete nodes in both families on exactly the retained queries.

    Earlier query answers remain teacher forced, as during probe extraction.
    Append a scoring-only ' |' to each original prompt so the final query also
    has a concrete termination target. Causal attention prevents this suffix
    from changing any original query state or earlier answer prediction.
    """
    q_ids = get_q_token_ids(tokenizer)
    separator_ids = {i for i in range(len(tokenizer))
                     if tokenizer.decode([i]).strip() == "|"}
    if not separator_ids:
        raise ValueError("Tokenizer has no standalone answer separator")
    scoring_texts = [sample["text"] + " |" for sample in eval_set]
    rows = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(
        query_metadata["prompt_ids"], query_metadata["query_indices"]))}
    n_rows = query_metadata["prompt_ids"].size
    if len(rows) != n_rows:
        raise ValueError("Duplicate retained prompt/query identifiers")
    arrays = {name: np.zeros(n_rows, dtype=bool) for name in
              ("answer_span_correct", "boundary_correct", "complete_node_correct")}
    evaluated = np.zeros(n_rows, dtype=bool)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_set), batch_size):
            chunk = eval_set[start:start + batch_size]
            tok = tokenize_batch(scoring_texts[start:start + batch_size], tokenizer,
                                 max_len=max_len)
            tokens = tok["tokens"]
            mask = tok["attention_mask"]
            if isinstance(model, LoopedTransformer) and K is not None:
                logits = model(tokens.to(device), mask.to(device), K=K)
            else:
                logits = model(tokens.to(device), mask.to(device))
            predictions = logits.argmax(-1).cpu().tolist()
            for b, sample in enumerate(chunk):
                token_row = tokens[b].tolist()
                spans = complete_answer_spans(
                    token_row, mask[b].tolist(), q_ids, separator_ids, tokenizer,
                    [q["answer"] for q in sample["meta"]["queries"]],
                )
                for q_index, (answer_start, boundary) in enumerate(spans):
                    row = rows.get((start + b, q_index))
                    if row is None:
                        continue
                    scores = score_complete_answer_span(
                        predictions[b], token_row, answer_start, boundary)
                    for name, value in zip(arrays, scores):
                        arrays[name][row] = value
                    evaluated[row] = True
    if not evaluated.all():
        raise ValueError("Complete-answer scoring failed to cover every retained query")
    strata = summarize_query_accuracy(arrays["complete_node_correct"], query_metadata)
    summary = {
        "protocol": "teacher_forced_answer_tokens_plus_delimiter_v1",
        "previous_query_answers": "teacher_forced",
        "termination_target": "|",
        "scoring_suffix": " |",
        "scoring_suffix_note": "Appended after the final gold node; never scored as padding or EOS.",
        "eval_text_sha256": hash_evaluation_texts([s["text"] for s in eval_set]),
        "scoring_text_sha256": hash_evaluation_texts(scoring_texts),
        "n_evaluated": n_rows,
        "complete_node_accuracy": strata["all"]["accuracy"],
        "answer_span_accuracy": float(arrays["answer_span_correct"].mean()),
        "boundary_accuracy": float(arrays["boundary_correct"].mean()),
        "strata": strata,
    }
    return summary, {"prompt_ids": query_metadata["prompt_ids"],
                     "query_indices": query_metadata["query_indices"], **arrays}


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
    collect_correctness: bool = False,
    per_layer: bool = False,
    query_metadata_out: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Run model forward on eval set and extract per-(query, iteration) probe inputs.

    Returns:
        states: float32 array (n_total, n_levels, dim) where
                n_total = sum over samples of (number of '?' positions found),
                n_levels = K+1 for looped, n_layers+1 for deep.
        targets: int64 array (n_total, H_eval+1) of node IDs (n_0, ..., n_H).
        correct: bool array (n_total,) of model next-token correctness at the
                 probe position (full-K forward), or None unless
                 collect_correctness is set.
        prompt_ids: int64 array (n_total,) linking retained queries to their
                    original evaluation prompt, including when samples skip.
    """
    q_token_ids = get_q_token_ids(tokenizer)

    all_states = []
    all_targets = []
    all_correct = []
    all_prompt_ids = []

    for start in range(0, len(eval_set), batch_size):
        chunk = eval_set[start : start + batch_size]
        texts = [s["text"] for s in chunk]
        tok = tokenize_batch(texts, tokenizer, max_len=max_len)

        if is_looped:
            states = extract_states_looped(
                model, tok["tokens"], tok["attention_mask"],
                K=K_or_layers, device=device, per_layer=per_layer,
            )
        else:
            states = extract_states_deep(
                model, tok["tokens"], tok["attention_mask"], device=device,
            )
        # states: (n_levels, B, T, D)

        preds = None
        if collect_correctness:
            with torch.no_grad():
                tokens_dev = tok["tokens"].to(device)
                mask_dev = tok["attention_mask"].to(device)
                if is_looped:
                    logits = model(tokens_dev, mask_dev, K=K_or_layers)
                else:
                    logits = model(tokens_dev, mask_dev)
                preds = logits.argmax(-1).cpu()

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
                hidden = copy_query_state(states, b, pos)
                target = np.array(q["intermediates"], dtype=np.int64)  # (H+1,)
                if target.shape[0] != H_eval + 1:
                    continue
                if collect_correctness:
                    if pos + 1 >= tok["tokens"].size(1):
                        continue
                    all_correct.append(
                        preds[b, pos].item() == tok["tokens"][b, pos + 1].item()
                    )
                all_states.append(hidden)
                all_targets.append(target)
                all_prompt_ids.append(start + b)
                if query_metadata_out is not None:
                    query_metadata_out.append(pointer_query_record(sample["meta"], start + b, q_idx))

    correct = np.array(all_correct, dtype=bool) if collect_correctness else None
    return (np.stack(all_states), np.stack(all_targets), correct,
            np.asarray(all_prompt_ids, dtype=np.int64))


def copy_query_state(states: torch.Tensor, batch_index: int, position: int) -> np.ndarray:
    """Own only one query's states, releasing the full batch tensor afterward."""
    return states[:, batch_index, position, :].numpy().copy()


def make_probe_split(
    n_total: int,
    train_frac: float,
    rng: np.random.Generator,
    prompt_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split queries, optionally keeping every retained prompt entirely together."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie strictly between 0 and 1")
    if prompt_ids is None:
        perm = rng.permutation(n_total)
        n_train = int(train_frac * n_total)
        train_idx, test_idx = perm[:n_train], perm[n_train:]
    else:
        if prompt_ids.shape != (n_total,):
            raise ValueError("prompt_ids must contain one ID per probe query")
        groups = rng.permutation(np.unique(prompt_ids))
        n_train_groups = int(train_frac * groups.size)
        is_train = np.isin(prompt_ids, groups[:n_train_groups])
        train_idx, test_idx = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    if not train_idx.size or not test_idx.size:
        raise ValueError("probe split must contain both training and test queries")
    return train_idx, test_idx


def seed_probe_training(seed: int) -> None:
    """Seed probe initialization and minibatch order after model construction."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shuffle_training_labels(
    targets: np.ndarray, train_idx: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Construct the grouped-split control without using held-out labels."""
    return targets[train_idx][rng.permutation(train_idx.size)]


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
    return_model: bool = False,
    return_predictions: bool = False,
):
    """Train a probe and return test accuracy (optionally also the probe)."""
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
    if return_model:
        return acc, model
    if return_predictions:
        return acc, pred.cpu().numpy()
    return acc


def run_probe_grid(
    states: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    train_frac: float,
    seed: int,
    device: str,
    probe_types: tuple[str, ...] = ("linear", "mlp"),
    prompt_ids: np.ndarray | None = None,
    query_strata: dict[str, np.ndarray] | None = None,
    stratified_accuracies: dict[str, np.ndarray] | None = None,
) -> dict:
    """Train probes for every (probe_type, level k, hop h) and (real, shuffled-control).

    Returns a dict of np.ndarrays indexed by f"{probe_type}_{kind}" where kind in
    {"real", "shuffled"}, each of shape (n_levels, H+1).
    If supplied, stratified_accuracies is populated with held-out accuracies
    for the SAME fitted probes, indexed as e.g. linear_real__first_query.
    """
    n_total, n_levels, dim = states.shape
    H_plus_1 = targets.shape[1]

    rng = np.random.default_rng(seed)
    train_idx, test_idx = make_probe_split(n_total, train_frac, rng, prompt_ids)
    seed_probe_training(seed)

    test_strata = {}
    if query_strata is not None:
        if stratified_accuracies is None:
            raise ValueError("query_strata requires a stratified_accuracies output dictionary")
        for name, mask in query_strata.items():
            if mask.shape != (n_total,) or mask.dtype != np.bool_:
                raise ValueError("Each query stratum must be a boolean mask over all probe rows")
            test_strata[name] = mask[test_idx]

    out = {}
    for probe_type in probe_types:
        for kind in ("real", "shuffled"):
            out[f"{probe_type}_{kind}"] = np.full(
                (n_levels, H_plus_1), np.nan, dtype=np.float32
            )
            for stratum in test_strata:
                stratified_accuracies[f"{probe_type}_{kind}__{stratum}"] = np.full(
                    (n_levels, H_plus_1), np.nan, dtype=np.float32)

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
                    return_predictions=bool(test_strata),
                )

                # Hewitt-Liang shuffled-label control: same hidden states,
                # labels randomly permuted on the train set. We then evaluate
                # on the original test labels (which are NOT decodable from the
                # randomly-mapped train signal), so chance accuracy = 1/n_classes.
                if prompt_ids is None:
                    # Preserve the legacy control distribution for legacy runs.
                    yh_shuf_train = yh[rng.permutation(yh.shape[0])][train_idx]
                else:
                    yh_shuf_train = shuffle_training_labels(yh, train_idx, rng)
                acc_ctrl = train_probe(
                    Xk[train_idx], yh_shuf_train,
                    Xk[test_idx], yh[test_idx],
                    n_classes=n_classes, probe_type=probe_type, device=device,
                    return_predictions=bool(test_strata),
                )

                if test_strata:
                    acc_real, pred_real = acc_real
                    acc_ctrl, pred_ctrl = acc_ctrl
                    for kind, predicted in (("real", pred_real), ("shuffled", pred_ctrl)):
                        for stratum, mask in test_strata.items():
                            if mask.any():
                                stratified_accuracies[f"{probe_type}_{kind}__{stratum}"][k, h] = (
                                    predicted[mask] == yh[test_idx][mask]).mean()

                out[f"{probe_type}_real"][k, h] = acc_real
                out[f"{probe_type}_shuffled"][k, h] = acc_ctrl

        elapsed = time.time() - t_start
        print(f"  level {k}/{n_levels-1} done ({elapsed:.0f}s elapsed)")

    return out


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney); labels in {0,1}. NaN if one class absent."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # midranks for ties
    allsc = np.concatenate([pos, neg])
    for v in np.unique(allsc):
        m = allsc == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def run_correctness_split(
    states: np.ndarray,
    targets: np.ndarray,
    correct: np.ndarray,
    n_classes: int,
    train_frac: float,
    seed: int,
    device: str,
    prompt_ids: np.ndarray | None = None,
) -> dict:
    """Per-(level, hop) linear-probe accuracy split by model correctness,
    plus a logistic probe predicting model correctness from each level's state.

    `correct` is a bool array (n_total,) = model answered this query correctly.
    Probes are trained on the full train split; accuracy is reported separately
    over the model-correct and model-incorrect test examples. The correctness
    probe is a 2-class linear head; AUC per level via rank statistic.
    """
    n_total, n_levels, _ = states.shape
    H_plus_1 = targets.shape[1]

    rng = np.random.default_rng(seed)
    train_idx, test_idx = make_probe_split(n_total, train_frac, rng, prompt_ids)
    seed_probe_training(seed)
    te_correct = correct[test_idx]

    acc_c = np.full((n_levels, H_plus_1), np.nan, dtype=np.float32)
    acc_i = np.full((n_levels, H_plus_1), np.nan, dtype=np.float32)
    auc = np.full(n_levels, np.nan, dtype=np.float32)

    t_start = time.time()
    for k in range(n_levels):
        Xk = states[:, k, :].astype(np.float32)
        Xte = torch.from_numpy(Xk[test_idx]).to(device)
        for h in range(H_plus_1):
            yh = targets[:, h]
            _, probe = train_probe(
                Xk[train_idx], yh[train_idx], Xk[test_idx], yh[test_idx],
                n_classes=n_classes, probe_type="linear", device=device,
                return_model=True,
            )
            probe.eval()
            with torch.no_grad():
                pred = probe(Xte).argmax(-1).cpu().numpy()
            hit = pred == yh[test_idx]
            if te_correct.sum() > 0:
                acc_c[k, h] = float(hit[te_correct].mean())
            if (~te_correct).sum() > 0:
                acc_i[k, h] = float(hit[~te_correct].mean())

        # logistic probe: model correctness from the level-k state
        _, cprobe = train_probe(
            Xk[train_idx], correct[train_idx].astype(np.int64),
            Xk[test_idx], correct[test_idx].astype(np.int64),
            n_classes=2, probe_type="linear", device=device, return_model=True,
        )
        cprobe.eval()
        with torch.no_grad():
            score = torch.softmax(cprobe(Xte), dim=-1)[:, 1].cpu().numpy()
        auc[k] = _auc_from_scores(score, te_correct.astype(np.int64))
        print(f"  split level {k}/{n_levels-1} done, AUC={auc[k]:.3f} "
              f"({time.time()-t_start:.0f}s elapsed)")

    return {
        "acc_model_correct": acc_c,
        "acc_model_incorrect": acc_i,
        "correctness_auc_per_level": auc,
        "n_test_correct": int(te_correct.sum()),
        "n_test_incorrect": int((~te_correct).sum()),
    }


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
    p.add_argument("--split_by_prompt", action="store_true",
                   help="Hold out complete evaluation prompts, keeping all their "
                        "queries together; shuffled controls use training labels only. "
                        "Default retains the legacy query-row split.")
    p.add_argument("--complete_answer_eval", action="store_true",
                   help="Score every retained complete node plus its | boundary, "
                        "save query metadata, and report first/fresh/repeated-start strata. "
                        "Previous query answers remain teacher forced.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--random_init", action="store_true",
                   help="Skip checkpoint load (random-init control)")
    p.add_argument("--use_nope", default=None, action=argparse.BooleanOptionalAction,
                   help="Override training use_nope. Default: read from training args.json.")
    p.add_argument("--zero_init", default=None, action=argparse.BooleanOptionalAction,
                   help="Override training zero_init. Default: read from training args.json.")
    p.add_argument("--probe_types", default="linear,mlp")
    p.add_argument("--per_layer", action="store_true",
                   help="Looped models only: probe after every layer inside "
                        "every iteration (K*L+1 levels) instead of at block "
                        "boundaries (K+1 levels), for a depth-matched axis "
                        "against untied stacks.")
    p.add_argument("--k_sweep", action="store_true",
                   help="Run forward at K_eval in {1..K_max} and record model-head "
                        "accuracy per K_eval (looped models only).")
    p.add_argument("--split_by_correctness", action="store_true",
                   help="Report per-(level, hop) linear-probe accuracy "
                        "separately over model-correct and model-incorrect "
                        "queries, plus a correctness-probe AUC per level; write "
                        "correctness_split.npz and exit (skips the regular grid).")
    p.add_argument("--sanity_min_acc", type=float, default=0.6,
                   help="Abort if model next-token-acc at probe pos is below this. "
                        "Catches PE mismatch (fallback baseline ~0.4375). "
                        "Set to 0.0 to skip; --random_init implicitly skips.")
    args = p.parse_args()

    # Resolve runtime config from training args (use_nope / zero_init / max_len)
    train_args = {} if args.random_init else load_train_args(args.checkpoint)
    if args.use_nope is None:
        # No silent default: a RoPE-trained checkpoint loaded as NoPE (or the
        # reverse) is evaluated in the wrong position regime.
        if not args.random_init and "use_nope" not in train_args:
            raise RuntimeError(
                f"[probe] use_nope is not recorded in the training args for "
                f"{args.checkpoint}. Pass --use_nope or --no-use_nope explicitly "
                f"after checking how the run was trained."
            )
        args.use_nope = bool(train_args.get("use_nope", True))
    if args.zero_init is None:
        args.zero_init = bool(train_args.get("zero_init", True))
    args.no_timestep_embedding = bool(train_args.get("no_timestep_embedding", False))
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
    tokenizer = build_tokenizer(
        args.N, M=args.M,
        H_min=int(train_args.get("H_min", 1)),
        H_max=int(train_args.get("H_max", 15)),
    )
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    print(f"[probe] vocab_size={vocab_size} pad_id={pad_id}")

    # Model
    print("[probe] creating model...")
    model, config = create_model(
        args.config_name, vocab_size=vocab_size,
        max_sequence_len=args.max_len, pad_token_idx=pad_id,
        use_nope=args.use_nope, zero_init=args.zero_init,
        use_timestep_embedding=not args.no_timestep_embedding,
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
        n_levels_label = "layer" if args.per_layer else "iteration"
        K = config["K_max"]
    else:
        n_levels_label = "layer"
        K = config["n_layers"]
    if args.per_layer and not is_looped:
        print("[probe] --per_layer is a no-op for untied models (already per-layer)")
    print(f"[probe] is_looped={is_looped}, {n_levels_label}s per level, K_or_layers={K}, "
          f"per_layer={args.per_layer}")

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
    query_records = [] if args.complete_answer_eval else None
    states, targets, correct, prompt_ids = build_probe_inputs(
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
        collect_correctness=args.split_by_correctness,
        per_layer=args.per_layer and is_looped,
        query_metadata_out=query_records,
    )
    print(f"[probe] states shape={states.shape}, targets shape={targets.shape} "
          f"({time.time()-t0:.0f}s)")

    if states.shape[0] < 1000:
        print(f"WARNING: only {states.shape[0]} probe inputs. Tokenization may "
              f"be dropping samples (see build_probe_inputs).")

    split_groups = prompt_ids if args.split_by_prompt else None
    train_idx, test_idx = make_probe_split(
        states.shape[0], args.train_frac, np.random.default_rng(args.probe_seed),
        split_groups,
    )
    split_hash = hashlib.sha256()
    for values in (prompt_ids, train_idx, test_idx):
        split_hash.update(np.asarray(values, dtype="<i8").tobytes())
    split_provenance = {
        "unit": "prompt" if args.split_by_prompt else "query",
        "eval_seed": args.eval_seed,
        "probe_seed": args.probe_seed,
        "torch_probe_seed": args.probe_seed,
        "train_frac": args.train_frac,
        "n_train_queries": int(train_idx.size),
        "n_test_queries": int(test_idx.size),
        "n_train_prompts": int(np.unique(prompt_ids[train_idx]).size),
        "n_test_prompts": int(np.unique(prompt_ids[test_idx]).size),
        "n_shared_prompts": int(np.intersect1d(
            prompt_ids[train_idx], prompt_ids[test_idx]).size),
        "control_labels": "training_partition_only" if args.split_by_prompt else "legacy_all_queries",
        "eval_text_sha256": hash_evaluation_texts([s["text"] for s in eval_set]),
        "split_arrays_sha256": split_hash.hexdigest(),
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    np.savez(out_dir / "probe_split.npz", prompt_ids=prompt_ids,
             train_idx=train_idx, test_idx=test_idx)
    with open(out_dir / "probe_split.json", "w") as f:
        json.dump(split_provenance, f, indent=2)

    complete_answer_summary = None
    query_strata = None
    probe_strata = None
    if args.complete_answer_eval:
        query_metadata = {
            name: np.asarray([row[name] for row in query_records],
                             dtype=bool if name.startswith("is_") else np.int64)
            for name in query_records[0]
        }
        if not np.array_equal(query_metadata["prompt_ids"], prompt_ids):
            raise ValueError("Query metadata lost alignment with extracted states")
        np.savez(out_dir / "query_metadata.npz", **query_metadata)
        query_strata = query_stratum_masks(query_metadata)
        probe_strata = {
            name: {"n_train_queries": int(mask[train_idx].sum()),
                   "n_test_queries": int(mask[test_idx].sum())}
            for name, mask in query_strata.items()
        }
        print("[probe] scoring complete nodes and answer boundaries on all retained queries...")
        complete_answer_summary, complete_answer_arrays = model_complete_answer_at_q(
            eval_set, tokenizer, model, args.device, args.max_len, args.batch_size,
            query_metadata, K=K if is_looped else None,
        )
        np.savez(out_dir / "model_complete_answer.npz", **complete_answer_arrays)
        with open(out_dir / "model_complete_answer.json", "w") as f:
            json.dump(complete_answer_summary, f, indent=2)
        print(f"[probe] complete node + boundary accuracy = "
              f"{complete_answer_summary['complete_node_accuracy']:.3f} "
              f"on {complete_answer_summary['n_evaluated']} queries")

    # Correctness-split mode: dedicated output, then exit.
    if args.split_by_correctness:
        n_inc = int((~correct).sum())
        print(f"[probe] correctness-split mode: {int(correct.sum())} correct / "
              f"{n_inc} incorrect queries")
        if n_inc < 50:
            print(f"[probe] WARN: only {n_inc} model-incorrect queries; the "
                  f"incorrect-side accuracies will be noisy.")
        cs = run_correctness_split(
            states=states, targets=targets, correct=correct, n_classes=args.N,
            train_frac=args.train_frac, seed=args.probe_seed, device=args.device,
            prompt_ids=split_groups,
        )
        np.savez(out_dir / "correctness_split.npz",
                 acc_model_correct=cs["acc_model_correct"],
                 acc_model_incorrect=cs["acc_model_incorrect"],
                 correctness_auc_per_level=cs["correctness_auc_per_level"])
        with open(out_dir / "correctness_split_summary.json", "w") as f:
            json.dump({
                "config_name": args.config_name,
                "checkpoint": args.checkpoint,
                "H_eval": args.H_eval,
                "n_levels": int(states.shape[1]),
                "n_probe_examples": int(states.shape[0]),
                "probe_split": split_provenance,
                "n_correct": int(correct.sum()),
                "n_incorrect": n_inc,
                "n_test_correct": cs["n_test_correct"],
                "n_test_incorrect": cs["n_test_incorrect"],
                "model_prediction_sanity": sanity,
                "auc_per_level": [float(x) for x in cs["correctness_auc_per_level"]],
            }, f, indent=2)
        print(f"[probe] correctness split DONE. max AUC="
              f"{np.nanmax(cs['correctness_auc_per_level']):.3f}. Artifacts in {out_dir}")
        return

    # Probes
    probe_types = tuple(args.probe_types.split(","))
    print(f"[probe] training probes ({probe_types})...")
    t0 = time.time()
    stratified_accuracies = {}
    accs = run_probe_grid(
        states=states, targets=targets,
        n_classes=args.N, train_frac=args.train_frac,
        seed=args.probe_seed, device=args.device,
        probe_types=probe_types,
        prompt_ids=split_groups,
        query_strata=query_strata,
        stratified_accuracies=stratified_accuracies,
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
    if stratified_accuracies:
        np.savez(out_dir / "probe_stratified_accuracies.npz", **stratified_accuracies)
    np.savez(out_dir / "trajectory.npz", **traj)
    np.savez(out_dir / "states_targets.npz", states=states, targets=targets,
             prompt_ids=prompt_ids)
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
        "probe_split": split_provenance,
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
        "model_complete_answer": complete_answer_summary,
        "probe_strata": probe_strata,
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
