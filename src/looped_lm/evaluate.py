"""
Evaluation utilities for H1 experiments.

- evaluate_accuracy: per-sample accuracy on synthetic tasks
- evaluate_per_hop: accuracy at each hop count separately
- evaluate_depth_extrapolation: accuracy at K > K_train
- sweep_K: evaluate a looped model at different K values without retraining
- extract_critical_K: fit sigmoid and extract inflection point
"""

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from looped_lm.model import LoopedTransformer
from looped_lm.utils import unwrap_model


_Q_TOKEN_IDS_CACHE = {}

def _get_q_token_ids(tokenizer) -> set[int]:
    """Get '?' token IDs, thread-safe. Cached after first call."""
    tok_id = id(tokenizer)
    if tok_id in _Q_TOKEN_IDS_CACHE:
        return _Q_TOKEN_IDS_CACHE[tok_id]

    # Do the tokenizer call once (before any background threads)
    if hasattr(tokenizer, 'old_tokenizer'):
        old_tok = tokenizer.old_tokenizer
        # Use convert_tokens_to_ids which is a dict lookup, thread-safe
        vocab = old_tok.get_vocab()
        raw_ids = [v for k, v in vocab.items() if '?' in k]
        if tokenizer.map_old_to_new is not None:
            result = {tokenizer.map_old_to_new.get(t, t) for t in raw_ids}
        else:
            result = set(raw_ids)
    else:
        vocab = tokenizer.get_vocab()
        result = {v for k, v in vocab.items() if '?' in k}

    _Q_TOKEN_IDS_CACHE[tok_id] = result
    return result


def evaluate_accuracy(
    model,
    dataset,
    tokenizer,
    device: str,
    config: dict,
    K: int = None,
    batch_size: int = 64,
) -> dict:
    """Evaluate exact-match accuracy on a synthetic task dataset.

    For each sample, finds the '?' token, extracts ground-truth answer tokens,
    greedy-predicts from logits, and compares decoded strings.

    Args:
        model: trained model
        dataset: OnlineSyntheticDataset with samples_answers
        tokenizer: tokenizer for decoding
        device: 'cuda' or 'cpu'
        config: model config dict (needs vocab_size)
        K: loop iterations for LoopedTransformer (None for standard)
        batch_size: evaluation batch size

    Returns:
        dict with 'accuracy', 'correct', 'total'
    """
    raw_model = unwrap_model(model)
    raw_model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Token IDs for parsing (thread-safe)
    q_token_ids = _get_q_token_ids(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_tokens = batch["tokens"].to(device)
            batch_mask = batch["attention_mask"].to(device)

            # Forward pass
            if isinstance(raw_model, LoopedTransformer) and K is not None:
                logits = model(batch_tokens, batch_mask, K=K)
            else:
                logits = model(batch_tokens, batch_mask)

            for i in range(batch_tokens.size(0)):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= len(dataset.samples_answers):
                    break

                tokens_list = batch_tokens[i].tolist()
                gt_answer = dataset.samples_answers[sample_idx]

                # Find last '?' position
                q_pos = None
                for pos in range(len(tokens_list) - 1, -1, -1):
                    if tokens_list[pos] in q_token_ids:
                        q_pos = pos
                        break
                if q_pos is None:
                    continue

                # Extract ground truth answer tokens (after '?', before padding)
                gt_tokens = []
                for pos in range(q_pos + 1, len(tokens_list)):
                    if tokens_list[pos] == pad_id:
                        break
                    gt_tokens.append(tokens_list[pos])

                if not gt_tokens:
                    continue

                # Greedy predict each answer token position
                pred_tokens = []
                for j in range(len(gt_tokens)):
                    pred_pos = q_pos + j  # logits[pos] predicts token at pos+1
                    if pred_pos >= logits.size(1):
                        break
                    pred_token = logits[i, pred_pos].argmax(dim=-1).item()
                    pred_tokens.append(pred_token)

                # Compare decoded strings
                pred_str = tokenizer.decode(pred_tokens).strip()
                gt_str = gt_answer.strip()

                if pred_str == gt_str:
                    correct += 1
                total += 1

    accuracy = correct / max(total, 1)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def sweep_K(
    model,
    dataset,
    tokenizer,
    device: str,
    config: dict,
    K_values: list[int],
    batch_size: int = 64,
) -> dict[int, float]:
    """Evaluate a looped model at multiple K values without retraining.

    This is the key advantage of looped models: a single trained model
    yields the full accuracy-vs-K curve.

    Args:
        model: trained LoopedTransformer
        dataset: evaluation dataset
        tokenizer: tokenizer for decoding
        device: 'cuda' or 'cpu'
        config: model config dict
        K_values: list of K values to evaluate at
        batch_size: evaluation batch size

    Returns:
        dict mapping K -> accuracy
    """
    results = {}
    for K in K_values:
        metrics = evaluate_accuracy(
            model, dataset, tokenizer, device, config, K=K, batch_size=batch_size
        )
        results[K] = metrics["accuracy"]
    return results


def evaluate_per_hop(
    model,
    tokenizer,
    device: str,
    config: dict,
    N: int = 16,
    M: int = 5,
    H_values: list[int] = None,
    K: int = None,
    n_samples_per_hop: int = 500,
    eval_seed: int = 99999,
) -> dict[int, float]:
    """Evaluate accuracy at each hop count separately.

    Generates fresh data for each H and evaluates independently.
    Uses direct batch generation to avoid thread-safety issues with
    OnlineSyntheticDataset's background tokenizer thread.

    Returns:
        dict mapping H -> accuracy
    """
    from looped_lm.datasets.pointer_chasing import generate_pointer_chasing_sample

    if H_values is None:
        H_values = list(range(1, 21))

    raw_model = unwrap_model(model)
    raw_model.eval()
    q_ids = _get_q_token_ids(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = config.get("max_sequence_len", 512)
    results = {}

    for H in H_values:
        rng = random.Random(eval_seed + H)
        texts = []
        answers_list = []
        for _ in range(n_samples_per_hop):
            text, answer, _ = generate_pointer_chasing_sample(rng, N, H, M)
            texts.append(text)
            answers_list.append(answer)

        # Tokenize all at once (single thread, no background)
        out = tokenizer(
            texts, truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt", return_attention_mask=True,
        )
        tokens = out["input_ids"].long()
        masks = out["attention_mask"].to(torch.bool)

        # Evaluate in batches
        correct = 0
        total = 0
        bs = 64
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                end = min(start + bs, len(texts))
                batch_tokens = tokens[start:end].to(device)
                batch_mask = masks[start:end].to(device)

                if isinstance(raw_model, LoopedTransformer) and K is not None:
                    logits = model(batch_tokens, batch_mask, K=K)
                else:
                    logits = model(batch_tokens, batch_mask)

                for i in range(batch_tokens.size(0)):
                    sample_idx = start + i
                    row = batch_tokens[i].tolist()
                    gt_answer = answers_list[sample_idx]

                    # Find last '?'
                    q_pos = None
                    for pos in range(len(row) - 1, -1, -1):
                        if row[pos] in q_ids:
                            q_pos = pos
                            break
                    if q_pos is None:
                        continue

                    gt_tokens = []
                    for pos in range(q_pos + 1, len(row)):
                        if row[pos] == pad_id:
                            break
                        gt_tokens.append(row[pos])
                    if not gt_tokens:
                        continue

                    pred_tokens = []
                    for j in range(len(gt_tokens)):
                        pred_pos = q_pos + j
                        if pred_pos >= logits.size(1):
                            break
                        pred_tokens.append(logits[i, pred_pos].argmax(dim=-1).item())

                    pred_str = tokenizer.decode(pred_tokens).strip()
                    if pred_str == gt_answer.strip():
                        correct += 1
                    total += 1

        results[H] = correct / max(total, 1)

    return results


def evaluate_depth_extrapolation(
    model,
    tokenizer,
    device: str,
    config: dict,
    N: int = 16,
    M: int = 5,
    K_values: list[int] = None,
    H_values: list[int] = None,
    n_samples_per_hop: int = 500,
    eval_seed: int = 99999,
) -> dict[int, dict[int, float]]:
    """Evaluate at multiple K values (including K > K_train) across hop counts.

    Returns:
        dict mapping K -> {H -> accuracy}
    """
    from looped_lm.datasets.pointer_chasing import PointerChasingDataset

    if H_values is None:
        H_values = list(range(1, 21))
    if K_values is None:
        raw_model = unwrap_model(model)
        K_max = raw_model.K_max if hasattr(raw_model, 'K_max') else 8
        K_values = list(range(1, 2 * K_max + 1))

    results = {}
    for K in K_values:
        results[K] = {}
        for H in H_values:
            ds = PointerChasingDataset(
                N=N, H=H, M=M, tokenizer=tokenizer,
                max_length=config.get("max_sequence_len", 512),
                n_samples=n_samples_per_hop, base_seed=eval_seed + H,
            )
            metrics = evaluate_accuracy(model, ds, tokenizer, device, config, K=K)
            results[K][H] = metrics["accuracy"]

    return results


def extract_critical_K(
    accuracies: dict[int, float],
    threshold: float = 0.5,
) -> float:
    """Extract the critical K value where accuracy crosses a threshold.

    Uses linear interpolation between the two K values that bracket
    the threshold. If accuracy never crosses, returns None.

    Args:
        accuracies: dict mapping K -> accuracy
        threshold: accuracy threshold (default 0.5 = 50% of max)

    Returns:
        interpolated K value at threshold, or None
    """
    sorted_Ks = sorted(accuracies.keys())
    if not sorted_Ks:
        return None

    # Scale threshold relative to max accuracy
    max_acc = max(accuracies.values())
    target = threshold * max_acc

    # Find crossing point
    for i in range(len(sorted_Ks) - 1):
        K_lo = sorted_Ks[i]
        K_hi = sorted_Ks[i + 1]
        acc_lo = accuracies[K_lo]
        acc_hi = accuracies[K_hi]

        if acc_lo <= target <= acc_hi:
            # Linear interpolation
            if acc_hi == acc_lo:
                return float(K_lo)
            frac = (target - acc_lo) / (acc_hi - acc_lo)
            return K_lo + frac * (K_hi - K_lo)

    return None


def extract_critical_K_sigmoid(
    accuracies: dict[int, float],
    L_r: int = 1,
) -> dict:
    """Fit a sigmoid to accuracy-vs-K data and extract parameters.

    Fits: accuracy(K) = L / (1 + exp(-r * (K*L_r - K_c)))

    Args:
        accuracies: dict mapping K -> accuracy
        L_r: number of recurrent layers (to compute effective depth)

    Returns:
        dict with 'K_c' (critical K), 'r' (steepness), 'L' (plateau),
        'K_c_effective' (K_c * L_r), 'fit_error'
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        return {"error": "scipy not available"}

    Ks = sorted(accuracies.keys())
    accs = [accuracies[k] for k in Ks]
    x = np.array([k * L_r for k in Ks], dtype=float)
    y = np.array(accs, dtype=float)

    def sigmoid(x, L, r, K_c):
        return L / (1.0 + np.exp(-r * (x - K_c)))

    try:
        popt, _ = curve_fit(
            sigmoid, x, y,
            p0=[max(y), 1.0, np.median(x)],
            bounds=([0, 0.01, 0], [1.5, 10.0, max(x) * 2]),
            maxfev=5000,
        )
        L, r, K_c = popt
        y_pred = sigmoid(x, L, r, K_c)
        mse = float(np.mean((y - y_pred) ** 2))

        return {
            "K_c_effective": float(K_c),
            "K_c": float(K_c / L_r),
            "r": float(r),
            "L": float(L),
            "fit_mse": mse,
        }
    except RuntimeError:
        return {"error": "sigmoid fit failed"}
