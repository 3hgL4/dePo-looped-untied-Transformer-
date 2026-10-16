"""
Per-hop accuracy of the evaluation during training (evaluate_per_hop).
"""

import random

import torch

from src.datasets.pointer_chasing import generate_pointer_chasing_sample
from src.model import LoopedTransformer
from src.utils import unwrap_model


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

    This is the evaluation during training: n_samples_per_hop fresh prompts per H
    from a generator seeded with eval_seed + H, scored on the digits of the LAST
    query's answer under teacher forcing (decoded and compared as a string). The
    last query ends the sequence, so no delimiter follows it.

    Returns:
        dict mapping H -> accuracy
    """
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
