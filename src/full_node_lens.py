"""
Full-node logit lens: which node would the model write if its loop stopped after t iterations?

The first-token lens (src/logit_lens_h2.py) reads one token, so it cannot tell nodes 1 and 10-15 apart (they share the leading "1"). This lens scores whole nodes instead: for every
iteration budget t = 1..T it runs the model with the loop stopped after t iterations at every position (the model's
own final norm and unembedding, nothing trained) and computes, for every node v, the probability of writing v's
digits followed by the "|" delimiter after the first query's readout position:

    P_t(v) = prod_i p_t(token_i of "v |" | prompt up to the readout position, earlier tokens of "v |").

The candidates are disjoint sequences, so sum_v P_t(v) <= 1. Reading P_t at each node of the query's chain
(start, hops 1..H-1, answer) gives the lens staircase. The node the model writes greedily is the candidate whose
every token is the argmax (with the space after "?" predicted too), which is the full-answer protocol
teacher_forced_answer_tokens_plus_delimiter_v1 at loop budget t; at t = T it must reproduce the probe report's
first-query full-answer accuracy on the same evaluation set (eval_seed 12345), the lens-versus-forward check.

Outputs (in --out_dir):
    full_node_lens.npz        P (n, T, N) node probabilities, written (n, T) greedy node or -1, chain (n, H+1)
    full_node_lens_summary.json  lens[h][t-1] = mean P_t(chain node h), answer_accuracy[t-1], mean probability mass
                              on the chain, and the run's settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.intervention_h2 import answer_token_sequences
from src.model_configs import create_model
from src.probe_h2 import (build_tokenizer, find_query_positions, generate_eval_set, get_q_token_ids,
                          load_train_args, tokenize_batch)


def build_rows(eval_set, tokenizer, max_len, gold):
    """One row per (prompt, candidate node): the prompt cut after the first query's readout position (left padding
    dropped) followed by the candidate's tokens. Returns padded tokens, mask, the readout index per row and the
    candidate token positions."""
    q_ids = get_q_token_ids(tokenizer)
    prefixes = []
    for sample in eval_set:
        tok = tokenize_batch([sample["text"]], tokenizer, max_len=max_len)
        valid = tok["attention_mask"][0]
        start = int(valid.int().argmax())
        pos = find_query_positions(tok["tokens"], q_ids)[0][0]
        prefixes.append(tok["tokens"][0, start:pos + 1])
    n_gen = max(len(g) for g in gold)
    W = max(len(p) for p in prefixes) + n_gen
    n, N = len(prefixes), len(gold)
    tokens = torch.full((n * N, W), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((n * N, W), dtype=torch.bool)
    readout = torch.zeros(n * N, dtype=torch.long)
    for i, p in enumerate(prefixes):
        for v, g in enumerate(gold):
            r = i * N + v
            tokens[r, :len(p)] = p
            tokens[r, len(p):len(p) + len(g)] = torch.tensor(g)
            mask[r, :len(p) + len(g)] = True
            readout[r] = len(p) - 1
    return tokens, mask, readout


@torch.no_grad()
def run_lens(model, tokens, mask, readout, gold, space_tok, T, batch_size, device):
    """P[row, t-1] = probability of the row's candidate at loop budget t; ok[row, t-1] = every candidate token
    (and the space after "?") is the argmax, i.e. the model writes this candidate greedily."""
    R = tokens.shape[0]
    N = len(gold)
    lens = torch.tensor([len(gold[r % N]) for r in range(R)])
    P = np.zeros((R, T), dtype=np.float32)
    ok = np.zeros((R, T), dtype=bool)
    for a in range(0, R, batch_size):
        tk, mk = tokens[a:a + batch_size].to(device), mask[a:a + batch_size].to(device)
        ro, ln = readout[a:a + batch_size].to(device), lens[a:a + batch_size].to(device)
        rows = torch.arange(tk.shape[0], device=device)
        for t in range(1, T + 1):
            logp = torch.log_softmax(model(tk, mk, K=t).float(), dim=-1)
            total = torch.zeros(tk.shape[0], device=device)
            good = logp[rows, ro - 1].argmax(-1) == tk[rows, ro]  # the space after "?"
            for i in range(ln.max().item()):
                live = i < ln
                pos = (ro + i).clamp(max=tk.shape[1] - 2)
                tgt = tk[rows, pos + 1]
                step = logp[rows, pos]
                total += torch.where(live, step[rows, tgt], torch.zeros_like(total))
                good &= ~live | (step.argmax(-1) == tgt)
            P[a:a + batch_size, t - 1] = total.exp().cpu().numpy()
            ok[a:a + batch_size, t - 1] = good.cpu().numpy()
    return P, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--H_eval", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--random_init", action="store_true", help="Untrained model, same architecture (control).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train_args = load_train_args(args.checkpoint)  # architecture flags also for the random-init control
    max_len = int(train_args.get("max_len", 200))
    tokenizer = build_tokenizer(args.N)
    torch.manual_seed(args.eval_seed)
    model, config = create_model(
        args.config_name, vocab_size=len(tokenizer), max_sequence_len=max_len, pad_token_idx=tokenizer.pad_token_id,
        use_nope=bool(train_args.get("use_nope", True)), zero_init=bool(train_args.get("zero_init", True)),
        use_timestep_embedding=not bool(train_args.get("no_timestep_embedding", False)))
    ckpt_sha = None
    if not args.random_init:
        raw = Path(args.checkpoint).read_bytes()
        ckpt_sha = hashlib.sha256(raw).hexdigest()
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(args.device).eval()
    T = config["K_max"]

    gold, space_tok = answer_token_sequences(tokenizer, args.N, max_len)
    eval_set = generate_eval_set(N=args.N, H_eval=args.H_eval, M=args.M, n_samples=args.n_samples,
                                 seed=args.eval_seed)
    chain = np.array([s["meta"]["queries"][0]["intermediates"] for s in eval_set], dtype=np.int64)
    assert chain.shape == (args.n_samples, args.H_eval + 1)
    t0 = time.time()
    tokens, mask, readout = build_rows(eval_set, tokenizer, max_len, gold)
    P, ok = run_lens(model, tokens, mask, readout, gold, space_tok, T, args.batch_size, args.device)
    P = P.reshape(args.n_samples, args.N, T).transpose(0, 2, 1)       # (n, T, N)
    ok = ok.reshape(args.n_samples, args.N, T).transpose(0, 2, 1)
    assert ok.sum(-1).max() <= 1, "two candidates cannot both be greedy"
    written = np.where(ok.any(-1), ok.argmax(-1), -1)                  # (n, T)
    print(f"[lens] {args.n_samples} prompts x {T} budgets in {time.time() - t0:.0f}s")

    rows = np.arange(args.n_samples)
    lens = [[float(P[rows, t, chain[:, h]].mean()) for t in range(T)] for h in range(args.H_eval + 1)]
    acc = [float((written[:, t] == chain[:, -1]).mean()) for t in range(T)]
    mass = [float(P[:, t].sum(-1).mean()) for t in range(T)]
    summary = {
        "config_name": args.config_name, "checkpoint": args.checkpoint, "checkpoint_sha256": ckpt_sha,
        "random_init": args.random_init, "H_eval": args.H_eval, "T": T, "n_samples": args.n_samples,
        "eval_seed": args.eval_seed, "query": "first query of each prompt",
        "protocol": "full-node lens: loop stopped after t iterations at every position; P(node digits + '|'); "
                    "greedy written node = teacher_forced_answer_tokens_plus_delimiter_v1 at budget t",
        "lens": lens, "answer_accuracy": acc, "node_probability_mass": mass,
        "learning_rate": train_args.get("lr"),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "full_node_lens.npz", P=P.astype(np.float32), written=written, chain=chain)
    (out / "full_node_lens_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[lens] answer accuracy by t: {[round(a, 3) for a in acc]}")
    print(f"[lens] lens diagonal P(hop t | t): {[round(lens[min(t + 1, args.H_eval)][t], 3) for t in range(T)]}")


if __name__ == "__main__":
    main()
