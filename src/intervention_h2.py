"""
Causal intervention: activation patching across prompts in looped models.

The probes and logit lens are correlational: they show the residual stream after
iteration k linearly encodes hop k, but not that this state is causally the
running pointer. This script patches the state from a donor prompt into a
receiver prompt after iteration j and lets the loop finish.

Design (paired, single-start, single-H):
    For each trial we sample TWO independent N-cycles, pi_A (donor) and pi_B
    (receiver), a shared start node s, and a fixed hop count H. We build prompt_A
    and prompt_B (identical query "hop H s ?", different edge lists).

    * Baseline: model(prompt_B) answers pi_B^H(s).
    * Patch at iteration j: run the receiver on prompt_B; immediately AFTER loop
      iteration j, overwrite the residual stream at the answer-readout position(s)
      with the donor's residual at iteration j (captured from model(prompt_A));
      then let the loop finish and read the answer.

    If each iteration advances a running pointer by exactly one hop and that
    pointer lives in the residual stream (the paper's mechanism), then after the
    swap the pointer holds pi_A^j(s) and the remaining H-j iterations advance it
    under the RECEIVER's permutation pi_B, yielding the composed node

        composed(j) = pi_B^{H-j}( pi_A^{j}(s) ),   0 <= j <= H,

    which interpolates from pure_B = pi_B^H(s) at j=0 to pure_A = pi_A^H(s) at
    j>=H. A high match-rate to composed(j) (and specifically NOT to pure_B) is
    causal evidence for the one-hop-per-iteration pointer.

Null / control: a random-init model, or a model whose state is patched at a
position that does not carry the pointer, should show no such composition.

Scoring. Every patched (and baseline) run is scored two ways:
    * full answer (the paper's measure): the model
      writes the whole answer greedily after the readout position, every digit
      plus the "|" delimiter, and the written node is compared by identity with
      each target. Earlier query answers stay teacher forced (the receiver's
      gold), the space after "?" must be predicted too, and greedy exact match
      against a target equals teacher-forced exactness against that target, so
      this is the probe reports' teacher_forced_answer_tokens_plus_delimiter_v1
      applied to the patched model. Chance is 1/N.
    * first token (kept for comparison with earlier runs): argmax over
      entity first tokens at the readout position. Nodes 1 and 10-15 share the
      leading "1", so this flatters any model (7/16 floor for a constant guess).

Outputs (in --out_dir):
    intervention.npz     per-(trial,query,j) predicted node (first token and full
                         answer) + the three target nodes (pure_A, pure_B,
                         composed_j) and the query index.
    intervention_summary.json  aggregate match-rate curves vs j: full-answer
                         block ("full_answer", all queries and first query) and
                         the first-token curves with their unambiguous subset.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.model import LoopedTransformer
from src.model_configs import create_model
from src.logit_lens_h2 import build_node_token_map
from src.probe_h2 import (
    build_tokenizer,
    find_query_positions,
    get_q_token_ids,
    load_train_args,
    tokenize_batch,
)


# ---------------------------------------------------------------------------
# Paired-prompt generation with recorded permutations
# ---------------------------------------------------------------------------

def _cycle_maps(order: list[int]) -> tuple[dict, dict]:
    """Given a cycle order[i] -> order[(i+1)%N], return (fwd, bwd) node->node maps."""
    N = len(order)
    fwd, bwd = {}, {}
    for i in range(N):
        s, d = order[i], order[(i + 1) % N]
        fwd[s] = d
        bwd[d] = s
    return fwd, bwd


def _apply_power(fwd: dict, node: int, power: int) -> int:
    """Apply the permutation `power` times (power >= 0)."""
    x = node
    for _ in range(power):
        x = fwd[x]
    return x


def _edges_str(order: list[int], rng: random.Random) -> str:
    N = len(order)
    edges = [(order[i], order[(i + 1) % N]) for i in range(N)]
    rng.shuffle(edges)
    return " , ".join(f"{s} > {d}" for s, d in edges)


def generate_paired_trials(
    N: int, H: int, M: int, n_trials: int, seed: int, mode: str = "compose"
) -> list[dict]:
    """Each trial builds a donor prompt A and a receiver prompt B, plus per-query
    target nodes. One patched forward resolves all M queries at once.

    mode="compose" (cross-permutation): A and B use two
        INDEPENDENT N-cycles pi_A, pi_B but the SAME start s and hop H. Injecting
        donor state (hop j+1 = pi_A^{j+1}(s)) after receiver iteration j and
        finishing under pi_B predicts the composed node
            composed[j] = pi_B^{H-(j+1)}( pi_A^{j+1}(s) )   (pure_A once j+1>=H).
        This is the "combination of appropriate powers of both permutations".

    mode="redirect" (same-permutation cross-start, robust binary flip): A and B
        share ONE cycle pi but use DIFFERENT start nodes s_d (donor), s_r
        (receiver). If the query-position residual causally carries the running
        pointer, injecting the donor pointer at iteration j redirects the answer
        to the donor chain pi^H(s_d) for every j where enough iterations remain;
        composed[j] = pi^H(s_d) = pure_A for all j. pure_B = pi^H(s_r).
    """
    rng = random.Random(seed)
    trials = []
    for _ in range(n_trials):
        order_A = rng.sample(range(N), N)
        order_B = order_A[:] if mode == "redirect" else rng.sample(range(N), N)
        fwd_A, _ = _cycle_maps(order_A)
        fwd_B, _ = _cycle_maps(order_B)

        q_A, q_B, targets = [], [], []
        for _q in range(M):
            s_d = order_A[rng.randint(0, N - 1)]  # donor start
            if mode == "redirect":
                # receiver start differs from donor start
                s_r = s_d
                while s_r == s_d:
                    s_r = order_A[rng.randint(0, N - 1)]
            else:
                s_r = s_d  # compose: shared start
            ans_A = _apply_power(fwd_A, s_d, H)   # donor answer
            ans_B = _apply_power(fwd_B, s_r, H)   # receiver (baseline) answer
            q_A.append(f"hop {H} {s_d} ? {ans_A}")
            q_B.append(f"hop {H} {s_r} ? {ans_B}")
            # composed[j] = model output predicted if the pointer is carried,
            # for an injection AFTER receiver iteration j (0-indexed), i.e. the
            # injected donor state encodes hop (j+1). Length H+1, index by j but
            # we evaluate j in 0..K-1 downstream (clamped).
            composed = []
            for j in range(H + 1):
                donor_hop = min(j + 1, H)
                mid = _apply_power(fwd_A, s_d, donor_hop)          # pi_A^{j+1}(s_d)
                remaining = max(H - donor_hop, 0)
                composed.append(_apply_power(fwd_B, mid, remaining))
            targets.append({
                "start_donor": s_d,
                "start_recv": s_r,
                "pure_A": ans_A,
                "pure_B": ans_B,
                "composed": composed,  # composed[H]=pure_A; redirect: all=pure_A
            })

        text_A = _edges_str(order_A, rng) + " | " + " | ".join(q_A)
        text_B = _edges_str(order_B, rng) + " | " + " | ".join(q_B)
        trials.append({
            "text_A": text_A,
            "text_B": text_B,
            "targets": targets,  # length M
            "H": H,
            "mode": mode,
        })
    return trials


# ---------------------------------------------------------------------------
# Patched forward pass for the looped model
# ---------------------------------------------------------------------------

@torch.no_grad()
def looped_forward_with_injection(
    model: LoopedTransformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    inject_iter: int | torch.Tensor,
    inject_values: torch.Tensor | None,   # (B, T, D) donor residual, or None
    inject_mask: torch.Tensor | None,     # (B, T) bool, positions to overwrite
    K: int,
) -> torch.Tensor:
    """Mirror LoopedTransformer.forward, but after loop iteration `inject_iter`
    overwrite the residual at masked positions with donor values. Returns logits.

    inject_iter is 0-indexed over completed iterations: inject_iter=j overwrites
    the state that get_intermediate_states would return as states[j] (i.e. after
    the (j+1)-th application of the recurrent block). inject_iter<0 disables.
    A (B,) tensor gives each row its own iteration, so one batch holds every j.
    """
    if inject_values is not None and not torch.is_tensor(inject_iter):
        inject_iter = torch.full((tokens.shape[0],), inject_iter, device=tokens.device)
    x = model.token_embeddings(tokens)
    for k in range(K):
        if model.use_timestep_embedding:
            t_idx = min(k, model.K_max - 1)
            x = x + model.timestep_embeddings.weight[t_idx]
        for layer in model.recurrent_layers:
            x = layer(x, attention_mask)
        if inject_values is not None:
            sel = inject_mask & (inject_iter == k).unsqueeze(-1)
            x = torch.where(sel.unsqueeze(-1), inject_values, x)
    x = model.norm(x)
    return model.out_proj(x)


@torch.no_grad()
def capture_donor_states(
    model: LoopedTransformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    K: int,
) -> list[torch.Tensor]:
    """Return donor per-iteration residual states (list length K, each (B,T,D))."""
    return model.get_intermediate_states(tokens, attention_mask=attention_mask, K=K)


# ---------------------------------------------------------------------------
# Core intervention loop
# ---------------------------------------------------------------------------

def answer_token_sequences(tokenizer, N: int, max_len: int) -> tuple[list[list[int]], int]:
    """Gold tokens written after the readout position for each node (its digits
    and the "|" delimiter), read off a tokenized query, plus the space token that
    follows "?" (the readout position itself)."""
    q_token_ids = get_q_token_ids(tokenizer)
    seqs, space = [], None
    for n in range(N):
        tok = tokenize_batch([f"hop 1 0 ? {n} |"], tokenizer, max_len=max_len)
        row, valid = tok["tokens"][0].tolist(), tok["attention_mask"][0].tolist()
        pos = find_query_positions(tok["tokens"], q_token_ids)[0][0]
        end = max(i for i, v in enumerate(valid) if v)
        assert tokenizer.decode(row[pos + 1:end]).strip() == str(n) and tokenizer.decode([row[end]]).strip() == "|"
        seqs.append(row[pos + 1:end + 1])
        assert space in (None, row[pos])
        space = row[pos]
    return seqs, space


@torch.no_grad()
def run_intervention(
    trials: list[dict],
    tokenizer,
    model: LoopedTransformer,
    device: str,
    K: int,
    H: int,
    max_len: int,
    node_to_tok: np.ndarray,
    ambiguous_nodes: set[int],
    patch_answer_digit: bool = False,
) -> dict:
    """For each trial: capture donor states from prompt_A, then for the baseline
    and every patch iteration j in 0..K-1 run the receiver prompt_B with the
    donor's readout-position residuals injected after iteration j, and record
    (i) the first-answer-token argmax over entity tokens (node id, or -1) and
    (ii) the node the model writes greedily, digits plus delimiter (node id, or
    -1 if the written tokens are no node followed by "|").

    One batch per trial holds a row per (j, query): row (j, q) is prompt_B cut
    after query q's readout position, with every readout position up to q
    patched. Causal attention makes the cut exact, and the shared left padding is
    dropped (masked keys, relative positions).

    patch_answer_digit also overwrites the position after each readout (the first
    answer digit, which predicts the second digit or the delimiter) with the
    donor's state at its own first-digit position, same iteration j. For the
    current query that position holds the digit the model wrote.
    """
    q_token_ids = get_q_token_ids(tokenizer)
    tok2node = {int(t): v for v, t in enumerate(node_to_tok.tolist())}
    entity_tok = torch.tensor(sorted(set(int(t) for t in node_to_tok.tolist())),
                              device=device)
    gold, space_tok = answer_token_sequences(tokenizer, len(node_to_tok), max_len)
    n_gen = max(len(g) for g in gold)

    def decode_first_token(logits_row: torch.Tensor) -> int:
        """Restrict to entity first-tokens, return node id of argmax (or -1)."""
        sub = logits_row[entity_tok]
        best = int(entity_tok[int(sub.argmax())].item())
        return tok2node.get(best, -1)

    def written_node(written: list[int]) -> int:
        return next((n for n, g in enumerate(gold) if written[:len(g)] == g), -1)

    records = []  # one dict per (trial, query)
    n_iter_grid = K + 1  # baseline handled separately; patch j in 0..K-1, row K unused

    for trial in trials:
        tok_A = tokenize_batch([trial["text_A"]], tokenizer, max_len=max_len)
        tok_B = tokenize_batch([trial["text_B"]], tokenizer, max_len=max_len)
        A_tokens = tok_A["tokens"].to(device)
        A_mask = tok_A["attention_mask"].to(device)

        # Readout positions per query for both prompts
        A_pos = find_query_positions(tok_A["tokens"], q_token_ids)[0]
        B_pos = find_query_positions(tok_B["tokens"], q_token_ids)[0]
        M = len(trial["targets"])
        if len(A_pos) < M or len(B_pos) < M:
            continue  # tokenization dropped a query; skip trial

        # Donor states (list length K, each (1,T,D))
        donor_states = capture_donor_states(model, A_tokens, A_mask, K=K)
        D = donor_states[0].shape[-1]

        valid_B = tok_B["attention_mask"][0]
        start = int(valid_B.int().argmax())  # first real token (left padding)
        assert valid_B[start:].all()
        rel = [p - start for p in B_pos]
        W = rel[M - 1] + 1 + n_gen
        js = list(range(-1, K))  # -1 = baseline
        R = len(js) * M
        tokens = torch.full((R, W), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros((R, W), dtype=torch.bool, device=device)
        it = torch.full((R,), -1, dtype=torch.long, device=device)
        inj_vals = torch.zeros((R, W, D), dtype=donor_states[0].dtype, device=device)
        inj_mask = torch.zeros((R, W), dtype=torch.bool, device=device)
        cur = torch.zeros(R, dtype=torch.long, device=device)
        prompt = tok_B["tokens"][0, start:].to(device)
        for a, j in enumerate(js):
            for q in range(M):
                r = a * M + q
                tokens[r, :rel[q] + 1] = prompt[:rel[q] + 1]
                mask[r, :rel[q] + 1] = True
                cur[r] = rel[q] + 1
                if j >= 0:
                    it[r] = j
                    for qq in range(q + 1):
                        inj_vals[r, rel[qq]] = donor_states[j][0, A_pos[qq]]
                        inj_mask[r, rel[qq]] = True
                        if patch_answer_digit:
                            inj_vals[r, rel[qq] + 1] = donor_states[j][0, A_pos[qq] + 1]
                            inj_mask[r, rel[qq] + 1] = True

        first = np.full(R, -2, dtype=np.int64)
        space_ok = np.zeros(R, dtype=bool)
        written = [[] for _ in range(R)]
        rows = torch.arange(R, device=device)
        for step in range(n_gen):
            logits = looped_forward_with_injection(
                model, tokens, mask, inject_iter=it,
                inject_values=inj_vals, inject_mask=inj_mask, K=K,
            )
            nxt = logits[rows, cur - 1].argmax(-1)
            if step == 0:
                space_ok = (logits[rows, cur - 2].argmax(-1) == space_tok).cpu().numpy()
                first = np.array([decode_first_token(row) for row in logits[rows, cur - 1]])
            tokens[rows, cur] = nxt
            mask[rows, cur] = True
            for r, t in enumerate(nxt.tolist()):
                written[r].append(t)
            cur += 1
        full = np.array([written_node(w) if ok else -1 for w, ok in zip(written, space_ok)])

        pred_by_j = np.full((n_iter_grid, M), -2, dtype=np.int64)
        full_by_j = np.full((n_iter_grid, M), -2, dtype=np.int64)
        written_by_j = np.full((n_iter_grid + 1, M, n_gen), -2, dtype=np.int64)  # row K+1 = baseline
        written = np.array(written, dtype=np.int64).reshape(len(js), M, n_gen)
        written_by_j[K + 1] = written[0]
        for a, j in enumerate(js[1:], start=1):
            pred_by_j[j] = first[a * M:(a + 1) * M]
            full_by_j[j] = full[a * M:(a + 1) * M]
            written_by_j[j] = written[a]

        for q in range(M):
            tgt = trial["targets"][q]
            records.append({
                "q": q,
                "pure_A": tgt["pure_A"],
                "pure_B": tgt["pure_B"],
                "composed": tgt["composed"],      # length H+1
                "base_pred": int(first[q]),
                "base_full": int(full[q]),
                "pred_by_j": pred_by_j[:, q].tolist(),  # length K+1 (last unused)
                "full_by_j": full_by_j[:, q].tolist(),
                "written_by_j": written_by_j[:, q].tolist(),  # (K+2, n_gen) tokens; row K+1 = baseline
            })

    return {"records": records, "K": K, "H": H}


def _first_tok(node: int, node_to_tok: np.ndarray) -> int:
    return int(node_to_tok[node]) if node >= 0 else -999


def summarize_full_answer(recs: list[dict], K: int, H: int) -> dict:
    """Match rates of the written node (digits plus delimiter, by node identity)
    against composed(j), pure_A and pure_B. "distinct" restricts to queries whose
    composed(j) differs from both pure answers (compose mode, j < H-1); it is None
    where that subset is empty, as in redirect mode where composed = pure_A."""
    Jmax = min(K, H)
    n = len(recs)
    base = sum(r["base_full"] == r["pure_B"] for r in recs)
    comp, pa, pb, dist, dist_n = (np.zeros(Jmax + 1) for _ in range(5))
    for r in recs:
        for j in range(Jmax + 1):
            pred, c = r["full_by_j"][j], r["composed"][j]
            comp[j] += pred == c
            pa[j] += pred == r["pure_A"]
            pb[j] += pred == r["pure_B"]
            if c not in (r["pure_A"], r["pure_B"]):
                dist_n[j] += 1
                dist[j] += pred == c
    rate = lambda a, c: [None if c[i] == 0 else float(a[i] / c[i]) for i in range(len(a))]
    return {"n": n, "baseline_receiver_acc": base / max(n, 1), "j_grid": list(range(Jmax + 1)),
            "match_composed": rate(comp, [n] * (Jmax + 1)), "match_pure_A": rate(pa, [n] * (Jmax + 1)),
            "match_pure_B": rate(pb, [n] * (Jmax + 1)), "match_composed_distinct": rate(dist, dist_n),
            "distinct_counts": dist_n.tolist()}


def summarize(out: dict, node_to_tok: np.ndarray, ambiguous_nodes: set[int]) -> dict:
    """Aggregate match-rate curves. Compare the patched prediction at iteration j
    against composed(j), pure_A, pure_B, using first-token identity (robust to the
    N=16 multi-digit tokenization). The unambiguous-subset headline restricts to
    queries whose pure_A / pure_B / composed(j) all have distinct first tokens.
    """
    recs = out["records"]
    K, H = out["K"], out["H"]
    Jmax = min(K, H)  # composed defined for j in 0..H

    def ft(n):
        return _first_tok(n, node_to_tok)

    match_composed = np.zeros(Jmax + 1)
    match_pureA = np.zeros(Jmax + 1)
    match_pureB = np.zeros(Jmax + 1)
    counts = np.zeros(Jmax + 1)
    # unambiguous headline
    u_match_composed = np.zeros(Jmax + 1)
    u_counts = np.zeros(Jmax + 1)

    base_correct = 0
    for r in recs:
        # baseline: receiver should answer pure_B
        if ft(r["base_pred"]) == ft(r["pure_B"]):
            base_correct += 1

    for r in recs:
        comp = r["composed"]  # length H+1
        for j in range(Jmax + 1):
            pred = r["pred_by_j"][j]
            fp = ft(pred)
            counts[j] += 1
            if fp == ft(comp[j]):
                match_composed[j] += 1
            if fp == ft(r["pure_A"]):
                match_pureA[j] += 1
            if fp == ft(r["pure_B"]):
                match_pureB[j] += 1
            # unambiguous: composed(j), pure_A, pure_B distinct first tokens and
            # none of the three nodes is a multi-digit ambiguous node
            trio = [comp[j], r["pure_A"], r["pure_B"]]
            fts = [ft(x) for x in trio]
            if (len(set(fts)) == 3
                    and all(x not in ambiguous_nodes for x in trio)):
                u_counts[j] += 1
                if fp == ft(comp[j]):
                    u_match_composed[j] += 1

    def rate(a, c):
        # None where the denominator is empty, so an undefined metric cannot be
        # misread as a measured zero. In redirect mode composed[j] == pure_A by
        # construction (see the mode docs above), so the unambiguous trio
        # [composed, pure_A, pure_B] can never hold three distinct first tokens
        # and match_composed_unambiguous is undefined for every j, not 0.0.
        return [None if c[i] == 0 else float(a[i] / c[i]) for i in range(len(a))]

    return {
        "n_records": len(recs),
        "H": H,
        "K": K,
        "full_answer": {"all": summarize_full_answer(recs, K, H),
                        "first_query": summarize_full_answer([r for r in recs if r["q"] == 0], K, H)},
        "baseline_receiver_acc": base_correct / max(len(recs), 1),
        "j_grid": list(range(Jmax + 1)),
        "match_composed": rate(match_composed, counts),
        "match_pure_A": rate(match_pureA, counts),
        "match_pure_B": rate(match_pureB, counts),
        "match_composed_unambiguous": rate(u_match_composed, u_counts),
        "unambiguous_counts": u_counts.tolist(),
        "counts": counts.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--H", type=int, default=6,
                   help="Hop count for the paired trials. Keep <= K and modest so "
                        "the composed path is legible. Default 6.")
    p.add_argument("--n_trials", type=int, default=400)
    p.add_argument("--mode", choices=["compose", "redirect"], default="compose",
                   help="compose: cross-permutation composition. "
                        "redirect: same-permutation cross-start (robust binary flip).")
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--eval_seed", type=int, default=20260727)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patch_answer_digit", action="store_true",
                   help="Also patch the first-answer-digit position (which predicts the second digit).")
    p.add_argument("--random_init", action="store_true",
                   help="Random-init control (mechanism should vanish).")
    p.add_argument("--use_nope", default=None, action=argparse.BooleanOptionalAction)
    p.add_argument("--zero_init", default=None, action=argparse.BooleanOptionalAction)
    p.add_argument("--no_timestep_embedding", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="Override tau_k setting. Default: read from training args.json.")
    args = p.parse_args()

    # The random-init control keeps the trained model's architecture flags
    # (use_nope, zero_init, tau); only the weights are left at their seeded init.
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

    print(f"[intv] config={args.config_name} ckpt={args.checkpoint}")
    print(f"[intv] N={args.N} H={args.H} n_trials={args.n_trials} "
          f"use_nope={args.use_nope} zero_init={args.zero_init} "
          f"no_tau={args.no_timestep_embedding} max_len={args.max_len}")

    tokenizer = build_tokenizer(args.N)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    torch.manual_seed(args.eval_seed)  # reproducible random-init control
    model, config = create_model(
        args.config_name, vocab_size=vocab_size,
        max_sequence_len=args.max_len, pad_token_idx=pad_id,
        use_nope=args.use_nope, zero_init=args.zero_init,
        use_timestep_embedding=not args.no_timestep_embedding,
    )
    model = model.to(args.device)
    assert isinstance(model, LoopedTransformer), "intervention is for looped models"
    if not args.random_init:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[intv] RANDOM-INIT control")
    model.eval()

    K = config["K_max"]
    assert args.H <= K, f"H={args.H} must be <= K={K}"

    node_to_tok, ambiguous = build_node_token_map(tokenizer, args.N)
    ambiguous_set = set(ambiguous)

    trials = generate_paired_trials(
        N=args.N, H=args.H, M=args.M, n_trials=args.n_trials, seed=args.eval_seed,
        mode=args.mode,
    )

    t0 = time.time()
    out = run_intervention(
        trials, tokenizer, model, args.device, K=K, H=args.H,
        max_len=args.max_len, node_to_tok=node_to_tok, ambiguous_nodes=ambiguous_set,
        patch_answer_digit=args.patch_answer_digit,
    )
    print(f"[intv] {len(out['records'])} query-records in {time.time()-t0:.0f}s")

    summary = summarize(out, node_to_tok, ambiguous_set)
    summary.update({
        "mode": args.mode,
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint),
        "random_init": bool(args.random_init),
        "patch_answer_digit": bool(args.patch_answer_digit),
        "no_timestep_embedding": bool(args.no_timestep_embedding),
        "use_nope": bool(args.use_nope),
        "eval_seed": args.eval_seed,
        "full_answer_protocol": "greedy_answer_tokens_plus_delimiter_v1 (earlier answers teacher forced; "
                                "equals teacher_forced_answer_tokens_plus_delimiter_v1 per target)",
    })
    fa = summary["full_answer"]["all"]
    print(f"[intv] FULL ANSWER baseline receiver acc = {fa['baseline_receiver_acc']:.3f}")
    for key in ("match_composed", "match_composed_distinct", "match_pure_A", "match_pure_B"):
        print(f"[intv] FULL ANSWER {key:24s} = {[None if x is None else round(x, 3) for x in fa[key]]}")
    print(f"[intv] first-token baseline receiver acc = {summary['baseline_receiver_acc']:.3f}")
    for key in ("match_composed", "match_composed_unambiguous", "match_pure_B"):
        print(f"[intv] first-token {key:26s} = {[None if x is None else round(x, 3) for x in summary[key]]}")

    # Persist records (compact) + summary
    np.savez_compressed(
        out_dir / "intervention.npz",
        pure_A=np.array([r["pure_A"] for r in out["records"]], dtype=np.int64),
        pure_B=np.array([r["pure_B"] for r in out["records"]], dtype=np.int64),
        composed=np.array([r["composed"] for r in out["records"]], dtype=np.int64),
        base_pred=np.array([r["base_pred"] for r in out["records"]], dtype=np.int64),
        pred_by_j=np.array([r["pred_by_j"] for r in out["records"]], dtype=np.int64),
        base_full=np.array([r["base_full"] for r in out["records"]], dtype=np.int64),
        full_by_j=np.array([r["full_by_j"] for r in out["records"]], dtype=np.int64),
        query_index=np.array([r["q"] for r in out["records"]], dtype=np.int64),
        written_by_j=np.array([r["written_by_j"] for r in out["records"]], dtype=np.int64),
        answer_tokens=np.array([g + [-1] * (3 - len(g)) for g in answer_token_sequences(tokenizer, args.N, args.max_len)[0]],
                               dtype=np.int64),
        node_to_tok=node_to_tok,
    )
    with open(out_dir / "intervention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[intv] wrote {out_dir}/intervention.npz and intervention_summary.json")


if __name__ == "__main__":
    main()
