"""
Behavioural ladder diagnostic for a letters-format checkpoint (2026-10-27).

Question: the letters model answers h1 at 1.000 and h2/h3/h4 at chance. *Why*
does it stop? Reading nothing but the model's free-running argmax, this script
asks where the wrong answer lands on the chain it was given:

    k = 0   -> it echoes the start node            (no chase at all)
    k = 1   -> it does exactly one hop then stops   (the "one-step" failure)
    k = H   -> correct
    other   -> none of the chain nodes

A pure "uniform noise" failure puts ~(N-1)/N of the mass in `other` and leaves
each ladder rung at 1/16 = 0.0625. A "stops after one hop" failure piles mass on
k=1. The two look identical in accuracy alone, which is why accuracy is not
enough to say anything about the mechanism.

No probing, no interventions, no training: one forward pass per prompt, scoring
the free-running argmax at each query's answer position, exactly where
src/evaluate.py reads it. That keeps the numbers comparable with the training
trajectories -- the H=1 rung doubling as a hard sanity gate (it must reproduce
the ~1.000 from the trajectory, else the node<->token mapping here is wrong).

Usage (inside WSL):

    .venv/bin/python tools/h2_ladder.py \
        --checkpoint runs/iso/ouro-4x8_s44_letters_resume_hmax4/checkpoint.pt \
        --config_name ouro-4x8 --node_format letters \
        --H_eval 1,2,3,4 --n_samples 1000 --out_json experiments/iso_letters/ladder_take2.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.pointer_chasing import generate_pointer_chasing_sample  # noqa: E402
from src.model import LoopedTransformer  # noqa: E402
from src.model_configs import create_model  # noqa: E402
from src.node_tokens import node_symbols  # noqa: E402
from src.utils import TokenMappingTokenizer  # noqa: E402


def build_tokenizer(N, M, H_min, H_max, node_format,
                    base_name="hf-internal-testing/llama-tokenizer"):
    """Same construction as src/train_uniform_h.py (map is sorted, so it is
    order-independent; the same token SET gives the same compact ids)."""
    base = AutoTokenizer.from_pretrained(base_name)
    if base.pad_token_id is None:
        base.pad_token_id = base.eos_token_id
    rng = random.Random(999)
    map_texts = [
        generate_pointer_chasing_sample(rng, N, H=h, M=M, node_format=node_format)[0]
        for h in range(H_min, H_max + 1)
        for _ in range(50)
    ]
    tok = TokenMappingTokenizer(base)
    tok.estimate_token_map_from_dataset(map_texts)
    return tok


def node_token_lookup(tok, N, node_format):
    """compact token id -> node index, via the token string itself.

    letters: the answer is one token "▁A".."▁P" (the leading space merges into
    the symbol), so the id of the token after "?" IS the answer.
    plain:   "0".."15" is one or two bare digit tokens, so a single argmax can
    only see the first digit; the caller must not use this path for plain.
    """
    syms = node_symbols(N, node_format)
    lookup = {}
    for i, s in enumerate(syms):
        # Encode in the same context the answer actually appears in: one space
        # after the '?'. letters gives ["▁?", "▁A"] = 2 tokens; plain gives
        # ["▁?", "▁", "1", "0"] and is rejected here on purpose.
        # TokenMappingTokenizer only wraps __call__/decode, so encode through the
        # base tokenizer and map the ids ourselves.
        raw = tok.old_tokenizer.encode("? " + s, add_special_tokens=False)
        ids = [tok.map_old_to_new[t] for t in raw]
        if len(ids) != 2:
            raise SystemExit(
                f"node symbol {s!r} is {len(ids) - 1} token(s) after '?' under "
                f"{node_format}; the ladder diagnostic only supports single-token "
                f"node symbols")
        lookup[int(ids[-1])] = i
    if len(set(lookup.values())) != N:
        raise SystemExit("node symbols do not give %d distinct tokens" % N)
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--node_format", default="letters")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--H_eval", default="1,2,3,4")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--eval_seed", type=int, default=20260727)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    train_args = {}
    for cand in (ckpt_path.parent / "args.json",):
        if cand.exists():
            with open(cand) as f:
                train_args = json.load(f)
    print(f"[ladder] train args: H_min={train_args.get('H_min')} "
          f"H_max={train_args.get('H_max')} max_len={train_args.get('max_len')} "
          f"node_format={train_args.get('node_format')}")
    if train_args.get("node_format") not in (None, args.node_format):
        print(f"[ladder] WARNING: checkpoint trained with node_format="
              f"{train_args.get('node_format')}, scoring with {args.node_format}")

    tok = build_tokenizer(args.N, args.M,
                          int(train_args.get("H_min", 1)),
                          int(train_args.get("H_max", 15)),
                          args.node_format)
    vocab_size, pad_id = len(tok), tok.pad_token_id
    print(f"[ladder] vocab_size={vocab_size} pad_id={pad_id}")

    model, config = create_model(
        args.config_name, vocab_size=vocab_size,
        max_sequence_len=args.max_len, pad_token_idx=pad_id,
        use_nope=train_args.get("use_nope", False),
        zero_init=train_args.get("zero_init", False),
        use_timestep_embedding=not train_args.get("no_timestep_embedding", False),
    )
    model = model.to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    K = int(config["K_max"]) if isinstance(model, LoopedTransformer) else None
    print(f"[ladder] loaded step={ckpt.get('step')} K={K}")

    node_of_token = node_token_lookup(tok, args.N, args.node_format)

    q_ids = set()
    for tid in range(vocab_size):
        if "?" in tok.decode([tid]):
            q_ids.add(tid)
    if not q_ids:
        raise SystemExit("no '?' token id found")

    H_values = [int(x) for x in args.H_eval.split(",") if x.strip()]
    report = {}
    for H in H_values:
        rng = random.Random(args.eval_seed + H)
        texts, metas = [], []
        for _ in range(args.n_samples):
            text, _ans, meta = generate_pointer_chasing_sample(
                rng, args.N, H, args.M, node_format=args.node_format)
            texts.append(text)
            metas.append(meta)

        enc = tok(texts, truncation=True, max_length=args.max_len,
                  padding="max_length", return_tensors="pt",
                  return_attention_mask=True)
        tokens = enc["input_ids"].long()
        masks = enc["attention_mask"].to(torch.bool)

        # ladder[h] = counts over k in 0..H (k=0 echoes the start node) + 'other'
        ladder = {"last": [0] * (H + 1), "all": [0] * (H + 1)}
        other = {"last": 0, "all": 0}
        total = {"last": 0, "all": 0}
        n_q = {"last": 0, "all": 0}
        # Which node does it actually say? Distinguishes "collapsed onto one
        # favourite node" (mass concentrated) from "uniform noise" (mass spread).
        pred_hist = [0] * args.N
        pred_hist_last = [0] * args.N

        bs = 64
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                end = min(start + bs, len(texts))
                bt = tokens[start:end].to(args.device)
                bm = masks[start:end].to(args.device)
                logits = model(bt, bm, K=K) if K is not None else model(bt, bm)
                logits = logits.float()

                for i in range(bt.size(0)):
                    row = bt[i].tolist()
                    meta = metas[start + i]
                    qpos = [p for p, t in enumerate(row) if t in q_ids]
                    # The '?' of query j is followed in letters format by exactly
                    # one answer token, so argmax at qpos predicts it.
                    for j, p in enumerate(qpos):
                        if j >= len(meta["queries"]):
                            break
                        q = meta["queries"][j]
                        pred_id = int(logits[i, p].argmax(dim=-1).item())
                        pred_node = node_of_token.get(pred_id)
                        chain = q["intermediates"]           # [start, pi, ..., pi^H]
                        k = None if pred_node is None else (
                            chain.index(pred_node) if pred_node in chain else None)
                        for scope in ("all",) + (("last",) if j == len(qpos) - 1 else ()):
                            total[scope] += 1
                            if k is None:
                                other[scope] += 1
                            else:
                                ladder[scope][k] += 1
                        if pred_node is not None:
                            pred_hist[pred_node] += 1
                            if j == len(qpos) - 1:
                                pred_hist_last[pred_node] += 1
                    n_q["last"] += 1

        def rates(scope):
            t = max(total[scope], 1)
            return {"acc": ladder[scope][H] / t,
                    **{f"k={k}": ladder[scope][k] / t for k in range(H + 1)},
                    "other": other[scope] / t,
                    "n": total[scope]}

        report[str(H)] = {"last_query": rates("last"), "all_queries": rates("all"),
                          "pred_hist": pred_hist, "pred_hist_last": pred_hist_last}
        rl = report[str(H)]["last_query"]
        print(f"\nH={H}  n={rl['n']}  (chance per node = 1/{args.N} = "
              f"{1 / args.N:.4f})")
        print("  last query:  " + "  ".join(
            f"{k}={v:.3f}" for k, v in rl.items() if k != "n"))
        ra = report[str(H)]["all_queries"]
        print(f"  all queries: acc={ra['acc']:.3f}  k=1={ra.get('k=1', 0):.3f}  "
              f"other={ra['other']:.3f}  n={ra['n']}")
        tot_h = max(sum(pred_hist), 1)
        top = sorted(range(args.N), key=lambda i: -pred_hist[i])[:3]
        print("  predicted node shares (all queries): "
              + ", ".join(f"{node_symbols(args.N, args.node_format)[i]}="
                          f"{pred_hist[i] / tot_h:.3f}" for i in top)
              + f"   [uniform would be {1 / args.N:.3f} each]")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"checkpoint": str(ckpt_path), "step": ckpt.get("step"),
                       "node_format": args.node_format, "N": args.N,
                       "eval_seed": args.eval_seed, "n_samples": args.n_samples,
                       "ladder": report}, f, indent=2)
        print(f"\n[ladder] wrote {out}")


if __name__ == "__main__":
    main()
