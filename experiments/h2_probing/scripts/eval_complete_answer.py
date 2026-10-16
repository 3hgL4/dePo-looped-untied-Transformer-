"""Full-answer accuracy of a trained checkpoint, after every iteration count of a looped model.

Every accuracy in the paper is full-answer accuracy: every answer token of the queried node plus
its `|` delimiter must be the argmax, teacher forced along the gold prefix, with earlier query
answers teacher forced (src/probe_h2.model_complete_answer_at_q, protocol
teacher_forced_answer_tokens_plus_delimiter_v1). Chance is 1/16: only node 1 is answered "1|".
This script scores the checkpoints whose probes used the earlier row split, which only recorded
first-answer-token accuracy, on the same evaluation set as the prompt-disjoint probe reports
(dePo N=16, M=5 queries, 5,000 prompts, eval_seed 12345, the training max_len, batch size 32), so
every panel stands on one footing. Query retention follows build_probe_inputs, so the 25,000
scored queries and their first/fresh/repeated-start strata match the probe reports row for row.

Looped models are scored after every iteration count t = 1..T (the loop stopped after t
iterations, the head read out as usual); untied models once. When a shipped validation report or
row-split probe report covers the same run, its numbers are attached for comparison.

    uv run python experiments/h2_probing/scripts/eval_complete_answer.py \
        --checkpoint runs/h1_clean_sweep/ouro-4x8_s42/checkpoint.pt \
        --out_dir probe_outputs/complete_answer_20261009

Writes <out_dir>/<run>_H<H>.json (run = the checkpoint's directory name) and rewrites
<out_dir>/summary.md over every result in out_dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.model import LoopedTransformer  # noqa: E402
from src.model_configs import create_model  # noqa: E402
from src.probe_h2 import (  # noqa: E402
    build_tokenizer, find_query_positions, generate_eval_set, get_q_token_ids, hash_evaluation_texts,
    model_complete_answer_at_q, pointer_query_record, tokenize_batch,
)

RES = ROOT / "experiments/h2_probing/results"
N, M = 16, 5
# The evaluation set of the prompt-disjoint probe reports at H=4 (5,000 prompts, eval_seed 12345).
EVAL_TEXT_SHA_H4 = "fc89833490947492a169bada20cbd083c0824016aac2dfa8e96ca6bc83bd2e1c"
REPORTS = [RES / "matched_lr3e4_20260915/validation_production.json",
           RES / "geometry_lr1e3_20260920/validation.json"]
ROW_SPLIT = [RES / "clean_sweep_5seed_h4", RES / "aux_rerun_h4"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def reference(run, H):
    """The full-answer result of a run in a shipped validation report, if one covers it."""
    for path in REPORTS:
        if not path.exists():
            continue
        for d in json.loads(path.read_text())["conditions"]:
            if d["run"] == run and d["H_eval"] == H:
                return {"report": rel(path), "checkpoint_sha256": d["checkpoint_sha256"],
                        "complete_answer": d["complete_answer"],
                        "first_answer_token_accuracy": d["first_answer_token_accuracy"]}
    return None


def first_token(run, H):
    """First-answer-token accuracy (500-prompt sanity check) and head k-sweep of a row-split probe report."""
    for sub in ROW_SPLIT:
        p = sub / run / "probe_results.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d["H_eval"] == H:
                return {"source": rel(p),
                        "accuracy": d["model_prediction_sanity"]["next_token_accuracy_at_probe_pos"],
                        "k_sweep_head_accuracy": d.get("k_sweep_head_accuracy")}
    ref = reference(run, H)
    if ref:
        return {"source": ref["report"], "accuracy": ref["first_answer_token_accuracy"],
                "k_sweep_head_accuracy": None}
    return None


def query_metadata(eval_set, tokenizer, max_len, H, batch_size):
    """Retained queries, by the retention rule of build_probe_inputs, without extracting states."""
    q_ids = get_q_token_ids(tokenizer)
    rows = []
    for start in range(0, len(eval_set), batch_size):
        chunk = eval_set[start:start + batch_size]
        tokens = tokenize_batch([s["text"] for s in chunk], tokenizer, max_len)["tokens"]
        positions = find_query_positions(tokens, q_ids)
        for b, sample in enumerate(chunk):
            queries = sample["meta"]["queries"]
            if len(positions[b]) < len(queries):
                continue
            for q_idx, q in enumerate(queries):
                if len(q["intermediates"]) == H + 1:
                    rows.append(pointer_query_record(sample["meta"], start + b, q_idx))
    return {name: np.asarray([r[name] for r in rows], dtype=bool if name.startswith("is_") else np.int64)
            for name in rows[0]}


def evaluate(ckpt: Path, run: str, args):
    train_args = json.loads((ckpt.parent / "args.json").read_text())
    assert train_args["N"] == N, "this scorer is for the N=16 task"
    max_len = int(train_args.get("max_len", 512))
    tokenizer = build_tokenizer(N, M=M, H_min=int(train_args.get("H_min", 1)),
                                H_max=int(train_args.get("H_max", 15)))
    model, config = create_model(train_args["config_name"], vocab_size=len(tokenizer), max_sequence_len=max_len,
                                 pad_token_idx=tokenizer.pad_token_id, use_nope=bool(train_args["use_nope"]),
                                 zero_init=bool(train_args.get("zero_init", True)),
                                 use_timestep_embedding=not bool(train_args.get("no_timestep_embedding", False)))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.to(args.device).eval()

    eval_set = generate_eval_set(N=N, H_eval=args.H_eval, M=M, n_samples=args.n_samples, seed=args.eval_seed)
    text_sha = hash_evaluation_texts([s["text"] for s in eval_set])
    if (args.H_eval, args.n_samples, args.eval_seed) == (4, 5000, 12345):
        assert text_sha == EVAL_TEXT_SHA_H4, "evaluation set differs from the probe reports"
    meta = query_metadata(eval_set, tokenizer, max_len, args.H_eval, args.batch_size)

    looped = isinstance(model, LoopedTransformer)
    T = config["K_max"] if looped else None
    per_t = []
    for t in (range(1, T + 1) if looped else [None]):
        summary, _ = model_complete_answer_at_q(eval_set, tokenizer, model, args.device, max_len,
                                                args.batch_size, meta, K=t)
        per_t.append(summary)
        print(f"  {run} t={t}: full-answer first-query {summary['strata']['first_query']['accuracy']:.4f}, "
              f"all {summary['complete_node_accuracy']:.4f}", flush=True)
    final = per_t[-1]
    out = {
        "run": run, "config_name": train_args["config_name"], "seed": train_args["seed"],
        "learning_rate": train_args["lr"], "aux_deep_supervision": bool(train_args.get("aux_deep_supervision", False)),
        "H_eval": args.H_eval, "N": N, "M": M, "n_samples": args.n_samples, "eval_seed": args.eval_seed,
        "max_len": max_len, "batch_size": args.batch_size, "device": args.device, "eval_text_sha256": text_sha,
        "checkpoint_source": str(ckpt), "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "protocol": final["protocol"], "chance_full_answer": 1 / 16, "chance_first_token": 7 / 16,
        "complete_answer": final,
        "first_query_accuracy": final["strata"]["first_query"]["accuracy"],
        "first_token_on_disk": first_token(run, args.H_eval),
    }
    if looped:
        out["T"] = T
        out["per_t"] = [{"t": t, "first_query_accuracy": s["strata"]["first_query"]["accuracy"],
                         "complete_node_accuracy": s["complete_node_accuracy"], "strata": s["strata"],
                         "answer_span_accuracy": s["answer_span_accuracy"], "boundary_accuracy": s["boundary_accuracy"]}
                        for t, s in enumerate(per_t, start=1)]
    ref = reference(run, args.H_eval)
    if ref:
        same = ref["complete_answer"]["strata"] == final["strata"] and ref["checkpoint_sha256"] == out["checkpoint_sha256"]
        out["sanity_check"] = {"reference": ref["report"], "reference_strata": ref["complete_answer"]["strata"],
                               "reproduced_exactly": same}
        print(f"  {run}: reproduces {ref['report']} exactly: {same}", flush=True)
    return out


def summarize(out_dir: Path):
    results = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_H*.json"))]
    lines = ["# Full-answer accuracy", "",
             "Written by `experiments/h2_probing/scripts/eval_complete_answer.py`. Protocol "
             "`teacher_forced_answer_tokens_plus_delimiter_v1`: every answer token of the queried node plus its `|` "
             "delimiter, earlier answers teacher forced; 5000 prompts (25000 queries), eval_seed 12345, the evaluation "
             "set of the prompt-disjoint probe reports. First query is the stratum shown in the figures. Chance is 1/16 "
             "for full answers and 7/16 for the first answer token. First-token numbers are the probe sanity check "
             "(500 prompts, first token at every query).", "",
             "| Run | H | lr | Full answer, first query | Full answer, all queries | First token (probe report) | Sanity check |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        ft = r["first_token_on_disk"]
        ft_text = f"{ft['accuracy']:.4f}" if ft else "no report"
        sc = r.get("sanity_check")
        check = ""
        if sc:
            check = "reproduces " + Path(sc["reference"]).parent.name if sc["reproduced_exactly"] else "MISMATCH"
        lines.append(f"| {r['run']} | {r['H_eval']} | {r['learning_rate']:g} | {r['first_query_accuracy']:.4f} | "
                     f"{r['complete_answer']['complete_node_accuracy']:.4f} | {ft_text} | {check} |")
    looped = [r for r in results if "per_t" in r]
    if looped:
        lines += ["", "## Loop stopped after t iterations (full answer, first query)", "",
                  "| Run | " + " | ".join(f"t={t}" for t in range(1, 17)) + " | first t >= 0.9 |",
                  "|---|" + "---|" * 17]
        for r in looped:
            accs = [p["first_query_accuracy"] for p in r["per_t"]]
            first = next((t for t, a in enumerate(accs, start=1) if a >= .9), None)
            cells = [f"{a:.3f}" for a in accs] + [""] * (16 - len(accs))
            lines.append(f"| {r['run']} | " + " | ".join(cells) + f" | {first if first else 'never'} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                   help="checkpoint.pt files, each with the run's args.json beside it")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--H_eval", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in args.checkpoint:
        run = ckpt.parent.name
        print(f"[complete-answer] {run} on {args.device}", flush=True)
        result = evaluate(ckpt, run, args)
        (args.out_dir / f"{run}_H{args.H_eval}.json").write_text(json.dumps(result, indent=1) + "\n")
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
