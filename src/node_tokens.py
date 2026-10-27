"""
Answer formatting for the 16 nodes, and the floor it implies (P5).

Background
----------
The task writes node ids as decimal strings (`"0 > 3 , 3 > 7 , ..."`). With the
LLaMA/Qwen tokenizer, `" 10"` is [space, "1", "0"], so 1 and 10..15 share the
first token "1". Any analysis that reads only the first answer token therefore
has 10 distinguishable classes out of 16 nodes and a FREE 7/16 = 0.4375 for a
model that always answers "1". The main metric (`per_hop_acc`, full-string
comparison) is not affected: its floor is 1/16 = 0.0625.

Zero-padding to "00".."15" was proposed as a fix and is WRONG: it makes every
answer two tokens, the first token becomes "0" for 0..9 and "1" for 10..15, and
the floor rises to 10/16 = 0.625. Measured, not argued (proposal Sec 4.2).

What this module provides
-------------------------
1. `node_symbols(N, fmt)` - the label for each node under a format:
     "plain"   `0..15`    (current runs; 10 single-token + 6 two-token answers)
     "padded"  `00..15`   (measured worse; kept so the audit can show it)
     "letters" `A..P`     (16 single tokens, 16 distinct first tokens -> floor 1/16)
2. `audit(...)` - measure the tokenizer's actual behaviour for every format and
   report tokens-per-answer, distinct first tokens, and both floors.

`run_node_audit.py`-style usage:
    .venv/bin/python -m src.node_tokens --tokenizer hf-internal-testing/llama-tokenizer

Switching training to a new format is a NEW BENCHMARK (P5): it needs a retrain
(20-40 h) and its results cannot be mixed with the H_max=15 decimal runs. This
module only measures and prepares it; `--node_format letters` is wired through
the trainer as an opt-in, default "plain".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NODE_FORMATS = ("plain", "padded", "letters")


def node_symbols(N: int, fmt: str = "plain") -> list[str]:
    """Label for each node id under a format."""
    if fmt == "plain":
        return [str(v) for v in range(N)]
    if fmt == "padded":
        return [f"{v:02d}" for v in range(N)]
    if fmt == "letters":
        if N > 26:
            raise ValueError(f"letters format supports N<=26, got {N}")
        return [chr(ord("A") + v) for v in range(N)]
    raise ValueError(f"unknown node format {fmt!r}; expected one of {NODE_FORMATS}")


def format_answer(node: int, fmt: str = "plain", N: int = 16) -> str:
    """Answer string for one node (used by eval's string comparison)."""
    return node_symbols(N, fmt)[node]


def audit_format(tokenizer, N: int, fmt: str) -> dict:
    """Measure how `tokenizer` splits each node label under one format.

    `tokenizer` may be a raw HuggingFace tokenizer or the training
    `TokenMappingTokenizer` (ids are remapped, grouping is unchanged).

    Returns a dict with the table row plus the group structure.
    """
    labels = node_symbols(N, fmt)
    n_tokens, first_tok = [], []
    for label in labels:
        ids = tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not ids:
            raise RuntimeError(f"tokenizer produced 0 tokens for ' {label}'")
        # The task writes the answer right after "? ", so " 7" tokenizes as
        # [space, "7"]: drop the leading space and count the ANSWER tokens.
        n_tokens.append(len(ids) - 1 if len(ids) > 1 else 1)
        first_tok.append(int(ids[1]) if len(ids) > 1 else int(ids[0]))

    groups: dict[int, list[int]] = {}
    for node, tok in enumerate(first_tok):
        groups.setdefault(tok, []).append(node)
    group_list = [sorted(g) for g in sorted(groups.values(), key=lambda g: min(g))]
    max_group = max(len(g) for g in group_list)

    return {
        "format": fmt,
        "labels": labels,
        "n_tokens_per_answer": n_tokens,
        "n_single_token_answers": int(sum(1 for t in n_tokens if t == 1)),
        "first_tokens": first_tok,
        "groups": group_list,
        "n_groups": len(group_list),
        "max_group_size": max_group,
        "first_token_floor": max_group / N,
        "group_baseline_mass": float(sum((len(g) / N) ** 2 for g in group_list)),
        "distinct_first_tokens": len(group_list) == N,
    }


def audit(tokenizer, N: int = 16, formats=NODE_FORMATS) -> dict:
    """Audit every format and pick the best floor-free candidate."""
    rows = [audit_format(tokenizer, N, fmt) for fmt in formats]
    clean = [r for r in rows if r["distinct_first_tokens"]]
    best = min(rows, key=lambda r: (r["first_token_floor"], r["n_tokens_per_answer"]))
    return {
        "N": N,
        "formats": rows,
        "floor_free_formats": [r["format"] for r in clean],
        "best": best["format"],
        "recommended": clean[0]["format"] if clean else None,
    }


def format_report(result: dict) -> str:
    lines = [
        f"N={result['N']}  answer-format audit (first-token readout)",
        f"{'format':<9} {'1-tok':>6} {'groups':>7} {'max grp':>8} "
        f"{'floor':>7} {'sum p^2':>8} {'distinct':>9}",
        "-" * 58,
    ]
    for r in result["formats"]:
        lines.append(
            f"{r['format']:<9} {r['n_single_token_answers']:>6} {r['n_groups']:>7} "
            f"{r['max_group_size']:>8} {r['first_token_floor']:>7.4f} "
            f"{r['group_baseline_mass']:>8.4f} {str(r['distinct_first_tokens']):>9}"
        )
    lines.append("")
    lines.append(f"floor-free formats: {result['floor_free_formats'] or 'none'}")
    lines.append(f"lowest floor overall: {result['best']}")
    if result["recommended"]:
        lines.append(
            f"recommended (distinct first tokens, fewest tokens): {result['recommended']} "
            f"-> floor 1/{result['N']} = {1 / result['N']:.4f}"
        )
    else:
        lines.append("recommended: none - no format gives 16 distinct first tokens")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Audit answer formats against a tokenizer (P5)")
    p.add_argument("--tokenizer", default="hf-internal-testing/llama-tokenizer")
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--formats", default=",".join(NODE_FORMATS))
    p.add_argument("--out", default=None, help="Write the audit as JSON here")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    result = audit(tok, N=args.N, formats=tuple(args.formats.split(",")))
    result["tokenizer"] = args.tokenizer
    print(format_report(result))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
