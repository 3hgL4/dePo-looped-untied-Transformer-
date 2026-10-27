"""
Floor-free readouts for the first-token logit lens (P4).

Problem this fixes
------------------
`logit_lens_h2.py` reads node identity off the FIRST answer token only. With
the LLaMA/Qwen byte-level tokenizer, `" 10"` splits into [space, "1", "0"], so
nodes 1, 10, 11, 12, 13, 14, 15 all project onto the single token "1": sixteen
classes collapse to ten groups, the largest holding 7 nodes. A model that always
emits "1" therefore scores 7/16 = 0.4375 on that readout *for free*, and every
reported "accuracy" sits on top of that floor.

What is done instead
--------------------
Group the 16 nodes by shared first token and report three numbers per level:

* `node_top1_unambig`  - node-level accuracy over queries whose ANSWER node has
  a unique first token, chance = 1/16. Comparable to full-string accuracy; the
  honest "can it name the node" number, but it discards the 7 ambiguous nodes.
* `group_top1`         - accuracy of predicting the answer's token group,
  chance = sum_g (|g|/N)^2 (the group-frequency baseline, 0.2266 for N=16).
* `group_balanced`     - mean per-group recall (each of the 10 groups weighted
  equally), chance = 1/G = 0.10. This is the one that actually removes the
  0.4375 floor: a constant "1" prediction scores 0.10 here, not 0.4375.

Also reported: `p_answer` = mean probability mass on the ground-truth answer
node over the whole 16-way distribution (no argmax, so no floor at all; chance
= 1/16 = 0.0625).

At least one ambiguity-free number must accompany any first-token claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def node_groups(node_to_tok: np.ndarray | list[int]) -> list[list[int]]:
    """Group node ids by their first answer token, in ascending node order."""
    toks = [int(t) for t in np.asarray(node_to_tok).tolist()]
    by_tok: dict[int, list[int]] = {}
    for node, tok in enumerate(toks):
        by_tok.setdefault(tok, []).append(node)
    return [sorted(g) for g in sorted(by_tok.values(), key=lambda g: min(g))]


def group_of_node(groups: list[list[int]], n_nodes: int) -> np.ndarray:
    """Array node -> group index."""
    out = np.full(n_nodes, -1, dtype=np.int64)
    for gi, members in enumerate(groups):
        for node in members:
            out[node] = gi
    if (out < 0).any():
        raise ValueError("groups do not cover every node")
    return out


def first_token_floor(groups: list[list[int]], n_nodes: int) -> float:
    """Best constant-strategy score on the first-token readout (0.4375 @ N=16)."""
    return max(len(g) for g in groups) / n_nodes


def group_baseline_mass(groups: list[list[int]], n_nodes: int) -> float:
    """sum_g (|g|/N)^2: chance level of group_top1 under the marginal."""
    return float(sum((len(g) / n_nodes) ** 2 for g in groups))


def unambiguous_nodes(groups: list[list[int]]) -> list[int]:
    """Nodes whose first token is unique (singletons, 10..15 excluded at N=16)."""
    return sorted(g[0] for g in groups if len(g) == 1)


def readout_metrics(
    probs: np.ndarray,
    answer_nodes: np.ndarray,
    groups: list[list[int]],
) -> dict:
    """Per-level metrics for a lens tensor.

    Args:
        probs: (Q, L, N) first-token probabilities over the 16 nodes.
        answer_nodes: (Q,) ground-truth answer node id per query.
        groups: output of node_groups().

    Returns:
        dict of per-level lists plus the two scalar baselines.
    """
    probs = np.asarray(probs, dtype=np.float64)
    Q, L, N = probs.shape
    gof = group_of_node(groups, N)
    G = len(groups)

    pred_node = probs.argmax(-1)                    # (Q, L)
    pred_group = gof[pred_node]                     # (Q, L)
    gt_group = gof[answer_nodes]                    # (Q,)
    unambig = np.isin(answer_nodes, unambiguous_nodes(groups))
    gt_node_hit = pred_node == answer_nodes[:, None]  # (Q, L)

    group_top1, group_balanced, node_unambig, p_answer = [], [], [], []
    for lvl in range(L):
        hit = pred_group[:, lvl] == gt_group
        group_top1.append(float(hit.mean()) if Q else float("nan"))

        recalls = [float(hit[gt_group == g].mean()) if (gt_group == g).any() else float("nan")
                   for g in range(G)]
        recalls = [r for r in recalls if not np.isnan(r)]
        group_balanced.append(float(np.mean(recalls)) if recalls else float("nan"))

        node_unambig.append(float(gt_node_hit[unambig, lvl].mean())
                            if unambig.any() else float("nan"))
        p_answer.append(float(probs[np.arange(Q), lvl, answer_nodes].mean()) if Q else float("nan"))

    return {
        "n_queries": int(Q),
        "n_levels": int(L),
        "n_nodes": int(N),
        "groups": groups,
        "n_groups": G,
        "first_token_floor": first_token_floor(groups, N),
        "group_top1_baseline": group_baseline_mass(groups, N),
        "group_balanced_baseline": (1.0 / G) if G else float("nan"),
        "node_unambig_baseline": 1.0 / N,
        "p_answer_baseline": 1.0 / N,
        "n_unambig_queries": int(unambig.sum()),
        "node_top1_unambig": node_unambig,
        "group_top1": group_top1,
        "group_balanced": group_balanced,
        "p_answer": p_answer,
        "node_top1_all": [float(gt_node_hit[:, l].mean()) for l in range(L)],
    }


_METRIC_KEYS = (
    ("node_top1_unambig", True),
    ("group_top1", True),
    ("group_balanced", True),
    ("p_answer", False),
)


def format_table(name: str, met: dict) -> str:
    """One text table per lens output: rows = levels, columns = the four readouts."""
    lines = [
        f"== {name} ==",
        f"  queries={met['n_queries']}  levels={met['n_levels']}  groups={met['n_groups']}  "
        f"groups={met['groups']}",
        f"  baselines: first-token floor={met['first_token_floor']:.4f}  "
        f"group_top1={met['group_top1_baseline']:.4f}  "
        f"group_balanced={met['group_balanced_baseline']:.4f}  "
        f"p_answer={met['p_answer_baseline']:.4f}",
        f"  {'level':>5} {'node_unambig':>12} {'group_top1':>10} {'group_bal':>9} {'p_answer':>8}",
    ]
    for lvl in range(met["n_levels"]):
        lines.append(
            f"  {lvl:>5} {met['node_top1_unambig'][lvl]:>12.4f} "
            f"{met['group_top1'][lvl]:>10.4f} {met['group_balanced'][lvl]:>9.4f} "
            f"{met['p_answer'][lvl]:>8.4f}"
        )
    return "\n".join(lines)


def load_lens(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read probs / answer nodes / node->token map out of a logit_lens.npz."""
    with np.load(npz_path) as d:
        probs = d["probs"]
        chains = d["chains"]
        node_to_tok = d["node_to_tok"]
    return probs, chains[:, -1], [int(t) for t in node_to_tok.tolist()]


def summarize_file(npz_path: Path, json_out: Path | None = None) -> dict:
    probs, answer_nodes, node_to_tok = load_lens(npz_path)
    met = readout_metrics(probs, answer_nodes, node_groups(node_to_tok))
    met["source"] = str(npz_path)
    print(format_table(str(npz_path.parent), met))
    if json_out is not None:
        json_out.write_text(json.dumps(met, indent=2))
        print(f"  -> {json_out}")
    return met


def main() -> None:
    p = argparse.ArgumentParser(description="Floor-free readouts for logit-lens outputs (P4)")
    p.add_argument("paths", nargs="+", help="logit_lens.npz files or directories holding one")
    p.add_argument("--json", action="store_true",
                   help="Also write lens_metrics.json next to each input")
    p.add_argument("--summary", type=str, default=None,
                   help="Write a single JSON array with every analysed file")
    args = p.parse_args()

    npz_paths: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            npz_paths.extend(sorted(path.glob("*/logit_lens.npz")) or [path / "logit_lens.npz"])
        else:
            npz_paths.append(path)

    all_metrics = []
    for npz_path in npz_paths:
        if not npz_path.exists():
            print(f"skip (missing): {npz_path}")
            continue
        out = npz_path.parent / "lens_metrics.json" if args.json else None
        all_metrics.append(summarize_file(npz_path, out))
        print()

    if args.summary:
        Path(args.summary).write_text(json.dumps(all_metrics, indent=2))
        print(f"wrote {args.summary} ({len(all_metrics)} files)")


if __name__ == "__main__":
    main()
