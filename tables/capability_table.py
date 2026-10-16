"""Table 1 (looped 4x8 against untied 32-layer) and the capability numbers in the text, from the shipped
training trajectories.

Solve rule (the training-time evaluation, src/evaluate.py, logged in trajectory.jsonl every 4k steps):
the solved-chain depth H* of an evaluation is the largest H such that every hop length 1..H has
last-query full-answer accuracy >= 0.8. A seed solves the N=16 task when H* = 15 at the final evaluation
(step 100k, which every run must have reached). The step at which a seed solves is the first evaluation
with H* = 15.

    uv run python tables/capability_table.py [--out tables/capability_table.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/h1_clean_sweep"
THRESHOLD, MAX_HOP = 0.8, 15
SEEDS = (42, 43, 44, 45, 46)

# Where each run directory lives: the first grid, the rate and auxiliary-loss controls (seeds 42-44 of
# the rate controls, all of the auxiliary loss and tau_t), the 1 Oct gap fill (seeds 45-46 of the rate
# controls, untied 16 layers) and N=32.
RUN_DIRS = ("runs", "runs_rate_controls", "runs_gapfill_20261001", "runs_n32", "runs_n32_rerun_20261005")
N32_DIRS = {42: "runs_n32", 43: "runs_n32_rerun_20261005", 44: "runs_n32_rerun_20261005"}


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path} (see data/MANIFEST.sha256 for the shipped files)")
    return path


def solved_depth(per_hop: dict) -> int:
    h = 0
    while h < MAX_HOP and per_hop.get(str(h + 1), 0.0) >= THRESHOLD:
        h += 1
    return h


def find(run: str, where=RUN_DIRS) -> Path:
    hits = [EXP / d / run for d in where if (EXP / d / run).is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f"missing input: run {run} found in {len(hits)} of {where} (expected 1)")
    return hits[0]


def trajectory(run: str, where=RUN_DIRS, lr=None, N=16):
    d = find(run, where)
    args = json.loads(need(d / "args.json").read_text())
    rows = [json.loads(l) for l in need(d / "trajectory.jsonl").read_text().splitlines() if l.strip()]
    assert rows[-1]["step"] == args["total_steps"] == 100_000, f"{run}: final step {rows[-1]['step']}"
    assert args["N"] == N, f"{run}: N={args['N']}"
    if lr is not None:
        assert args["lr"] == lr, f"{run}: lr {args['lr']}, expected {lr}"
    return args, rows


def seed_record(run, **kw):
    args, rows = trajectory(run, **kw)
    depths = [solved_depth(r["per_hop_acc"]) for r in rows]
    first = next((r["step"] for r, h in zip(rows, depths) if h == MAX_HOP), None)
    return {"run": run, "H_star": depths[-1], "solves": depths[-1] == MAX_HOP, "first_solve_step": first}


def cell(prefix, lr, seeds=SEEDS, **kw):
    return [seed_record(f"{prefix}_s{s}", lr=lr, **kw) for s in seeds]


def rate(records):
    return f"{sum(r['solves'] for r in records)}/{len(records)}"


def mean_first_solve(records):
    steps = [r["first_solve_step"] for r in records if r["solves"]]
    return sum(steps) / len(steps) / 1000, len(steps)


def compute():
    c = {
        ("N16", "1e-3", "looped"): cell("ouro-4x8", 1e-3),
        ("N16", "1e-3", "untied"): cell("deep-32", 1e-3),
        ("N16", "3e-4", "looped"): cell("ouro-4x8-lr3e4", 3e-4),
        ("N16", "3e-4", "untied"): cell("deep-32-lr3e4", 3e-4),
        ("N16", "1e-4", "looped"): cell("ouro-4x8-lr1e4", 1e-4),
        ("N16", "1e-4", "untied"): cell("deep-32-lr1e4", 1e-4),
        ("N16 untied 16 layers", "3e-4", "untied"): cell("deep-16-lr3e4", 3e-4),
        ("N16 auxiliary loss", "1e-3", "untied"): cell("deep-32-aux", 1e-3),
        ("N16 no tau_t", "1e-3", "looped"): cell("ouro-4x8-notau", 1e-3, seeds=(42, 43, 44, 45)),
    }
    for fam, prefix in (("looped", "ouro-4x8"), ("untied", "deep-32")):
        c[("N32", "3e-4", fam)] = [seed_record(f"{prefix}_n32_s{s}", where=(N32_DIRS[s],), lr=3e-4, N=32)
                                   for s in (42, 43, 44)]
    c[("N32 untied 4 layers", "3e-4", "untied")] = [seed_record(f"deep-4_n32_s{s}", where=("runs_n32",), lr=3e-4,
                                                                N=32) for s in (42, 43, 46, 47)]
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "tables/capability_table.md")
    args = p.parse_args()
    c = compute()
    hstar = lambda rs: "H* = " + ", ".join(str(r["H_star"]) for r in rs)
    table = [  # (task, lr, looped, untied, paper looped, paper untied)
        ("N=16", "1e-3", rate(c[("N16", "1e-3", "looped")]), rate(c[("N16", "1e-3", "untied")]), "5/5", "0/5"),
        ("N=16", "3e-4", rate(c[("N16", "3e-4", "looped")]), rate(c[("N16", "3e-4", "untied")]), "5/5", "4/5"),
        ("N=16", "1e-4", rate(c[("N16", "1e-4", "looped")]), rate(c[("N16", "1e-4", "untied")]), "1/5", "1/5"),
        ("N=16, untied 16 layers", "3e-4", "--", rate(c[("N16 untied 16 layers", "3e-4", "untied")]), "--", "1/5"),
        ("N=16, auxiliary loss", "1e-3", "--", rate(c[("N16 auxiliary loss", "1e-3", "untied")]), "--", "2/5"),
        ("N=32", "3e-4", hstar(c[("N32", "3e-4", "looped")]), hstar(c[("N32", "3e-4", "untied")]),
         "H* = 6, 0, 1", "H* = 1, 1, 1"),
    ]
    lines = ["# Table 1: looped 4x8 against untied 32-layer", "",
             "Seeds that solve all 15 hops at N=16 (H* = 15 at the final evaluation), and H* per seed at N=32.", "",
             "| Task | lr | Looped 4x8 | Untied 32-layer | Paper (looped / untied) |", "|---|---|---|---|---|"]
    lines += [f"| {t} | {lr} | {a} | {b} | {pa} / {pb} |" for t, lr, a, b, pa, pb in table]
    lm, ln = mean_first_solve(c[("N16", "3e-4", "looped")])
    um, un = mean_first_solve(c[("N16", "3e-4", "untied")])
    lines += ["", "# Numbers in the text", "",
              "| Quantity | Value | Paper |", "|---|---|---|",
              f"| Mean first-solve step at lr 3e-4, looped 4x8 (solving seeds) | {lm:.1f}k over {ln} | 45k |",
              f"| Mean first-solve step at lr 3e-4, untied 32-layer (solving seeds) | {um:.1f}k over {un} | 55k |",
              f"| Looped 4x8 without tau_t, lr 1e-3, seeds that solve | {rate(c[('N16 no tau_t', '1e-3', 'looped')])} "
              f"| three of four |",
              f"| Untied 4-layer at N=32, lr 3e-4, H* per seed (42, 43, 46, 47) | "
              f"{', '.join(str(r['H_star']) for r in c[('N32 untied 4 layers', '3e-4', 'untied')])} | at most 1, four seeds |",
              "", "# Every seed", "", "| Condition | Run | H* (final) | First step with H* = 15 |", "|---|---|---|---|"]
    for (task, lr, fam), rs in c.items():
        for r in rs:
            lines.append(f"| {task}, lr {lr}, {fam} | {r['run']} | {r['H_star']} | {r['first_solve_step'] or '--'} |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
