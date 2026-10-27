"""Three-way frontier table for finished runs (P1), straight from trajectory.jsonl.

Zero GPU: reads the recorded per-hop accuracies and re-derives the numbers, so it
also works on the runs made before the trainer started logging fwd/rev/max.

    .venv/bin/python tools/frontier_report.py                       # runs/slice/*
    .venv/bin/python tools/frontier_report.py runs/slice/ouro-4x8_s44
    .venv/bin/python tools/frontier_report.py --sweep --json out.json

Columns: fwd/rev/max at the paper threshold 0.8, both counted in EFFECTIVE hops
(h_eff = min(H, 16-H), so 8 is the ceiling), the per-run span over thresholds
0.70/0.75/0.80/0.85, and the last recorded step that was read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import frontier_sweep, frontier_three, h_eff_vectors  # noqa: E402


def load_trajectory(run_dir: Path) -> list[dict]:
    path = run_dir / "trajectory.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def final_row(rows: list[dict]) -> dict | None:
    return max(rows, key=lambda r: r["step"]) if rows else None


def main() -> None:
    p = argparse.ArgumentParser(description="fwd/rev/max frontier of finished runs")
    p.add_argument("runs", nargs="*", help="run directories (default: runs/slice/*)")
    p.add_argument("--sweep", action="store_true", help="also print the threshold sweep")
    p.add_argument("--vectors", action="store_true", help="also print the 8-dim h_eff vectors")
    p.add_argument("--json", default=None, help="write the table as JSON here")
    args = p.parse_args()

    run_dirs = [Path(r) for r in args.runs] or sorted(Path("runs/slice").glob("*"))
    table, skipped = [], []
    for run_dir in run_dirs:
        rows = load_trajectory(run_dir)
        last = final_row(rows)
        if last is None or "per_hop_acc" not in last:
            skipped.append(run_dir.name)
            continue
        per_hop = last["per_hop_acc"]
        front = frontier_three(per_hop)
        sweep = frontier_sweep(per_hop)
        entry = {
            "run": run_dir.name,
            "step": last["step"],
            "loss": last.get("loss"),
            **{k: front[k] for k in ("fwd", "rev", "max")},
            "sweep_fwd": sweep["fwd"],
            "span_fwd": sweep["span_fwd"],
            "span_max": sweep["span_max"],
            "recorded": {k: last.get(k) for k in
                         ("frontier_fwd", "frontier_rev", "frontier_max")},
        }
        if args.vectors:
            entry["vectors"] = h_eff_vectors(per_hop)
        table.append(entry)

    hdr = f"{'run':<18} {'step':>7} {'fwd':>4} {'rev':>4} {'max':>4} " \
          f"{'fwd@0.7/0.75/0.8/0.85':>21} {'span':>5}"
    print(hdr)
    print("-" * len(hdr))
    for e in table:
        sw = "/".join(str(x) for x in e["sweep_fwd"])
        print(f"{e['run']:<18} {e['step']:>7} {e['fwd']:>4} {e['rev']:>4} {e['max']:>4} "
              f"{sw:>21} {e['span_fwd']:>5}")

    changed = [e for e in table
               if e["recorded"]["frontier_max"] is not None
               and e["recorded"]["frontier_max"] != e["max"]]
    if changed:
        print("\nrecorded frontier_max disagrees with the recomputation:")
        for e in changed:
            print(f"  {e['run']}: recorded {e['recorded']} vs computed max={e['max']}")
    elif table and table[0]["recorded"]["frontier_max"] is not None:
        print("\nrecorded frontier_* columns match the recomputation")

    risk = [e for e in table if e["span_fwd"] >= 3]
    if risk:
        print("\nthreshold-sensitive runs (fwd moves >= 3 levels between 0.70 and 0.85):")
        for e in risk:
            print(f"  {e['run']}: {e['sweep_fwd']}")

    if args.vectors:
        for e in table:
            v = e["vectors"]
            print(f"\n{e['run']}  fwd(H=1..8) = "
                  + " ".join(f"{x if x is None else round(x, 3)}" for x in v["fwd_acc"]))
            print(f"{'':<{len(e['run'])}}  rev(H=15..8) = "
                  + " ".join(f"{x if x is None else round(x, 3)}" for x in v["rev_acc"]))

    if skipped:
        print(f"\nskipped (no per_hop_acc): {', '.join(skipped)}")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
