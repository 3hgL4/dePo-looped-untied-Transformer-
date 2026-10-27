"""C2 crossing-point comparison: at which level of the stack does each chain
position peak, and how far apart are the two seeds?

Levels are counted in LAYER EVALUATIONS (level 0 = post-embedding, level j =
after the j-th layer evaluation, so j runs to K*L = 32 for ouro-4x8 with
--per_layer). One loop iteration = L = 4 level steps, which is the unit the
"spends n extra iterations" claim is written in.

WHY NOT "first level above 0.8": measured on ouro-4x8_s42 @ H_eval=8, the linear
probe for the INTERMEDIATE chain positions peaks at 0.43-0.59 (pos3 0.467,
pos4 0.459, pos5 0.425), so a 0.8 threshold never fires for them. What does move
cleanly is WHERE the peak sits: 5, 11, 15, 19, 23, 27, 31 for pos 0..6, i.e.
~+4 levels (= 1 iteration) per hop. The tool therefore reports, per chain
position: peak accuracy, peak level, and the "half-rise" level (first level
reaching 50% of that position's own peak, scale-free). The 0.8 crossing is kept
as a column for the FINAL position only, where the curve does reach ~0.87.

Usage:
    .venv/bin/python tools/c2_crossing.py \
        --pair s42:probe_outputs/c2/ouro-4x8_s42_H6 \
        --pair s44:probe_outputs/c2/ouro-4x8_s44_H6 \
        [--json out.json]

Notes:
  * Only compare runs with the SAME H_eval in one delta table; the tool refuses
    to print deltas across different rulers (exits with a warning instead).
  * linear_shuffled is the control: if the shuffled curve peaks as high as the
    real one, the real curve is label memorisation, not decodability.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def half_rise(curve):
    thr = 0.5 * float(curve.max())
    idx = np.where(curve >= thr)[0]
    return int(idx[0]) if idx.size else None


def first_cross(curve, thr):
    idx = np.where(curve >= thr)[0]
    return int(idx[0]) if idx.size else None


def load(run_dir):
    d = Path(run_dir)
    z = np.load(d / "probe_accuracies.npz")
    res = json.loads((d / "probe_results.json").read_text())
    cfg = res.get("model_config") or {}
    n_levels = res["n_levels"]
    K = cfg.get("K_max") or cfg.get("K")
    L = (n_levels - 1) // K if K else None
    return {
        "dir": str(d),
        "real": z["linear_real"],
        "shuf": z["linear_shuffled"],
        "H_eval": res["H_eval"],
        "n_levels": n_levels,
        "K": K,
        "L": L,
        "k_sweep": res.get("k_sweep_head_accuracy"),
        "n_train_queries": res.get("probe_split", {}).get("n_train_queries"),
    }


def per_position(run):
    """-> list of dicts, one per chain position 0..H_eval."""
    real, shuf = run["real"], run["shuf"]
    out = []
    for k in range(real.shape[1]):
        c = real[:, k]
        out.append({
            "pos": k,
            "peak": round(float(c.max()), 4),
            "peak_level": int(c.argmax()),
            "half_rise_level": half_rise(c),
            "cross08": first_cross(c, 0.8),
            "shuf_peak": round(float(shuf[:, k].max()), 4),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    help="label:dir, e.g. s42:probe_outputs/c2/ouro-4x8_s42_H6")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runs = []
    for p in args.pair:
        label, _, d = p.partition(":")
        r = load(d)
        r["label"] = label
        r["table"] = per_position(r)
        runs.append(r)

    payload = {"runs": []}
    for r in runs:
        print(f"=== {r['label']}   {r['dir']}")
        print(f"    H_eval={r['H_eval']}  levels={r['n_levels']}  K={r['K']}  L={r['L']}  "
              f"train_queries={r['n_train_queries']}")
        print("  pos |   peak  peak_lvl  half_rise  0.8_cross | shuffled_peak")
        for row in r["table"]:
            print(f"  {row['pos']:>3} | {row['peak']:>6.3f} {row['peak_level']:>9} "
                  f"{str(row['half_rise_level']):>10} {str(row['cross08']):>9} | "
                  f"{row['shuf_peak']:>8.3f}")
        if r["k_sweep"]:
            print("  k_sweep head acc (K_eval=1..): " +
                  " ".join(f"{x:.3f}" for x in r["k_sweep"]))
        payload["runs"].append({"label": r["label"], "dir": r["dir"],
                                "H_eval": r["H_eval"], "K": r["K"], "L": r["L"],
                                "rows": r["table"], "k_sweep": r["k_sweep"]})
        print()

    same_ruler = len(runs) == 2 and runs[0]["H_eval"] == runs[1]["H_eval"]
    if not same_ruler:
        if len(runs) == 2:
            print(f"!! mixed H_eval ({runs[0]['H_eval']} vs {runs[1]['H_eval']}): "
                  f"no delta table, the numbers are not on the same ruler.")
    else:
        a, b = runs
        L = a["L"]
        print(f"=== delta ({b['label']} - {a['label']}); 1 iteration = {L} levels")
        print("  pos | peak_lvl a  b | d_lvl  d_iter | half_rise a  b | d_lvl  d_iter | peak a     b")
        deltas = []
        for ra, rb in zip(a["table"], b["table"]):
            dp = rb["peak_level"] - ra["peak_level"]
            dh = (None if (ra["half_rise_level"] is None or rb["half_rise_level"] is None)
                  else rb["half_rise_level"] - ra["half_rise_level"])
            row = {
                "pos": ra["pos"],
                "peak_level": [ra["peak_level"], rb["peak_level"]],
                "d_peak_lvl": dp, "d_peak_iter": round(dp / L, 2) if L else None,
                "half_rise": [ra["half_rise_level"], rb["half_rise_level"]],
                "d_half_lvl": dh,
                "d_half_iter": (round(dh / L, 2) if (dh is not None and L) else None),
                "peak_acc": [ra["peak"], rb["peak"]],
            }
            deltas.append(row)
            print(f"  {ra['pos']:>3} | {ra['peak_level']:>10} {rb['peak_level']:>2} | "
                  f"{dp:>5} {str(row['d_peak_iter']):>7} | "
                  f"{str(ra['half_rise_level']):>11} {str(rb['half_rise_level']):>2} | "
                  f"{str(dh):>5} {str(row['d_half_iter']):>7} | "
                  f"{ra['peak']:.3f} {rb['peak']:.3f}")
        payload["delta"] = {"a": a["label"], "b": b["label"], "L": L, "rows": deltas}
        print()
        print("  read: d_peak_iter = how many extra loop iterations seed b needs before "
              "the probe for that chain position peaks.")

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
