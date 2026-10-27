"""C4 matched-control 2x2 table with trial-clustered bootstrap CIs.

Reads intervention_outputs/p3p5c4_control/<cell>/{intervention_summary.json,
intervention.npz} and prints, per cell and per injection iteration j:

    real->real   real activation, real target      (mechanism: HIGH)
    ctrl->ctrl   control activation, control target (mechanism: HIGH)
    ctrl->real   control activation, real target    (leakage/position: HIGH)
    real->ctrl   real activation, control target    (leakage/position: HIGH)

Null is 1/N = 0.0625 for all four: the comparison is against the written node
(digits plus delimiter), so no ambiguity filtering is involved.

CIs resample TRIALS, not records: the M=5 queries of one trial share the same
donor prompt and the same permutation pair, so records are not independent.
The trial index is recovered as record_index // M (records are appended trial
by trial and no trial was skipped; both facts are asserted).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

METRICS = ("real_vs_real", "ctrl_vs_ctrl", "ctrl_vs_real", "real_vs_ctrl")


def load_cell(d: str) -> dict:
    s = json.load(open(os.path.join(d, "intervention_summary.json")))
    z = np.load(os.path.join(d, "intervention.npz"))
    # Valid injection iterations are j = 0..K-1. When K == H the composed target
    # also has an entry for j = H, but no row was ever run for it (pred stays at
    # the -2 fill), so that column is an artifact that reads as 0.0 — drop it.
    J = min(s["H"] + 1, s["K"])
    agree = [(z["full_by_j"][:, j] == z["ctrl_full_by_j"][:, j]).mean() for j in range(s["K"])]
    return {"name": os.path.basename(d.rstrip("/")), "summary": s,
            "composed": z["composed"][:, :J], "composed_ctrl": z["composed_ctrl"][:, :J],
            "full": z["full_by_j"][:, :J], "ctrl_full": z["ctrl_full_by_j"][:, :J],
            "agree": agree, "dropped_j": s["H"] + 1 - J}


def cell_rates(cell: dict, idx: np.ndarray) -> np.ndarray:
    """(4, J) match rates over the rows selected by `idx`."""
    comp = cell["composed"][idx]
    ctrl = cell["composed_ctrl"][idx]
    full = cell["full"][idx]
    cfull = cell["ctrl_full"][idx]
    ok = (ctrl >= 0).all(axis=1)          # rows with a valid partner
    comp, ctrl, full, cfull = comp[ok], ctrl[ok], full[ok], cfull[ok]
    J = comp.shape[1]
    full, cfull = full[:, :J], cfull[:, :J]
    out = np.zeros((4, J))
    n = max(len(comp), 1)
    out[0] = ((full == comp).sum(axis=0)) / n
    out[1] = ((cfull == ctrl).sum(axis=0)) / n
    out[2] = ((cfull == comp).sum(axis=0)) / n
    out[3] = ((full == ctrl).sum(axis=0)) / n
    return out


def clustered_ci(cell: dict, M: int, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rec = len(cell["composed"])
    assert n_rec % M == 0, f"{n_rec} records not divisible by M={M}"
    n_tr = n_rec // M
    point = cell_rates(cell, np.arange(n_rec))
    rng = np.random.default_rng(seed)
    boots = np.zeros((draws, 4, point.shape[1]))
    for b in range(draws):
        pick = rng.integers(0, n_tr, n_tr)
        idx = (pick[:, None] * M + np.arange(M)[None, :]).ravel()
        boots[b] = cell_rates(cell, idx)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return point, lo, hi


def fmt(v, lo, hi, chance):
    star = "*" if lo > chance else " "
    return f"{v:.3f}{star}[{lo:.3f},{hi:.3f}]"


def pooled_rates(cell: dict, idx: np.ndarray, j0: int, j1: int) -> np.ndarray:
    """Four match rates pooled over injection iterations j0..j1 (inclusive).

    j0..H-2 is the COMPOSITION window (one or more receiver iterations still to
    run, so composed(j) is a real mixture of both permutations); j >= H-1 is the
    IDENTITY window, where donor_hop clamps to H and composed(j) collapses to
    pure_A. Pooling is only meaningful inside one window.
    """
    comp = cell["composed"][idx][:, j0:j1 + 1]
    ctrl = cell["composed_ctrl"][idx][:, j0:j1 + 1]
    full = cell["full"][idx][:, j0:j1 + 1]
    cfull = cell["ctrl_full"][idx][:, j0:j1 + 1]
    ok = (ctrl >= 0).all(axis=1)
    comp, ctrl, full, cfull = comp[ok], ctrl[ok], full[ok], cfull[ok]
    denom = max(comp.size, 1)
    return np.array([(full == comp).sum(), (cfull == ctrl).sum(),
                     (cfull == comp).sum(), (full == ctrl).sum()]) / denom


def pooled_ci(cell: dict, M: int, j0: int, j1: int, draws: int, seed: int):
    n_rec = len(cell["composed"])
    n_tr = n_rec // M
    point = pooled_rates(cell, np.arange(n_rec), j0, j1)
    rng = np.random.default_rng(seed)
    boots = np.zeros((draws, 4))
    for b in range(draws):
        pick = rng.integers(0, n_tr, n_tr)
        idx = (pick[:, None] * M + np.arange(M)[None, :]).ravel()
        boots[b] = pooled_rates(cell, idx, j0, j1)
    return point, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="intervention_outputs/p3p5c4_control")
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chance", type=float, default=1 / 16)
    ap.add_argument("--comp_start", type=int, default=1,
                    help="First injection iteration pooled into the composition window. "
                         "j=0 is a verified no-op: two provably different injections there "
                         "give identical predictions (washout 1.000), so it only adds a "
                         "structural zero.")
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()

    print("columns: real->real | ctrl->ctrl | ctrl->real | real->ctrl "
          f"(chance {args.chance:.4f}); * = CI lower bound above chance")

    cells = [load_cell(d) for d in sorted(glob.glob(os.path.join(args.root, "*/")))
             if os.path.isfile(os.path.join(d, "intervention_summary.json"))]
    if not cells:
        raise SystemExit(f"no summaries under {args.root}")

    payload = {}
    for c in cells:
        point, lo, hi = clustered_ci(c, args.M, args.draws, seed=0)
        s = c["summary"]
        print(f"\n### {c['name']}  (H={s['H']} K={s['K']} mode={s['mode']} "
              f"randinit={s.get('random_init')} n_rec={s['n_records']} "
              f"n_ctrl={s['control_matched']['n_ctrl'] if 'control_matched' in s else '-'})")
        if "control_matched" in s:
            print(f"    json ctrl_vs_ctrl = "
                  f"{[None if x is None else round(x, 3) for x in s['control_matched']['ctrl_vs_ctrl']]}")
        print("    | j | real->real | ctrl->ctrl | ctrl->real | real->ctrl |")
        print("    |---|---|---|---|---|")
        for j in range(point.shape[1]):
            row = " | ".join(fmt(point[i, j], lo[i, j], hi[i, j], args.chance) for i in range(4))
            print(f"    | {j} | {row} |")
        K, H = s["K"], s["H"]
        if c["dropped_j"]:
            print(f"    (dropped j>={H}: no row exists, K=H)")
        print("    washout real==ctrl " + " ".join(f"{x:.2f}" for x in c["agree"]))
        pooled = {}
        windows = [("COMP j>=1", max(1, args.comp_start), min(H - 2, K - 1)),
                   ("COMP j>=0", 0, min(H - 2, K - 1)),
                   ("IDENTITY", H - 1, min(H, K - 1))]
        for label, j0, j1 in windows:
            if j0 > j1:
                print(f"    {label:12s} window empty")
                continue
            p, l, h = pooled_ci(c, args.M, j0, j1, args.draws, seed=1)
            print(f"    {label:12s} j={j0}..{j1}  " +
                  "  ".join(fmt(p[i], l[i], h[i], args.chance) for i in range(4)))
            pooled[label] = {"j0": j0, "j1": j1,
                             **{m: {"rate": float(p[i]), "lo": float(l[i]), "hi": float(h[i])}
                                for i, m in enumerate(METRICS)}}
        payload[c["name"]] = {
            "H": s["H"], "K": s["K"], "random_init": s.get("random_init"),
            "n_records": s["n_records"], "dropped_j": c["dropped_j"],
            "j_grid": list(range(point.shape[1])), "pooled": pooled,
            **{m: {"rate": point[i].tolist(), "lo": lo[i].tolist(), "hi": hi[i].tolist()}
               for i, m in enumerate(METRICS)},
        }

    print("\n(* = 95% trial-clustered bootstrap CI lower bound above chance "
          f"{args.chance:.4f}; italic-free plain text on purpose)")
    if args.json_out:
        json.dump(payload, open(args.json_out, "w"), indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
