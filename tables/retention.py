"""Retention of the looped 4x8 model at H=8 (App. "Probing details", paragraph "Retention").

After its peak, a hop's selectivity stays at or above half that peak for a number of further
iterations (consecutive, counted from the peak), its retention. Averaged over hops 1 to 7 (the answer
has no iterations left), for three probe sets:
  lr 3e-4, seeds 42-44, prompt-disjoint probes after every layer (results/matched_lr3e4_20260915),
      counted in layers and divided by 4 (layers per iteration);
  lr 1e-3, seed 42, the prompt-disjoint re-probe after every layer (results/geometry_lr1e3_20260920), likewise;
  lr 1e-3, seeds 42-45, row-split probes after every iteration (results/clean_sweep_5seed), in iterations.

    uv run python tables/retention.py [--out tables/retention.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/h2_probing/scripts"))
import figcommon as fc  # noqa: E402

import numpy as np  # noqa: E402

H = 8
LAYERS_PER_ITERATION = 4


def retention(sel, H=H):
    """Levels after the peak with selectivity >= half the peak, for each hop 1..H-1."""
    S = np.asarray(sel)
    out = []
    for h in range(1, H):
        s = S[:, h]
        peak = int(np.argmax(s))
        n = 0
        for v in s[peak + 1:]:
            if v < s[peak] / 2:
                break
            n += 1
        out.append(n)
    return out


def rows():
    m = fc.matched()
    g = fc.geometry("ouro-4x8_s42")[H]
    assert g["learning_rate"] == 1e-3
    p1 = [retention(m[("ouro-4x8", s, H)]["selectivity"]) for s in fc.SEEDS]
    reprobe = [retention(g["selectivity"])]
    old = []
    for s in (42, 43, 44, 45):
        d = fc.res("clean_sweep_5seed", f"ouro-4x8_s{s}")
        info = fc.load_json(d / "probe_results.json")
        arr = fc.load_npz(d / "probe_accuracies.npz")
        assert info["H_eval"] == H and info["n_levels_label"] == "iteration" and arr["linear_real"].shape == (9, H + 1)
        old.append(retention(fc.selectivity(arr)))
    return [
        ("lr 3e-4, seeds 42-44, prompt-disjoint, per layer / 4", np.mean(p1) / LAYERS_PER_ITERATION, p1, "2.2"),
        ("lr 1e-3, seed 42 re-probe, prompt-disjoint, per layer / 4", np.mean(reprobe) / LAYERS_PER_ITERATION,
         reprobe, "2.6"),
        ("lr 1e-3, seeds 42-45, row split, per iteration", float(np.mean(old)), old, "1.7"),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "tables/retention.md")
    args = p.parse_args()
    lines = ["# Retention, looped 4x8 at H=8, mean over hops 1-7 (iterations)", "",
             "| Probe set | Retention | Paper | Per seed, hops 1-7 (levels) |", "|---|---|---|---|"]
    for label, value, per_seed, paper in rows():
        lines.append(f"| {label} | {value:.2f} | {paper} | {per_seed} |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
