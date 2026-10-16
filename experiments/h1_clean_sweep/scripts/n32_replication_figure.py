"""Figure A4: N=32 comparison of looped 4x8 and untied 32-layer, where no backward shortcut exists.

Per-hop accuracy at the final step (100k) for the three seeds per family of Table 1's N=32 row (lr 3e-4,
384-token context). At N=32 with H_max=15 every tested hop has H < N-H, so the backward route that
exists at N=16 for H>8 is unavailable.

Metric: the per_hop_acc column of each run's trajectory.jsonl, written by src/evaluate.py during
training: teacher-forced, full-answer exact match on the last of the M=5 queries in each prompt, about
533 prompts per hop. A uniform guess over 32 nodes scores 1/32; the last query can also copy an earlier
query's answer when it repeats that query's start node, which lifts the floor of a model that has not
solved H to about 0.15 to 0.2.

Seed sources:
  s42       experiments/h1_clean_sweep/runs_n32/<arch>_n32_s42/
  s43, s44  experiments/h1_clean_sweep/runs_n32_rerun_20261005/<arch>_n32_s<seed>/

Drawn at its printed size (3.3 x 2.0 in, included at 0.6 of the 5.5 in text width). Looped seed 44
coincides with the untied seeds (solved-chain depth 1, then the copy floor), so the looped lines are drawn
above the untied ones and each family is shifted by a small horizontal offset.

    uv run python experiments/h1_clean_sweep/scripts/n32_replication_figure.py [--out_dir figures/build]

Writes fig_n32.{pdf,png} and checks the solved-chain depths against Table 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.figstyle import ACCENT_RED, MARKERS_L, PALETTE_L, apply_style  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]  # resolved at call time, so a test can point it elsewhere
DEFAULT_OUT = EXP_DIR.parents[1] / "figures/build"
TOTAL_STEPS = 100_000
H_AXIS = np.arange(1, 16)
EXPECTED_DEPTHS = {"ouro-4x8": [6, 0, 1], "deep-32": [1, 1, 1]}  # Table 1, N=32 row

SEED_DIRS = {42: "runs_n32", 43: "runs_n32_rerun_20261005", 44: "runs_n32_rerun_20261005"}

# Printed size and text sizes (points at 100% scale), matching the main-text figures.
FIGSIZE = (3.3, 2.0)
LABEL_PT, TICK_PT, LEGEND_PT = 7.5, 6.5, 6.5

# (arch, legend label, colour, marker, x offset, zorder): looped red on top, untied grey below.
ARCHS = [
    ("deep-32", "Untied 32-layer", PALETTE_L[1], MARKERS_L[1], +0.12, 2),
    ("ouro-4x8", r"Looped 4$\times$8", PALETTE_L[8], MARKERS_L[8], -0.12, 3),
]


def final_per_hop(arch: str, seed: int) -> np.ndarray:
    p = EXP_DIR / SEED_DIRS[seed] / f"{arch}_n32_s{seed}" / "trajectory.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p} (see data/MANIFEST.sha256 for the shipped files)")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    last = rows[-1]
    if last["step"] != TOTAL_STEPS:
        raise SystemExit(f"{p}: last step {last['step']}, expected {TOTAL_STEPS}")
    return np.array([float(last["per_hop_acc"][str(h)]) for h in H_AXIS])


def solved_prefix(acc: np.ndarray, thr: float = 0.8) -> int:
    below = np.flatnonzero(acc < thr)
    return int(below[0]) if below.size else len(acc)


def main(out_dir: Path = DEFAULT_OUT):
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    handles = {}
    for arch, label, color, marker, dx, z in ARCHS:
        depths = []
        for seed in sorted(SEED_DIRS):
            acc = final_per_hop(arch, seed)
            depths.append(solved_prefix(acc))
            (line,) = ax.plot(H_AXIS + dx, acc, color=color, marker=marker, markersize=2.6,
                              markeredgewidth=0.5, markerfacecolor="white" if arch == "deep-32" else color,
                              linewidth=0.8, zorder=z, label=label, clip_on=False)
            handles[label] = line
        print(f"{arch:9s} solved-chain depth per seed {sorted(SEED_DIRS)}: {depths}")
        if depths != EXPECTED_DEPTHS[arch]:
            raise SystemExit(f"{arch}: depths {depths} differ from Table 1 {EXPECTED_DEPTHS[arch]}")

    ax.axhline(0.8, color=ACCENT_RED, linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xlabel(r"Hop length $H$", fontsize=LABEL_PT, labelpad=2)
    ax.set_ylabel("Accuracy", fontsize=LABEL_PT, labelpad=2)
    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(range(1, 16, 2))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(which="major", labelsize=TICK_PT, width=0.5, length=2.5, pad=2)
    ax.tick_params(which="minor", width=0.4, length=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    order = [r"Looped 4$\times$8", "Untied 32-layer"]
    ax.legend([handles[k] for k in order], order, loc="upper right", fontsize=LEGEND_PT,
              handlelength=1.8, borderpad=0.35, labelspacing=0.3, borderaxespad=0.4,
              frameon=True).get_frame().set_linewidth(0.5)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_n32.{ext}"
        fig.savefig(out, bbox_inches=fig.bbox_inches, dpi=300)  # exact printed size
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    main(p.parse_args().out_dir)
