"""Figure A1: training loss against optimizer step for the three matched-depth models at lr 1e-3.

Untied 32-layer, looped 4x8 and looped 2x16, five seeds each. The source is each run's train.log
(runs/<cell>_s<seed>/train.log), which records the single-minibatch training loss (120 prompts) every
200 steps. The step-1 line (the loss at initialisation) is dropped, and each seed is smoothed with a
trailing mean over a 2,000-step window (the ten logged losses at steps in (s - 2000, s]); thin
translucent lines are seeds, the bold line is the cross-seed mean of the smoothed curves. The dashed
vertical line marks the end of the hop curriculum (7k + 23k = 30k steps), after which every chain
length up to H_max = 15 is sampled. Prints the smoothed loss at 30k and 100k for every seed.

    uv run python experiments/h1_clean_sweep/scripts/undertraining_figure.py [--out_dir figures/build]

Writes fig_undertraining.{pdf,png}.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.figstyle import PALETTE_L, apply_style  # noqa: E402

EXP = Path(__file__).resolve().parents[1]  # resolved at call time, so a test can point it elsewhere
DEFAULT_OUT = EXP.parents[1] / "figures/build"
WINDOW = 2000
SEEDS = (42, 43, 44, 45, 46)
CURRICULUM_END = 30_000
CELLS = [  # (run prefix, legend label, colour)
    ("deep-32", "Untied 32-layer", PALETTE_L[1]),
    ("ouro-4x8", r"Looped 4$\times$8", PALETTE_L[8]),
    ("ouro-2x16", r"Looped 2$\times$16", PALETTE_L[2]),
]
LINE_RE = re.compile(r"^step=\s*(\d+)\s+loss=([0-9.]+)")


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path} (see data/MANIFEST.sha256 for the shipped files)")
    return path


def smoothed_log_loss(log: Path):
    logged = {}
    for line in need(log).read_text().splitlines():
        m = LINE_RE.match(line)
        if m and int(m.group(1)) > 1:
            logged[int(m.group(1))] = float(m.group(2))
    steps = np.array(sorted(logged))
    if steps.size == 0 or steps[-1] < 100_000:
        raise ValueError(f"{log}: training loss not logged to step 100000")
    loss = np.array([logged[s] for s in steps])
    smooth = np.array([loss[(steps > s - WINDOW) & (steps <= s)].mean() for s in steps])
    return steps, smooth


def at(steps, values, step):
    return float(values[np.searchsorted(steps, step)])


def main(out_dir: Path = DEFAULT_OUT) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.41, 2.12))
    ax.axvline(CURRICULUM_END / 1000, color="0.55", linestyle=(0, (3, 2)), linewidth=0.6, zorder=1)
    print(f"Fig. A1 loss table (trailing {WINDOW}-step mean of train.log losses)")
    for cell, label, color in CELLS:
        curves, steps_ref = [], None
        for seed in SEEDS:
            steps, smooth = smoothed_log_loss(EXP / "runs" / f"{cell}_s{seed}" / "train.log")
            if steps_ref is None:
                steps_ref = steps
            assert np.array_equal(steps, steps_ref), f"{cell}_s{seed}: logged steps differ"
            curves.append(smooth)
            ax.plot(steps / 1000, smooth, color=color, alpha=0.25, linewidth=0.5, zorder=2)
            l30, l100 = at(steps, smooth, 30_000), at(steps, smooth, 100_000)
            print(f"  {cell:<10} s{seed}  loss@30k={l30:.4f}  loss@100k={l100:.4f}  drop={l30 - l100:+.4f}")
        mean = np.mean(np.stack(curves), axis=0)
        ax.plot(steps_ref / 1000, mean, color=color, linewidth=1.3, label=label, zorder=3)

    fs_label, fs_tick = 7.5, 6.5
    ax.set_xlabel("Training step (thousands)", fontsize=fs_label)
    ax.set_ylabel("Training loss", fontsize=fs_label)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", which="major", labelsize=fs_tick, length=3.0, width=0.6)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.legend(loc="upper right", fontsize=fs_tick, frameon=True,
              handlelength=1.6, borderpad=0.4, labelspacing=0.3).get_frame().set_linewidth(0.5)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_undertraining.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_undertraining.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_dir / 'fig_undertraining.pdf'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    main(p.parse_args().out_dir)
