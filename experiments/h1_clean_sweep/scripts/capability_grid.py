"""Figure 3: the capability grid at learning rate 1e-3.

Layers per block L against iterations T (untied stacks are the T=1 column), one node per trained cell
on its line of constant effective depth D_eff = L*T, colour and label = seeds that solve all 15 hops
(k/n). A seed solves when the final training-time evaluation (trajectory.jsonl, every hop 1..H at
accuracy >= 0.8) reaches H = 15. Counts are derived from runs/ and runs_gapfill_20261001/
<cell>_s<seed>/trajectory.jsonl and written beside the figure as capability_grid_counts.csv.
Every cell must have finished runs for seeds 42-46; anything else is an error.

    uv run python experiments/h1_clean_sweep/scripts/capability_grid.py [--out_dir figures/build]
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
import numpy as np
from src.figstyle import apply_style, add_subtle_grid

apply_style()

EXP = Path(__file__).resolve().parents[1]  # resolved at call time, so a test can point it elsewhere
RUN_DIRS = ("runs", "runs_gapfill_20261001")  # the first grid, and the gap fill (16x2, 4x3, 4x6, 8x3)
DEFAULT_OUT = EXP.parents[1] / "figures/build"
SEEDS = tuple(range(42, 47))
THRESHOLD, MAX_HOP = 0.8, 15
# Seed fraction is one of six values (k of 5), so the colour scale is discrete: white for none, then five
# steps of one blue hue. Kept apart from SELECTIVITY_CMAP (white to red) so the grid is not read as a
# probe map or as the looped-red family colour.
N_SEEDS = 5
FRACTION_CMAP = ListedColormap(["white"] + [plt.cm.Blues(v) for v in np.linspace(0.3, 0.95, N_SEEDS)])
FRACTION_NORM = BoundaryNorm(np.linspace(-0.1, 1.1, N_SEEDS + 2), FRACTION_CMAP.N)
CELL_RE = re.compile(r"^(?:deep-(\d+)|ouro-(\d+)x(\d+))_s(\d+)$")


def solved_prefix(per_hop: dict) -> int:
    h = 0
    while h < MAX_HOP and per_hop.get(str(h + 1), 0.0) >= THRESHOLD:
        h += 1
    return h


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path} (see data/MANIFEST.sha256 for the shipped files)")
    return path


def load_cells() -> dict[tuple[int, int], list[int]]:
    """(L, T) -> final solved prefix per seed, for the lr 1e-3 grid cells."""
    cells: dict[tuple[int, int], dict[int, int]] = {}
    for name in RUN_DIRS:
        for run in sorted(need(EXP / name).iterdir()):
            m = CELL_RE.match(run.name)
            if not m:
                continue  # the gap fill's -lr3e4/-lr1e4 runs belong to the rate controls
            deep, L, T, seed = m.groups()
            L, T = (int(deep), 1) if deep else (int(L), int(T))
            args = json.loads(need(run / "args.json").read_text())
            assert args.get("lr") == 1e-3, f"{run.name}: lr {args.get('lr')}, expected 1e-3"
            last = json.loads(need(run / "trajectory.jsonl").read_text().strip().splitlines()[-1])
            assert last["step"] == args["total_steps"], f"{run.name}: unfinished ({last['step']} steps)"
            assert int(seed) not in cells.get((L, T), {}), f"{run.name}: seed found in two run directories"
            cells.setdefault((L, T), {})[int(seed)] = solved_prefix(last["per_hop_acc"])
    if not cells:
        raise FileNotFoundError(f"missing input: no grid runs under {EXP}")
    for key, seeds in cells.items():
        assert tuple(sorted(seeds)) == SEEDS, f"cell {key}: seeds {sorted(seeds)}, expected {SEEDS}"
    return {key: [seeds[s] for s in sorted(seeds)] for key, seeds in sorted(cells.items())}



def effective_depth_arrow(fig, ax, lims, tail=(2.55, 2.55), length_pt=46) -> None:
    """Arrow pointing up and to the right, perpendicular on the page to the constant-D_eff diagonals.

    Its tail sits just past the D_eff=32 diagonal, between the 8x4 and 4x8 nodes, so it reads as
    continuing from the largest effective depth trained.

    Black and bold, the one exception to the no-bold figure rule. The angle depends on the axes'
    printed aspect, so this runs after the final tight_layout.
    """
    x0, x1, y0, y1 = lims
    fig.canvas.draw()
    box = ax.get_window_extent()
    sx, sy = box.width / (x1 - x0), box.height / (y1 - y0)  # display units per data unit
    # A diagonal runs along (sx, -sy) on the page, so the perpendicular is (sy, sx) = (sin, cos) of this angle.
    angle = np.arctan2(sy, sx)
    length = length_pt * fig.dpi / 72  # arrow length in display units
    dx, dy = length * np.sin(angle) / sx, length * np.cos(angle) / sy
    tx, ty = tail
    cx, cy = tx + dx / 2, ty + dy / 2
    ax.annotate("", xy=(tx + dx, ty + dy), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6, mutation_scale=13))
    bold = FontProperties(fname=Path(matplotlib.get_data_path()) / "fonts/ttf/cmb10.ttf", size=9)
    # The label sits on the arrow's upper-left side, offset along the page normal (-cos, sin) and rotated with it.
    ax.annotate(r"increasing $\mathbf{D}_{\mathbf{eff}}$", xy=(cx, cy),
                xytext=(-4 * np.cos(angle), 4 * np.sin(angle)), textcoords="offset points",
                rotation=90 - np.degrees(angle), rotation_mode="anchor", ha="center", va="bottom",
                fontproperties=bold, color="black")


def main(out_dir: Path = DEFAULT_OUT) -> None:
    cells = load_cells()
    solved = {key: sum(h == MAX_HOP for h in hs) for key, hs in cells.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "capability_grid_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "T", "D_eff", "solved_prefix_per_seed", "solved", "seeds"])
        for (L, T), hs in cells.items():
            w.writerow([L, T, L * T, " ".join(map(str, hs)), solved[(L, T)], len(hs)])

    # Built at the printed size (NeurIPS text width 5.5 in), so the font sizes below are the printed ones.
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    x0, x1, y0, y1 = -0.5, 4.55, 0.5, 5.55
    for D in sorted({L * T for L, T in cells}):
        d, power = np.log2(D), D & (D - 1) == 0
        ax.plot([x0, d - y0], [d - x0, y0], ls="-", lw=0.6 if power else 0.4,
                color="0.72" if power else "0.86", zorder=1)
        if power:  # labelled where the diagonal enters at the left edge, between the untied nodes
            label = rf"$D_{{\mathrm{{eff}}}}{{=}}{D}$" if D == 2 else f"{D}"
            ax.text(x0 + 0.04, d - x0 - 0.04, label, ha="left", va="top", fontsize=7, color="0.35")
    ax.axvline(0.5, ls="--", lw=0.8, color="0.35", zorder=2)  # untied (T=1) | looped (T>=2)
    ax.text(0.42, y1 - 0.2, "untied", ha="right", va="center", fontsize=8, color="0.35")
    ax.text(0.58, y1 - 0.2, "looped", ha="left", va="center", fontsize=8, color="0.35")

    for (L, T), k in solved.items():
        x, y, n = np.log2(T), np.log2(L), len(cells[(L, T)])
        ax.scatter(x, y, s=120, color=FRACTION_CMAP(FRACTION_NORM(k / n)), edgecolor="black", linewidth=0.5,
                   zorder=3)
        ax.text(x, y, f"{k}/{n}", ha="center", va="center", fontsize=5.2,
                color="white" if k / n >= 0.6 else "black", zorder=4)

    ax.set_xticks(range(5), ["1", "2", "4", "8", "16"])
    ax.set_yticks(range(1, 6), ["2", "4", "8", "16", "32"])
    ax.minorticks_off()
    add_subtle_grid(ax, axis="both")
    ax.tick_params(labelsize=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Iterations $T$", fontsize=9)
    ax.set_ylabel("Layers in the block $L$", fontsize=9)
    ax.set_title("Seeds that solve all 15 hops", fontsize=9)  # the learning rate is in the caption
    sm = plt.cm.ScalarMappable(cmap=FRACTION_CMAP, norm=FRACTION_NORM)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, ticks=np.linspace(0, 1, N_SEEDS + 1))
    cb.set_label("Fraction of seeds", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.minorticks_off()

    fig.tight_layout()
    effective_depth_arrow(fig, ax, (x0, x1, y0, y1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"capability_grid.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir}/capability_grid.{{png,pdf}} and capability_grid_counts.csv")
    for (L, T), k in solved.items():
        print(f"  L={L:2d} T={T:2d} D_eff={L*T:2d}  {k}/{len(cells[(L, T)])}  {cells[(L, T)]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    main(p.parse_args().out_dir)
