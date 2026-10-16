"""Figure 5: how the loop spends its iterations at H=8, lr 1e-3.

(a)-(c) First decodable iteration of each reference node (layer / L, so within-iteration timing stays
visible), one line per seed, ordered by budget: (a) extra loops, looped 2x16 (T=16 > H); (b) no slack, 4x8
(T=8 = H); (c) loop deficit, 8x4 (T=4 < H). The grey diagonal is one hop per iteration, and nodes that
never reach selectivity 0.5 sit in a "never" column right of the last iteration. Onset rule in
figcommon.py. Prompt-disjoint probes after every layer: seed 42 of 4x8 and 2x16 from
results/geometry_lr1e3_20260920, every other run from results/fig5_h8_lr1e3_20261009.

A run is drawn if its full-answer first-query accuracy at H=8 is at least 0.95 (from its probe report),
and only seeds 42, 43, 44 and 46 can be drawn, since seed 45 solves in no budget. Seed 45 of the 2x16
model does not solve the task in training and was not probed.

    uv run python experiments/h2_probing/scripts/build_fig5_dynamics.py [--out_dir figures/build]

Writes fig5_dynamics.{pdf,png} and prints the onsets and accuracies of every probed run.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcommon as fc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np
from src.figstyle import SEED_PALETTE

TEXTWIDTH = 5.5
H = 8
SEEDS = (42, 43, 44, 45, 46)
NOT_PROBED = {("ouro-2x16", 45)}
SOLVED = .95
FAMILIES = {  # cell: (layers per block L, loop count T, title)
    "ouro-2x16": (2, 16, "(a) Extra loops: $\\mathit{T>H}$"),
    "ouro-4x8": (4, 8, "(b) No slack: $\\mathit{T=H}$"),
    "ouro-8x4": (8, 4, "(c) Loop deficit: $\\mathit{T<H}$"),
}
SEED_OFFSET = {s: .07 * (s - 44) for s in SEEDS}  # small vertical offsets so coincident seeds stay visible
# Every seed that solves in some panel, one colour (SEED_PALETTE) and one marker each, the same in every
# panel, so a seed can be followed across budgets and into the never column.
ONSET_SEEDS = (42, 43, 44, 46)
SEED_MARKERS = {42: "o", 43: "s", 44: "^", 46: "D"}


def probe_dir(cell, seed):
    if seed == 42 and cell in ("ouro-4x8", "ouro-2x16"):
        return fc.res("geometry_lr1e3_20260920/results/H8", f"{cell}_s{seed}")
    return fc.res("fig5_h8_lr1e3_20261009", f"{cell}_s{seed}_H{H}")


def load_probe(cell, seed):
    """Selectivity (33 levels x 9 targets) and full-answer first-query accuracy of one probed run."""
    d = probe_dir(cell, seed)
    info = fc.load_json(d / "probe_results.json")
    assert info["H_eval"] == H and info["config_name"] == cell and info["probe_split"]["unit"] == "prompt"
    assert info["args"]["n_samples"] == 5000 and info["args"]["eval_seed"] == 12345, d
    acc = fc.load_npz(d / "probe_accuracies.npz")
    assert acc["linear_real"].shape == (33, H + 1)
    full = fc.load_json(d / "model_complete_answer.json")
    assert full["protocol"] == fc.FULL_ANSWER
    return fc.selectivity(acc), full["strata"]["first_query"]["accuracy"]


def panel_title(ax, text):
    """Title as text at the same height in every panel (fc.save resets axes titles)."""
    ax.text(.5, 1.04, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def onset_panel(ax, cell, record):
    block, T, title = FAMILIES[cell]
    never = T * 1.22
    for t in range(1, T + 1):  # iteration boundaries: hops between two neighbouring lines share an iteration
        ax.axvline(t, color=".9", lw=.35, zorder=0)
    ax.plot([0, T], [0, T], color=".6", lw=.6, zorder=0)  # one hop per iteration
    ax.axvline(T * 1.11, color=".75", lw=.4, zorder=0)
    for seed in SEEDS:
        if (cell, seed) in NOT_PROBED:
            continue
        sel, full = load_probe(cell, seed)
        ons_all = fc.onsets(np.asarray(sel))  # start node, hops 1-7, answer
        drawn = full >= SOLVED and seed in ONSET_SEEDS
        record[cell][seed] = {"full_answer_first_query": round(float(full), 4), "onset_layer": ons_all[1:],
                              "drawn": drawn}
        if not drawn:
            continue
        y = np.arange(0, H + 1) + SEED_OFFSET[seed]
        x = np.array([np.nan if o is None else o / block for o in ons_all], dtype=float)
        color, marker = SEED_PALETTE[seed], SEED_MARKERS[seed]
        ms = 2.3 if marker == "D" else 2.6  # diamonds read larger
        ax.plot(x, y, color=color, lw=.6, marker=marker, ms=ms, mfc="white", mew=.6)
        miss = np.isnan(x)  # never decodable: the seed's own marker in the never column
        ax.plot(np.full(miss.sum(), never), y[miss], ls="none", color=color, marker=marker, ms=ms, mfc="white", mew=.6)
    panel_title(ax, title)
    ax.set_xlim(-T * .04, T * 1.3)
    ticks = list(range(0, T + 1, T // 4))
    ax.set_xticks(ticks + [never], [str(t) for t in ticks] + ["never"])
    ax.set_ylim(-.5, H + .5)
    ax.set_yticks(range(H + 1))
    ax.tick_params(labelleft=False)
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_locator(FixedLocator([]))


def main(out_dir=fc.DEFAULT_OUT):
    fc.style()
    record = {cell: {} for cell in FAMILIES}
    fig = plt.figure(figsize=(TEXTWIDTH, 2.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[.3, 1, 1, 1], wspace=.1, left=.06, right=.99, bottom=.17, top=.9)
    chain = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, c]) for c in (1, 2, 3)]
    for ax, cell in zip(axes, FAMILIES):
        onset_panel(ax, cell, record)
    fc.hop_chain(chain, H)  # the reference nodes, as in Fig. 4
    fc.row_label(fig, chain)
    x0, x1 = axes[0].get_position().x0, axes[2].get_position().x1
    fig.text((x0 + x1) / 2, .015, "First decodable iteration", ha="center", va="bottom", fontsize=8)
    handles = [Line2D([], [], color=".6", lw=.6, label="One hop per iteration")]
    handles += [Line2D([], [], color=SEED_PALETTE[sd], lw=.6, marker=SEED_MARKERS[sd], ms=3, mfc="white", mew=.6,
                       label=f"Seed {sd}") for sd in ONSET_SEEDS]
    # (a)'s empty lower right, ending at the never column's separator; opaque white box over the iteration lines.
    sep = (16 * 1.11 - axes[0].get_xlim()[0]) / (axes[0].get_xlim()[1] - axes[0].get_xlim()[0])
    leg = axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(sep - .01, .015), fontsize=5,
                         frameon=True, fancybox=False, framealpha=1, edgecolor="black", handlelength=1.2,
                         labelspacing=.2, handletextpad=.4, borderpad=.35, borderaxespad=0)
    leg.get_frame().set_linewidth(.4)
    fc.save(fig, "fig5_dynamics", out_dir)
    for cell, runs in record.items():
        for seed, r in runs.items():
            print(cell, seed, json.dumps(r))


if __name__ == "__main__":
    main(fc.out_dir_arg(__doc__))
