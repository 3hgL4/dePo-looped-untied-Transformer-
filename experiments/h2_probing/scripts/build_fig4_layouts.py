"""Figure 4: probe-selectivity maps at H=4, seed 42 unless stated, in a 2x2 grid.

  (a) looped 4x8, lr 1e-3: prompt-disjoint re-probe (results/geometry_lr1e3_20260920);
  (b) untied 32-layer, lr 1e-3, which does not solve the task: row-split probes (results/clean_sweep_5seed_h4);
  (c) untied 32-layer, lr 3e-4: prompt-disjoint probes (results/matched_lr3e4_20260915);
  (d) untied 32-layer with the auxiliary loss on layers 8, 16 and 24, lr 1e-3, seed 45: row-split probes
      (results/aux_rerun_h4); the dashed lines mark the supervised layers.
Every panel reports full-answer accuracy on first queries: (a) and (c) from their probe reports, (b) and (d)
from results/complete_answer_20261009 (eval_complete_answer.py, the same evaluation set).

    uv run python experiments/h2_probing/scripts/build_fig4_layouts.py [--out_dir figures/build]

Writes fig4_layouts.{pdf,png}.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcommon as fc

import matplotlib.pyplot as plt

AUX_MAP_SEED = 45  # seed 44, the other solving seed, answers 0.855 of first queries
AUX_LAYERS = (8, 16, 24)  # the trainer's --aux_layers (1-indexed layer outputs)
TEXTWIDTH = 5.5  # text width in inches
TITLES = ("(a) Looped", "(b) Untied", "(c) Untied + lower LR", "(d) Untied + aux. loss")


def row_split_selectivity(sub, run):
    """Selectivity (33 levels x 5 targets) of a row-split H=4 probe run of an untied model."""
    d = fc.res(sub, run)
    arr = fc.load_npz(d / "probe_accuracies.npz")
    info = fc.load_json(d / "probe_results.json")
    assert info["H_eval"] == 4 and arr["linear_real"].shape == (33, 5)
    return fc.selectivity(arr)


def panels():
    """(selectivity, full-answer first-query accuracy, block size) of panels (a) to (d)."""
    g = fc.geometry("ouro-4x8_s42")[4]
    p1 = fc.matched()[("deep-32", 42, 4)]
    assert g["learning_rate"] == 1e-3
    return [(g["selectivity"], fc.first_query_accuracy(g), 4),
            (row_split_selectivity("clean_sweep_5seed_h4", "deep-32_s42"),
             fc.complete_answer("deep-32_s42")["first_query_accuracy"], None),
            (p1["selectivity"], fc.first_query_accuracy(p1), None),
            (row_split_selectivity("aux_rerun_h4", f"deep-32-aux_s{AUX_MAP_SEED}"),
             fc.complete_answer(f"deep-32-aux_s{AUX_MAP_SEED}")["first_query_accuracy"], None)]


def main(out_dir=fc.DEFAULT_OUT):
    fc.style()
    content = panels()
    fig = plt.figure(figsize=(.8 * TEXTWIDTH, 3.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[.3, 1, 1, .05], wspace=.1, hspace=.42,
                          left=.07, right=.9, bottom=.12, top=.93)
    chains = [fig.add_subplot(gs[r, 0]) for r in (0, 1)]
    maps = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (1, 2)]
    cax = fig.add_subplot(gs[:, 3])
    for chain in chains:
        fc.hop_chain(chain, 4)
        fc.row_label(fig, chain)
    for ax, (sel, acc, block), title in zip(maps, content, TITLES):
        im = fc.selectivity_map(ax, sel, 4, acc, block)
        # Titles as text: fc.save resets axes titles to 9 pt, too wide here.
        ax.text(.5, 1.035, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
    for layer in AUX_LAYERS:  # (d): the states the auxiliary loss reads, under the accuracy box
        maps[3].axvline(layer, color=".25", lw=.7, ls=(0, (2.2, 1.6)), zorder=.9)
    fig.text((maps[2].get_position().x0 + maps[3].get_position().x1) / 2, .02, "Completed layer evaluations",
             ha="center", fontsize=8)
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.set_label("Probe selectivity", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6)
    fc.save(fig, "fig4_layouts", out_dir)
    print(json.dumps({t: round(float(a), 4) for t, (_, a, _) in zip(TITLES, content)}))


if __name__ == "__main__":
    main(fc.out_dir_arg(__doc__))
