"""Figure 1: probe-selectivity maps at H=8 of the looped 4x8 and untied 32-layer models (lr 3e-4, seed 42).

Prompt-disjoint probes after every layer, from results/matched_lr3e4_20260915/validation_production.json.
Each map carries the first-crossing frontier (onset rule in figcommon.py), a cross for a hop that never
reaches selectivity 0.5, and the model's full-answer accuracy on first queries.

    uv run python experiments/h2_probing/scripts/build_fig1_teaser.py [--out_dir figures/build]

Writes fig1_teaser.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcommon as fc

import matplotlib.pyplot as plt

TITLES = {"ouro-4x8": "Looped", "deep-32": "Untied"}  # the caption gives the architectures
NOTES = {"ouro-4x8": (.05, .67, "left", "one new hop readable\nper iteration"),
         "deep-32": (.40, .58, "center", "hops readable only\nin the last layers")}


def main(out_dir=fc.DEFAULT_OUT):
    fc.style()
    data = fc.matched()
    fig = plt.figure(figsize=(6.75, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[.34, 1, 1, .045], wspace=.1,
                          left=.055, right=.94, bottom=.2, top=.88)
    chain = fig.add_subplot(gs[0, 0])
    maps = [fig.add_subplot(gs[0, c]) for c in (1, 2)]
    cax = fig.add_subplot(gs[0, 3])
    fc.hop_chain(chain, 8)
    for ax, family in zip(maps, fc.FAMILIES):
        d = data[(family, 42, 8)]
        im = fc.selectivity_map(ax, d["selectivity"], 8, fc.first_query_accuracy(d), fc.BLOCK.get(family),
                                show_missing=True)
        ax.set_title(TITLES[family], pad=4)
        x, y, ha, text = NOTES[family]
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="center", fontsize=7, linespacing=1.3,
                style="italic")
        ax.set_xlabel("Completed layer evaluations")
    fc.row_label(fig, chain)
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.set_label("Probe selectivity", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6)
    fc.save(fig, "fig1_teaser", out_dir)


if __name__ == "__main__":
    main(fc.out_dir_arg(__doc__))
