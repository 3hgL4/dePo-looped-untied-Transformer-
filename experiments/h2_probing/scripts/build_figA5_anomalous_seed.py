"""Figure A5: the looped 2x16 seed with weak middle hops, probes and logit lens together.

(a) Peak probe selectivity per hop at H=4 for seeds 42, 44 and 46 of the looped 2x16 model at lr 1e-3
    (row-split probes after each iteration, results/rebuttal/probe_o2x16_s*).
(b), (c) Full-node logit lens at H=4 (src/full_node_lens.py, results/full_node_lens_20261010): probability
    that the model writes each reference node when its loop is stopped after t iterations, for seed 44 (b)
    and seed 42 (c).

    uv run python experiments/h2_probing/scripts/build_figA5_anomalous_seed.py [--out_dir figures/build]

Writes figA5_anomalous_seed.{pdf,png} and prints the plotted values.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcommon as fc

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
from src.figstyle import SELECTIVITY_CMAP, SEED_PALETTE, disable_categorical_minor_ticks

SEEDS = (42, 44, 46)
H = 4
ROWS = ["start", "1", "2", "3", "ans."]
FIG_W = 6.35  # drawn width (in); included at the 5.5 in text width, fc.save's 8/7 pt text prints at ~7.3/6.4 pt


def probe_peaks(seed):
    sel = fc.selectivity(fc.load_npz(fc.res("rebuttal", f"probe_o2x16_s{seed}", "probe_accuracies.npz")))
    assert sel.shape[1] == H + 1
    return sel.max(0)


def lens(seed):
    s = fc.load_json(fc.res("full_node_lens_20261010", f"ouro-2x16_s{seed}_H{H}", "full_node_lens_summary.json"))
    assert s["H_eval"] == H and s["config_name"] == "ouro-2x16" and not s["random_init"]
    return np.array(s["lens"]), s["answer_accuracy"][-1]


def main(out_dir=fc.DEFAULT_OUT):
    fc.style()
    record = {}
    fig = plt.figure(figsize=(FIG_W, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=.27, left=.075, right=.905, bottom=.2, top=.82)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    pos = c.get_position()
    cax = fig.add_axes([pos.x1 + .012, pos.y0, .012, pos.height])  # colour bar close to (c)
    w = .26
    hops = np.arange(1, H + 1)
    for i, seed in enumerate(SEEDS):
        p = probe_peaks(seed)
        record[seed] = {"probe_peak": [round(float(v), 3) for v in p]}
        a.bar(hops + (i - 1) * w, p[1:], width=w, color=SEED_PALETTE[seed], edgecolor="black", linewidth=.4,
              label=f"Seed {seed}")
    a.axhline(.5, color=".5", lw=.5, ls=(0, (2, 2)))  # the selectivity threshold that defines a probe onset
    a.set_xticks(hops, ["1", "2", "3", "ans."])
    disable_categorical_minor_ticks(a)
    a.set_ylim(0, 1.05)
    a.set_xlabel("Hop")
    a.set_ylabel("Peak probe selectivity", labelpad=2)
    # Legend in one row between the panel and its title, clear of the bars.
    a.legend(fontsize=6.4, loc="lower center", bbox_to_anchor=(.5, 1.008), ncol=3, frameon=False,
             handlelength=.8, handletextpad=.3, columnspacing=.6, borderaxespad=0, borderpad=0)
    title_y = 1.1  # every panel title at the same height, above (a)'s legend
    a.text(.5, title_y, "(a) Probes", transform=a.transAxes, ha="center", va="bottom", fontsize=8)
    for ax, seed, title in ((b, 44, "(b) Logit lens, weak-middle seed"), (c, 42, "(c) Logit lens, typical seed")):
        m, acc = lens(seed)
        record[seed].update({"lens": m.round(3).tolist(), "answer_accuracy_t_T": acc})
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=SELECTIVITY_CMAP, vmin=0, vmax=1,
                       extent=(.5, m.shape[1] + .5, -.5, H + .5), interpolation="nearest")
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.set_yticks(range(H + 1), ROWS if ax is b else [""] * (H + 1))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel("Loop stopped after iteration")
        ax.text(.5, title_y, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    cb = plt.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.set_label("Probability of writing the node", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=6)
    fc.save(fig, "figA5_anomalous_seed", out_dir)
    print(json.dumps(record))


if __name__ == "__main__":
    main(fc.out_dir_arg(__doc__))
