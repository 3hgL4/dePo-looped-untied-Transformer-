"""Figure A2 (every-seed onsets, with its table) and Figure A3 (forward-walk probes).

Figure A2 (H=4): the first decodable layer (onset rule in figcommon.py, start node excluded) of hops 1-3 and
the answer, for every probed seed of the Fig. 4 conditions plus the looped model at the lower rate:
  (a) looped 4x8, lr 1e-3, seeds 42-46: results/clean_sweep_5seed_h4 (row split of the query rows, probed
      after each of the 8 iterations, so level t is drawn at layer 4t), plus the prompt-disjoint seed-42
      re-probe (results/geometry_lr1e3_20260920, 33 layer levels, filled markers);
  (b) untied 32-layer, lr 1e-3, seeds 42-46: results/clean_sweep_5seed_h4 (row split, 33 levels);
  (c) untied 32-layer, lr 3e-4, seeds 42-44: results/matched_lr3e4_20260915 (prompt-disjoint, 33 levels);
  (d) untied 32-layer with the auxiliary loss at layers 8, 16, 24, lr 1e-3, seeds 44 and 45: results/aux_rerun_h4
      (row split);
  (e) looped 4x8, lr 3e-4, seeds 42-44: results/matched_lr3e4_20260915 (prompt-disjoint, 33 levels).
Hops never decodable are crosses in a "never" column. Every probed seed is shown; none is filtered.

Accuracy in the table is full-answer accuracy on the first query: from the prompt-disjoint probe reports
(complete_answer.strata.first_query) and from results/complete_answer_20261009/<run>_H4.json
(eval_complete_answer.py, the same evaluation set) for the row-split checkpoints. The table rows are
written to figA2_every_seed_onsets.json (tables/every_seed_table.py prints Table A1 from conditions()).

Figure A3: peak linear-probe selectivity for each forward node pi^d(s), d = 0..15, looped 2x16, seed 42,
lr 1e-3, H=12 (row-split probes after every iteration, results/rebuttal/dirprobe_o2x16_h12).

    uv run python experiments/h2_probing/scripts/build_appendix_probe_figures.py [--out_dir figures/build]

Writes figA2_every_seed_onsets.{pdf,png,json} and figA3_directional.{pdf,png}.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcommon as fc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Patch
import numpy as np
from src.figstyle import ACCENT_RED, SEED_PALETTE, disable_categorical_minor_ticks

COLORS = fc.COLORS  # looped red (ouro-4x8), untied grey (deep-32)
SEED_MARKERS = {42: "o", 43: "s", 44: "^", 45: "v", 46: "D"}
SEED_OFFSET, REPROBE_OFFSET = .1, -.3  # vertical offsets in hops: seed s at SEED_OFFSET * (s - 44), the re-probe below
TEXTWIDTH = 5.5  # text width in inches
NEVER = 47       # x position of the "never decodable" column
AUX_LAYERS = (8, 16, 24)
REPROBE = "prompt-disjoint re-probe"


def row_split(sub, run, levels):
    """Selectivity of a row-split probe run (real minus shuffled-label linear-probe accuracy)."""
    d = fc.res(sub, run)
    arr = fc.load_npz(d / "probe_accuracies.npz")
    info = fc.load_json(d / "probe_results.json")
    assert info["H_eval"] == 4 and arr["linear_real"].shape == (levels, 5)
    assert info["args"]["train_frac"] == .8 and not info["args"].get("split_by_prompt")
    return fc.selectivity(arr)


def full_answer(run):
    return fc.complete_answer(run)["first_query_accuracy"]


def conditions():
    """(key, title, family, lr, entries); an entry is (seed, onset layers of hops 1-4, accuracy, protocol, block)."""
    data, geo = fc.matched(), fc.geometry("ouro-4x8_s42")
    looped_1e3 = []
    for seed in (42, 43, 44, 45, 46):
        ons = fc.onsets(row_split("clean_sweep_5seed_h4", f"ouro-4x8_s{seed}", 9))[1:]
        looped_1e3.append((seed, [None if t is None else 4 * t for t in ons], full_answer(f"ouro-4x8_s{seed}"),
                           "row split, per iteration", 4))
    g = geo[4]
    assert g["learning_rate"] == 1e-3 and g["split_counts"]["n_shared_prompts"] == 0
    looped_1e3.append((42, fc.onsets(np.asarray(g["selectivity"]))[1:], fc.first_query_accuracy(g), REPROBE, 4))
    untied_1e3 = [(seed, fc.onsets(row_split("clean_sweep_5seed_h4", f"deep-32_s{seed}", 33))[1:],
                   full_answer(f"deep-32_s{seed}"), "row split", None) for seed in (42, 43, 44, 45, 46)]
    p1 = lambda fam, seed: data[(fam, seed, 4)]
    untied_3e4 = [(s, fc.onsets(np.asarray(p1("deep-32", s)["selectivity"]))[1:], fc.first_query_accuracy(p1("deep-32", s)),
                   "prompt-disjoint", None) for s in fc.SEEDS]
    aux = [(s, fc.onsets(row_split("aux_rerun_h4", f"deep-32-aux_s{s}", 33))[1:], full_answer(f"deep-32-aux_s{s}"),
            "row split", None) for s in (44, 45)]
    looped_3e4 = [(s, fc.onsets(np.asarray(p1("ouro-4x8", s)["selectivity"]))[1:], fc.first_query_accuracy(p1("ouro-4x8", s)),
                   "prompt-disjoint", 4) for s in fc.SEEDS]
    for d in [p1(f, s) for f in fc.FAMILIES for s in fc.SEEDS]:
        assert d["split_counts"]["n_shared_prompts"] == 0 and d["H_eval"] == 4
    return [("a", "(a) Looped\n$\\eta=10^{-3}$", "ouro-4x8", "1e-3", looped_1e3),
            ("b", "(b) Untied\n$\\eta=10^{-3}$", "deep-32", "1e-3", untied_1e3),
            ("c", "(c) Untied\n$\\eta=3{\\times}10^{-4}$", "deep-32", "3e-4", untied_3e4),
            ("d", "(d) Untied + aux. loss\n$\\eta=10^{-3}$", "deep-32", "1e-3", aux),
            ("e", "(e) Looped\n$\\eta=3{\\times}10^{-4}$", "ouro-4x8", "3e-4", looped_3e4)]


def table_rows(conds=None):
    """One row per (condition, seed): the content of Table A1."""
    rows = []
    for key, _, fam, lr, entries in conds or conditions():
        for seed, ons, acc, protocol, block in entries:
            rows.append({"panel": key, "family": fam, "lr": lr, "seed": seed, "protocol": protocol,
                         "onset_layers_hops_1_3_answer": ons,
                         "onset_iterations": None if block is None else
                         [None if o is None else math.ceil(o / block) for o in ons],
                         "full_answer_first_query": acc})
    return rows


def text_sizes(fig, label=7.5, tick=6.5):
    """Printed sizes for a figure built at its include width."""
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(label)
        ax.yaxis.label.set_fontsize(label)
        ax.tick_params(which="major", labelsize=tick, width=.5, length=2.5)
        ax.tick_params(which="minor", width=.4, length=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(.55)


def framed_legend(owner, handles, **kw):
    """Thin black square border, opaque white."""
    leg = owner.legend(handles=handles, fontsize=6, frameon=True, fancybox=False, framealpha=1, edgecolor="black",
                       borderpad=.35, **kw)
    leg.get_frame().set_linewidth(.4)
    return leg


def fig_every_seed(out_dir):
    """Seeds in SEED_PALETTE colours with their own markers, hollow; the prompt-disjoint seed-42 re-probe in (a)
    filled. Seed s is drawn SEED_OFFSET * (s - 44) hops off its row, the re-probe REPROBE_OFFSET hops."""
    conds = conditions()
    fig = plt.figure(figsize=(TEXTWIDTH, 1.95))
    gs = fig.add_gridspec(1, 5, wspace=.12, left=.085, right=.99, bottom=.3, top=.8)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    for ax, (key, title, fam, lr, entries) in zip(axes, conds):
        for seed, ons, acc, protocol, block in entries:
            reprobe = protocol == REPROBE
            color = SEED_PALETTE[seed]
            y = np.arange(1, 5) + (REPROBE_OFFSET if reprobe else SEED_OFFSET * (seed - 44))
            ax.plot([np.nan if o is None else o for o in ons], y, color=color, lw=.6, marker=SEED_MARKERS[seed],
                    ms=2.6, mfc=color if reprobe else "white", mew=.55, zorder=3 if reprobe else 2)
            miss = [yy for yy, o in zip(y, ons) if o is None]
            ax.plot([NEVER] * len(miss), miss, "x", color=color, ms=2.8, mew=.6)
        if key == "d":
            for layer in AUX_LAYERS:
                ax.axvline(layer, color=".6", lw=.45, zorder=0)
        if fam == "ouro-4x8":
            ax.plot([0, 20], [0, 5], color=".7", lw=.5, zorder=0)  # one hop per iteration
        ax.text(.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, linespacing=1.1)
        ax.set_xlim(-2, 51)
        ax.set_xticks([0, 8, 16, 24, 32, NEVER], ["0", "", "16", "", "32", "never"])
        ax.set_ylim(.55, 4.45)
        ax.set_yticks([1, 2, 3, 4], ["1", "2", "3", "answer"])
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        if key != "a":
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Reference hop", labelpad=2)
    text_sizes(fig)
    fig.text((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2, .175, "First decodable layer",
             ha="center", va="center", fontsize=7.5)
    handles = [Line2D([], [], color=SEED_PALETTE[s], lw=.6, marker=SEED_MARKERS[s], ms=2.6, mfc="white", mew=.55,
                      label=f"Seed {s}") for s in (42, 43, 44, 45, 46)]
    handles += [Line2D([], [], color=SEED_PALETTE[42], lw=.6, marker=SEED_MARKERS[42], ms=2.6, mew=.55,
                       label="Seed 42, prompt-disjoint re-probe")]
    framed_legend(fig, handles, loc="lower center", ncol=6, handlelength=1.8, columnspacing=1.2,
                  bbox_to_anchor=(.53, .02))
    fc.save_raw(fig, "figA2_every_seed_onsets", out_dir)
    table = table_rows(conds)
    (Path(out_dir) / "figA2_every_seed_onsets.json").write_text(json.dumps(table, indent=1) + "\n")
    for row in table:
        print(row["panel"], row["family"], row["lr"], row["seed"], row["protocol"], row["onset_layers_hops_1_3_answer"],
              row["onset_iterations"], row["full_answer_first_query"])


def fig_directional(out_dir):
    """Peak linear-probe selectivity of each forward node pi^d(s), d = 0..15. d = 1..H is the forward walk,
    d = H+1..15 the backward route of length N - H = 4."""
    s = fc.load_json(fc.res("rebuttal/dirprobe_o2x16_h12/directional_probe_summary.json"))
    assert s["config_name"] == "ouro-2x16" and s["H_eval"] == 12 and s["N"] == 16
    assert s["checkpoint"] == "experiments/h1_clean_sweep/runs/ouro-2x16_s42/checkpoint.pt"
    d, peak = np.array(s["forward_distance_d"]), np.array(s["peak_selectivity_per_distance"])
    H = s["forward_walk_length"]
    assert H == 12 and s["backward_shortcut_length"] == 4 and d.tolist() == list(range(16))
    looped, untied = COLORS["ouro-4x8"], COLORS["deep-32"]
    colors = [".85" if x == 0 else looped if x <= H else untied for x in d]
    fig, ax = plt.subplots(figsize=(.6 * TEXTWIDTH, 1.9))
    ax.bar(d, peak, color=colors, width=.75, edgecolor="black", linewidth=.4, zorder=2)
    ax.axhline(.5, color=ACCENT_RED, lw=.6, ls="--", zorder=1)
    ax.axvline(H + .5, color="black", lw=.5, ls=":", zorder=1)
    ax.set_xlim(-.7, 15.7)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 16))
    disable_categorical_minor_ticks(ax)
    ax.set_xlabel("Forward distance from the start node")
    ax.set_ylabel("Peak probe selectivity", labelpad=2)
    text_sizes(fig)
    framed_legend(ax, [Patch(facecolor=looped, edgecolor="black", lw=.4, label=f"Forward walk (hops 1 to {H})"),
                       Patch(facecolor=untied, edgecolor="black", lw=.4, label="Backward route"),
                       Patch(facecolor=".85", edgecolor="black", lw=.4, label="Start node")],
                  loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, handlelength=1.0, handleheight=.8,
                  columnspacing=.8, borderaxespad=0)
    fc.save_raw(fig, "figA3_directional", out_dir)
    print("directional peak selectivity by d:", [round(float(v), 3) for v in peak])
    print("peak level by d:", s["peak_level_per_distance"])


def main(out_dir=fc.DEFAULT_OUT):
    fc.style()
    fig_every_seed(out_dir)
    fig_directional(out_dir)


if __name__ == "__main__":
    main(fc.out_dir_arg(__doc__))
