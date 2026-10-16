"""Figure 6 (logit lens and patching) and Figure A6 (patching, every seed), from shipped summaries.

Figure 6
(a) Full-node logit lens, no slack (looped 4x8, H=8, lr 1e-3, seed 42): mean probability over 5,000 prompts
    that the model writes each reference node (digits plus delimiter) when its loop is stopped after t
    iterations (src/full_node_lens.py, results/full_node_lens_20261010). Rows start to answer.
(b) Same-permutation patching of the looped 4x8 model at H=6, seeds 42-44: share of answers that are the
    donor's (solid) and the receiver's own (dashed) when the donor's states at the readout position and the
    first answer digit replace the receiver's after iteration j (results/patching_fullanswer_digit_20261010).
    Coincident seeds are offset slightly.
(c) The patching of (b), illustrated (plot_patching_schematic.py).

Figure A6
(a) Same permutation, the donor's answer, seeds 42-44 and an untrained model.
(b) Different permutations: full-answer match to the node composed from both walks, on queries where it
    differs from both single-permutation answers; chance 1/16. Composition is defined only while the
    receiver still has hops left (patched after iterations 1 to H - 1 = 5).

    uv run python experiments/h2_probing/scripts/build_fig6_use.py [--out_dir figures/build]

Writes fig6_use.{pdf,png} and figA6_patching_every_seed.{pdf,png}.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcommon as fc
import plot_patching_schematic as schematic

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np
from src.figstyle import ACCENT_RED, SELECTIVITY_CMAP, SEED_PALETTE

TEXTWIDTH = 5.5
H = 8
SEED_MARKERS = {42: "o", 43: "s", 44: "^", 46: "D"}  # as in build_fig5_dynamics.py
PATCH_SEEDS = (42, 43, 44)  # the 4x8 seeds Fig. 5 draws
LENS_SEED = 42
OFFSET = {42: (0, 0.), 43: (-.06, .02), 44: (.06, -.02)}  # (x, y) offsets so coincident seeds stay visible
APPX_W = 4.75  # appendix figure width (in), printed at 0.75 of the text width


def lens_summary(cell, seed, h=H):
    s = fc.load_json(fc.res("full_node_lens_20261010", f"{cell}_s{seed}_H{h}", "full_node_lens_summary.json"))
    assert s["H_eval"] == h and s["config_name"] == cell and not s["random_init"] and s["n_samples"] == 5000
    return s


def patch_summary(seed, mode, randinit=False):
    p = fc.res("patching_fullanswer_digit_20261010", f"o4x8_s{seed}_{mode}{'_randinit' if randinit else ''}",
               "intervention_summary.json")
    s = fc.load_json(p)
    assert s["patch_answer_digit"] and s["H"] == 6 and s["mode"] == mode and s["random_init"] == randinit
    return s["full_answer"]["all"]


def panel_title(ax, text):
    ax.text(.5, 1.04, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def legend(ax, handles, **kw):
    leg = ax.legend(handles=handles, fontsize=5, frameon=True, fancybox=False, framealpha=1, edgecolor="black",
                    handlelength=1.8, labelspacing=.2, borderpad=.35, **kw)
    leg.get_frame().set_linewidth(.4)


def lens_panel(ax, cax, record):
    m = np.array(lens_summary("ouro-4x8", LENS_SEED)["lens"])  # (H+1, T)
    record["lens_seed"] = LENS_SEED
    record["lens_mean"] = m.round(3).tolist()
    im = ax.imshow(m, origin="lower", aspect="auto", cmap=SELECTIVITY_CMAP, vmin=0, vmax=1,
                   extent=(.5, m.shape[1] + .5, -.5, H + .5), interpolation="nearest")
    ax.set_xticks(range(1, m.shape[1] + 1))
    ax.set_yticks(range(H + 1))
    ax.tick_params(labelleft=False)
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Loop stopped after iteration")
    panel_title(ax, "(a) Logit lens, no slack")
    cb = plt.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.set_label("Probability of writing the node", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6)


def main_patch_panel(ax, record):
    """Fig. 6(b): the donor's answer (solid) and the receiver's own answer (dashed); the rest are neither's."""
    for seed in PATCH_SEEDS:
        s = patch_summary(seed, "redirect")
        dx, dy = OFFSET[seed]
        for key, ls in (("match_pure_A", "-"), ("match_pure_B", "--")):
            y = np.array(s[key], dtype=float)
            ax.plot(np.arange(1, len(y) + 1) + dx, y + dy, color=SEED_PALETTE[seed], lw=.7, ls=ls,
                    marker=SEED_MARKERS[seed], ms=2.8, mfc="white", mew=.6)
        record.setdefault("fig6b", {})[seed] = {"donor": s["match_pure_A"], "receiver": s["match_pure_B"]}
    ax.set_ylim(-.04, 1.06)
    ax.set_xlim(.6, 7.4)
    ax.set_xticks(range(1, 8))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Patched after iteration")
    ax.set_ylabel("Share of answers", labelpad=2)
    panel_title(ax, "(b) Patching, same permutation")
    legend(ax, [Line2D([], [], color=SEED_PALETTE[s], lw=.7, marker=SEED_MARKERS[s], ms=2.8, mfc="white", mew=.6,
                       label=f"Seed {s}") for s in PATCH_SEEDS]
               + [Line2D([], [], color="black", lw=.7, ls=ls, label=lab)
                  for ls, lab in (("-", "Donor's answer"), ("--", "Receiver's answer"))],
           loc="center right", bbox_to_anchor=(.98, .5))


def patch_panel(ax, mode, record):
    """Fig. A6 panel. Seeds take OFFSET in (a), where 43 and 44 coincide at 1.0, and only its horizontal part
    in (b), where the values sit near chance. The untrained control is 0 everywhere and is drawn above the
    seed lines in (b) so the trained points near 0 cannot hide it."""
    key = "match_pure_A" if mode == "redirect" else "match_composed_distinct"
    record[mode] = {}
    if mode == "compose":
        ax.axhline(1 / 16, color=ACCENT_RED, lw=.6, ls="--")
    for seed in PATCH_SEEDS:
        s = patch_summary(seed, mode)
        y = np.array([np.nan if v is None else v for v in s[key]], dtype=float)
        x = np.arange(1, len(y) + 1)  # patch after iteration j + 1
        dx, dy = OFFSET[seed] if mode == "redirect" else (OFFSET[seed][0], 0.)
        color, marker = SEED_PALETTE[seed], SEED_MARKERS[seed]
        ax.plot(x + dx, y + dy, color=color, lw=.6, marker=marker, ms=2.6, mfc="white", mew=.6)
        record[mode][seed] = {key: s[key], "match_pure_B": s["match_pure_B"]}
    r = patch_summary(42, mode, randinit=True)
    yr = np.array([np.nan if v is None else v for v in r[key]], dtype=float)
    ax.plot(np.arange(1, len(yr) + 1), yr, color=".55", lw=.8, ls="-", zorder=1.5 if mode == "redirect" else 3)
    record[mode]["random_init"] = r[key]
    ax.set_ylim(-.05, 1.05)
    xmax = int(np.flatnonzero(~np.isnan(yr)).max()) + 1  # last patch iteration with data: 7 in (a), 5 in (b)
    ax.set_xlim(.6, xmax + .4)
    ax.set_xticks(range(1, xmax + 1))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Patched after iteration")
    if mode == "redirect":
        ax.set_ylabel("Writes the donor's answer", labelpad=2)
        panel_title(ax, "(a) Same permutation")
    else:
        ax.set_ylabel("Writes the composed node", labelpad=2)
        panel_title(ax, "(b) Different permutations")
        ax.text(xmax + .3, 1 / 16 + .025, "chance", color=ACCENT_RED, fontsize=6, ha="right", va="bottom")


def figure_6(out_dir, record):
    fig = plt.figure(figsize=(TEXTWIDTH, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[.3, 1, .05, .45, 1.1], wspace=.08,
                          left=.07, right=.985, bottom=.585, top=.94)  # column 3 is a spacer for the colour bar label
    sch = fig.add_axes([0, .0, 1, 2.1 / 4.4])  # (c) as a strip below (a) and (b)
    schematic.draw(sch)
    fig.text(.5, 2.1 / 4.4, "(c) Patching, illustrated", ha="center", va="bottom", fontsize=8)
    chain = fig.add_subplot(gs[0, 0])
    lens_ax, cax = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    redirect_ax = fig.add_subplot(gs[0, 4])
    lens_panel(lens_ax, cax, record)
    fc.hop_chain(chain, H)
    fc.row_label(fig, chain)
    main_patch_panel(redirect_ax, record)
    fc.save(fig, "fig6_use", out_dir)


def figure_a6(out_dir, record):
    # Drawn at APPX_W and included at 0.75 of the 5.5 in text width, so fc.save's 8 pt labels and 7 pt
    # ticks print at about 7.3 pt and 6.4 pt.
    fig, (a, b) = plt.subplots(1, 2, figsize=(APPX_W, 2.25), gridspec_kw={"wspace": .36})
    fig.subplots_adjust(left=.1, right=.98, bottom=.2, top=.87)
    patch_panel(a, "redirect", record)
    patch_panel(b, "compose", record)
    legend(a, [Line2D([], [], color=SEED_PALETTE[s], lw=.6, marker=SEED_MARKERS[s], ms=2.6, mfc="white", mew=.6,
                      label=f"Seed {s}") for s in PATCH_SEEDS] + [Line2D([], [], color=".55", lw=.8, label="Untrained")],
           loc="upper left", bbox_to_anchor=(.25, .9))
    for t in a.get_legend().get_texts():
        t.set_fontsize(6.2)
    fc.save(fig, "figA6_patching_every_seed", out_dir)


def main(out_dir=fc.DEFAULT_OUT):
    fc.style()
    record = {}
    figure_6(out_dir, record)
    figure_a6(out_dir, record)
    print(json.dumps({k: v for k, v in record.items() if k != "lens_mean"}))


if __name__ == "__main__":
    main(fc.out_dir_arg(__doc__))
