"""Shared pieces of the probe figures: input loading, the probe-onset rule, the heat-map frontier,
the hop chain beside the maps, and the save step.

Every figure reads only shipped summaries under experiments/h2_probing/results/ (no model
execution). A missing input raises FileNotFoundError naming the file; nothing is skipped.

Onset rule everywhere: the first state level (0 = embedding, l = after layer l, or after
iteration l for the per-iteration probes) at which probe selectivity, real minus shuffled-label
linear-probe accuracy, reaches 0.5.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
from src.figstyle import apply_style, PALETTE_L, MARKERS_L, SELECTIVITY_CMAP  # noqa: F401

# Every input path is resolved against DATA at call time, so a test can point it at an empty tree.
DATA = ROOT
DEFAULT_OUT = ROOT / "figures/build"

# Colours and markers are keyed by loop count: ouro-4x8 is four layers looped eight times (L=8),
# ouro-2x16 two layers looped sixteen times (L=16), deep-32 is unlooped (L=1).
COLORS = {"ouro-4x8": PALETTE_L[8], "ouro-2x16": PALETTE_L[16], "deep-32": PALETTE_L[1]}
MARKERS = {"ouro-4x8": MARKERS_L[8], "ouro-2x16": MARKERS_L[16], "deep-32": MARKERS_L[1]}
BLOCK = {"ouro-4x8": 4, "ouro-2x16": 2}  # layers per shared block, for the boundary lines
FAMILIES = ("ouro-4x8", "deep-32")
SEEDS = (42, 43, 44)  # seeds of the lr 3e-4 prompt-disjoint probes

FULL_ANSWER = "teacher_forced_answer_tokens_plus_delimiter_v1"  # protocol tag of full-answer scoring


def res(*parts) -> Path:
    """Path under experiments/h2_probing/results of the current DATA root."""
    return DATA.joinpath("experiments/h2_probing/results", *parts)


def need(path: Path) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(f"missing input: {path} (see data/MANIFEST.sha256 for the shipped files)")
    return Path(path)


def load_json(path: Path):
    return json.loads(need(path).read_text())


def load_npz(path: Path):
    return np.load(need(path))


def selectivity(arr) -> np.ndarray:
    """Probe selectivity: real minus shuffled-label linear-probe accuracy."""
    return arr["linear_real"] - arr["linear_shuffled"]


def matched_report() -> Path:
    """Prompt-disjoint probes of looped 4x8 and untied 32-layer at lr 3e-4, seeds 42-44, H=4 and H=8."""
    return res("matched_lr3e4_20260915/validation_production.json")


def geometry_report() -> Path:
    """Prompt-disjoint re-probes of looped 4x8 and 2x16, seed 42, lr 1e-3, H=4 and H=8."""
    return res("geometry_lr1e3_20260920/validation.json")


def matched():
    r = load_json(matched_report())
    assert r["status"] == "passed" and not r["errors"]
    return {(d["family"], d["seed"], d["H_eval"]): d for d in r["conditions"]}


def geometry(run="ouro-2x16_s42"):
    r = load_json(geometry_report())
    assert r["status"] == "passed" and not r["errors"]
    out = {d["H_eval"]: d for d in r["conditions"] if d["run"] == run}
    if not out:
        raise FileNotFoundError(f"missing input: no {run} conditions in {geometry_report()}")
    return out


def first_query_accuracy(condition) -> float:
    """Full-answer accuracy on the first query, from a validation-report condition."""
    return condition["complete_answer"]["strata"]["first_query"]["accuracy"]


def complete_answer(run, H=4):
    """Full-answer result of eval_complete_answer.py for a row-split checkpoint (first-query accuracy,
    and for looped runs the accuracy with the loop stopped after each iteration)."""
    r = load_json(res("complete_answer_20261009", f"{run}_H{H}.json"))
    assert r["protocol"] == FULL_ANSWER and r["H_eval"] == H
    return r


def style():
    apply_style()


def save(fig, stem, out_dir):
    """Write <stem>.pdf and .png, scaling text on these manuscript-sized axes without changing the shared rcParams."""
    for ax in fig.axes:
        is_colorbar = ax.get_position().width < .03 or ax.get_label() == "<colorbar>"
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(7 if is_colorbar else 8)
        ax.yaxis.label.set_fontsize(7 if is_colorbar else 8)
        ax.xaxis.offsetText.set_fontsize(6)
        ax.yaxis.offsetText.set_fontsize(6)
        ax.tick_params(which="major", labelsize=6 if is_colorbar else 7, width=.5, length=2.5)
        ax.tick_params(which="minor", width=.4, length=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(.55)
    for legend in fig.legends:
        for label in legend.get_texts():
            label.set_fontsize(7)
    save_raw(fig, stem, out_dir)


def save_raw(fig, stem, out_dir):
    """Write <stem>.pdf and .png with the text sizes the figure already has."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=.025, dpi=240)
    plt.close(fig)
    print("wrote", out_dir / f"{stem}.pdf")


def out_dir_arg(doc: str) -> Path:
    p = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args().out_dir


def onsets(matrix):
    """First level at which each target reaches selectivity 0.5 (None if it never does)."""
    return [next((l for l, vals in enumerate(matrix) if vals[h] >= .5), None)
            for h in range(len(matrix[0]))]


def frontier(ax, ons):
    for hop, layer in enumerate(ons):
        if layer is None:
            continue
        ax.plot([layer - .5] * 2, [hop - .5, hop + .5], color="black", lw=.6)
        if hop and ons[hop - 1] is not None:
            ax.plot([ons[hop - 1] - .5, layer - .5], [hop - .5] * 2, color="black", lw=.6)
        ax.plot(layer, hop, "o", ms=2.6, mfc="white", mec="black", mew=.4)


# Illustrative reference trace for the hop chains: Figure 2's example (3 -> 11 -> 6 -> 2 -> 9)
# continued round the same cycle, so H=4 and H=8 share their start and first three hops.
EXAMPLE_TRACE = [3, 11, 6, 2, 9, 0, 14, 5, 12]


def hop_chain(ax, hop, label_size=6.5, max_ms=8.2):
    """Vertical chain of reference nodes beside the maps: start at the bottom, answer at the top.

    One node per heat-map row with its example node id inside, one small arrow per hop.
    """
    ax.set_xlim(-1, 1)
    ax.set_ylim(-.5, hop + .5)
    ax.axis("off")
    # Size the nodes from the row pitch so the H=8 chain does not overlap itself.
    fig = ax.figure
    row_pt = ax.get_position().height * fig.get_figheight() * 72 / (hop + 1)
    ms = min(max_ms, .72 * row_pt)
    pad = ms / 2 / row_pt + .07  # arrow clearance from each node edge, in rows
    labels = ["start"] + [str(h) for h in range(1, hop)] + ["answer"]
    for h in range(hop + 1):
        face = ".88" if h in (0, hop) else "white"
        ax.plot(.68, h, "o", ms=ms, mfc=face, mec="black", mew=.45, zorder=3, clip_on=False)
        ax.text(.68, h, str(EXAMPLE_TRACE[h]), ha="center", va="center", fontsize=.62 * ms, zorder=4)
        ax.text(.14, h, labels[h], ha="right", va="center", fontsize=label_size, clip_on=False)
        if h:
            ax.annotate("", xy=(.68, h - pad), xytext=(.68, h - 1 + pad),
                        arrowprops={"arrowstyle": "-|>", "lw": .45, "color": "black",
                                    "shrinkA": 0, "shrinkB": 0, "mutation_scale": 3.5 * max_ms / 8.2})


def accuracy_box(ax, acc):
    """Boxed two-line full-answer accuracy label, under the frontier dots, over the map."""
    ax.text(.035, .955, f"Accuracy\n{f'{100 * acc:.1f}'.removesuffix('.0')}%", transform=ax.transAxes, va="top",
            fontsize=6.5, linespacing=1.05,
            bbox={"boxstyle": "square,pad=0.3", "facecolor": "white", "edgecolor": "black", "linewidth": .8},
            zorder=1)


def selectivity_map(ax, sel, hop, acc, block=None, show_missing=False):
    """One probe-selectivity heat map with the first-crossing frontier, block boundaries and accuracy box."""
    m = np.asarray(sel)
    im = ax.imshow(m.T, origin="lower", aspect="auto", interpolation="nearest",
                   extent=(-.5, 32.5, -.5, hop + .5), cmap=SELECTIVITY_CMAP, vmin=0, vmax=1)
    ons = onsets(m)
    frontier(ax, ons)
    if show_missing:
        for h, layer in enumerate(ons):
            if layer is None:  # never reaches 0.5: a cross past the last layer of that row
                ax.plot(32, h, "x", ms=3.2, color="black", mew=.6)
    if block:
        for depth in range(block, 33, block):
            ax.axvline(depth, color="black", lw=.3, alpha=.22 if block >= 4 else .14)
    accuracy_box(ax, acc)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks(list(range(hop + 1)))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(labelleft=False)  # the hop chain at left labels the rows
    return im


def row_label(fig, chain, text="Reference hop"):
    """Axis name for the heat-map rows, set beside the hop chain (the chain axis itself is off)."""
    pos = chain.get_position()
    fig.text(pos.x0 - .004, (pos.y0 + pos.y1) / 2, text, rotation=90, ha="right", va="center", fontsize=8)
