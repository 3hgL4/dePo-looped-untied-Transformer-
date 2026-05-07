"""Section 5 figures: per-iteration probe staircase and logit-lens panels.

Builds the four §5 figures from a multi-seed, uniform-H re-probe:
    fig_h2_full_chain_hero      — multi-seed per-mMLRD staircase, looped cells
    fig_h2_full_chain_appendix  — extended onset-iter panel
    fig_h2_commit_deferral      — k-sweep commitment-deferral curve
    fig_h2_lens_hero            — single-query logit-lens trajectory

Inputs (under --results_root):
    <results_root>/probe_h4/<cell>_s<seed>/        (probe_accuracies.npz, probe_results.json)
    <results_root>/lens_h4/<cell>_s<seed>/         (logit_lens.npz, logit_lens_summary.json)
    <results_root>/probe_permlrd/<cell>_s<seed>/   (per-mMLRD probes for hero panel)
    <results_root>/lens_permlrd/<cell>_s<seed>/    (per-mMLRD lens for hero panel)
    <results_root>/lens_perh/<cell>_s<seed>/       (per-H lens, optional)

Outputs go under --out_dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from looped_lm.figstyle import (
    ACCENT_RED, MARKERS_L, PALETTE_L, apply_style, disable_categorical_minor_ticks,
)

apply_style()

# These globals are populated from CLI args in main(). Functions below read them.
H2_RES: Path = Path(".")
OUT: Path = Path(".")
PROBE_H4: Path = Path(".")
LENS_H4: Path = Path(".")
PROBE_PERMLRD: Path = Path(".")
LENS_PERMLRD: Path = Path(".")
LENS_PERH: Path = Path(".")

# Per-cell H_eval, mirroring the probe full-chain hero (Fig 4).
PER_CELL_H = {"ouro-4x8": 8, "ouro-2x16": 12}

# Inclusion thresholds. Looped models must actually solve H=4 (sanity > 0.95);
# deep-32 by design doesn't saturate H=4 (H1 finding) so we use a lax floor and
# keep its panel as a charitable "even when probe finds h0, no staircase" view.
SANITY_LOOPED = 0.95
SANITY_DEEP = 0.0
LOOPED_CELLS = ["ouro-4x8", "ouro-2x16"]
DEEP_CELL = "deep-32"
SEEDS = [42, 43, 44, 45, 46]


def load_probe(cell: str, seed: int):
    """Return (selectivity, sanity, k_sweep_head, real, shuffled) or None if missing/aborted."""
    d = PROBE_H4 / f"{cell}_s{seed}"
    pa = d / "probe_accuracies.npz"
    pr = d / "probe_results.json"
    if not (pa.exists() and pr.exists()):
        return None
    arrs = np.load(pa)
    info = json.load(open(pr))
    sanity = info["model_prediction_sanity"]["next_token_accuracy_at_probe_pos"]
    return {
        "real": arrs["linear_real"],          # (levels, H+1)
        "shuf": arrs["linear_shuffled"],
        "selectivity": arrs["linear_real"] - arrs["linear_shuffled"],
        "sanity": sanity,
        "k_sweep_head": info.get("k_sweep_head_accuracy"),  # list or None
    }


def stack_seeds(cell: str, sanity_floor: float):
    """Return (selectivity_stack, k_sweep_stack, real_stack, kept_seeds)."""
    sels, ksw, reals, kept = [], [], [], []
    for s in SEEDS:
        d = load_probe(cell, s)
        if d is None or d["sanity"] < sanity_floor:
            continue
        sels.append(d["selectivity"])
        reals.append(d["real"])
        if d["k_sweep_head"] is not None:
            ksw.append(np.asarray(d["k_sweep_head"], dtype=np.float32))
        kept.append(s)
    if not sels:
        return None, None, None, []
    return (
        np.stack(sels, axis=0),
        np.stack(ksw, axis=0) if ksw else None,
        np.stack(reals, axis=0),
        kept,
    )


# --------------------------------------------------------------------------
# Heatmap helper, used by the per-mMLRD full-chain hero figure.
# --------------------------------------------------------------------------

def heatmap_panel(ax, mat, title, hop_labels=None, cbar=False):
    """Heatmap with iteration on x-axis, hop on y-axis.

    `mat` is shape (n_levels, H+1) as returned by the probe pipeline; we
    transpose so that the time axis (iteration / layer) reads left-to-right.
    """
    im = ax.imshow(mat.T, cmap="viridis", vmin=0, vmax=1, aspect="auto",
                   origin="lower")
    ax.set_title(title)
    is_deep = "deep" in title.lower()
    ax.set_xlabel("layer $\\ell$" if is_deep else "iteration $k$")
    ax.set_ylabel("hop index $h$")
    H_plus_1 = mat.shape[1]
    ax.set_yticks(range(H_plus_1))
    if hop_labels is None:
        hop_labels = [f"$h_{{{h}}}$" + (" (ans)" if h == H_plus_1 - 1 else "")
                      for h in range(H_plus_1)]
    ax.set_yticklabels(hop_labels)
    n_levels = mat.shape[0]
    if n_levels <= 10:
        ax.set_xticks(range(n_levels))
    else:
        ax.set_xticks(range(0, n_levels, max(1, n_levels // 8)))
    disable_categorical_minor_ticks(ax)
    return im


# --------------------------------------------------------------------------
# Fig B: commitment deferral (k-sweep on the head)
# --------------------------------------------------------------------------

def fig_commit_deferral():
    fig, ax_c = plt.subplots(1, 1, figsize=(7.5, 5.0))

    for cell in LOOPED_CELLS:
        _, ksw, _, kept = stack_seeds(cell, SANITY_LOOPED)
        if ksw is None: continue
        L = int(cell.split("-")[1].split("x")[0])
        K = int(cell.split("x")[1])
        mean = ksw.mean(axis=0); std = ksw.std(axis=0)
        x = np.arange(1, mean.shape[0] + 1)
        col = PALETTE_L.get(L, "k")
        ax_c.plot(x, mean, marker=MARKERS_L.get(L, "o"), color=col, linewidth=1.6,
                  markersize=7, label=f"loop-{L}x{K} ($n={len(kept)}$)")
        ax_c.fill_between(x, mean - std, mean + std, color=col, alpha=0.18, linewidth=0)

    ax_c.axhline(7/16, color=ACCENT_RED, linestyle="--", linewidth=0.9,
                 label="fallback $0.4375$")
    ax_c.set_xlabel("$K_{\\mathrm{eval}}$")
    ax_c.set_ylabel("head accuracy on $h_H$")
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.legend(loc="lower right")

    fig.tight_layout()
    out = OUT / "fig_h2_commit_deferral"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}.{{png,pdf}}")


# --------------------------------------------------------------------------
# Fig C: logit lens hero
# --------------------------------------------------------------------------

def _lens_seed_mean(lens_dir: Path):
    """Mean P(h_h | iter k) across solved queries for one seed.

    Returns (mean (K+1, H+1), sanity_solved_rate, n_used) or None if missing.
    Filters to solved AND first-token-distinct queries so each hop maps to a
    unique vocabulary slot the lens can actually express.
    """
    f = lens_dir / "logit_lens.npz"
    if not f.exists():
        return None
    data = np.load(f)
    probs = data["probs"]            # (N, K+1, n_nodes)
    chains = data["chains"]          # (N, H+1) -- node ids
    solved = data["solved"].astype(bool)
    n2t = data["node_to_tok"]
    H_plus_1 = chains.shape[1]
    mask_distinct = np.array([len(set(c.tolist())) == H_plus_1 for c in chains])
    mask_ftd = np.array([len(set(n2t[c].tolist())) == H_plus_1 for c in chains])
    mask = solved & mask_distinct & mask_ftd
    if mask.sum() < 50:
        mask = solved if solved.sum() > 50 else np.ones_like(solved, bool)
    Kp1 = probs.shape[1]
    idx = chains[:, None, :].repeat(Kp1, axis=1)            # (N, K+1, H+1)
    p_hop = np.take_along_axis(probs, idx, axis=2)          # (N, K+1, H+1)
    return p_hop[mask].mean(axis=0), float(solved.mean()), int(mask.sum())


def lens_lineplot_panel(ax, cell: str, title: str, root: Path,
                         show_ylabel: bool = True):
    """Cross-seed mean P(chain[h] | iter k) line chart, one line per hop.

    Loads all SEEDS for `cell` from `root`, drops sub-SANITY_LOOPED seeds,
    plots the per-hop trace (mean over seeds + ±1 SD band) on a viridis
    colormap. Sparse legend: only h_0, h_{H/2}, h_H to avoid a wall of
    labels on the H=12 panel.
    """
    per_seed = []
    kept = []
    for s in SEEDS:
        out = _lens_seed_mean(root / f"{cell}_s{s}")
        if out is None:
            continue
        mean_q, sanity, _ = out
        if sanity < SANITY_LOOPED:
            continue
        per_seed.append(mean_q)
        kept.append(s)
    if not per_seed:
        ax.set_title(f"{title}\n(no saturating seeds)"); ax.axis("off"); return
    stack = np.stack(per_seed, axis=0)                       # (S, K+1, H+1)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    n_levels, H_plus_1 = mean.shape
    H = H_plus_1 - 1
    cmap = plt.cm.viridis
    xs = np.arange(n_levels)
    for h in range(H_plus_1):
        col = cmap(h / max(H, 1))
        is_answer = (h == H)
        label = f"$h_{{{h}}}$" + (" (ans)" if is_answer else "")
        ax.plot(xs, mean[:, h], color=col,
                linewidth=1.4, marker="o", markersize=3.5,
                alpha=0.92 if is_answer else 0.8,
                label=label,
                zorder=3 if is_answer else 2)
    # chance baseline
    ax.axhline(1.0 / 16.0, color=ACCENT_RED, linestyle=":", linewidth=0.8,
               label="chance $1/16$")
    ax.set_xlabel("iteration $k$")
    if show_ylabel:
        ax.set_ylabel("$P(\\mathrm{chain\\ node}\\ |\\ \\mathrm{iter}\\ k)$")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    if n_levels <= 10:
        ax.set_xticks(xs)
    else:
        ax.set_xticks(xs[::max(1, n_levels // 8)])
    ncol_leg = 3 if H_plus_1 >= 10 else 2
    ax.legend(loc="upper left", fontsize=10, ncol=ncol_leg, framealpha=0.95,
              columnspacing=0.8, handlelength=1.2, handletextpad=0.4)


def fig_lens_hero():
    # figsize matched to fig 2's on-page scale (~0.66 \linewidth/inch).
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2),
                              gridspec_kw={"wspace": 0.24, "width_ratios": [1, 1.6]})
    lens_lineplot_panel(axes[0], "ouro-4x8",
                        "loop-4x8, $H_{\\mathrm{eval}}{=}8$",
                        LENS_PERH, show_ylabel=True)
    lens_lineplot_panel(axes[1], "ouro-2x16",
                        "loop-2x16, $H_{\\mathrm{eval}}{=}12$",
                        LENS_PERH, show_ylabel=False)
    out = OUT / "fig_h2_lens_hero"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}.{{png,pdf}}")


# --------------------------------------------------------------------------
# Fig D (per-mMLRD hero): full-chain staircase on s42 hero seed
# --------------------------------------------------------------------------

def _load_permlrd_seed(cell_seed: str):
    """Return (selectivity, sanity) or None if missing."""
    d = PROBE_PERMLRD / cell_seed
    if not (d / "probe_accuracies.npz").exists():
        return None
    arrs = np.load(d / "probe_accuracies.npz")
    info = json.load(open(d / "probe_results.json"))
    sel = arrs["linear_real"] - arrs["linear_shuffled"]
    sanity = info["model_prediction_sanity"]["next_token_accuracy_at_probe_pos"]
    return sel, sanity


def fig_full_chain_hero():
    """s42 hero: per-mMLRD selectivity for both looped cells.

    H_eval per cell matches K to stress-test each architecture:
      loop-4x8  (K=8):  H=8  -> K=H, zero slack
      loop-2x16 (K=16): H=12 -> 4 iterations of slack, avoids the N=16 shortcut zone
    Cross-seed reproducibility lives in fig_full_chain_appendix (per-seed
    onset offsets and one seed with a non-iterative encoding).
    """
    cells = [("ouro-4x8", 4, 8, 8),
             ("ouro-2x16", 2, 16, 12)]
    panel_data = []
    for cfg, L, K, H_eval in cells:
        r = _load_permlrd_seed(f"{cfg}_s42")
        if r is None:
            continue
        sel, _ = r
        panel_data.append((cfg, L, K, H_eval, sel))

    if not panel_data:
        print("  (per-mMLRD data not yet pulled, skipping)")
        return

    # figsize matched to fig 2's on-page scale; ~4.6in per panel keeps fonts
    # readable when LaTeX renders at 0.92\linewidth.
    fig, axes = plt.subplots(1, len(panel_data), figsize=(4.6 * len(panel_data), 4.6),
                             gridspec_kw={"wspace": 0.7}, squeeze=False)
    axes = axes[0]
    last_im = None
    for ax, (cfg, L, K, H_eval, sel) in zip(axes, panel_data):
        title = f"loop-{L}x{K}, $H{{=}}{H_eval}$"
        last_im = heatmap_panel(ax, sel, title)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.022, pad=0.02)
        cbar.set_label("probe selectivity")
    out = OUT / "fig_h2_full_chain_hero"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}.{{png,pdf}}")


# --------------------------------------------------------------------------
# Appendix: per-seed staircase reproducibility
# --------------------------------------------------------------------------

def fig_full_chain_appendix():
    """Per-seed view of the staircase across saturating training seeds.

    Two rows: each cell renders one heatmap per seed (sanity >= 0.95 only).
    Plus a final row with onset-iteration-vs-hop traces overlaid across seeds,
    which makes the phase-shift effect (averaging blurs the hero) visible
    and shows that one ouro-2x16 seed solves the task without storing
    intermediate node identities (chance-level middle hops).
    """
    cells = [("ouro-4x8", 4, 8, 8, [42, 43, 44, 45]),
             ("ouro-2x16", 2, 16, 12, [42, 43, 44, 46])]
    onset_threshold = 0.5

    # Two heatmap rows (one per cell, 4 seeds each) + one onset row (two panels).
    # Figsize tuned so that 4 heatmap panels per row (with default figstyle font
    # sizes) don't overlap, while still giving big on-page fonts at 0.96\linewidth.
    fig = plt.figure(figsize=(17, 12))
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85], hspace=0.75)

    last_im = None
    for row_idx, (cfg, L, K, H_eval, seeds) in enumerate(cells):
        gs_row = gs_outer[row_idx].subgridspec(1, len(seeds) + 1,
                                               width_ratios=[1.0] * len(seeds) + [0.04],
                                               wspace=0.85)
        for col_idx, s in enumerate(seeds):
            r = _load_permlrd_seed(f"{cfg}_s{s}")
            ax = fig.add_subplot(gs_row[0, col_idx])
            if r is None:
                ax.set_title(f"s{s}\n(missing)", fontsize=16); ax.axis("off"); continue
            sel, sanity = r
            kept = sanity >= SANITY_LOOPED
            tag = f"s{s}, sanity={sanity:.2f}" + ("" if kept else "\n[dropped]")
            last_im = heatmap_panel(ax, sel, tag)
            ax.title.set_fontsize(16)
            if col_idx == 0:
                ax.text(-0.45, 1.25,
                        f"loop-{L}x{K}, $H{{=}}{H_eval}$",
                        transform=ax.transAxes, fontsize=18,
                        ha="left", va="bottom")
        if last_im is not None:
            cax = fig.add_subplot(gs_row[0, -1])
            cbar = fig.colorbar(last_im, cax=cax)
            cbar.set_label("probe selectivity")

    gs_bot = gs_outer[2].subgridspec(1, 2, wspace=0.28)
    for col_idx, (cfg, L, K, H_eval, seeds) in enumerate(cells):
        ax = fig.add_subplot(gs_bot[0, col_idx])
        ax.set_title(f"loop-{L}x{K}: first iteration above selectivity {onset_threshold}")
        for s in seeds:
            r = _load_permlrd_seed(f"{cfg}_s{s}")
            if r is None: continue
            sel, sanity = r
            onset = []
            for h in range(sel.shape[1]):
                idx = np.where(sel[:, h] > onset_threshold)[0]
                onset.append(idx[0] if len(idx) else np.nan)
            hops = np.arange(sel.shape[1])
            ls = "-" if sanity >= SANITY_LOOPED else "--"
            ax.plot(hops, onset, marker="o", markersize=4, linestyle=ls,
                    linewidth=1.2,
                    label=f"s{s} (sanity={sanity:.2f})")
        ax.plot([0, H_eval], [0, H_eval], color="0.5", linestyle=":",
                linewidth=0.8, label="$k=h$ ideal")
        ax.set_xlabel("hop $h$"); ax.set_ylabel("onset iter $k$")
        ax.set_xlim(-0.3, H_eval + 0.3)
        ax.legend(loc="lower right", fontsize=11)
        disable_categorical_minor_ticks(ax)

    out = OUT / "fig_h2_full_chain_appendix"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}.{{png,pdf}}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results_root", required=True, type=Path,
                   help="Directory containing probe_h4/, lens_h4/, probe_permlrd/, lens_permlrd/ subdirs.")
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Output directory for figures (will be created).")
    args = p.parse_args()

    global H2_RES, OUT, PROBE_H4, LENS_H4, PROBE_PERMLRD, LENS_PERMLRD, LENS_PERH
    H2_RES = args.results_root
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    PROBE_H4 = H2_RES / "probe_h4"
    LENS_H4 = H2_RES / "lens_h4"
    PROBE_PERMLRD = H2_RES / "probe_permlrd"
    LENS_PERMLRD = H2_RES / "lens_permlrd"
    LENS_PERH = H2_RES / "lens_perh"

    fig_full_chain_hero()
    fig_full_chain_appendix()
    fig_commit_deferral()
    fig_lens_hero()
    print("Done.")


if __name__ == "__main__":
    main()
