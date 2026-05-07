"""H1 paper figures from a uniform-H clean sweep.

Figures:
  fig_matched_deff32.{png,pdf}    Matched D_eff=32 saturation: deep-32 vs ouro-2x16 vs ouro-4x8.
                                  Per-seed scatter + mean bar + saturation rate (mMLRD>=15).
  fig_saturation_gradient.{png,pdf}  Probability-of-saturation vs D_eff, one curve per L.
  fig_perhop_headline.{png,pdf}   Per-hop accuracy at headline cells, all seeds.
  table_seed_distribution.csv     Seed-by-seed mMLRD per cell.

Per-seed mMLRD = largest H such that ALL h in 1..H have acc >= 0.8 at the
final training step (monotone-from-bottom).

Usage:
    uv run python scripts/plot_h1_saturation.py \\
        --runs_dir runs/clean_sweep \\
        --outdir   figures/h1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from looped_lm.figstyle import (
    ACCENT_RED,
    MARKERS_L,
    PALETTE_L,
    add_subtle_grid,
    apply_style,
    disable_categorical_minor_ticks,
)

apply_style()


def parse_arch(name: str):
    if name.startswith("ouro-"):
        L, K = name[5:].split("x")
        return int(L), int(K), int(L)*int(K), "looped"
    if name.startswith("deep-"):
        K = int(name[5:])
        return 1, K, K, "unlooped"
    return None, None, None, "unknown"


def per_seed_mlrd(per_hop: dict, threshold: float = 0.8) -> int:
    """Largest H such that ALL h in 1..H have acc >= threshold."""
    if not per_hop:
        return 0
    ks = sorted(int(k) for k in per_hop.keys())
    m = 0
    for h in ks:
        if per_hop[str(h)] >= threshold:
            m = h
        else:
            break
    return m


def load_runs(runs_dir: Path):
    """Return dict[(cell, seed)] -> {final_per_hop, all_steps_per_hop}."""
    out = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        traj = run_dir / "trajectory.jsonl"
        argf = run_dir / "args.json"
        if not traj.exists() or not argf.exists():
            continue
        args = json.loads(argf.read_text())
        cell = args["config_name"]
        seed = int(args["seed"])
        rows = []
        with open(traj) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            continue
        out[(cell, seed)] = {
            "rows": rows,
            "final": rows[-1]["per_hop_acc"],
            "args": args,
        }
    return out


def fig_matched_deff32(runs, outdir):
    cells = ["deep-32", "ouro-2x16", "ouro-4x8", "ouro-8x4"]
    labels = [r"$\mathtt{deep\text{-}32}$",
              r"$\mathtt{loop\text{-}2{\times}16}$",
              r"$\mathtt{loop\text{-}4{\times}8}$",
              r"$\mathtt{loop\text{-}8{\times}4}$"]
    short_labels = [r"$\mathtt{deep\text{-}32}$",
                    r"$\mathtt{loop\text{-}2{\times}16}$",
                    r"$\mathtt{loop\text{-}4{\times}8}$",
                    r"$\mathtt{loop\text{-}8{\times}4}$"]
    # Use the shared L-keyed palette / markers (ouro family + deep=L=1).
    cell_L = [1, 2, 4, 8]
    colors = [PALETTE_L[L] for L in cell_L]
    markers = [MARKERS_L[L] for L in cell_L]

    # Larger per-figure font sizes (override global defaults on this figure).
    # figsize matched to fig 2's on-page scale: width ~ width_fraction / 0.1025
    # so fonts read at the same size as fig 2 after LaTeX scaling.
    fs_title = 22
    fs_label = 24
    fs_tick = 19
    fs_legend = 17
    fs_annot = 17

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 5.0),
                                    gridspec_kw={"width_ratios": [1, 1.25]})

    # Left panel: translucent mean bar + per-seed scatter on top.
    rng = np.random.default_rng(0)
    for i, cell in enumerate(cells):
        ms = []
        for seed in [42, 43, 44, 45, 46]:
            if (cell, seed) in runs:
                m = per_seed_mlrd(runs[(cell, seed)]["final"])
                ms.append(m)
        if not ms:
            continue
        mean_m = float(np.mean(ms))
        ax1.bar(i, mean_m, width=0.62, color=colors[i], alpha=0.28,
                edgecolor=colors[i], linewidth=1.0, zorder=2)
        x = np.full(len(ms), i, dtype=float) + rng.uniform(-0.10, 0.10, size=len(ms))
        ax1.scatter(x, ms, s=70, color=colors[i], marker=markers[i],
                    edgecolor="black", linewidth=0.6, zorder=3, alpha=0.95)

    ax1.set_xticks(range(len(cells)))
    ax1.set_xticklabels(labels, fontsize=fs_tick, rotation=20, ha="right")
    disable_categorical_minor_ticks(ax1)
    ax1.tick_params(axis="y", labelsize=fs_tick)
    ax1.set_ylabel(r"$H^{\star}@0.8$", fontsize=fs_label)
    ax1.set_ylim(0, 16)
    ax1.set_xlim(-0.6, len(cells) - 0.4)
    ax1.axhline(15, color=ACCENT_RED, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.text(-0.55, 15.2, "Saturation",
             color=ACCENT_RED, fontsize=fs_tick - 4, ha="left", va="bottom")
    add_subtle_grid(ax1, axis="y")

    # Right panel: per-hop accuracy curves H=1..15, mean +/- std across seeds.
    H_vals = list(range(1, 16))

    for i, cell in enumerate(cells):
        accs = []  # shape (n_seeds, len(H_vals))
        for seed in [42, 43, 44, 45, 46]:
            if (cell, seed) not in runs:
                continue
            ph = runs[(cell, seed)]["final"]
            row = [ph[str(h)] for h in H_vals if str(h) in ph]
            if len(row) == len(H_vals):
                accs.append(row)
        if not accs:
            continue
        accs = np.array(accs)
        mean = accs.mean(axis=0)
        std = accs.std(axis=0)
        ax2.fill_between(H_vals, np.clip(mean - std, 0, 1),
                         np.clip(mean + std, 0, 1),
                         color=colors[i], alpha=0.14, zorder=2,
                         linewidth=0)
        ax2.plot(H_vals, mean, color=colors[i], linewidth=1.4,
                 marker=markers[i], markersize=4.5,
                 markeredgecolor=colors[i], markerfacecolor=colors[i],
                 markeredgewidth=0.6,
                 label=short_labels[i], zorder=3)

    ax2.set_xlabel(r"Hop $H$", fontsize=fs_label)
    ax2.set_ylabel("Per-hop accuracy", fontsize=fs_label)
    ax2.set_xticks(H_vals)
    ax2.set_xticklabels([str(h) if h % 2 == 1 else "" for h in H_vals])
    disable_categorical_minor_ticks(ax2)
    ax2.tick_params(axis="both", labelsize=fs_tick)
    ax2.set_xlim(0.5, 15.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.legend(loc="center right", ncol=2, bbox_to_anchor=(0.99, 0.65),
               handlelength=1.4, borderpad=0.35, labelspacing=0.22,
               columnspacing=0.9,
               fontsize=fs_legend - 3)
    add_subtle_grid(ax2, axis="y")
    fig.tight_layout()
    out = outdir / "fig_matched_deff32.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_saturation_gradient(runs, outdir):
    """Saturation rate vs D_eff, one curve per L.

    Linear x-axis (ticks every 2), bold titles, paper notation.
    Solid markers indicate n=5 seeds; open markers + dashed segments
    indicate n=3 seeds (noted in caption).
    """
    cells_by_L = defaultdict(list)
    for cell in {c for (c, _) in runs.keys()}:
        L, K, Deff, mt = parse_arch(cell)
        cells_by_L[L].append((Deff, cell))
    for L in cells_by_L:
        cells_by_L[L].sort()

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = PALETTE_L
    markers = MARKERS_L
    labels = {
        1: r"unlooped (deep-$N$)",
        2: r"looped $L{=}2$",
        4: r"looped $L{=}4$",
        8: r"looped $L{=}8$",
    }

    for L in sorted(cells_by_L.keys()):
        Ds, sats, ns = [], [], []
        for Deff, cell in cells_by_L[L]:
            ms = [per_seed_mlrd(runs[(cell, s)]["final"])
                  for s in [42, 43, 44, 45, 46] if (cell, s) in runs]
            if not ms:
                continue
            Ds.append(Deff)
            sats.append(100.0 * sum(1 for m in ms if m >= 15) / len(ms))
            ns.append(len(ms))
        if not Ds:
            continue
        # Plot connecting line: solid segment between two n>=5 endpoints,
        # dashed segment if either endpoint has n<5.
        for i in range(len(Ds) - 1):
            style = "-" if (ns[i] >= 5 and ns[i + 1] >= 5) else "--"
            ax.plot(Ds[i:i + 2], sats[i:i + 2], style, linewidth=2.2,
                    color=colors.get(L, "black"), zorder=2)
        # Plot markers: filled for n>=5, open for n<5.
        for x, y, n in zip(Ds, sats, ns):
            if n >= 5:
                ax.plot(x, y, marker=markers.get(L, "o"), markersize=11,
                        color=colors.get(L, "black"),
                        markerfacecolor=colors.get(L, "black"),
                        markeredgecolor="black", markeredgewidth=1.0,
                        linestyle="None", zorder=3)
            else:
                ax.plot(x, y, marker=markers.get(L, "o"), markersize=11,
                        color=colors.get(L, "black"),
                        markerfacecolor="white",
                        markeredgecolor=colors.get(L, "black"),
                        markeredgewidth=1.6, linestyle="None", zorder=3)
        # Single legend entry per family (use a proxy line so legend shows
        # solid line + filled marker even though we plotted segments).
        ax.plot([], [], marker=markers.get(L, "o"), linewidth=2.2,
                markersize=11, color=colors.get(L, "black"),
                markerfacecolor=colors.get(L, "black"),
                markeredgecolor="black", markeredgewidth=1.0,
                label=labels.get(L, f"$L{{=}}{L}$"))

    ax.set_xticks(list(range(2, 34, 2)))
    ax.set_xlim(1, 33)
    ax.set_xlabel(r"Effective Model Depth $D_{\mathrm{eff}}$")
    ax.set_ylabel(r"% seeds with $H^{\star}{=}15$")
    ax.set_ylim(-5, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = outdir / "fig_saturation_gradient.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_perhop_headline(runs, outdir):
    cells = [("deep-32", "deep-32 (unlooped, D_eff=32)", "#888888"),
             ("ouro-2x16", "ouro-2×16 (L=2, D_eff=32)", "#1f77b4"),
             ("ouro-4x8", "ouro-4×8 (L=4, D_eff=32)", "#2ca02c")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (cell, title, color) in zip(axes, cells):
        for seed in [42, 43, 44, 45, 46]:
            if (cell, seed) not in runs:
                continue
            ph = runs[(cell, seed)]["final"]
            hs = sorted(int(k) for k in ph.keys())
            accs = [ph[str(h)] for h in hs]
            ax.plot(hs, accs, marker="o", color=color, alpha=0.55,
                    label=f"s={seed}")
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("hop H")
        ax.set_xticks([1, 4, 8, 12, 15])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-hop accuracy")
    axes[0].legend(fontsize=11, loc="upper right")
    fig.suptitle("Per-hop accuracy at end of training (5 seeds, matched D_eff=32)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out = outdir / "fig_perhop_headline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def write_seed_table(runs, outdir):
    cells = sorted({c for (c, _) in runs.keys()},
                   key=lambda c: (parse_arch(c)[0] or 99, parse_arch(c)[2] or 99))
    rows = []
    for cell in cells:
        L, _, Deff, _ = parse_arch(cell)
        per = {}
        for s in [42, 43, 44, 45, 46]:
            per[s] = per_seed_mlrd(runs[(cell, s)]["final"]) if (cell, s) in runs else None
        sat = sum(1 for v in per.values() if v is not None and v >= 15)
        n = sum(1 for v in per.values() if v is not None)
        rows.append({
            "cell": cell, "L": L, "D_eff": Deff,
            "s42": per[42], "s43": per[43], "s44": per[44], "s45": per[45], "s46": per[46],
            "sat_at_15": f"{sat}/{n}",
            "mean_mMLRD": np.mean([v for v in per.values() if v is not None]) if n else None,
        })

    out = outdir / "table_seed_distribution.csv"
    with open(out, "w") as f:
        keys = ["cell", "L", "D_eff", "s42", "s43", "s44", "s45", "s46", "sat_at_15", "mean_mMLRD"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) if r[k] is not None else "" for k in keys) + "\n")
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    print("Loading runs...")
    runs = load_runs(runs_dir)
    print(f"  loaded {len(runs)} (cell, seed) pairs from {len({c for (c,_) in runs})} cells")

    print("Generating figures...")
    fig_matched_deff32(runs, outdir)
    fig_saturation_gradient(runs, outdir)
    fig_perhop_headline(runs, outdir)
    write_seed_table(runs, outdir)
    print("Done.")


if __name__ == "__main__":
    main()
