"""Publication-clean figure style for the looped-LMs paper.

Single source of truth for matplotlib rcParams, color palette, and marker
shapes used across every figure in this repo. Any new figure-generation
script should start with:

    from looped_lm.figstyle import apply_style, PALETTE_L, MARKERS_L, ACCENT_RED
    apply_style()

This guarantees consistent typography (Computer Modern serif), thin black
box on all four sides, inward major + minor ticks, no grid, and a shared
color/marker mapping keyed by the loop count L.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


# Color palette indexed by loop count L. deep-N corresponds to L=1.
PALETTE_L = {
    1: "#555555",  # unlooped / deep-N        - neutral grey
    2: "#1f77b4",  # ouro L=2                  - blue
    4: "#2ca02c",  # ouro L=4                  - green
    8: "#d62728",  # ouro L=8                  - red
}

# Marker shape indexed by L. Same key set as PALETTE_L; consistent across
# every panel of every figure so a reader can identify a model family by
# shape alone.
MARKERS_L = {
    1: "s",
    2: "o",
    4: "D",
    8: "^",
}

# Reference accent for threshold lines (0.8 accuracy, H_max saturation, ...).
ACCENT_RED = "#c0392b"


_RC: dict = {
    # Typography
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.titleweight": "normal",
    "axes.labelsize": 22,
    "axes.labelweight": "normal",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "figure.titlesize": 24,
    "figure.titleweight": "normal",
    # Legend frame: thin black border, opaque, square corners.
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    # Axes box: thin black border on all four sides.
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    # Inward major + minor ticks on all four sides.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    # No grid by default; rely on tick marks for visual structure.
    "axes.grid": False,
    # Saving: tight bounding box, embed TrueType (Type 42) for arXiv/Overleaf.
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Install the paper figure style into the global matplotlib state."""
    plt.rcParams.update(_RC)


def disable_categorical_minor_ticks(ax) -> None:
    """Hide minor ticks on a categorical axis.

    Inward minor ticks look noisy when the x-axis is categorical (model
    names, condition labels). Call this on the ax in question after
    setting xticks/xticklabels.
    """
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def add_subtle_grid(ax, axis: str = "both") -> None:
    """Add a muted grid for figures where reading values across panels matters.

    This is the only sanctioned way to add a grid in this repo: subtle major
    lines, no minor lines, sits behind the data. Use sparingly, only when a
    reader genuinely needs to read off values from the axes.
    """
    ax.grid(axis=axis, which="major", color="0.85", linewidth=0.6,
            linestyle="-", zorder=0)
    ax.set_axisbelow(True)
