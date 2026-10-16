"""Figure 2: a worked dePo example at the 5.5-inch text width.

    uv run python experiments/h2_probing/scripts/plot_task_schematic.py [--out_dir figures/build]

The reference chain and omitted return path derive from one valid 16-cycle. No inputs.
Writes fig2_task.{pdf,png}.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as DrawPath

from src.figstyle import apply_style


DEFAULT_OUT = ROOT / "figures/build"
# Starts with figcommon.EXAMPLE_TRACE, so every figure draws the same walk.
CYCLE = [3, 11, 6, 2, 9, 0, 14, 5, 12, 7, 1, 10, 8, 4, 13, 15]
START, HOPS = 3, 4
VISIBLE_NODES = 7
INK, MUTED = "#222222", "#777777"
ANSWER, ANSWER_FILL = "#20756e", "#eaf4f2"


def main(out_dir=DEFAULT_OUT):
    apply_style()
    successor = dict(zip(CYCLE, CYCLE[1:] + CYCLE[:1]))
    assert len(CYCLE) == len(set(CYCLE)) == 16
    assert set(CYCLE) == set(range(16))
    chain = [START]
    assert HOPS < VISIBLE_NODES < len(CYCLE)
    for _ in range(VISIBLE_NODES - 1):
        chain.append(successor[chain[-1]])
    omitted_nodes = len(CYCLE) - len(chain)
    assert successor[CYCLE[-1]] == START

    fig = plt.figure(figsize=(5.5, 1.88))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 5.5), ylim=(0, 1.88), aspect="equal")
    ax.axis("off")

    def label(x, y, text, size=9, color=INK, **kwargs):
        return ax.text(x, y, text, fontsize=size, color=color,
                       ha=kwargs.pop("ha", "left"), va="center", **kwargs)

    # Annotate the graph itself: where to start, how far to go, what to predict.
    xs = [.40 + .78 * j for j in range(VISIBLE_NODES)]
    label(xs[0], .40, "Start", size=9, ha="center")
    label((xs[0] + xs[HOPS]) / 2, 1.75, rf"$H={HOPS}$ hops", size=10, ha="center")
    label(xs[HOPS], .40, "Answer", size=9, ha="center", color=ANSWER)
    ax.plot([xs[0], xs[0], xs[HOPS], xs[HOPS]], [1.49, 1.57, 1.57, 1.49],
            color=MUTED, linewidth=.6)
    y, radius = .92, .15
    state_labels = [r"$z_0=s$"] + [rf"$z_{j}$" for j in range(1, HOPS)]
    state_labels.append(rf"$z_{HOPS}=\pi^{HOPS}(s)$")
    for j, (x, node) in enumerate(zip(xs, chain)):
        endpoint = j == HOPS
        ax.add_patch(Circle((x, y), radius, facecolor=ANSWER_FILL if endpoint else "white",
                            edgecolor=ANSWER if endpoint else INK, linewidth=.9))
        label(x, y, str(node), size=10, ha="center", color=ANSWER if endpoint else INK)
        if j <= HOPS:
            label(x, .61, state_labels[j], size=10, ha="center", color=ANSWER if endpoint else INK)
        # Ordinary successor edges continue beyond the requested endpoint.
        if j < VISIBLE_NODES - 1:
            ax.add_patch(FancyArrowPatch((x + radius + .055, y),
                                        (xs[j + 1] - radius - .055, y),
                                        arrowstyle="-|>", mutation_scale=8,
                                        linewidth=.8, color=INK, shrinkA=0, shrinkB=0))
        # Each requested hop has its own curved arrow, above the graph edge.
        if j < HOPS:
            next_x = xs[j + 1]
            hop_path = DrawPath(
                [(x + .055, y + radius + .025),
                 (x + .12, 1.40), (next_x - .12, 1.40),
                 (next_x - .055, y + radius + .025)],
                [DrawPath.MOVETO, DrawPath.CURVE4,
                 DrawPath.CURVE4, DrawPath.CURVE4],
            )
            ax.add_patch(FancyArrowPatch(path=hop_path, arrowstyle="-|>",
                                        mutation_scale=8, linewidth=1.0,
                                        color=ANSWER))
            label((x + next_x) / 2, 1.43, f"hop {j + 1}", size=8.5,
                  ha="center", color=ANSWER)
    # The dashed return path abbreviates ten edges through nine more nodes.
    # It must not look like an additional single-hop edge completing a seven-cycle.
    return_path = DrawPath(
        [(xs[-1] + radius + .04, y), (5.48, y), (5.48, .17), (xs[-1], .17),
         (xs[0], .17), (.02, .17), (.02, y), (xs[0] - radius - .04, y)],
        [DrawPath.MOVETO, DrawPath.CURVE4, DrawPath.CURVE4, DrawPath.CURVE4,
         DrawPath.LINETO, DrawPath.CURVE4, DrawPath.CURVE4, DrawPath.CURVE4],
    )
    ax.add_patch(FancyArrowPatch(path=return_path, arrowstyle="-|>", mutation_scale=8,
                                linewidth=.75, linestyle=(0, (3, 2.5)), color=MUTED,
                                zorder=0))
    label(2.74, .17, rf"{omitted_nodes} more nodes ($N={len(CYCLE)}$ total)",
          size=9, ha="center", color=MUTED,
          bbox=dict(facecolor="white", edgecolor="none", pad=4))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_task.{ext}", dpi=300,
                    bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"Visible chain: {chain}; {HOPS}-hop answer: {chain[HOPS]}; "
          f"saved {out_dir / 'fig2_task.pdf'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    main(p.parse_args().out_dir)
