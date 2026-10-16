"""Activation-patching schematic (Figure 6c), in the style of Figure 2.

Both prompts share one permutation (Figure 2's 16-cycle), so the figure draws the same
row of ten cycle nodes twice. The receiver starts at 3 and walks 3, 11, 6, 2, 9; the
donor starts at 0 and walks 0, 14, 5, 12, 7. Top row, no intervention: the receiver
takes its 4 hops and writes 9. Bottom row, intervention after iteration 2: the
receiver's state at node 6 is replaced with the donor's state after iteration 2 (node
5, reached by the donor's own two hops drawn below the row), and the receiver then
continues 5, 12, 7 and writes 7, the donor's answer. The drawing is idealised as one
hop per iteration.

``draw(ax)`` draws everything into a given axes; build_fig6_use.py places it as panel (c).
``uv run python experiments/h2_probing/scripts/plot_patching_schematic.py [--out_dir figures/build]``
writes the standalone patching_schematic.{pdf,png}.
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
VISIBLE = 10
HOPS, PATCH_AFTER = 4, 2
RECEIVER_START, DONOR_START = 3, 0
INK, MUTED, PATCH = "#222222", "#777777", "#444444"
ANSWER, ANSWER_FILL = "#20756e", "#eaf4f2"

# Drawing extent in inches: draw() uses equal data units, so W x H is the aspect.
W, H = 5.5, 2.1
R = .12                                   # node radius
XS = [1.30 + .445 * j for j in range(VISIBLE)]
Y_TOP, Y_BOT = 1.58, .62


def walk(start, hops):
    successor = dict(zip(CYCLE, CYCLE[1:] + CYCLE[:1]))
    chain = [start]
    for _ in range(hops):
        chain.append(successor[chain[-1]])
    return chain


def check_chains():
    assert len(CYCLE) == len(set(CYCLE)) == 16 and set(CYCLE) == set(range(16))
    receiver, donor = walk(RECEIVER_START, HOPS), walk(DONOR_START, HOPS)
    # states after iterations 0..4 with the patch: the iteration-2 state becomes the donor's
    patched = receiver[:PATCH_AFTER] + walk(donor[PATCH_AFTER], HOPS - PATCH_AFTER)
    assert receiver == [3, 11, 6, 2, 9]
    assert donor == [0, 14, 5, 12, 7]
    assert patched == [3, 11, 5, 12, 7] and patched[-1] == donor[-1] != receiver[-1]
    row = walk(CYCLE[0], VISIBLE - 1)
    assert row == CYCLE[:VISIBLE] and set(receiver + donor) <= set(row)
    return row, receiver, donor


def _arrow(ax, start=None, end=None, color=INK, lw=.8, ls="solid", z=2, path=None):
    kwargs = dict(arrowstyle="-|>", mutation_scale=7, linewidth=lw, color=color,
                  linestyle=ls, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(FancyArrowPatch(path=path, **kwargs) if path is not None
                 else FancyArrowPatch(start, end, **kwargs))


def _arc(ax, x0, x1, y, depth, color, lw, label=None, label_size=8, lift0=.02,
         lift1=.02, ls="solid"):
    """Curved hop arrow, above the row for depth > 0 and below it for depth < 0."""
    s = 1 if depth > 0 else -1
    path = DrawPath([(x0 + .04, y + s * (R + lift0)), (x0 + .085, y + depth),
                     (x1 - .085, y + depth), (x1 - .04, y + s * (R + lift1))],
                    [DrawPath.MOVETO, DrawPath.CURVE4, DrawPath.CURVE4, DrawPath.CURVE4])
    _arrow(ax, color=color, lw=lw, ls=ls, path=path)
    if label:
        ax.text((x0 + x1) / 2, y + depth + s * .065, label, fontsize=label_size,
                color=color, ha="center", va="center")


def _row(ax, y, row, answer=None, replaced=None, ringed=None):
    """Figure 2's node row: circles joined by straight black successor edges."""
    for j, (x, node) in enumerate(zip(XS, row)):
        face, edge, color = "white", INK, INK
        if node == answer:
            face, edge, color = ANSWER_FILL, ANSWER, ANSWER
        elif node == replaced:
            edge, color = MUTED, MUTED
        ax.add_patch(Circle((x, y), R, facecolor=face, edgecolor=edge, linewidth=.85,
                            zorder=3))
        ax.text(x, y, str(node), fontsize=8.5, color=color, ha="center", va="center",
                zorder=4)
        if node == ringed:
            ax.add_patch(Circle((x, y), R + .035, facecolor="none", edgecolor=PATCH,
                                linewidth=.9, linestyle=(0, (2.5, 1.5)), zorder=3))
        if j < len(row) - 1:
            gap0 = .075 if node == ringed else .04
            gap1 = .075 if row[j + 1] == ringed else .04
            _arrow(ax, (x + R + gap0, y), (XS[j + 1] - R - gap1, y))


def draw(ax):
    """Draw the schematic into ``ax`` (data units are inches, W x H)."""
    row, receiver, donor = check_chains()
    at = {node: XS[j] for j, node in enumerate(row)}
    ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
    ax.axis("off")

    def text(x, y, s, color=INK, size=8, ha="center", **kw):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", **kw)

    # ---- No intervention --------------------------------------------------------
    _row(ax, Y_TOP, row, answer=receiver[-1])
    for k in range(HOPS):
        _arc(ax, at[receiver[k]], at[receiver[k + 1]], Y_TOP, .3, ANSWER, .9,
             f"hop {k + 1}")
    text(.02, Y_TOP, "No intervention", size=9, ha="left")
    text(at[receiver[0]], Y_TOP - .25, "start")
    text(at[receiver[-1]], Y_TOP - .25, f"writes {receiver[-1]}", color=ANSWER)

    # ---- Intervention after iteration 2 ------------------------------------------
    copied, replaced = donor[PATCH_AFTER], receiver[PATCH_AFTER]
    _row(ax, Y_BOT, row, answer=donor[-1], replaced=replaced, ringed=copied)
    for k in range(PATCH_AFTER):
        _arc(ax, at[receiver[k]], at[receiver[k + 1]], Y_BOT, .3, ANSWER, .9,
             f"hop {k + 1}")
    for k in range(PATCH_AFTER, HOPS):
        _arc(ax, at[donor[k]], at[donor[k + 1]], Y_BOT, .3, ANSWER, .9, f"hop {k + 1}",
             lift0=.055 if donor[k] == copied else .02)
    # the donor's own first two hops, below the row, bring its state to node 5
    for k in range(PATCH_AFTER):
        _arc(ax, at[donor[k]], at[donor[k + 1]], Y_BOT, -.28, MUTED, .7,
             f"hop {k + 1}", label_size=7.5,
             lift1=.055 if donor[k + 1] == copied else .02)
    text(at[donor[0]], Y_BOT - .5, f"donor starts at {donor[0]}", color=MUTED)
    # the receiver's state at node 6 jumps to the donor's node 5
    x0, x1 = at[replaced], at[copied]
    jump = DrawPath([(x0 + .02, Y_BOT + R + .02), (x0 + .25, Y_BOT + .66),
                     (x1 - .25, Y_BOT + .66), (x1 - .02, Y_BOT + R + .055)],
                    [DrawPath.MOVETO, DrawPath.CURVE4, DrawPath.CURVE4, DrawPath.CURVE4])
    _arrow(ax, color=PATCH, lw=.9, ls=(0, (3, 2)), path=jump)
    text((x0 + x1) / 2, Y_BOT + .33, "state replaced with the donor's", color=PATCH)
    text(.02, Y_BOT, "Intervention\nafter iteration 2", size=9, ha="left",
         linespacing=1.15)
    text(at[receiver[0]], Y_BOT - .25, "start")
    text(W - .02, Y_BOT - .32, f"writes {donor[-1]}, the\ndonor's answer", color=ANSWER,
         ha="right", linespacing=1.15)


def main(out_dir=DEFAULT_OUT):
    apply_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"patching_schematic.{ext}", dpi=200, bbox_inches=None,
                    pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"saved {out_dir / 'patching_schematic.pdf'} and .png ({W} x {H} in)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    main(p.parse_args().out_dir)
