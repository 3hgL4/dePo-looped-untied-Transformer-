"""
Report metrics for the dePo benchmark.

Implements P1 (three-way frontier: fwd / rev / max, plus threshold sweep) and
P2 (evaluation hop list) from
`notes/2026-10-26_基准测量缺陷与改进提案.md`.

Why the frontier needs three numbers
-----------------------------------
H and 16-H are the same question: pi^15 = pi^-1, so hop H is the reverse of hop
16-H (75000 checks, 0 counterexamples; see main report Sec 12.2 N1). The
paper's frontier only counts H = 1, 2, ... upward, so a model that solves a
chain *backwards* (official ouro-8x2 seed 42: fwd = 3, rev = 8) reads as "does
not solve it". `frontier_three` reports forward, reverse and their max; the
values are measured in EFFECTIVE hops h_eff = min(H, 16 - H), so 8 is the
ceiling for both directions, not 15.

Conventions
-----------
* `per_hop` maps hop length to accuracy. Keys may be ints or strings (the
  trajectory jsonl stores strings).
* A direction's frontier is the largest h_eff with every hop from 1 up to it
  at or above the threshold (continuous prefix, same rule as the paper).
  Missing hops stop the count: a frontier can never be claimed from data that
  was not evaluated.
* Nothing here rewrites the paper's number. `frontier_three(...)["fwd"]` is
  exactly the old `H_sol@0.8` when the full 15-hop grid is present.
"""

from __future__ import annotations

DEFAULT_THR = 0.8


def _get(per_hop: dict, H: int) -> float | None:
    """Look up one hop, tolerating int and str keys; None when not evaluated."""
    for key in (str(H), H):
        if key in per_hop:
            try:
                return float(per_hop[key])
            except (TypeError, ValueError):
                return None
    return None


def _contiguous(seq: list[float | None], thr: float) -> int:
    """Length of the leading run of values >= thr (stops at a miss or a gap)."""
    frontier = 0
    for i, value in enumerate(seq, 1):
        if value is None or value < thr:
            break
        frontier = i
    return frontier


def h_eff_of(H: int, N: int = 16) -> int:
    """Effective hop count of a hop length: min(H, N - H)."""
    return min(H, N - H)


def frontier_three(per_hop: dict, thr: float = DEFAULT_THR, N: int = 16) -> dict:
    """Forward / reverse / max frontier, all counted in effective hops.

    Args:
        per_hop: hop length -> accuracy (int or str keys).
        thr: accuracy threshold, 0.8 by default (the paper's).
        N: cycle length; the reverse hop of H is N - H.

    Returns:
        {"fwd": int, "rev": int, "max": int, "thr": float, "n": N}
        fwd counts H = 1, 2, ...; rev counts H = N-1, N-2, ... (both stop at
        h_eff = N//2, i.e. 8 hops for N = 16).
    """
    half = N // 2
    fwd_axis = list(range(1, half + 1))
    rev_axis = list(range(N - 1, N - 1 - half, -1))
    fwd = _contiguous([_get(per_hop, H) for H in fwd_axis], thr)
    rev = _contiguous([_get(per_hop, H) for H in rev_axis], thr)
    return {"fwd": fwd, "rev": rev, "max": max(fwd, rev), "thr": thr, "n": N}


def h_eff_vectors(per_hop: dict, N: int = 16) -> dict:
    """The two 8-dimensional accuracy vectors the frontier scalar is derived from.

    The proposal (Sec 2.4) asks for the curve, not only the scalar: a run whose
    hops all sit near the threshold can move several levels when the threshold
    moves 0.05, and the scalar hides that.
    """
    half = N // 2
    fwd_axis = list(range(1, half + 1))
    rev_axis = list(range(N - 1, N - 1 - half, -1))
    return {
        "h_eff": list(range(1, half + 1)),
        "fwd_H": fwd_axis,
        "rev_H": rev_axis,
        "fwd_acc": [_get(per_hop, H) for H in fwd_axis],
        "rev_acc": [_get(per_hop, H) for H in rev_axis],
    }


def frontier_sweep(
    per_hop: dict,
    thresholds=(0.70, 0.75, 0.80, 0.85),
    N: int = 16,
) -> dict:
    """Frontier at several thresholds (proposal Sec 2.4 item 3).

    Returns {"thr": [...], "fwd": [...], "rev": [...], "max": [...],
             "span_fwd": int, "span_max": int} where span = max - min. The
    proposal flags a configuration as "boundary undecided" when a threshold
    move of 0.05 shifts the reads by 2 or more levels.
    """
    rows = [frontier_three(per_hop, thr=t, N=N) for t in thresholds]
    fwd = [r["fwd"] for r in rows]
    rev = [r["rev"] for r in rows]
    mx = [r["max"] for r in rows]
    return {
        "thr": [float(t) for t in thresholds],
        "fwd": fwd,
        "rev": rev,
        "max": mx,
        "span_fwd": max(fwd) - min(fwd),
        "span_max": max(mx) - min(mx),
    }


def parse_eval_hops(spec: str | None, H_min: int, H_max: int, N: int = 16) -> list[int]:
    """Evaluation hop list (P2).

    Accepted specs:
        ""/"all"     every hop in [H_min, H_max] (the paper's 15-hop grid)
        "half"       H_min .. N//2, i.e. one representative per effective hop
                     (8 items for N=16); saves 46.7% of eval wall-clock
        "1,2,3"      explicit list, validated against [H_min, H_max]

    H = N//2+1 .. N-1 are repeats of the low half (pi^(N-1) = pi^-1), so the
    `half` grid carries exactly the same information as the full one.
    """
    if spec is None or spec == "" or spec == "all":
        return list(range(H_min, H_max + 1))
    if spec == "half":
        hops = list(range(H_min, N // 2 + 1))
    else:
        try:
            hops = [int(x) for x in spec.split(",") if x.strip() != ""]
        except ValueError as exc:
            raise ValueError(f"--eval_hops: cannot parse {spec!r}") from exc
        if not hops:
            raise ValueError(f"--eval_hops: empty list in {spec!r}")
        bad = [h for h in hops if not (H_min <= h <= H_max)]
        if bad:
            raise ValueError(
                f"--eval_hops: {bad} outside [H_min={H_min}, H_max={H_max}]")
        hops = sorted(set(hops))
    return hops


def fold_note(H_max: int, N: int = 16) -> str | None:
    """Warn when the training grid folds the difficulty axis (B2/P3)."""
    if H_max <= N // 2:
        return None
    return (
        f"H_max={H_max} > N//2={N // 2}: hops {N // 2 + 1}..{N - 1} repeat the low "
        f"half (pi^{N - 1} = pi^-1), so the difficulty axis folds into a U shape "
        f"and {N - 1 - N // 2} of {N - 1} eval hops are redundant. Set --H_max "
        f"{N // 2} for a monotone axis (that is a different benchmark: results "
        f"cannot be mixed with runs trained at H_max={N - 1})."
    )
