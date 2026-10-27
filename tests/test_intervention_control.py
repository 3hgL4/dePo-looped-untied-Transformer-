"""C4 matched control: pairing invariants and the 2x2 arithmetic.

The control has to hold everything about the injected activation fixed except
which node it is about, so these tests pin the two properties that could
silently break that:

1. the control pairing never lands on the same chain (provably different node
   at every iteration), and never perturbs the real targets or the prompts;
2. the recorded control target equals an independent recomputation from the
   edge strings, including the j+1 > H clamp.
"""

import numpy as np
import pytest

from src.intervention_h2 import (
    generate_paired_trials,
    summarize_control,
    _apply_power,
    _cycle_maps,
)


def _fwd_maps(trial):
    """Rebuild (pi_A, pi_B) from the edge segments of the two prompts."""
    maps = []
    for key in ("text_A", "text_B"):
        edges = trial[key].split(" | ")[0]
        order = []
        pairs = [e.strip() for e in edges.split(",")]
        # edges were shuffled; rebuild the cycle by following s > d from any edge
        fwd_edge = {}
        for p in pairs:
            s, d = [int(x) for x in p.split(">")]
            fwd_edge[s] = d
        start = next(iter(fwd_edge))
        cur = start
        for _ in range(len(fwd_edge)):
            order.append(cur)
            cur = fwd_edge[cur]
        assert cur == start, "edges do not form a single cycle"
        fwd, _ = _cycle_maps(order)
        maps.append(fwd)
    return maps[0], maps[1]


def test_control_pairing_is_a_different_chain():
    trials = generate_paired_trials(16, H=4, M=5, n_trials=30, seed=11,
                                    mode="compose", with_control=True)
    n_with_ctrl = 0
    for trial in trials:
        for q, tgt in enumerate(trial["targets"]):
            assert tgt["ctrl_q"] is not None or tgt["composed_ctrl"] is None
            if tgt["ctrl_q"] is None:
                continue
            n_with_ctrl += 1
            assert tgt["ctrl_q"] != q
            assert len(tgt["composed_ctrl"]) == len(tgt["composed"])
            # same iteration, different node => different composed target at
            # every j in 0..H, including the j+1 > H clamp
            for j in range(len(tgt["composed"])):
                assert tgt["composed_ctrl"][j] != tgt["composed"][j]
    assert n_with_ctrl >= 30 * 4, "too many queries failed to find a partner"


def test_control_does_not_touch_prompts_or_real_targets():
    plain = generate_paired_trials(16, H=5, M=5, n_trials=20, seed=7, mode="compose")
    withc = generate_paired_trials(16, H=5, M=5, n_trials=20, seed=7, mode="compose",
                                   with_control=True)
    for a, b in zip(plain, withc):
        assert a["text_A"] == b["text_A"] and a["text_B"] == b["text_B"]
        for ta, tb in zip(a["targets"], b["targets"]):
            assert ta["pure_A"] == tb["pure_A"]
            assert ta["pure_B"] == tb["pure_B"]
            assert ta["composed"] == tb["composed"]
            assert ta["start_donor"] == tb["start_donor"]
            # default path keeps the original key set (no downstream surprises)
            assert "composed_ctrl" not in ta
            assert "ctrl_q" not in ta


def test_control_target_matches_independent_recomputation():
    H, N = 6, 16
    trials = generate_paired_trials(N, H=H, M=4, n_trials=25, seed=3,
                                    mode="compose", with_control=True)
    for trial in trials:
        fwd_A, fwd_B = _fwd_maps(trial)
        for tgt in trial["targets"]:
            # real target from the shared start
            recomputed = []
            for j in range(H + 1):
                hop = min(j + 1, H)
                mid = _apply_power(fwd_A, tgt["start_donor"], hop)
                recomputed.append(_apply_power(fwd_B, mid, max(H - hop, 0)))
            assert recomputed == tgt["composed"]
            if tgt["ctrl_q"] is None:
                continue
            partner = trial["targets"][tgt["ctrl_q"]]
            recomputed_c = []
            for j in range(H + 1):
                hop = min(j + 1, H)
                mid = _apply_power(fwd_A, partner["start_donor"], hop)
                recomputed_c.append(_apply_power(fwd_B, mid, max(H - hop, 0)))
            assert recomputed_c == tgt["composed_ctrl"]


def _rec(q, pure_A, pure_B, composed, ctrl, real_preds, ctrl_preds):
    return {
        "q": q,
        "pure_A": pure_A,
        "pure_B": pure_B,
        "composed": composed,
        "composed_ctrl": ctrl,
        "ctrl_q": (q + 1) % 5,
        "full_by_j": real_preds,
        "ctrl_full_by_j": ctrl_preds,
    }


def test_summarize_control_perfect_mechanism():
    """A perfect pointer mechanism must score high in ctrl_vs_ctrl and
    real_vs_real, and 0 in the two cross cells."""
    K = H = 3
    J = H + 1
    recs = []
    for q in range(5):
        composed = [10 + j for j in range(J)]
        ctrl = [0 + j for j in range(J)]
        recs.append(_rec(q, 13, 2, composed, ctrl, composed, ctrl))
    m = summarize_control(recs, np.arange(16), set(), K, H)
    assert m["ctrl_vs_ctrl"] == [1.0] * J
    assert m["real_vs_real"] == [1.0] * J
    assert m["ctrl_vs_real"] == [0.0] * J
    assert m["real_vs_ctrl"] == [0.0] * J
    assert m["chance"] == pytest.approx(1 / 16)
    assert m["n_ctrl"] == 5


def test_summarize_control_no_mechanism_is_chance():
    """Independent random outputs put all four cells at chance."""
    K = H = 3
    J = H + 1
    rng = np.random.default_rng(0)
    recs = []
    for q in range(400):
        composed = [int(x) for x in rng.integers(0, 16, J)]
        ctrl = [int(x) for x in rng.integers(0, 16, J)]
        real_preds = [int(x) for x in rng.integers(0, 16, J)]
        ctrl_preds = [int(x) for x in rng.integers(0, 16, J)]
        recs.append(_rec(q, 13, 2, composed, ctrl, real_preds, ctrl_preds))
    m = summarize_control(recs, np.arange(16), set(), K, H)
    for key in ("ctrl_vs_ctrl", "ctrl_vs_real", "real_vs_ctrl", "real_vs_real"):
        for v in m[key]:
            assert abs(v - 1 / 16) < 0.05, (key, v)


def test_summarize_control_skips_records_without_partner():
    K = H = 2
    recs = [_rec(0, 13, 2, [3, 4, 5], [6, 7, 8], [3, 4, 5], [6, 7, 8]),
            _rec(1, 13, 2, [3, 4, 5], None, [3, 4, 5], [-2, -2, -2])]
    recs[1]["ctrl_q"] = None
    m = summarize_control(recs, np.arange(16), set(), K, H)
    assert m["n_records"] == 2 and m["n_ctrl"] == 1
    assert m["ctrl_vs_ctrl"][0] == 1.0
