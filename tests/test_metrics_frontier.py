"""P1/P2: three-way frontier, threshold sweep, eval-hop list."""

import pytest

from src.metrics import (
    fold_note,
    frontier_sweep,
    frontier_three,
    h_eff_of,
    h_eff_vectors,
    parse_eval_hops,
)

# Official ouro-8x2 seed 42, exact float values from experiments/bench_review.
OURO_8X2_S42 = {
    1: 1.000, 2: 1.000, 3: 0.977, 4: 0.7861, 5: 0.7486,
    6: 0.7880, 7: 0.8049, 8: 0.8311,
    9: 0.830, 10: 0.890, 11: 0.910, 12: 0.960, 13: 0.990, 14: 1.000, 15: 1.000,
}


def test_three_way_on_reverse_only_run():
    # The paper's one-way frontier reads 3; the same weights solve the chain
    # backwards for all 8 effective hops.
    front = frontier_three(OURO_8X2_S42)
    assert (front["fwd"], front["rev"], front["max"]) == (3, 8, 8)
    assert front["thr"] == 0.8


def test_fwd_is_the_paper_number():
    # A run that only solves forward: fwd unchanged, rev 0, max == fwd.
    per_hop = {h: (0.9 if h <= 2 else 0.3) for h in range(1, 16)}
    front = frontier_three(per_hop)
    assert (front["fwd"], front["rev"], front["max"]) == (2, 0, 2)


def test_string_keys_are_accepted():
    per_hop = {str(h): v for h, v in OURO_8X2_S42.items()}
    assert frontier_three(per_hop)["max"] == 8


def test_missing_hops_stop_the_count():
    # Only hops 1..3 were evaluated: no direction can claim more than 3.
    partial = {h: 1.0 for h in (1, 2, 3)}
    front = frontier_three(partial)
    assert (front["fwd"], front["rev"], front["max"]) == (3, 0, 3)


def test_gap_breaks_continuity():
    per_hop = {1: 1.0, 2: 1.0, 4: 1.0, 5: 1.0}
    assert frontier_three(per_hop)["fwd"] == 2


def test_threshold_sweep_reproduces_the_boundary_case():
    # Proposal Sec 2.2: 0.70 -> 8, 0.75 -> 4 (H5 = 0.7486 misses by 0.0014),
    # 0.80 -> 3 (H4 = 0.7861), 0.85 -> 3.
    sweep = frontier_sweep(OURO_8X2_S42, thresholds=(0.70, 0.75, 0.80, 0.85))
    assert sweep["fwd"] == [8, 4, 3, 3]
    assert sweep["span_fwd"] == 5
    # The reverse side is fragile too, but only above 0.80: H9 = 0.830 drops
    # out at 0.85, so rev = 8,8,8,6 and the max column moves 2 levels, not 5.
    # Reporting max alone therefore HIDES almost all of the boundary fragility.
    assert sweep["rev"] == [8, 8, 8, 6]
    assert sweep["max"] == [8, 8, 8, 6]
    assert sweep["span_max"] == 2


def test_h_eff_vectors_cover_both_directions():
    vec = h_eff_vectors(OURO_8X2_S42)
    assert vec["h_eff"] == list(range(1, 9))
    assert vec["fwd_H"] == list(range(1, 9))
    assert vec["rev_H"] == list(range(15, 7, -1))
    assert vec["fwd_acc"][3] == 0.7861
    assert vec["rev_acc"][0] == 1.000  # H = 15


def test_h_eff_of():
    assert h_eff_of(1) == 1
    assert h_eff_of(15) == 1
    assert h_eff_of(8) == 8
    assert h_eff_of(9) == 7


def test_parse_eval_hops_full_grid_is_the_paper():
    assert parse_eval_hops("", 1, 15) == list(range(1, 16))
    assert parse_eval_hops(None, 1, 15) == list(range(1, 16))
    assert parse_eval_hops("all", 1, 15) == list(range(1, 16))


def test_parse_eval_hops_half_is_eight():
    assert parse_eval_hops("half", 1, 15) == list(range(1, 9))
    assert parse_eval_hops("half", 1, 15, N=32) == list(range(1, 17))


def test_parse_eval_hops_explicit_and_validated():
    assert parse_eval_hops("3,1,2", 1, 15) == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_eval_hops("1,16", 1, 15)
    with pytest.raises(ValueError):
        parse_eval_hops("abc", 1, 15)
    with pytest.raises(ValueError):
        parse_eval_hops(",", 1, 15)


def test_fold_note_only_when_the_axis_folds():
    assert fold_note(8) is None
    assert "fold" in fold_note(15)
