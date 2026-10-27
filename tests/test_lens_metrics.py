"""P4: the first-token floor and the two floor-free readouts."""

import numpy as np

from src.lens_metrics import (
    first_token_floor,
    group_baseline_mass,
    group_of_node,
    node_groups,
    readout_metrics,
    unambiguous_nodes,
)

# What the tokenizer actually does at N=16: node 1 and nodes 10..15 all start
# with the token "1"; every other node has a unique first token.
PAPER_FIRST_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 1, 1, 1]


def test_groups_match_the_measured_collision():
    groups = node_groups(PAPER_FIRST_TOKENS)
    assert len(groups) == 10
    assert [1, 10, 11, 12, 13, 14, 15] in groups
    assert first_token_floor(groups, 16) == 7 / 16
    assert group_baseline_mass(groups, 16) == 49 / 256 + 9 / 256


def test_unambiguous_nodes_are_the_singletons():
    assert unambiguous_nodes(node_groups(PAPER_FIRST_TOKENS)) == [0, 2, 3, 4, 5, 6, 7, 8, 9]


def test_group_of_node_is_a_partition():
    g = group_of_node(node_groups(PAPER_FIRST_TOKENS), 16)
    assert g[1] == g[10] == g[15]
    assert len(set(g.tolist())) == 10


def _constant_probs(answer_token: int, n: int = 16, levels: int = 2, q: int = 160):
    """A model that always burns its mass on one node's token."""
    probs = np.full((q, levels, n), 1e-6)
    for node in range(n):
        if PAPER_FIRST_TOKENS[node] == answer_token:
            probs[:, :, node] = 1.0
    return probs


def test_constant_answer_scores_the_floor_raw_and_chance_when_balanced():
    q, levels = 160, 2
    answers = np.arange(q) % 16          # every node equally likely
    probs = _constant_probs(1, levels=levels, q=q)
    met = readout_metrics(probs, answers, node_groups(PAPER_FIRST_TOKENS))

    # raw first-token readout: 7/16 = 0.4375 "for free"
    assert abs(met["group_top1"][0] - 0.4375) < 1e-9
    # balanced over the 10 groups: 1/10
    assert abs(met["group_balanced"][0] - 0.1) < 1e-9
    # node-level accuracy restricted to unambiguous answers: 0
    assert met["node_top1_unambig"][0] == 0.0
    # baselines travel with the metric
    assert abs(met["first_token_floor"] - 0.4375) < 1e-9
    assert abs(met["group_top1_baseline"] - 0.2265625) < 1e-9
    assert abs(met["group_balanced_baseline"] - 0.1) < 1e-9


def test_perfect_model_scores_one_everywhere():
    q, levels = 160, 3
    answers = np.arange(q) % 16
    probs = np.zeros((q, levels, 16))
    probs[np.arange(q), :, answers] = 1.0
    met = readout_metrics(probs, answers, node_groups(PAPER_FIRST_TOKENS))
    for key in ("group_top1", "group_balanced", "node_top1_unambig", "p_answer"):
        assert all(abs(v - 1.0) < 1e-9 for v in met[key]), key


def test_distinct_token_format_has_no_floor():
    # "letters" gives 16 distinct first tokens: no free accuracy.
    groups = node_groups(list(range(16)))
    assert len(groups) == 16
    assert first_token_floor(groups, 16) == 1 / 16
    assert abs(group_baseline_mass(groups, 16) - 1 / 16) < 1e-12
