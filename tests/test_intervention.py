"""Activation patching (src/intervention_h2.py): receiver/donor construction and patched positions.

Runs on CPU with a tiny untrained looped model. The tokenizer is the base tokenizer the trainer
uses (downloaded on first use); the tests are skipped if it cannot be loaded.
"""

import numpy as np
import pytest
import torch

from src import intervention_h2 as intv
from src.logit_lens_h2 import build_node_token_map
from src.model_configs import create_model
from src.probe_h2 import find_query_positions, get_q_token_ids, tokenize_batch

N, M, MAX_LEN = 16, 5, 200


def parse_prompt(text: str):
    """Return the successor map of the edge list and the (H, start, answer) of every query."""
    edges, *queries = [part.strip() for part in text.split("|")]
    succ = {}
    for edge in edges.split(","):
        src, dst = edge.split(">")
        succ[int(src)] = int(dst)
    parsed = []
    for q in queries:
        word, H, s, _, ans = q.split()
        assert word == "hop"
        parsed.append((int(H), int(s), int(ans)))
    return succ, parsed


def walk(succ, node, steps):
    for _ in range(steps):
        node = succ[node]
    return node


def test_redirect_shares_the_permutation_and_differs_in_start():
    H = 6
    for trial in intv.generate_paired_trials(N, H, M, n_trials=20, seed=1, mode="redirect"):
        succ_A, q_A = parse_prompt(trial["text_A"])
        succ_B, q_B = parse_prompt(trial["text_B"])
        assert succ_A == succ_B and len(succ_A) == N
        cycle, node = {0}, walk(succ_A, 0, 1)
        while node != 0:  # a single N-cycle
            cycle.add(node)
            node = succ_A[node]
        assert len(cycle) == N
        for (h_a, s_d, ans_a), (h_b, s_r, ans_b), tgt in zip(q_A, q_B, trial["targets"]):
            assert h_a == h_b == H and s_d != s_r
            assert ans_a == walk(succ_A, s_d, H) == tgt["pure_A"]
            assert ans_b == walk(succ_B, s_r, H) == tgt["pure_B"]
            assert tgt["composed"] == [tgt["pure_A"]] * (H + 1)


def test_compose_shares_the_start_and_composes_two_permutations():
    H = 6
    for trial in intv.generate_paired_trials(N, H, M, n_trials=20, seed=2, mode="compose"):
        succ_A, q_A = parse_prompt(trial["text_A"])
        succ_B, q_B = parse_prompt(trial["text_B"])
        for (_, s_d, ans_a), (_, s_r, ans_b), tgt in zip(q_A, q_B, trial["targets"]):
            assert s_d == s_r
            assert ans_a == tgt["pure_A"] and ans_b == tgt["pure_B"]
            for j in range(H + 1):
                hop = min(j + 1, H)  # the donor state injected after iteration j carries hop j+1
                assert tgt["composed"][j] == walk(succ_B, walk(succ_A, s_d, hop), H - hop)
            assert tgt["composed"][H - 1] == tgt["composed"][H] == tgt["pure_A"]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from src.probe_h2 import build_tokenizer
        return build_tokenizer(N)
    except Exception as exc:  # no network and no cached tokenizer
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    model, config = create_model("ouro-2x3", vocab_size=len(tokenizer), max_sequence_len=MAX_LEN,
                                 pad_token_idx=tokenizer.pad_token_id, zero_init=False)
    return model.eval(), config["K_max"]


def run(tokenizer, model, K, trials, patch_answer_digit, capture=None):
    node_to_tok, ambiguous = build_node_token_map(tokenizer, N)
    if capture is not None:
        original = intv.looped_forward_with_injection

        def spy(model, tokens, attention_mask, inject_iter, inject_values, inject_mask, K):
            capture.append((tokens.clone(), inject_iter.clone(), inject_mask.clone()))
            return original(model, tokens, attention_mask, inject_iter, inject_values, inject_mask, K)
        intv.looped_forward_with_injection = spy
    try:
        return intv.run_intervention(trials, tokenizer, model, "cpu", K=K, H=trials[0]["H"],
                                     max_len=MAX_LEN, node_to_tok=node_to_tok,
                                     ambiguous_nodes=set(ambiguous),
                                     patch_answer_digit=patch_answer_digit)
    finally:
        if capture is not None:
            intv.looped_forward_with_injection = original


@pytest.mark.parametrize("patch_answer_digit", [False, True])
def test_patched_positions(tokenizer, tiny_model, patch_answer_digit):
    """Row (j, q) patches the readout position of every query up to q, and with
    --patch_answer_digit also the first answer digit after each, after iteration j only."""
    model, K = tiny_model
    trial = intv.generate_paired_trials(N, 3, M, n_trials=1, seed=3, mode="redirect")[0]
    calls = []
    run(tokenizer, model, K, [trial], patch_answer_digit, capture=calls)
    tokens, it, mask = calls[0]  # the first generation step: rows hold the prompt cut at each readout
    q_ids = get_q_token_ids(tokenizer)
    for r in range(tokens.shape[0]):
        j, q = r // M - 1, r % M
        assert int(it[r]) == j
        row = tokens[r].tolist()
        readouts = [p + 1 for p, t in enumerate(row) if t in q_ids]
        assert len(readouts) == q + 1  # the row ends at query q's readout position
        expected = set()
        if j >= 0:
            for p in readouts:
                expected.add(p)
                if patch_answer_digit:
                    expected.add(p + 1)
        assert set(np.flatnonzero(mask[r].numpy()).tolist()) == expected
        for p in readouts:  # the readout position is the space token after '?'
            assert tokenizer.decode([row[p]]).strip() == ""
        for p in readouts[:-1]:  # earlier queries keep their gold answers: p + 1 is the first digit
            assert tokenizer.decode([row[p + 1]]).strip().isdigit()


@pytest.mark.parametrize("patch_answer_digit", [False, True])
def test_self_patching_is_the_identity(tokenizer, tiny_model, patch_answer_digit):
    """Patching a prompt with its own states leaves every written answer unchanged.

    With --patch_answer_digit the current query's first-digit position holds the digit the
    receiver wrote, while the donor's state there was computed on the gold digit, so the
    identity holds from the second written token on only where the two digits agree.
    """
    model, K = tiny_model
    trials = intv.generate_paired_trials(N, 3, M, n_trials=2, seed=4, mode="compose")
    for trial in trials:
        trial["text_A"] = trial["text_B"]
    out = run(tokenizer, model, K, trials, patch_answer_digit)
    gold, _ = intv.answer_token_sequences(tokenizer, N, MAX_LEN)
    for rec in out["records"]:
        assert all(x == rec["base_pred"] for x in rec["pred_by_j"][:K])
        written = np.array(rec["written_by_j"])
        assert (written[:K, 0] == written[K + 1, 0]).all()  # the first written token
        if not patch_answer_digit or written[K + 1, 0] == gold[rec["pure_B"]][0]:
            assert (written[:K] == written[K + 1]).all()
            assert all(x == rec["base_full"] for x in rec["full_by_j"][:K])
    if not patch_answer_digit:
        assert len(out["records"]) == 2 * M


def test_answer_digit_position_follows_the_readout(tokenizer):
    """In a tokenized prompt, readout + 1 holds the answer's first digit token."""
    text = "3 > 12 , 12 > 3 | hop 1 3 ? 12 | hop 2 3 ? 3"
    tok = tokenize_batch([text], tokenizer, max_len=MAX_LEN)
    row = tok["tokens"][0].tolist()
    readouts = find_query_positions(tok["tokens"], get_q_token_ids(tokenizer))[0]
    assert [tokenizer.decode([row[p + 1]]).strip() for p in readouts] == ["1", "3"]
    assert tokenizer.decode(row[readouts[0] + 1:readouts[0] + 3]).strip() == "12"
