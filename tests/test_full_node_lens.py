"""Full-node lens (src/full_node_lens.py) at t = T reproduces the model's own forward pass.

On a tiny untrained looped model (CPU): the lens probability of the gold answer at t = T equals
the teacher-forced probability of its digits and delimiter in the full prompt; and with a
stand-in model that writes a fixed node, the lens's greedy written node is correct exactly where
the probe's full-answer scorer (model_complete_answer_at_q) marks the first query correct.
"""

import numpy as np
import pytest
import torch

from src.full_node_lens import build_rows, run_lens
from src.intervention_h2 import answer_token_sequences
from src.model_configs import create_model
from src.probe_h2 import (find_query_positions, generate_eval_set, get_q_token_ids,
                          model_complete_answer_at_q, pointer_query_record, tokenize_batch)

N, M, H, MAX_LEN, n = 16, 5, 2, 200, 12


@pytest.fixture(scope="module")
def setup():
    try:
        from src.probe_h2 import build_tokenizer
        tokenizer = build_tokenizer(N)
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")
    torch.manual_seed(0)
    model, config = create_model("ouro-2x2", vocab_size=len(tokenizer), max_sequence_len=MAX_LEN,
                                 pad_token_idx=tokenizer.pad_token_id, zero_init=False)
    with torch.no_grad():  # sharpen the random model so that it writes some nodes greedily
        model.out_proj.weight.mul_(8.0)
    return tokenizer, model.eval(), config["K_max"]


def test_lens_at_T_matches_the_forward_pass(setup):
    tokenizer, model, T = setup
    eval_set = generate_eval_set(N=N, H_eval=H, M=M, n_samples=n, seed=12345)
    gold, space = answer_token_sequences(tokenizer, N, MAX_LEN)
    tokens, mask, readout = build_rows(eval_set, tokenizer, MAX_LEN, gold)
    P, ok = run_lens(model, tokens, mask, readout, gold, space, T, batch_size=64, device="cpu")
    P = P.reshape(n, N, T).transpose(0, 2, 1)
    ok = ok.reshape(n, N, T).transpose(0, 2, 1)
    assert (P.sum(-1) <= 1 + 1e-5).all()
    assert ok.sum(-1).max() <= 1
    answers = np.array([s["meta"]["queries"][0]["answer"] for s in eval_set])

    # Teacher-forced probability of the gold first answer (digits + "|") in the full prompt.
    q_ids = get_q_token_ids(tokenizer)
    tok = tokenize_batch([s["text"] for s in eval_set], tokenizer, max_len=MAX_LEN)
    with torch.no_grad():
        logp = torch.log_softmax(model(tok["tokens"], tok["attention_mask"], K=T).float(), -1)
    for i, pos in enumerate(q[0] for q in find_query_positions(tok["tokens"], q_ids)):
        g = gold[answers[i]]
        assert tok["tokens"][i, pos + 1:pos + 1 + len(g)].tolist() == g
        p = sum(float(logp[i, pos + k, g[k]]) for k in range(len(g)))
        assert abs(np.exp(p) - P[i, T - 1, answers[i]]) < 1e-4


class BigramWriter(torch.nn.Module):
    """A stand-in model whose next token depends on the current token only: after '?' it writes
    the space, after the space a fixed digit, after that digit the '|' delimiter. It writes the
    same single-digit node greedily after every query, at every loop budget."""

    def __init__(self, tokenizer, digit: int):
        super().__init__()
        vocab = len(tokenizer)
        enc = lambda text: tokenize_batch([text], tokenizer, max_len=MAX_LEN)
        row = enc(f"hop 1 0 ? {digit} |")
        ids, valid = row["tokens"][0].tolist(), row["attention_mask"][0].tolist()
        start = valid.index(True)
        _, q, space, d, bar = ids[start:][-5:]
        self.table = torch.full((vocab, vocab), -10.0)
        self.table[:, 0] = 0.0
        self.table[q, space] = self.table[space, d] = self.table[d, bar] = 10.0

    def forward(self, tokens, attention_mask=None, K=None):
        return self.table[tokens]


def test_greedy_written_node_matches_the_probe_scorer(setup):
    tokenizer, _, _ = setup
    n_prompts = 64
    eval_set = generate_eval_set(N=N, H_eval=H, M=M, n_samples=n_prompts, seed=12345)
    answers = np.array([s["meta"]["queries"][0]["answer"] for s in eval_set])
    digit = int(np.bincount(answers[answers < 10], minlength=10).argmax())
    model = BigramWriter(tokenizer, digit).eval()
    gold, space = answer_token_sequences(tokenizer, N, MAX_LEN)
    tokens, mask, readout = build_rows(eval_set, tokenizer, MAX_LEN, gold)
    _, ok = run_lens(model, tokens, mask, readout, gold, space, 2, batch_size=256, device="cpu")
    ok = ok.reshape(n_prompts, N, 2).transpose(0, 2, 1)
    written = np.where(ok.any(-1), ok.argmax(-1), -1)
    assert (written == digit).all()

    records = [pointer_query_record(s["meta"], i, q) for i, s in enumerate(eval_set) for q in range(M)]
    metadata = {k: np.asarray([r[k] for r in records], dtype=bool if k.startswith("is_") else np.int64)
                for k in records[0]}
    _, arrays = model_complete_answer_at_q(eval_set, tokenizer, model, "cpu", MAX_LEN, 64, metadata)
    first = arrays["complete_node_correct"][metadata["is_first_query"]]
    assert 0 < first.sum() < n_prompts
    np.testing.assert_array_equal(written[:, -1] == answers, first)
