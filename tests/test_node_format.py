"""P5: answer formats - the generator is format-agnostic and the floor is measured."""

import random

import pytest

from src.datasets.pointer_chasing import generate_pointer_chasing_sample
from src.node_tokens import NODE_FORMATS, node_symbols


def _sample(fmt: str, seed: int = 7, H: int = 3):
    return generate_pointer_chasing_sample(
        random.Random(seed), N=16, H=H, M=5, node_format=fmt)


def test_plain_is_byte_identical_to_the_paper_format():
    # The default must not change a single character of an existing run's data.
    text, answer, meta = _sample("plain")
    assert " > " in text and " | hop 3 " in text
    assert answer == str(meta["queries"][-1]["answer"])
    assert all(isinstance(q["answer"], int) for q in meta["queries"])


def test_format_changes_text_but_not_the_task():
    plain_text, plain_ans, plain_meta = _sample("plain")
    for fmt in ("padded", "letters"):
        text, answer, meta = _sample(fmt)
        # same cycle, same queries, same answers - only the spelling differs
        assert meta["order"] == plain_meta["order"]
        assert [q["answer"] for q in meta["queries"]] == \
               [q["answer"] for q in plain_meta["queries"]]
        assert [q["intermediates"] for q in meta["queries"]] == \
               [q["intermediates"] for q in plain_meta["queries"]]
        assert text != plain_text
        assert answer == node_symbols(16, fmt)[meta["queries"][-1]["answer"]]


def test_letters_cover_the_whole_alphabet_prefix():
    symbols = node_symbols(16, "letters")
    assert symbols[0] == "A" and symbols[15] == "P"
    assert len(set(symbols)) == 16
    text, _, _ = _sample("letters")
    assert "A" in text or "B" in text


def test_padded_is_two_characters_everywhere():
    symbols = node_symbols(16, "padded")
    assert symbols[0] == "00" and symbols[15] == "15"
    assert all(len(s) == 2 for s in symbols)


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        node_symbols(16, "roman")
    with pytest.raises(ValueError):
        node_symbols(30, "letters")


def test_audit_measures_the_two_known_floors():
    """Tokenizer-level audit: plain 7/16, padded 10/16, letters 1/16."""
    transformers = pytest.importorskip("transformers")
    from src.node_tokens import audit

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer")
    except Exception as exc:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {exc}")

    result = audit(tok, N=16, formats=NODE_FORMATS)
    rows = {r["format"]: r for r in result["formats"]}

    assert rows["plain"]["n_groups"] == 10
    assert abs(rows["plain"]["first_token_floor"] - 7 / 16) < 1e-9
    assert rows["padded"]["first_token_floor"] >= rows["plain"]["first_token_floor"]
    assert rows["letters"]["distinct_first_tokens"] is True
    assert abs(rows["letters"]["first_token_floor"] - 1 / 16) < 1e-9
    assert result["recommended"] == "letters"
