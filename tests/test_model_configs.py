"""Every configuration in the paper builds, with the parameter counts the paper reports."""

import pytest
import torch

from src.model import LoopedTransformer, Transformer
from src.model_configs import PAPER_CONFIGS, create_model

VOCAB = 18  # the token-mapped vocabulary of the N=16 task


def n_params(name: str) -> int:
    model, _ = create_model(name, vocab_size=VOCAB, max_sequence_len=200, pad_token_idx=1)
    return sum(p.numel() for p in model.parameters())


def test_paper_parameter_counts():
    # Methods: 3.22M for the looped 4x8 model against 25.72M for the untied 32-layer model.
    assert n_params("ouro-4x8") == 3_224_832
    assert n_params("deep-32") == 25_715_968


@pytest.mark.parametrize("name", PAPER_CONFIGS)
def test_every_paper_config_builds(name):
    model, config = create_model(name, vocab_size=VOCAB, max_sequence_len=200, pad_token_idx=1,
                                 zero_init=name.startswith("ouro-"))
    if name.startswith("ouro-"):
        L, T = map(int, name[5:].split("x"))
        assert isinstance(model, LoopedTransformer)
        assert config["model_type"] == "looped" and config["K_max"] == T
        assert model.effective_depth() == L * T
    else:
        assert isinstance(model, Transformer)
        assert config["model_type"] == "standard" and config["n_layers"] == int(name[5:])
    tokens = torch.randint(0, VOCAB, (2, 12))
    assert model(tokens, torch.ones(2, 12, dtype=torch.bool)).shape == (2, 12, VOCAB)


def test_zero_init_zeroes_only_output_projections():
    model, _ = create_model("ouro-2x2", vocab_size=VOCAB, zero_init=True)
    for name, p in model.named_parameters():
        zero = float(p.abs().sum()) == 0.0
        assert zero == name.endswith(("attention.o_proj.weight", "feed_forward.w3.weight")), name


@pytest.mark.parametrize("name", ["share-4x8-cyclic2", "deepvec-32", "etd-1_2_1x4", "wide-4",
                                  "ouro-4", "deep-x"])
def test_removed_or_malformed_configs_raise(name):
    with pytest.raises(ValueError):
        create_model(name, vocab_size=VOCAB)
