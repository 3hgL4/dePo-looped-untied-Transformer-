"""
Model configurations for the depth-gating experiments.

Provides a unified factory that returns either a Transformer or LoopedTransformer
depending on the config name.

Naming conventions:
- "ouro-{L_r}x{K}": Looped model with L_r recurrent layers, K max iterations
- "deep-{N}": Standard transformer with N unique layers
- "wide-{L}": Standard transformer with L layers and wider dim (param-matched)
"""

from looped_lm.model import Transformer, LoopedTransformer


# Base hyperparameters shared across all configs
BASE = {
    "dim": 256,
    "n_heads": 8,
    "causal": True,
    "feed_forward_multiple_of": 64,
    "norm_eps": 1e-5,
}


def create_model(
    config_name: str,
    vocab_size: int,
    max_sequence_len: int = 256,
    pad_token_idx: int = None,
    use_nope: bool = False,
    zero_init: bool = False,
) -> tuple:
    """Create a model from a config name.

    Returns:
        (model, config_dict) where config_dict contains all model settings
        including 'model_type' ('looped' or 'standard').
    """
    config = BASE.copy()
    config["vocab_size"] = vocab_size
    config["max_sequence_len"] = max_sequence_len
    config["pad_token_idx"] = pad_token_idx

    if config_name.startswith("ouro-"):
        # Looped model: "ouro-{L_r}x{K_max}"
        parts = config_name[5:].split("x")
        assert len(parts) == 2, f"Expected ouro-LxK format, got {config_name}"
        L_r = int(parts[0])
        K_max = int(parts[1])

        config["model_type"] = "looped"
        config["n_recurrent_layers"] = L_r
        config["K_max"] = K_max
        config["n_prelude_layers"] = 0
        config["n_coda_layers"] = 0

        model = LoopedTransformer(
            dim=config["dim"],
            vocab_size=vocab_size,
            n_recurrent_layers=L_r,
            n_heads=config["n_heads"],
            max_sequence_len=max_sequence_len,
            K_max=K_max,
            n_prelude_layers=0,
            n_coda_layers=0,
            causal=config["causal"],
            feed_forward_multiple_of=config["feed_forward_multiple_of"],
            pad_token_idx=pad_token_idx,
            norm_eps=config["norm_eps"],
            use_positional_embedding=not use_nope,
        )
        if zero_init:
            model.zero_init_output_projections()
        return model, config

    elif config_name.startswith("etd-"):
        # ETD-style: "etd-{P}_{L_r}_{C}x{K_max}"
        spec, k_str = config_name[4:].split("x")
        parts = spec.split("_")
        assert len(parts) == 3, f"Expected etd-P_Lr_Cx format, got {config_name}"
        P, L_r, C = int(parts[0]), int(parts[1]), int(parts[2])
        K_max = int(k_str)

        config["model_type"] = "looped"
        config["n_recurrent_layers"] = L_r
        config["K_max"] = K_max
        config["n_prelude_layers"] = P
        config["n_coda_layers"] = C

        model = LoopedTransformer(
            dim=config["dim"],
            vocab_size=vocab_size,
            n_recurrent_layers=L_r,
            n_heads=config["n_heads"],
            max_sequence_len=max_sequence_len,
            K_max=K_max,
            n_prelude_layers=P,
            n_coda_layers=C,
            causal=config["causal"],
            feed_forward_multiple_of=config["feed_forward_multiple_of"],
            pad_token_idx=pad_token_idx,
            norm_eps=config["norm_eps"],
            use_positional_embedding=not use_nope,
        )
        if zero_init:
            model.zero_init_output_projections()
        return model, config

    elif config_name.startswith("deep-"):
        # Standard transformer: "deep-{N}"
        N = int(config_name[5:])

        config["model_type"] = "standard"
        config["n_layers"] = N

        model = Transformer(
            dim=config["dim"],
            vocab_size=vocab_size,
            n_layers=N,
            n_heads=config["n_heads"],
            max_sequence_len=max_sequence_len,
            causal=config["causal"],
            feed_forward_multiple_of=config["feed_forward_multiple_of"],
            pad_token_idx=pad_token_idx,
            norm_eps=config["norm_eps"],
            use_positional_embedding=not use_nope,
        )
        return model, config

    elif config_name.startswith("wide-"):
        # Wide-shallow transformer: "wide-{L}" with 2x dim
        L = int(config_name[5:])

        config["model_type"] = "standard"
        config["n_layers"] = L
        config["dim"] = 512  # 2x base dim for param matching
        config["n_heads"] = 16  # scale heads with dim

        model = Transformer(
            dim=config["dim"],
            vocab_size=vocab_size,
            n_layers=L,
            n_heads=config["n_heads"],
            max_sequence_len=max_sequence_len,
            causal=config["causal"],
            feed_forward_multiple_of=config["feed_forward_multiple_of"],
            pad_token_idx=pad_token_idx,
            norm_eps=config["norm_eps"],
            use_positional_embedding=not use_nope,
        )
        return model, config

    else:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Expected ouro-LxK, etd-P_Lr_CxK, deep-N, or wide-L."
        )


def list_configs():
    """Print all standard experiment configs with parameter counts."""
    import torch
    from looped_lm.utils import format_params

    dummy_vocab = 100  # typical for synthetic tasks after token mapping
    configs = [
        # Looped models (primary)
        "ouro-1x1", "ouro-1x2", "ouro-1x4", "ouro-1x8",
        "ouro-1x12", "ouro-1x16", "ouro-1x24", "ouro-1x32",
        # L_r=2 variants
        "ouro-2x1", "ouro-2x2", "ouro-2x4", "ouro-2x8", "ouro-2x16",
        # L_r=4 variants
        "ouro-4x1", "ouro-4x2", "ouro-4x4", "ouro-4x8",
        # Unlooped baselines
        "deep-1", "deep-2", "deep-4", "deep-8", "deep-16", "deep-32",
        # Wide baselines
        "wide-4",
    ]

    print(f"{'Config':<16} {'Type':<10} {'Eff.Depth':<10} {'Params':<10}")
    print("-" * 50)
    for name in configs:
        model, cfg = create_model(name, vocab_size=dummy_vocab)
        n_params = sum(p.numel() for p in model.parameters())
        if cfg["model_type"] == "looped":
            eff = model.effective_depth()
        else:
            eff = cfg["n_layers"]
        print(f"{name:<16} {cfg['model_type']:<10} {eff:<10} {format_params(n_params):<10}")
        del model


if __name__ == "__main__":
    list_configs()
