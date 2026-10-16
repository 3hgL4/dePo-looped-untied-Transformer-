"""
Model configurations: one factory that builds a looped or an untied model from its name.

Naming conventions (the paper writes a looped model as L x T):
- "ouro-{L}x{T}": looped model, a block of L weight-shared layers iterated T times
- "deep-{N}": untied model with N distinct layers, each applied once

    uv run python -m src.model_configs     # parameter counts of the paper's configurations
"""

from src.model import Transformer, LoopedTransformer


# Hyperparameters shared by every config
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
    use_timestep_embedding: bool = True,
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
        parts = config_name[5:].split("x")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Expected ouro-LxT, got {config_name}")
        L_r, K_max = int(parts[0]), int(parts[1])

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
            causal=config["causal"],
            feed_forward_multiple_of=config["feed_forward_multiple_of"],
            pad_token_idx=pad_token_idx,
            norm_eps=config["norm_eps"],
            use_positional_embedding=not use_nope,
            use_timestep_embedding=use_timestep_embedding,
        )
        config["use_timestep_embedding"] = use_timestep_embedding
        if zero_init:
            model.zero_init_output_projections()
        return model, config

    if config_name.startswith("deep-"):
        n = config_name[5:]
        if not n.isdigit():
            raise ValueError(f"Expected deep-N, got {config_name}")
        N = int(n)

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
        # The paper's untied runs use the default initialization (--no_zero_init).
        if zero_init:
            model.zero_init_output_projections()
        return model, config

    raise ValueError(f"Unknown config: {config_name}. Expected ouro-LxT or deep-N.")


# Every configuration trained for the paper (capability grid, rate controls and N=32).
PAPER_CONFIGS = (
    ["deep-2", "deep-4", "deep-8", "deep-16", "deep-32"]
    + [f"ouro-2x{t}" for t in (2, 4, 5, 6, 7, 8, 10, 12, 14, 16)]
    + [f"ouro-4x{t}" for t in (2, 3, 4, 6, 8)]
    + [f"ouro-8x{t}" for t in (2, 3, 4)]
    + ["ouro-16x2"]
)


def list_configs(vocab_size: int = 18):
    """Print every paper configuration with its effective depth and parameter count.

    The default vocabulary (18 tokens) is the token-mapped vocabulary of the N=16 task.
    """
    print(f"{'Config':<12} {'Type':<10} {'D_eff':<6} {'Params':>12}")
    print("-" * 44)
    for name in PAPER_CONFIGS:
        model, cfg = create_model(name, vocab_size=vocab_size, max_sequence_len=200)
        n_params = sum(p.numel() for p in model.parameters())
        eff = model.effective_depth() if cfg["model_type"] == "looped" else cfg["n_layers"]
        print(f"{name:<12} {cfg['model_type']:<10} {eff:<6} {n_params:>12,}")
        del model


if __name__ == "__main__":
    list_configs()
