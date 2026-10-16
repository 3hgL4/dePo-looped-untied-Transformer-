"""
Shared utilities: seeding, parameter counts, the compact-vocabulary tokenizer wrapper
and the learning-rate schedule.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def seed_all(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_params(n: int) -> str:
    """Format parameter count (e.g. 1_234_567 -> '1.2M')."""
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.1f}T"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/compiled to get the underlying model."""
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    if isinstance(model, nn.DataParallel):
        model = model.module
    return model


# ---------------------------------------------------------------------------
# Token mapping (vocab reduction for small models)
# ---------------------------------------------------------------------------

def map_token_ids(token_ids, mapping: dict):
    """Apply token-id mapping preserving structure and type.

    Supports torch.Tensor, np.ndarray, list (nested), int.
    """
    if mapping is None:
        return token_ids

    if isinstance(token_ids, torch.Tensor):
        device = token_ids.device
        dtype = token_ids.dtype
        flat = token_ids.view(-1).tolist()
        mapped = [mapping.get(t, t) for t in flat]
        return torch.tensor(mapped, device=device, dtype=dtype).view(token_ids.shape)

    if isinstance(token_ids, np.ndarray):
        flat = token_ids.reshape(-1)
        mapped = np.array([mapping.get(int(t), t) for t in flat], dtype=token_ids.dtype)
        return mapped.reshape(token_ids.shape)

    if isinstance(token_ids, list):
        return [map_token_ids(t, mapping) for t in token_ids]

    if isinstance(token_ids, (int, np.integer)):
        return mapping.get(int(token_ids), token_ids)

    raise TypeError(f"Unsupported token_ids type: {type(token_ids)}")


class TokenMappingTokenizer:
    """Wraps a HuggingFace tokenizer to remap token IDs to a compact range.

    For small models, the full 32K vocab embedding dominates parameter count.
    This filters to only tokens that appear in the dataset, reducing vocab to
    typically <100 tokens for synthetic tasks.
    """

    def __init__(self, tokenizer):
        self.old_tokenizer = tokenizer
        self.token_map = None
        self.map_new_to_old = None
        self.map_old_to_new = None
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def estimate_token_map_from_dataset(self, dataset):
        """Scan dataset to find all used tokens and build compact mapping."""
        max_seq_len = 0
        unique_tokens = set()
        unique_tokens.add(self.old_tokenizer.bos_token_id)
        unique_tokens.add(self.old_tokenizer.eos_token_id)
        if self.old_tokenizer.pad_token_id is not None:
            unique_tokens.add(self.old_tokenizer.pad_token_id)

        for text in tqdm(dataset, desc='Collecting tokens from dataset'):
            tokens = self.old_tokenizer(text)["input_ids"]
            max_seq_len = max(len(tokens), max_seq_len)
            for t in tokens:
                unique_tokens.add(t)

        unique_tokens = list(sorted(unique_tokens))
        self.map_new_to_old = {i: v for i, v in enumerate(unique_tokens)}
        self.map_old_to_new = {v: i for i, v in enumerate(unique_tokens)}

        self.bos_token_id = self.map_old_to_new[self.old_tokenizer.bos_token_id]
        self.eos_token_id = self.map_old_to_new[self.old_tokenizer.eos_token_id]
        if self.old_tokenizer.pad_token_id is not None:
            self.pad_token_id = self.map_old_to_new[self.old_tokenizer.pad_token_id]

        print(f'Reduced vocab size from {len(self.old_tokenizer)} to {len(self)}')
        print(f'Max seq_len is {max_seq_len}')

    def __call__(self, *args, **kwargs):
        out = self.old_tokenizer(*args, **kwargs)
        if "input_ids" in out:
            out["input_ids"] = map_token_ids(out["input_ids"], self.map_old_to_new)
        return out

    def decode(self, ids, **kwargs):
        old_ids = map_token_ids(ids, self.map_new_to_old)
        return self.old_tokenizer.decode(old_ids, **kwargs)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return [self.convert_ids_to_tokens(x) for x in ids]
        old_ids = map_token_ids(ids, self.map_new_to_old)
        return self.old_tokenizer.convert_ids_to_tokens(old_ids)

    def __len__(self):
        if self.map_new_to_old is None:
            return len(self.old_tokenizer)
        return len(self.map_new_to_old)


class CosineWarmupSchedule:
    """Cosine decay from 1.0 to 0.1 after linear warmup."""

    def __init__(self, total_steps, warmup_steps=1000):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
