"""
Base class for online synthetic datasets.

All datasets generate fresh random instances each epoch to prevent memorization.
Background thread pre-generates next epoch's data for zero-wait epoch transitions.
"""

import threading

import torch
from torch.utils.data import Dataset


class OnlineSyntheticDataset(Dataset):
    """Base class for online-generated synthetic datasets.

    Subclasses implement _generate_samples(seed) to produce a list of
    (text_string, answer_string, metadata_dict) tuples.

    Features:
    - Fresh random instances each epoch (deterministic via seed)
    - Background thread pre-generates next epoch's data
    - Pre-tokenized and padded for fast training
    """

    def __init__(self, tokenizer, max_length: int, n_samples: int, base_seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_samples = n_samples
        self.base_seed = base_seed
        self.epoch = 0

        # Populated by _tokenize_and_store
        self.all_tokens = None
        self.all_attention_masks = None
        self.samples_text = []
        self.samples_answers = []
        self.samples_metadata = []

        # Background pre-generation state
        self._next_data = None
        self._bg_thread = None

        # Generate initial data
        self._generate_and_tokenize(self.base_seed)
        # Pre-generate next epoch in background
        self._start_background_generate(self.epoch + 1)

    def _generate_samples(self, seed: int) -> list[tuple[str, str, dict]]:
        """Generate n_samples (text, answer, metadata) tuples. Override in subclass."""
        raise NotImplementedError

    def _generate_and_tokenize(self, seed: int):
        """Generate samples and tokenize them."""
        samples = self._generate_samples(seed)
        self.samples_text = [s[0] for s in samples]
        self.samples_answers = [s[1] for s in samples]
        self.samples_metadata = [s[2] for s in samples]

        out = self.tokenizer(
            self.samples_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        self.all_tokens = out["input_ids"].long()
        self.all_attention_masks = out["attention_mask"].to(torch.bool)

    def _start_background_generate(self, epoch: int):
        """Pre-generate data for the given epoch in a background thread."""
        def _bg():
            seed = self.base_seed + epoch
            samples = self._generate_samples(seed)
            texts = [s[0] for s in samples]
            answers = [s[1] for s in samples]
            metadata = [s[2] for s in samples]
            out = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True,
            )
            self._next_data = {
                "tokens": out["input_ids"].long(),
                "masks": out["attention_mask"].to(torch.bool),
                "texts": texts,
                "answers": answers,
                "metadata": metadata,
            }

        self._bg_thread = threading.Thread(target=_bg, daemon=True)
        self._bg_thread.start()

    def set_epoch(self, epoch: int):
        """Switch to data for the given epoch."""
        self.epoch = epoch

        # Wait for background generation
        if self._bg_thread is not None:
            self._bg_thread.join()
            self._bg_thread = None

        # Swap in pre-generated data if available
        if self._next_data is not None:
            self.all_tokens = self._next_data["tokens"]
            self.all_attention_masks = self._next_data["masks"]
            self.samples_text = self._next_data["texts"]
            self.samples_answers = self._next_data["answers"]
            self.samples_metadata = self._next_data["metadata"]
            self._next_data = None
        else:
            self._generate_and_tokenize(self.base_seed + epoch)

        # Pre-generate next epoch
        self._start_background_generate(epoch + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "tokens": self.all_tokens[idx],
            "attention_mask": self.all_attention_masks[idx],
        }
