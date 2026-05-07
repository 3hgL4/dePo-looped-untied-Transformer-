"""
Pointer-chasing dataset (a.k.a. dePo).

Given a random permutation of N nodes as directed edges, follow the chain
for exactly H hops from a query node to produce the answer node.

Conventions used here:
- Fixed H per dataset instance (not random uniform)
- Fixed N for all instances (constant input length across H values)
- Single query per sample (M=1) for clean per-sample accuracy
- Shuffled edges by default (prevents positional shortcuts)
- Online generation (fresh random permutation each sample/epoch)

Format: "0 > 3 , 3 > 7 , 7 > 1 , ... | hop H start ? answer"
"""

import random

from looped_lm.datasets.base import OnlineSyntheticDataset


def generate_pointer_chasing_sample(
    rng: random.Random,
    N: int,
    H: int,
    M: int = 5,
    shuffle_edges: bool = True,
) -> tuple[str, str, dict]:
    """Generate one pointer chasing sample with M queries.

    Args:
        rng: seeded Random instance
        N: number of nodes (permutation size)
        H: exact number of hops (sequential depth)
        M: number of queries per sample (more = stronger training signal)
        shuffle_edges: randomize edge presentation order

    Returns:
        (text, answer, metadata) where:
        - text: full input string including all queries and answers
        - answer: the last answer (for eval compatibility)
        - metadata: dict with all query results for per-hop eval
    """
    # Random circular permutation: order[i] -> order[(i+1) % N]
    order = rng.sample(range(N), N)

    # Build edges: "src > dst" pairs
    edges = []
    for i in range(N):
        src = order[i]
        dst = order[(i + 1) % N]
        edges.append((src, dst))

    if shuffle_edges:
        rng.shuffle(edges)

    edges_str = " , ".join(f"{s} > {d}" for s, d in edges)

    # Generate M queries, all at hop count H
    queries = []
    all_answers = []
    for _ in range(M):
        start_idx = rng.randint(0, N - 1)
        start_node = order[start_idx]

        # Trace the chain
        intermediates = [start_node]
        for step in range(H):
            next_idx = (start_idx + step + 1) % N
            intermediates.append(order[next_idx])

        answer_node = intermediates[-1]
        queries.append(f"hop {H} {start_node} ? {answer_node}")
        all_answers.append({"start": start_node, "answer": answer_node,
                           "answer_str": str(answer_node),
                           "intermediates": intermediates})

    # Format: "edges | query1 | query2 | ..."
    text = edges_str + " | " + " | ".join(queries)
    # For eval compatibility, return last answer as the primary answer
    answer = all_answers[-1]["answer_str"]

    metadata = {
        "H": H,
        "N": N,
        "M": M,
        "queries": all_answers,
    }

    return text, answer, metadata


class PointerChasingDataset(OnlineSyntheticDataset):
    """Online pointer chasing dataset with fixed hop count H.

    Each epoch generates fresh random permutations. H and N are fixed
    for the lifetime of the dataset instance.
    """

    def __init__(
        self,
        N: int,
        H: int,
        tokenizer,
        max_length: int,
        n_samples: int = 10000,
        base_seed: int = 42,
        shuffle_edges: bool = True,
        M: int = 5,
    ):
        self.N = N
        self.H = H
        self.M = M
        self.shuffle_edges = shuffle_edges
        super().__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            n_samples=n_samples,
            base_seed=base_seed,
        )

    def _generate_samples(self, seed: int) -> list[tuple[str, str, dict]]:
        rng = random.Random(seed)
        samples = []
        for _ in range(self.n_samples):
            sample = generate_pointer_chasing_sample(
                rng=rng,
                N=self.N,
                H=self.H,
                M=self.M,
                shuffle_edges=self.shuffle_edges,
            )
            samples.append(sample)
        return samples


def generate_curriculum_batch(
    rng: random.Random,
    N: int,
    current_max_hop: int,
    M: int,
    batch_size: int,
    tokenizer,
    max_length: int,
) -> dict:
    """Generate a training batch with hop counts sampled from {1,...,current_max_hop}.

    Each sample in the batch gets a hop count sampled uniformly, then M queries
    at that hop count on a fresh random permutation.

    Args:
        rng: seeded Random instance
        N: number of nodes
        current_max_hop: curriculum frontier (sample H from 1..current_max_hop)
        M: queries per sample
        batch_size: number of samples in batch
        tokenizer: for tokenization
        max_length: max sequence length for padding

    Returns:
        dict with 'tokens' (batch_size, max_length), 'attention_mask' (batch_size, max_length),
        'hop_counts' (list of H per sample)
    """
    import torch

    texts = []
    hop_counts = []
    answers = []

    for _ in range(batch_size):
        H = rng.randint(1, current_max_hop)
        text, answer, _ = generate_pointer_chasing_sample(
            rng=rng, N=N, H=H, M=M, shuffle_edges=True,
        )
        texts.append(text)
        hop_counts.append(H)
        answers.append(answer)

    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )

    return {
        "tokens": out["input_ids"].long(),
        "attention_mask": out["attention_mask"].to(torch.bool),
        "hop_counts": hop_counts,
        "answers": answers,
        "texts": texts,
    }


def demo():
    """Print sample pointer chasing instances for inspection."""
    rng = random.Random(42)

    print("=== Pointer Chasing Samples ===\n")
    for H in [1, 2, 4, 8]:
        text, answer, meta = generate_pointer_chasing_sample(rng, N=8, H=H)
        print(f"H={H}: {text}")
        print(f"  Answer: {answer}")
        print(f"  Chain: {' -> '.join(str(x) for x in meta['intermediates'])}")
        print()

    # Show sequence lengths for typical configs
    print("=== Sequence Length Analysis ===")
    for N, H in [(16, 4), (32, 8), (32, 16), (64, 16)]:
        texts = []
        for _ in range(100):
            text, _, _ = generate_pointer_chasing_sample(rng, N=N, H=H)
            texts.append(text)
        lengths = [len(t) for t in texts]
        print(
            f"N={N}, H={H}: "
            f"mean={sum(lengths)/len(lengths):.0f}, "
            f"max={max(lengths)}, "
            f"min={min(lengths)} chars"
        )


if __name__ == "__main__":
    demo()
