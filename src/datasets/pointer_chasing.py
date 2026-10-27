"""
Directed permutation traversal (dePo): the pointer-chasing task of the paper.

A random single N-cycle pi is shown as a shuffled list of directed edges, followed by
M queries that share the edge list and the hop count H. Each query asks for pi^H(s),
the node H steps along the cycle from its start node s.

Format: "0 > 3 , 3 > 7 , 7 > 1 , ... | hop H s ? answer | hop H s' ? answer' | ..."

`node_format` (addition 2026-10-26, P5 in
notes/2026-10-26_基准测量缺陷与改进提案.md) changes only how a node id is
WRITTEN; the sampled cycle, the queries and the metadata stay identical. "plain"
(default) reproduces the paper's bytes exactly; "letters" (A..P) gives 16 answers
whose first token is unique, which removes the 7/16 first-token floor of the lens
readout. Switching a run to "letters" makes it a new benchmark: its numbers
cannot be mixed with the decimal runs.
"""

import random

from src.node_tokens import node_symbols


def generate_pointer_chasing_sample(
    rng: random.Random,
    N: int,
    H: int,
    M: int = 5,
    shuffle_edges: bool = True,
    node_format: str = "plain",
) -> tuple[str, str, dict]:
    """Generate one pointer chasing sample with M queries.

    Args:
        rng: seeded Random instance
        N: number of nodes (permutation size)
        H: exact number of hops (sequential depth)
        M: number of queries per sample (more = stronger training signal)
        shuffle_edges: randomize edge presentation order
        node_format: how a node id is written: "plain" = "0".."15" (the
            paper's), "padded" = "00".."15", "letters" = "A".."P". Only the
            text changes; metadata always keeps integer node ids.

    Returns:
        (text, answer, metadata) where:
        - text: full input string including all queries and answers
        - answer: the last answer (for eval compatibility)
        - metadata: dict with all query results for per-hop eval
    """
    labels = node_symbols(N, node_format) if node_format != "plain" else None

    def label(node: int) -> str:
        return labels[node] if labels is not None else str(node)

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

    edges_str = " , ".join(f"{label(s)} > {label(d)}" for s, d in edges)

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
        queries.append(f"hop {H} {label(start_node)} ? {label(answer_node)}")
        answer_str_for_eval = label(answer_node)
        all_answers.append({"start": start_node, "answer": answer_node,
                           "answer_str": answer_str_for_eval,
                           "intermediates": intermediates,
                           "start_idx": start_idx})

    # Format: "edges | query1 | query2 | ..."
    text = edges_str + " | " + " | ".join(queries)
    # For eval compatibility, return last answer as the primary answer
    answer = all_answers[-1]["answer_str"]

    metadata = {
        "H": H,
        "N": N,
        "M": M,
        "queries": all_answers,
        # Full cycle order for this sample (all queries share it). The backward
        # analysis needs order + per-query start_idx to construct pi^-k(s).
        "order": order,
    }

    return text, answer, metadata


def demo():
    """Print a few samples for inspection."""
    rng = random.Random(42)
    for H in [1, 2, 4, 8]:
        text, answer, meta = generate_pointer_chasing_sample(rng, N=8, H=H, M=2)
        print(f"H={H}: {text}")
        print(f"  answer of the last query: {answer}")
        print(f"  chain of the first query: {' -> '.join(map(str, meta['queries'][0]['intermediates']))}")


if __name__ == "__main__":
    demo()
