"""A resumed run must see the batches an uninterrupted run would have seen.

The trainer's data rng is not checkpointed; on --resume it is replayed through the resumed step
(replay_data_rng). This checks that the replayed stream continues exactly where an uninterrupted
stream is, across a curriculum ramp, and that a stream without the replay does not.

Run: uv run python tests/test_resume_data_rng.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_uniform_h import replay_data_rng, sample_stratified_texts

SEED, RESUME, TOTAL = 42, 37, 60
KW = dict(N=16, H_min=1, total_batch=120, M=5)  # the paper recipe


def ceiling(step: int) -> int:
    """A ramp that crosses the resume step, like the trainer's curriculum at a smaller scale."""
    return 1 if step <= 10 else min(15, 1 + (step - 10) // 3)


def batches(rng: random.Random, first: int, last: int) -> list:
    return [sample_stratified_texts(rng, H_max_current=ceiling(s), **KW) for s in range(first, last + 1)]


def test_replay_matches_uninterrupted() -> None:
    uninterrupted = batches(random.Random(SEED), 1, TOTAL)
    rng = random.Random(SEED)
    replay_data_rng(rng, RESUME, ceiling, **KW)
    assert batches(rng, RESUME + 1, TOTAL) == uninterrupted[RESUME:]


def test_without_replay_differs() -> None:
    uninterrupted = batches(random.Random(SEED), 1, TOTAL)
    assert batches(random.Random(SEED), RESUME + 1, TOTAL) != uninterrupted[RESUME:]


if __name__ == "__main__":
    test_replay_matches_uninterrupted()
    test_without_replay_differs()
    print("ok")
