"""Memory/time probe for the ouro loop depth question (2026-10-26).

Question: is the ouro-4x10 (K_max=10) run slow because it does not fit in 12 GB
(allocator thrash) or because the extra iterations simply cost more compute?

Measures, for a few (config, batch, max_len) combos:
  * peak allocated / reserved VRAM over one forward+backward+step
  * wall time per optimizer step (after one warm-up)
No data generation involved: random token ids, same shapes/dtypes as the trainer
(bf16 autocast, vocab 18, pad 1).

Usage: .venv/bin/python experiments/bench_review/mem_probe.py
"""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model_configs import create_model  # noqa: E402

CASES = [
    ("ouro-4x8", 120, 200),
    ("ouro-4x10", 120, 200),
    ("ouro-4x10", 120, 176),
    ("ouro-4x10", 60, 200),
]
STEPS = 4  # 1 warm-up + 3 timed


def one_case(name, batch, max_len, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, cfg = create_model(name, vocab_size=18, max_sequence_len=max_len,
                              pad_token_idx=1)
    model = model.to(device).train()
    K = cfg.get("K_max") or cfg.get("K")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    toks = torch.randint(2, 18, (batch, max_len), device=device)
    mask = torch.ones_like(toks, dtype=torch.bool)

    times = []
    for i in range(STEPS):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(toks, mask, K=K) if K else model(toks, mask)
        logits = logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 18),
                               toks[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    timed = times[1:]
    print(f"{name:>10} batch={batch:>3} len={max_len:>3} K={K:>2} | "
          f"step={sum(timed) / len(timed):6.2f}s | "
          f"peak_alloc={torch.cuda.max_memory_allocated() / 2**30:5.2f}GiB "
          f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:5.2f}GiB",
          flush=True)
    del model, opt, toks, mask
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)} "
          f"total={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}GiB",
          flush=True)
    # Optional: pass a case index (0..3) to run exactly one case in a fresh
    # process. Needed because a case that does not fit leaves the allocator in a
    # state that makes every later case meaningless (and can hang for minutes).
    sel = [int(a) for a in sys.argv[1:] if a.isdigit()]
    cases = [CASES[i] for i in sel] if sel else CASES
    for case in cases:
        try:
            one_case(*case)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{case}: OOM -> {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
