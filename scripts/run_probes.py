"""
Probe + lens orchestrator: launches `looped_lm.probe_h2` (and optionally
`looped_lm.logit_lens_h2`) over a list of trained checkpoints.

Each target name has form `<config>_s<seed>` (e.g. `ouro-2x16_s42`); the
checkpoint is expected at:
    <runs_root>/<target>/checkpoint.pt

Per-target H_eval and bookkeeping tags live in `TARGETS` below — edit to
match your sweep.

Usage:
    uv run python scripts/run_probes.py \\
        --runs_root runs/clean_sweep \\
        --out_root  results/probes \\
        --gpu 0 \\
        --models ouro-4x8_s42 ouro-2x16_s42 ouro-2x16_s45 deep-32_s42

Run multiple invocations on disjoint --models / different --gpu values for
parallelism.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

# config_name (architecture key) is the cell prefix; checkpoint lives under
# <runs_root>/<target>/checkpoint.pt
TARGETS = {
    # Headline: matched D_eff=32 saturated-vs-failed comparison
    "ouro-2x16_s42": {"config": "ouro-2x16", "H_eval": 8, "tag": "sat"},
    "ouro-2x16_s45": {"config": "ouro-2x16", "H_eval": 8, "tag": "fail"},
    # Replication of prior staircase finding under uniform-H training
    "ouro-4x8_s42":  {"config": "ouro-4x8",  "H_eval": 8, "tag": "sat"},
    # Second saturated-vs-failed pair at lower D_eff
    "ouro-4x4_s43":  {"config": "ouro-4x4",  "H_eval": 5, "tag": "sat"},
    "ouro-4x4_s42":  {"config": "ouro-4x4",  "H_eval": 5, "tag": "fail"},
    # Matched-D_eff=32 unlooped control (expected: no staircase)
    "deep-32_s42":   {"config": "deep-32",   "H_eval": 4, "tag": "ctrl"},
}


def run_probe(target: str, gpu: int, n_samples: int, runs_root: Path,
              out_root: Path, k_sweep: bool) -> int:
    cfg = TARGETS[target]
    ckpt = runs_root / target / "checkpoint.pt"
    if not ckpt.exists():
        print(f"[orch] MISSING checkpoint: {ckpt}")
        return 1

    out_dir = out_root / target
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "-m", "looped_lm.probe_h2",
        "--checkpoint", str(ckpt),
        "--config_name", cfg["config"],
        "--out_dir", str(out_dir),
        "--H_eval", str(cfg["H_eval"]),
        "--n_samples", str(n_samples),
    ]
    if k_sweep:
        cmd.append("--k_sweep")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path = out_dir / "probe.log"
    print(f"[orch] {target} ({cfg['tag']}) -> {out_dir} gpu={gpu} log={log_path}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[orch] {target} FAILED rc={proc.returncode}. Tail of log:")
        try:
            print(open(log_path).read()[-2000:])
        except Exception:
            pass
    else:
        print(f"[orch] {target} OK")
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True, type=Path,
                   help="Directory containing per-target checkpoint folders (<runs_root>/<target>/checkpoint.pt).")
    p.add_argument("--out_root", required=True, type=Path,
                   help="Output directory for probe results.")
    p.add_argument("--models", nargs="+", default=list(TARGETS.keys()),
                   help="Targets, e.g. ouro-2x16_s42 ouro-2x16_s45")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--k_sweep", action="store_true",
                   help="Also run K-sweep model-head accuracy (commitment-deferral test).")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    rc_total = 0
    for t in args.models:
        if t not in TARGETS:
            print(f"[orch] unknown target {t}, skipping")
            continue
        rc_total += run_probe(t, args.gpu, args.n_samples, args.runs_root,
                              args.out_root, args.k_sweep)
    raise SystemExit(0 if rc_total == 0 else 1)


if __name__ == "__main__":
    main()
