#!/usr/bin/env bash
# Forward-walk control (App. "controls"): probe every forward distance d along the cycle in the
# looped 2x16 model at H=12, where the backward walk (4 steps) is shorter than the forward one.
# Row split, 3,000 prompts. Needs train_grid.sh.
source "$(dirname "$0")/common.sh"
run $PY -m src.directional_probe_h2 --checkpoint $RUNS/h1_clean_sweep/ouro-2x16_s42/checkpoint.pt \
    --config_name ouro-2x16 --out_dir $PROBES/rebuttal/dirprobe_o2x16_h12 \
    --N 16 --M 5 --H_eval 12 --n_samples 3000 --eval_seed 12345 --train_frac 0.8
