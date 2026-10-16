#!/usr/bin/env bash
# Iteration-embedding ablation (App. "controls"): the looped 4x8 model at lr 1e-3 without the
# iteration embedding tau_t. Seeds 42-45.
source "$(dirname "$0")/common.sh"
OUT=$RUNS/h1_clean_sweep
for seed in 42 43 44 45; do
  run $PY src/train_uniform_h.py --config_name ouro-4x8 --seed $seed $RECIPE --lr 1e-3 \
      --zero_init --no_timestep_embedding --save_dir $OUT/ouro-4x8-notau_s$seed
done
