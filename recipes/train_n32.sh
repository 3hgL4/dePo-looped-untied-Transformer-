#!/usr/bin/env bash
# Longer chains at N=32 (Table "capability-rates", App. "controls"): lr 3e-4, 384-token context,
# 2,000 warm-up steps. The looped 4x8 and untied 32-layer models with seeds 42-44, and the untied
# 4-layer control with seeds 42, 43, 46 and 47.
source "$(dirname "$0")/common.sh"
OUT=$RUNS/h1_clean_sweep
for config in ouro-4x8 deep-32; do
  for seed in 42 43 44; do
    run $PY src/train_uniform_h.py --config_name $config --seed $seed $RECIPE_N32 --lr 3e-4 \
        $(zero_init_flag $config) --save_dir $OUT/${config}_n32_s$seed
  done
done
for seed in 42 43 46 47; do
  run $PY src/train_uniform_h.py --config_name deep-4 --seed $seed $RECIPE_N32 --lr 3e-4 \
      --no_zero_init --save_dir $OUT/deep-4_n32_s$seed
done
