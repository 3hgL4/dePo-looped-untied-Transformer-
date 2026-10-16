#!/usr/bin/env bash
# Learning-rate controls (Table "capability-rates"): the looped 4x8 and the untied 32-layer model
# at lr 3e-4 and 1e-4, and the untied 16-layer model at lr 3e-4, seeds 42-46. The lr 1e-3 runs of
# the same models are part of train_grid.sh.
source "$(dirname "$0")/common.sh"
OUT=$RUNS/h1_clean_sweep
for config in ouro-4x8 deep-32; do
  for lr in 3e-4 1e-4; do
    for seed in 42 43 44 45 46; do
      run $PY src/train_uniform_h.py --config_name $config --seed $seed $RECIPE --lr $lr \
          $(zero_init_flag $config) --save_dir $OUT/$config-lr${lr/-/}_s$seed
    done
  done
done
for seed in 42 43 44 45 46; do
  run $PY src/train_uniform_h.py --config_name deep-16 --seed $seed $RECIPE --lr 3e-4 \
      --no_zero_init --save_dir $OUT/deep-16-lr3e4_s$seed
done
