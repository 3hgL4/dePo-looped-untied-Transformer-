#!/usr/bin/env bash
# Intermediate supervision (Table "capability-rates", auxiliary-loss row): the untied 32-layer
# model at lr 1e-3 with the next-token loss added at layers 8, 16 and 24, read out through the
# shared final norm and unembedding with total weight 0.3. Seeds 42-46.
source "$(dirname "$0")/common.sh"
OUT=$RUNS/h1_clean_sweep
for seed in 42 43 44 45 46; do
  run $PY src/train_uniform_h.py --config_name deep-32 --seed $seed $RECIPE --lr 1e-3 \
      --aux_deep_supervision --aux_layers 8,16,24 --aux_weight 0.3 --no_zero_init \
      --save_dir $OUT/deep-32-aux_s$seed
done
