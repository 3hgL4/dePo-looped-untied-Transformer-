#!/usr/bin/env bash
# The capability grid at lr 1e-3 (Figure "capability", the matched D_eff = 32 comparison and the
# training-loss appendix figure): every cell below with seeds 42-46.
#   untied:  deep-2, deep-4, deep-8, deep-16, deep-32
#   looped:  ouro-2x{2,4,5,6,7,8,10,12,14,16}, ouro-4x{2,3,4,6,8}, ouro-8x{2,3,4}, ouro-16x2
source "$(dirname "$0")/common.sh"
OUT=$RUNS/h1_clean_sweep
CELLS="deep-2 deep-4 deep-8 deep-16 deep-32"
for T in 2 4 5 6 7 8 10 12 14 16; do CELLS="$CELLS ouro-2x$T"; done
for T in 2 3 4 6 8; do CELLS="$CELLS ouro-4x$T"; done
for T in 2 3 4; do CELLS="$CELLS ouro-8x$T"; done
CELLS="$CELLS ouro-16x2"
for config in $CELLS; do
  for seed in 42 43 44 45 46; do
    run $PY src/train_uniform_h.py --config_name $config --seed $seed $RECIPE --lr 1e-3 \
        $(zero_init_flag $config) --save_dir $OUT/${config}_s$seed
  done
done
