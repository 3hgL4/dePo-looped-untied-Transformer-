#!/usr/bin/env bash
# Full-node logit lens (Methods, "Logit lens and activation patching"): the probability that the
# model writes each node of the chain when its loop is stopped after t iterations. Looped 2x16,
# 4x8 and 8x4 at H=8 and 2x16 and 4x8 at H=4, seeds 42-46, plus untrained controls with the
# architecture flags of seed 42. Needs train_grid.sh.
source "$(dirname "$0")/common.sh"
CK=$RUNS/h1_clean_sweep
OUT=$PROBES/full_node_lens_20261010
LENS="--N 16 --M 5 --n_samples 5000 --eval_seed 12345 --batch_size 512"
for config in ouro-2x16 ouro-4x8 ouro-8x4; do
  for seed in 42 43 44 45 46; do
    run $PY -m src.full_node_lens --checkpoint $CK/${config}_s$seed/checkpoint.pt --config_name $config \
        --out_dir $OUT/${config}_s${seed}_H8 $LENS --H_eval 8
  done
done
for config in ouro-2x16 ouro-4x8; do
  for seed in 42 43 44 45 46; do
    run $PY -m src.full_node_lens --checkpoint $CK/${config}_s$seed/checkpoint.pt --config_name $config \
        --out_dir $OUT/${config}_s${seed}_H4 $LENS --H_eval 4
  done
done
for config in ouro-2x16 ouro-4x8 ouro-8x4; do
  run $PY -m src.full_node_lens --checkpoint $CK/${config}_s42/checkpoint.pt --config_name $config \
      --out_dir $OUT/${config}_randinit_H8 $LENS --H_eval 8 --random_init
done
