#!/usr/bin/env bash
# Prompt-disjoint linear probes (Methods, "Linear probes"), one probe after every layer, with
# full-answer scoring. Needs the training recipes.
#   matched_lr3e4_20260915   looped 4x8 and untied 32-layer at lr 3e-4, seeds 42-44, H=4 and H=8
#   geometry_lr1e3_20260920  looped 4x8 and 2x16 at lr 1e-3, seed 42, H=4 and H=8
#   fig5_h8_lr1e3_20261009   looped 2x16 (seeds 43, 44, 46), 4x8 and 8x4 (seeds 42-46) at H=8
#   notau_matched_protocol_20260920  looped 4x8 without tau_t, seeds 42-45, H=4 and H=8
source "$(dirname "$0")/common.sh"
CK=$RUNS/h1_clean_sweep
probe() {  # run config out_dir H
  run $PY src/probe_h2.py --checkpoint $CK/$1/checkpoint.pt --config_name $2 --out_dir $3 $PROBE --H_eval $4
}
for config in ouro-4x8 deep-32; do
  for seed in 42 43 44; do
    for H in 4 8; do
      probe $config-lr3e4_s$seed $config $PROBES/matched_lr3e4_20260915/results/H$H/$config-lr3e4_s$seed $H
    done
  done
done
for H in 4 8; do
  for config in ouro-4x8 ouro-2x16; do
    probe ${config}_s42 $config $PROBES/geometry_lr1e3_20260920/results/H$H/${config}_s42 $H
  done
done
for r in ouro-2x16_s43 ouro-2x16_s44 ouro-2x16_s46 \
         ouro-4x8_s42 ouro-4x8_s43 ouro-4x8_s44 ouro-4x8_s45 ouro-4x8_s46 \
         ouro-8x4_s42 ouro-8x4_s43 ouro-8x4_s44 ouro-8x4_s45 ouro-8x4_s46; do
  probe $r ${r%_s*} $PROBES/fig5_h8_lr1e3_20261009/${r}_H8 8
done
for seed in 42 43 44 45; do
  for H in 4 8; do
    probe ouro-4x8-notau_s$seed ouro-4x8 $PROBES/notau_matched_protocol_20260920/results/H$H/ouro-4x8-notau_s$seed $H
  done
done
