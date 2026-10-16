#!/usr/bin/env bash
# Activation patching (Methods, "Logit lens and activation patching"; App. "controls"): after
# iteration j the receiver's states at its readout position and its first answer digit are
# replaced with the donor's. redirect: the donor shares the permutation and starts elsewhere;
# compose: the donor has another permutation and the same start. Full-answer scoring.
#   looped 4x8, H=6, 400 trials, seeds 42-46;  looped 2x16, H=8, 300 trials, seeds 42, 43, 44, 46
#   untrained controls with the architecture flags of seed 42 of each model
# Needs train_grid.sh.
source "$(dirname "$0")/common.sh"
CK=$RUNS/h1_clean_sweep
OUT=$PROBES/patching_fullanswer_digit_20261010
patch() {  # name config run H n_trials mode [extra]
  run $PY -m src.intervention_h2 --checkpoint $CK/$3/checkpoint.pt --config_name $2 --out_dir $OUT/$1 \
      --N 16 --M 5 --H $4 --n_trials $5 --mode $6 --eval_seed 20260727 --patch_answer_digit ${7:-}
}
for seed in 42 43 44 45 46; do
  for mode in compose redirect; do patch o4x8_s${seed}_$mode ouro-4x8 ouro-4x8_s$seed 6 400 $mode; done
done
for mode in compose redirect; do patch o4x8_s42_${mode}_randinit ouro-4x8 ouro-4x8_s42 6 400 $mode --random_init; done
for seed in 42 43 44 46; do
  for mode in compose redirect; do patch o2x16_s${seed}_$mode ouro-2x16 ouro-2x16_s$seed 8 300 $mode; done
done
for mode in compose redirect; do patch o2x16_s42_${mode}_randinit ouro-2x16 ouro-2x16_s42 8 300 $mode --random_init; done
