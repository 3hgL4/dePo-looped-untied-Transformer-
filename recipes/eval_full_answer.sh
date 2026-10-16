#!/usr/bin/env bash
# Full-answer accuracy at H=4 of the checkpoints probed with the row split, after every iteration
# count of the looped models (the head readout in the dynamics figure and the accuracies quoted
# beside the probe maps). Needs the training recipes.
source "$(dirname "$0")/common.sh"
CK=$RUNS/h1_clean_sweep
RUNS_H4=""
for seed in 42 43 44 45 46; do RUNS_H4="$RUNS_H4 deep-32_s$seed ouro-4x8_s$seed ouro-2x16_s$seed"; done
RUNS_H4="$RUNS_H4 deep-32-aux_s44 deep-32-aux_s45 deep-32-lr3e4_s42"
for r in $RUNS_H4; do
  run $PY experiments/h2_probing/scripts/eval_complete_answer.py --checkpoint $CK/$r/checkpoint.pt \
      --out_dir $PROBES/complete_answer_20261009 --H_eval 4 --n_samples 5000 --eval_seed 12345 --batch_size 32
done
