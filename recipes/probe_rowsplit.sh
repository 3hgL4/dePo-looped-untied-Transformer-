#!/usr/bin/env bash
# Probes with the earlier split by query row (App. "Probes"), at H=4. Looped models are probed
# after every iteration, untied ones after every layer. Needs the training recipes.
#   clean_sweep_5seed_h4  untied 32-layer (seeds 42-46), looped 4x8 (42-46) and 2x16 (42, 43, 44, 46)
#                         at lr 1e-3, linear and MLP probes, plus an untrained 4x8 control; the looped
#                         runs also record the head's accuracy with the loop stopped after t iterations
#   aux_rerun_h4          the untied 32-layer model with intermediate supervision, the two seeds
#                         that solve the task (44, 45)
#   rebuttal/probe_o2x16_s*  looped 2x16, seeds 42-46, 3,000 prompts (the weak-middle-hops seed)
#   clean_sweep_5seed     looped 4x8 at H=8, seeds 42-45 (the retention paragraph of App. "Probes")
source "$(dirname "$0")/common.sh"
CK=$RUNS/h1_clean_sweep
OUT=$PROBES/clean_sweep_5seed_h4
ROW="$PROBE_ROW --H_eval 4 --n_samples 5000 --batch_size 32 --probe_types linear,mlp"
for seed in 42 43 44 45 46; do
  run $PY src/probe_h2.py --checkpoint $CK/deep-32_s$seed/checkpoint.pt --config_name deep-32 \
      --out_dir $OUT/deep-32_s$seed $ROW --sanity_min_acc 0.0
done
for r in ouro-4x8_s42 ouro-4x8_s43 ouro-4x8_s44 ouro-4x8_s45 ouro-4x8_s46 \
         ouro-2x16_s42 ouro-2x16_s43 ouro-2x16_s44 ouro-2x16_s46; do
  run $PY src/probe_h2.py --checkpoint $CK/$r/checkpoint.pt --config_name ${r%_s*} \
      --out_dir $OUT/$r $ROW --sanity_min_acc 0.6 --k_sweep
done
# Untrained control. A random-init probe reads no training args, so it runs with the probe's
# defaults (NoPE, a 512-token context); the paper's control was produced this way.
run $PY src/probe_h2.py --checkpoint $CK/ouro-4x8_s42/checkpoint.pt --config_name ouro-4x8 \
    --out_dir $OUT/ouro-4x8_random $ROW --sanity_min_acc 0.0 --random_init
for seed in 44 45; do
  run $PY src/probe_h2.py --checkpoint $CK/deep-32-aux_s$seed/checkpoint.pt --config_name deep-32 \
      --out_dir $PROBES/aux_rerun_h4/deep-32-aux_s$seed $PROBE_ROW --H_eval 4 --n_samples 5000 \
      --batch_size 64 --probe_types linear,mlp --sanity_min_acc 0.6
done
for seed in 42 43 44 45 46; do
  run $PY src/probe_h2.py --checkpoint $CK/ouro-2x16_s$seed/checkpoint.pt --config_name ouro-2x16 \
      --out_dir $PROBES/rebuttal/probe_o2x16_s$seed $PROBE_ROW --H_eval 4 --n_samples 3000 \
      --batch_size 64 --probe_types linear --sanity_min_acc 0.0
done
for seed in 42 43 44 45; do
  run $PY src/probe_h2.py --checkpoint $CK/ouro-4x8_s$seed/checkpoint.pt --config_name ouro-4x8 \
      --out_dir $PROBES/clean_sweep_5seed/ouro-4x8_s$seed $PROBE_ROW --H_eval 8 --n_samples 5000 \
      --batch_size 32 --probe_types linear,mlp --sanity_min_acc 0.6
done
