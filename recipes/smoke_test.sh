#!/usr/bin/env bash
# End-to-end check of every code path on a few training steps, on CPU in about five minutes.
# Trains tiny runs of each recipe family (looped, untied, intermediate supervision, no tau_t,
# N=32), then runs both probe protocols, the full-answer scorer, the full-node lens, activation
# patching (redirect, compose and an untrained control) and the forward-walk probe on the looped
# checkpoint. Writes to SMOKE_DIR (default: a fresh temporary directory); nothing it produces is
# a result.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=""  # CPU only
OUT="${SMOKE_DIR:-$(if [ -n "${DRY_RUN:-}" ]; then echo SMOKE_DIR; else mktemp -d; fi)}"
CK=$OUT/runs
P=$OUT/probes
echo "smoke test outputs: $OUT"
S=4  # training steps per run
TINY="--N 16 --M 5 --H_min 1 --H_max 15 --total_batch 15 --max_len 200 --wd 0.01 --warmup_steps 1000 \
--grad_clip 1.0 --total_steps $S --curriculum_floor_steps 7000 --curriculum_ramp_steps 23000 \
--eval_interval $S --eval_n_samples 30 --log_interval 2 --ckpt_interval $S --lr 1e-3 --device cpu"
TINY_N32="${TINY/--N 16/--N 32} --max_len 384"

run $PY src/train_uniform_h.py --config_name ouro-4x8 --seed 42 $TINY --zero_init --save_dir $CK/ouro-4x8_s42
run $PY src/train_uniform_h.py --config_name deep-4 --seed 42 $TINY --no_zero_init --save_dir $CK/deep-4_s42
run $PY src/train_uniform_h.py --config_name deep-4 --seed 42 $TINY --no_zero_init \
    --aux_deep_supervision --aux_layers 1,2,3 --aux_weight 0.3 --save_dir $CK/deep-4-aux_s42
run $PY src/train_uniform_h.py --config_name ouro-2x2 --seed 42 $TINY --zero_init --no_timestep_embedding \
    --save_dir $CK/ouro-2x2-notau_s42
run $PY src/train_uniform_h.py --config_name deep-4 --seed 42 $TINY_N32 --no_zero_init \
    --warmup_steps 2000 --lr 3e-4 --save_dir $CK/deep-4_n32_s42
# A resumed run continues from its checkpoint and replays the data rng.
run $PY src/train_uniform_h.py --config_name deep-4 --seed 42 ${TINY/--total_steps $S/--total_steps $((S + 2))} \
    --no_zero_init --resume --save_dir $CK/deep-4_s42

C=$CK/ouro-4x8_s42/checkpoint.pt
SMALL="--N 16 --M 5 --H_eval 2 --n_samples 40 --eval_seed 12345 --device cpu"
run $PY src/probe_h2.py --checkpoint $C --config_name ouro-4x8 --out_dir $P/prompt_disjoint \
    ${PROBE/--n_samples 5000/} ${SMALL} --sanity_min_acc 0
run $PY src/probe_h2.py --checkpoint $CK/deep-4_s42/checkpoint.pt --config_name deep-4 --out_dir $P/rowsplit \
    $PROBE_ROW $SMALL --batch_size 32 --probe_types linear,mlp --sanity_min_acc 0
run $PY src/probe_h2.py --checkpoint $C --config_name ouro-4x8 --out_dir $P/rowsplit_ksweep \
    $PROBE_ROW $SMALL --batch_size 32 --probe_types linear --sanity_min_acc 0 --k_sweep
run $PY experiments/h2_probing/scripts/eval_complete_answer.py --checkpoint $C $CK/deep-4_s42/checkpoint.pt \
    --out_dir $P/complete_answer --H_eval 2 --n_samples 20 --device cpu
LENS_SMALL="${SMALL/--n_samples 40/--n_samples 10}"
run $PY -m src.full_node_lens --checkpoint $C --config_name ouro-4x8 --out_dir $P/lens $LENS_SMALL
run $PY -m src.full_node_lens --checkpoint $C --config_name ouro-4x8 --out_dir $P/lens_randinit $LENS_SMALL --random_init
for mode in redirect compose; do
  run $PY -m src.intervention_h2 --checkpoint $C --config_name ouro-4x8 --out_dir $P/patch_$mode \
      --N 16 --M 5 --H 3 --n_trials 2 --mode $mode --patch_answer_digit --device cpu
done
run $PY -m src.intervention_h2 --checkpoint $C --config_name ouro-4x8 --out_dir $P/patch_redirect_randinit \
    --N 16 --M 5 --H 3 --n_trials 2 --mode redirect --patch_answer_digit --random_init --device cpu
run $PY -m src.directional_probe_h2 --checkpoint $C --config_name ouro-4x8 --out_dir $P/directional \
    --N 16 --M 5 --H_eval 4 --n_samples 40 --device cpu

for f in $CK/ouro-4x8_s42/trajectory.jsonl $CK/deep-4-aux_s42/trajectory.jsonl $CK/ouro-2x2-notau_s42/trajectory.jsonl \
         $CK/deep-4_n32_s42/trajectory.jsonl $P/prompt_disjoint/probe_results.json $P/prompt_disjoint/model_complete_answer.json \
         $P/rowsplit/probe_results.json $P/rowsplit_ksweep/probe_results.json $P/complete_answer/summary.md \
         $P/lens/full_node_lens_summary.json $P/lens_randinit/full_node_lens_summary.json \
         $P/patch_redirect/intervention_summary.json $P/patch_compose/intervention_summary.json \
         $P/patch_redirect_randinit/intervention_summary.json $P/directional/directional_probe_summary.json; do
  [ -n "${DRY_RUN:-}" ] || [ -s "$f" ] || { echo "smoke test FAILED: missing $f"; exit 1; }
done
[ -n "${DRY_RUN:-}" ] || grep -q '"step": 6' $CK/deep-4_s42/trajectory.jsonl || { echo "smoke test FAILED: resume"; exit 1; }
echo "smoke test passed"
