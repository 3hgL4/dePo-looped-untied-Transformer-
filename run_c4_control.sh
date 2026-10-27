#!/usr/bin/env bash
# C4 matched control (mismatched-query pairing) on the EXISTING checkpoints.
#
# Why: the C4 headline "no composability evidence" could not be upgraded to
# "falsified" because there was no control for "any activation injected at that
# position changes the answer". The matched control injects the SAME donor
# residuals (same prompt, same readout-position class, same iteration j, same
# layer) but taken from a DIFFERENT query's chain, so they provably encode a
# different node. 2x2 readout (all vs the written node, chance 1/N = 0.0625):
#   ctrl_vs_ctrl  control act -> control target   mechanism predicts HIGH
#   real_vs_real  real act    -> real target      mechanism predicts HIGH
#   ctrl_vs_real  control act -> real target      leakage/position predicts HIGH
#   real_vs_ctrl  real act    -> control target   leakage/position predicts HIGH
#
# Step 1 is a regression: --control none at the same n_trials must reproduce
# the published k8_s42_H8 curves bit for bit, otherwise the refactor moved the
# real rows and nothing below can be compared to §13.5.
set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
DEVICE=${DEVICE:-cuda}
N_TRIALS=${N_TRIALS:-400}
OUT_ROOT=intervention_outputs/p3p5c4_control
LOG_DIR=runs/c4_control
mkdir -p "$OUT_ROOT" "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- step 1: regr
REGR_DIR=/tmp/c4_regr_none
if [ ! -f "$REGR_DIR/intervention_summary.json" ]; then
  log "=== REGRESSION: --control none on k8_s42_H8 (expect bit-identical) ==="
  $PY src/intervention_h2.py \
    --checkpoint runs/slice/ouro-4x8_s42/checkpoint.pt \
    --config_name ouro-4x8 --out_dir "$REGR_DIR" \
    --N 16 --M 5 --H 8 --n_trials "$N_TRIALS" \
    --eval_seed 20260727 --device "$DEVICE" >> "$LOG_DIR/regr.log" 2>&1
  echo "REGR rc=$?" >> "$LOG_DIR/regr.log"
else
  log "SKIP regression (already have $REGR_DIR)"
fi

$PY - <<'PYEOF' >> "$LOG" 2>&1
import json
a = json.load(open('intervention_outputs/p3p5c4/k8_s42_H8/intervention_summary.json'))
b = json.load(open('/tmp/c4_regr_none/intervention_summary.json'))
fa, fb = a['full_answer']['all'], b['full_answer']['all']
same = all(fa[k] == fb[k] for k in ('match_composed', 'match_composed_distinct', 'match_pure_A', 'match_pure_B'))
same &= a['match_composed_unambiguous'] == b['match_composed_unambiguous']
same &= fa['baseline_receiver_acc'] == fb['baseline_receiver_acc']
print('[regr] real-row curves identical to published run:', same)
if not same:
    print('[regr] MISMATCH — control runs below are not comparable to 13.5')
PYEOF

# ------------------------------------------------------- step 2: control cells
# Same 7 cells as the published C4 sweep, same seed, same H, same n_trials.
run_ctrl() {
  local name=$1 run=$2 cfg=$3 h=$4; shift 4
  local out="$OUT_ROOT/$name"
  if [ -f "$out/intervention_summary.json" ]; then
    log "SKIP  $name (done)"
    return 0
  fi
  log "START $name  ($run H=$h $*)"
  local t0=$(date +%s)
  $PY src/intervention_h2.py \
    --checkpoint "runs/slice/$run/checkpoint.pt" \
    --config_name "$cfg" --out_dir "$out" \
    --N 16 --M 5 --H "$h" --n_trials "$N_TRIALS" \
    --eval_seed 20260727 --device "$DEVICE" \
    --control mismatched_query "$@" >> "$LOG_DIR/$name.log" 2>&1
  local rc=$?
  log "$([ $rc -eq 0 ] && echo OK || echo FAIL) $name  rc=$rc  $(( $(date +%s) - t0 ))s"
}

log "=== C4 matched control ($N_TRIALS trials, device=$DEVICE) ==="
run_ctrl k10_s44_H6         ouro-4x10_s44 ouro-4x10 6
run_ctrl k10_s44_H7         ouro-4x10_s44 ouro-4x10 7
run_ctrl k10_s44_H8         ouro-4x10_s44 ouro-4x10 8
run_ctrl k10_s44_H8_randinit ouro-4x10_s44 ouro-4x10 8 --random_init
run_ctrl k8_s42_H8          ouro-4x8_s42  ouro-4x8  8
run_ctrl k8_s44_H7          ouro-4x8_s44  ouro-4x8  7
run_ctrl k8_s44_H8          ouro-4x8_s44  ouro-4x8  8
log "=== DONE; summaries in $OUT_ROOT/ ==="
