#!/usr/bin/env bash
# Isolation run: can the letters node format learn 2-hop composition AT ALL?
# Launched 2026-10-27 ~20:30 after P5 was stopped at step 31000.
#
# WHY: P5 (ouro-4x8, seed 44, node_format=letters, H_max=15) ran 101 min /
# 31000 steps with h1=1.00 and h2..h15 stuck at 0.04-0.09 on all 7 evals
# (chance 0.0625), while the plain-format twin (ouro-4x8_s44) hit h2=1.00 at
# step 12000. Eval, data, tokenizer, truncation, vocab and config were each
# ruled out by direct measurement (notes/2026-10-27_letters格式P5停滞排查.md).
# Two explanations survive:
#   E1 capacity/competition: H=2..15 data shares the model with H=5..15, so the
#      short hops never get learned before the long ones dominate the loss.
#   E2 format: letters breaks the task itself (e.g. every node is "▁X", so the
#      single-token embedding carries no position-independent handle to chase).
#
# DESIGN: same model (ouro-4x8), same seed (44), same node_format (letters),
# same batch (120 x accum 1), same max_len (128), same warmup (1000), same
# zero_init -- only two things change:
#   --H_max 4                  no H>=5 data exists at all, so E1's competition
#                              term is removed by construction.
#   floor 500 / ramp 1500      the P5 schedule (floor 7000 / ramp 23000) would
#                              spend 7000 of 8000 steps at H=1 and prove nothing.
#                              With this one, all of H=1..4 is in distribution
#                              from step 2000.
# 8000 steps ~ 25 min at the measured 5.2 sps.
#
# READING THE RESULT (pre-registered here, before launch):
#   h2 -> ~1.0 by step 4000       E1 survives, E2 dies: letters learns 2 hops
#                                 fine; P5's stall was long-horizon competition,
#                                 so P5 should be re-run with a slower H ramp
#                                 (or H_max staged 4 -> 8 -> 15), not abandoned.
#   h2 ~0.0625 at step 8000       E2 survives: with no long data to blame, the
#                                 format itself is the blocker. Then the fix is
#                                 at the format layer (e.g. digits 0-15 as
#                                 distinct bare tokens instead of "▁A".."▁P"),
#                                 and no amount of budget helps.
#   h3/h4 answer a second question for free: under H_max=4 the frontier should
#   read 4 if the model generalises, i.e. whether "learn 2 hops" also means
#   "learn the loop".
#
# Usage (inside WSL, tmux):
#   bash ~/projects/insidetheloop/run_iso_hmax4.sh
#   FORMAT=plain bash ... # same protocol with the plain format as a control
#                         # (a plain run at H_max=4 should also solve h2; if it
#                         # does not, the compressed curriculum is the culprit
#                         # rather than the format)
#   DRY_RUN=1 bash ...    # print the command only
set -uo pipefail
cd "$HOME/projects/insidetheloop"

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

FORMAT=${FORMAT:-letters}
STEPS=${STEPS:-8000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
DEVICE=${DEVICE:-cuda}
DRY_RUN=${DRY_RUN:-}

OUT_DIR=runs/iso
OUT="$OUT_DIR/ouro-4x8_s44_${FORMAT}_hmax4"
CAMPAIGN="$OUT_DIR/campaign.log"
mkdir -p "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CAMPAIGN"; }

CMD=(
    $PY src/train_uniform_h.py
    --config_name ouro-4x8 --seed 44
    --N 16 --M 5 --H_min 1 --H_max 4
    --total_batch 120 --grad_accum_steps 1 --max_len 128
    --wd 0.01 --warmup_steps 1000 --grad_clip 1.0
    --total_steps "$STEPS"
    --curriculum_floor_steps 500 --curriculum_ramp_steps 1500
    --eval_interval "$EVAL_INTERVAL" --eval_n_samples 8000
    --ckpt_interval 2000
    --zero_init --node_format "$FORMAT" --device "$DEVICE"
    --save_dir "$OUT"
)

log "=== ISO: H_max=4 isolation, ouro-4x8 seed 44, node_format=$FORMAT, $STEPS steps (~25 min) ==="
log "  cmd: ${CMD[*]}"
if [ -n "$DRY_RUN" ]; then
    log "DRY   not executed"
    exit 0
fi

t0=$SECONDS
if "${CMD[@]}" 2>&1 | tee -a "$OUT_DIR/train_iso_hmax4_${FORMAT}.log"; then
    log "OK    iso_$FORMAT  ($(( (SECONDS - t0) / 60 )) min)"
else
    log "FAIL  iso_$FORMAT  (see $OUT_DIR/train_iso_hmax4_${FORMAT}.log)"
fi
