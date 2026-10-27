#!/usr/bin/env bash
# Isolation run, TAKE 2: remove long-horizon data from an h1-competent letters
# model and ask whether 2-hop composition becomes learnable.
#
# WHY TAKE 1 (run_iso_hmax4.sh, H_max=4 from scratch, 8000 steps) TESTED NOTHING:
# compressing total_steps to 8000 also compressed the cosine LR schedule, so by
# step 5000 the LR was already 3.0e-4 and even h1 was still at chance (0.076 at
# steps 5000-7000). The letters twin on the full 100k schedule crosses h1 between
# step 4000 and 8000 and only then starts on h2 (plain twin: h2 = 1.00 by step
# 12000). "h2 stays at chance" in an 8000-step cosine run is an LR-schedule
# artefact, not evidence about the format. Take 1 is kept as the negative control
# for exactly that claim and is NOT used to judge the format.
#   take 1 result: h1/h2/h3/h4 = 0.072/0.065/0.070/0.069 at step 7000, every eval
#   at chance, frontier fwd = 0, 8000 steps / 27.0 min, LR 1e-3 -> 1e-4.
#
# TAKE 2: start from the aborted P5 checkpoint (step 30000: last eval (28000) h1 = 0.998, h2 =
# 0.043, i.e. h1 already learned, h2 not), copy it into a fresh run dir, and
# --resume with ONE substantive change: H = 5..15 data is gone.
#   * the h1-learning cost (~7000 steps) is not paid again;
#   * E1's mechanism ("the long hops eat the capacity / dominate the loss") has
#     nothing left to act with;
#   * the LR schedule is NOT restarted -- checkpoint.pt restores the scheduler,
#     so LR continues the parent's 100k cosine (~8.2e-4 at step 30000, slowly
#     falling), the same regime the plain twin was in when it learned h2.
# 8000 further steps ~ 26 min at the measured 5.1 sps.
#
# WHY --H_max 15 AND --H_sample_max 4 INSTEAD OF --H_max 4:
# the token map is estimated from a dataset generated over [H_min, H_max]. With
# H_max=4 the strings "hop 5".."hop 9" never appear, so the mapped vocab is 28
# instead of 34 and load_state_dict fails on the 34-row embedding:
#   size mismatch for token_embeddings.weight: [34, 256] vs [28, 256]
# --H_max 15 keeps the vocabulary at the parent's 34 while --H_sample_max 4 caps
# what the optimizer actually sees. --eval_hops must then be explicit, otherwise
# the eval grid would be 1..15 and 11 of 15 points would be out of distribution.
# Two log lines are expected noise under this configuration and are NOT errors:
#   [B2/P3] H_max=15 > N//2 ... (fold note, driven by --H_max on paper)
#   [P1] eval grid is not closed ... (rev frontier is 0 because H=12..15 is not
#        trained at all here; fwd frontier is the only readable one)
#
# PRE-REGISTERED READOUT (before launch):
#   h2 -> >= 0.8 within 8000 resumed steps   E1 confirmed: the format is fine and
#                                            P5's stall was long-horizon
#                                            competition -> P5 is worth re-running
#                                            with a staged H ramp (1 -> 4 -> 8 -> 15).
#   h2 ~ 0.0625 at step 39000                E2 confirmed: with no long data left
#                                            to blame and h1 already solved, 2 hops
#                                            still do not form -> the letters
#                                            format itself is the blocker, and no
#                                            budget or curriculum change helps.
#   h2 in between (0.15-0.7)                 graded result, report as is.
#
# CONFOUND TO STATE IN ANY WRITE-UP: the parent model did see H up to 15 for
# 31000 steps, so this is "no long data from now on", not "never any long data";
# a weights-level scar would survive the resume. The from-scratch version of the
# clean design needs the original 7000-step floor and a full-length LR schedule,
# i.e. ~2-3 h per arm, which is why this cheap version goes first.
#
# Usage (inside WSL, tmux):
#   bash ~/projects/insidetheloop/run_iso_resume_hmax4.sh
#   DRY_RUN=1 bash ...    # print only
set -uo pipefail
cd "$HOME/projects/insidetheloop"

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

SRC=runs/aborted/ouro-4x8_s44_letters_aborted_p5
OUT_DIR=runs/iso
OUT="$OUT_DIR/ouro-4x8_s44_letters_resume_hmax4"
CAMPAIGN="$OUT_DIR/campaign.log"
STEPS=${STEPS:-39000}          # parent stopped at 30000 -> 9000 resumed steps (this launch used 39000)
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
DEVICE=${DEVICE:-cuda}
DRY_RUN=${DRY_RUN:-}
mkdir -p "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CAMPAIGN"; }

# Seed the new run dir from the parent checkpoint. trajectory.jsonl is dropped on
# purpose: it belongs to the H_max=15 phase and mixing it into this run's axis
# would be exactly the label trap that made H_sol unreadable before.
if [ ! -f "$OUT/checkpoint.pt" ]; then
    mkdir -p "$OUT"
    cp "$SRC/checkpoint.pt" "$OUT/checkpoint.pt"
    cp "$SRC/model_config.json" "$OUT/model_config.json" 2>/dev/null || true
    log "seeded $OUT/checkpoint.pt from $SRC (parent step 30000, last eval step 28000: h1=0.998 h2=0.068)"
fi
# Remove anything a previous failed attempt left behind.
rm -f "$OUT/trajectory.jsonl" "$OUT/train.log"

CMD=(
    $PY src/train_uniform_h.py
    --config_name ouro-4x8 --seed 44
    --N 16 --M 5 --H_min 1 --H_max 15 --H_sample_max 4
    --total_batch 120 --grad_accum_steps 1 --max_len 128
    --wd 0.01 --warmup_steps 1000 --grad_clip 1.0
    --total_steps "$STEPS"
    --curriculum_floor_steps 500 --curriculum_ramp_steps 1500
    --eval_hops 1,2,3,4
    --eval_interval "$EVAL_INTERVAL" --eval_n_samples 8000
    --ckpt_interval 2000
    --zero_init --node_format letters --device "$DEVICE"
    --resume
    --save_dir "$OUT"
)

log "=== ISO-2: resume letters from h1-competent ckpt, H_sample_max=4, 31000 -> $STEPS ==="
log "  cmd: ${CMD[*]}"
if [ -n "$DRY_RUN" ]; then
    log "DRY   not executed"
    exit 0
fi

t0=$SECONDS
if "${CMD[@]}" 2>&1 | tee -a "$OUT_DIR/train_iso_resume_hmax4.log"; then
    log "OK    iso2_resume_hmax4  ($(( (SECONDS - t0) / 60 )) min)"
else
    log "FAIL  iso2_resume_hmax4  (see $OUT_DIR/train_iso_resume_hmax4.log)"
fi
