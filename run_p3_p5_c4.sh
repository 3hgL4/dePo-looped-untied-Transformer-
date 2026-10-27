#!/usr/bin/env bash
# P3 + P5 + C4 serial queue (2026-10-27).
#
# Order is cheapest-most-informative first, so a failure costs minutes, not a night:
#   C4  intervention / activation patching on EXISTING checkpoints. ~3-4 min per run
#       (measured: 20 trials = 7 s on the 4080), 7 runs ~ 25 min total.
#   P3  H_max = 8 retrain (ouro-4x10, seed 44, plain). Removes the U-shape: with
#       H <= 8 there is no reverse-direction shortcut, so hitting H=8 is genuine
#       8-hop composition instead of "find the edge pointing at me". ~10-11 h
#       (anchor: the K=10 run just took 649.1 min at batch 60 x accum 2).
#   P5  letters format retrain (ouro-4x8, seed 44, node_format=letters). Measured
#       this evening: vocab 34 (vs 18) and max sequence 104 tokens (vs 167), i.e.
#       ~37% shorter, so batch 120 fits with accum 1 = the paper's original
#       configuration, no deviation. ~6-8 h (anchor: ouro-4x8_s44 took 604.4 min
#       at 158-token mean sequences).
#
# Usage, inside WSL, in tmux. The WSL2 VM is terminated when the last wsl.exe
# client disconnects, killing everything inside it -- keep ONE WSL terminal open
# for the whole night and keep the laptop awake.
#
#   wsl
#   tmux new -s night
#   DRY_RUN=1 bash ~/projects/insidetheloop/run_p3_p5_c4.sh   # print commands
#   bash ~/projects/insidetheloop/run_p3_p5_c4.sh             # run all three
#   C4_ONLY=1 bash  ...   # interventions only (~25 min)
#   P5_ONLY=1 bash  ...   # letters retrain only
#
# Finished stages are skipped; a stage whose training process is still alive is
# skipped instead of being launched twice, so re-running after an interruption is
# safe. Progress: tail -f runs/night/campaign.log  (per-stage logs beside it).
set -uo pipefail
cd "$HOME/projects/insidetheloop"

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1          # tokenizer already in ~/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

OUT_DIR=runs/night
INTV_DIR=intervention_outputs/p3p5c4
CAMPAIGN="$OUT_DIR/campaign.log"
mkdir -p "$OUT_DIR" "$INTV_DIR"

STEPS=${STEPS:-100000}
DRY_RUN=${DRY_RUN:-}
C4_ONLY=${C4_ONLY:-}
P3_ONLY=${P3_ONLY:-}
P5_ONLY=${P5_ONLY:-}
DEVICE=${DEVICE:-cuda}
N_TRIALS=${N_TRIALS:-400}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CAMPAIGN"; }

# run_stage <name> <skip-test> <command...>
run_stage() {
    local name=$1 skip_test=$2; shift 2
    if eval "$skip_test"; then
        log "SKIP  $name (already done)"
        return 0
    fi
    log "START $name"
    log "  cmd: $*"
    if [ -n "$DRY_RUN" ]; then
        log "DRY   $name (not executed)"
        return 0
    fi
    local t0=$SECONDS
    if "$@" 2>&1 | tee -a "$OUT_DIR/$name.log"; then
        log "OK    $name  ($(( (SECONDS - t0) / 60 )) min)"
    else
        log "FAIL  $name  (see $OUT_DIR/$name.log)"
    fi
}

train_alive() { pgrep -f "train_uniform_h.py.*$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# C4: intervention / activation patching (plain checkpoints only --
#      src/intervention_h2.py has no --node_format, letters text would not
#      tokenize the same way, so letters runs must not be patched here)
# ---------------------------------------------------------------------------
# Targets, all at H where the two seeds differ:
#   ouro-4x10_s44  the K_max=10 run that finished tonight: 15/15 hops at 1.000
#   ouro-4x8_s44   the seed that stops at frontier 6 (h7 0.747 / h8 0.692)
#   ouro-4x8_s42   the seed that solves everything (control for "is the effect
#                  about the model or about the hop?")
# H=8 is the deepest effective hop (min(8, 16-8) = 8). --random_init is the
# mandatory negative control on the strongest target.
intv_one() {
    local name=$1 run=$2 cfg=$3 h=$4; shift 4
    local out="$INTV_DIR/$name"
    run_stage "intv_$name" "[ -f $out/intervention_summary.json ]" \
        $PY src/intervention_h2.py \
        --checkpoint "runs/slice/$run/checkpoint.pt" \
        --config_name "$cfg" \
        --out_dir "$out" \
        --N 16 --M 5 --H "$h" --n_trials "$N_TRIALS" \
        --eval_seed 20260727 --device "$DEVICE" "$@"
}

log "=== C4 interventions (~25 min total) ==="
intv_one k10_s44_H8 ouro-4x10_s44 ouro-4x10 8
intv_one k10_s44_H7 ouro-4x10_s44 ouro-4x10 7
intv_one k10_s44_H6 ouro-4x10_s44 ouro-4x10 6
intv_one k10_s44_H8_randinit ouro-4x10_s44 ouro-4x10 8 --random_init
intv_one k8_s44_H8  ouro-4x8_s44  ouro-4x8  8
intv_one k8_s44_H7  ouro-4x8_s44  ouro-4x8  7
intv_one k8_s42_H8  ouro-4x8_s42  ouro-4x8  8

[ -n "$C4_ONLY" ] && { log "C4_ONLY=1 -> stopping before the retrains"; exit 0; }
[ -n "$P5_ONLY" ] && { log "P5_ONLY=1 -> skipping P3"; } || {

# ---------------------------------------------------------------------------
# P3: H_max = 8. Same model, same seed, same budget as the K=10 run that just
# finished; the ONLY change is the hop ceiling, which removes the folding
# (pi^15 = pi^-1) from both training and eval. Prediction if the "K_max is the
# binding constraint" story is right: H=8 reached, frontier(raw-H) = 8 = ceiling.
# If instead it stalls at 6 again, the extra iterations only ever bought the
# reverse-direction shortcut, which is a much stronger statement.
# ---------------------------------------------------------------------------
P3_OUT="runs/slice/ouro-4x10_hmax8_s44"
p3_done() {
    [ -f "$P3_OUT/checkpoint.pt" ] && \
    tail -1 "$P3_OUT/trajectory.jsonl" 2>/dev/null | grep -q "\"step\": $STEPS"
}
log "=== P3: H_max=8 retrain, ouro-4x10 seed 44 (~10-11 h) ==="
if train_alive "$P3_OUT"; then
    log "SKIP  train_ouro-4x10_hmax8_s44 (already running, pid: $(pgrep -f "train_uniform_h.py.*$P3_OUT" | tr '\n' ' '))"
else
    run_stage "train_ouro-4x10_hmax8_s44" "p3_done" \
        $PY src/train_uniform_h.py \
        --config_name ouro-4x10 --seed 44 \
        --N 16 --M 5 --H_min 1 --H_max 8 \
        --total_batch 60 --grad_accum_steps 2 --max_len 200 \
        --wd 0.01 --warmup_steps 1000 --grad_clip 1.0 --total_steps "$STEPS" \
        --curriculum_floor_steps 7000 --curriculum_ramp_steps 23000 \
        --eval_interval 4000 --eval_n_samples 8000 --ckpt_interval 5000 \
        --zero_init --node_format plain --device "$DEVICE" \
        --save_dir "$P3_OUT"
fi
}

# ---------------------------------------------------------------------------
# P5: letters. "A".."P" instead of "0".."15" gives 16 DISTINCT first tokens
# (measured floor 0.0625 = 1/16, vs 0.4375 for plain), so the logit-lens /
# probe readout is no longer capped by the 7 nodes that share first token "1".
# Sequences are 104 tokens max (measured) so batch 120 x accum 1 -- the paper's
# own setting -- fits in 12 GB; that is why this stage has NO memory deviation.
# max_len 128 instead of 200 = pure padding, no truncation (measured max 104).
# ---------------------------------------------------------------------------
P5_OUT="runs/slice/ouro-4x8_s44_letters"
p5_done() {
    [ -f "$P5_OUT/checkpoint.pt" ] && \
    tail -1 "$P5_OUT/trajectory.jsonl" 2>/dev/null | grep -q "\"step\": $STEPS"
}
log "=== P5: letters retrain, ouro-4x8 seed 44 (~6-8 h) ==="
if train_alive "$P5_OUT"; then
    log "SKIP  train_ouro-4x8_s44_letters (already running, pid: $(pgrep -f "train_uniform_h.py.*$P5_OUT" | tr '\n' ' '))"
else
    run_stage "train_ouro-4x8_s44_letters" "p5_done" \
        $PY src/train_uniform_h.py \
        --config_name ouro-4x8 --seed 44 \
        --N 16 --M 5 --H_min 1 --H_max 15 \
        --total_batch 120 --grad_accum_steps 1 --max_len 128 \
        --wd 0.01 --warmup_steps 1000 --grad_clip 1.0 --total_steps "$STEPS" \
        --curriculum_floor_steps 7000 --curriculum_ramp_steps 23000 \
        --eval_interval 4000 --eval_n_samples 8000 --ckpt_interval 5000 \
        --zero_init --node_format letters --device "$DEVICE" \
        --save_dir "$P5_OUT"
fi

log "queue finished. Results:"
log "  C4 : $INTV_DIR/"
log "  P3 : $P3_OUT/"
log "  P5 : $P5_OUT/"
