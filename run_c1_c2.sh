#!/usr/bin/env bash
# C1 + C2 queue (2026-10-26, revised after the H_eval=8 collision).
#
#   C2  per-layer linear probe (+ Hewitt-Liang shuffled-label control) on the two
#       ouro-4x8 seeds that bracket the mechanism: s42 solves all 8 effective hops,
#       s44 stops at 6. Cheap, uses checkpoints that already exist. Runs first.
#   C1  retrain seed 44 with K_max = 10 (config name ouro-4x10) to test the
#       falsifiable prediction "s44's frontier 6 -> 15 once the loop has 10
#       iterations". FROM-SCRATCH retrain, not a resume: K_max sizes the iteration
#       embedding table, so an ouro-4x8 checkpoint cannot be loaded into ouro-4x10.
#
# Usage, inside WSL, in tmux (the WSL2 VM dies when the last wsl.exe client
# disconnects, taking background jobs with it):
#
#   wsl
#   tmux new -s c1c2
#   bash ~/projects/insidetheloop/run_c1_c2.sh             # real run
#   PROBE_ONLY=1 bash ~/projects/insidetheloop/run_c1_c2.sh  # C2 only, no training
#   DRY_RUN=1 bash ~/projects/insidetheloop/run_c1_c2.sh    # print the commands
#
# Stages already finished are skipped, and a C1 training that is still alive is
# never launched a second time, so re-running after an interruption is safe.
# Progress: tail -f runs/c1c2/campaign.log   (per-stage logs next to it)
set -uo pipefail
cd "$HOME/projects/insidetheloop"

export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1          # the tokenizer is already in ~/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

OUT_DIR=runs/c1c2
PROBE_DIR=probe_outputs/c2
CAMPAIGN="$OUT_DIR/campaign.log"
mkdir -p "$OUT_DIR" "$PROBE_DIR"

STEPS=${STEPS:-100000}           # training budget; override for a smoke test
TOTAL_BATCH=${TOTAL_BATCH:-120}  # samples per micro-batch (the paper's 120)
ACCUM=${ACCUM:-1}                 # micro-batches per optimizer step. 1 = original.
                                  # 2 with TOTAL_BATCH=60 keeps the effective batch at
                                  # 120 samples while halving activation memory; needed
                                  # because K_max=10 at batch 120 does not fit in 12 GB
                                  # (measured: 10.28 GiB peak at K=8, 6.39 GiB at K=10/b60)
MAXLEN=${MAXLEN:-200}            # 200 = the paper's padding. 168 = the longest
                                 # sequence actually produced (measured), which
                                 # saves ~16% activation memory but is a DEVIATION.
                                 # Only use it if the K=10 run does not fit in 12 GB.
DRY_RUN=${DRY_RUN:-}
PROBE_ONLY=${PROBE_ONLY:-}       # 1 = run the C2 probes and stop

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CAMPAIGN"; }

# run_stage <name> <skip-test-shell> <command...>
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

# ---------------------------------------------------------------------------
# C2: per-layer linear probes, prompt-disjoint, with shuffled control
# ---------------------------------------------------------------------------
# --per_layer   : one probe after every layer inside every iteration
#                 (K_max * L + 1 = 33 levels for ouro-4x8), so the crossing point
#                 can be read in layers, not just in whole iterations.
# --split_by_prompt / --complete_answer_eval : the paper's probe protocol.
# --k_sweep     : model-head accuracy for K_eval = 1..K_max (nearly free, and it
#                 is the model's own commitment curve next to the probe staircase).
# --sanity_min_acc 0.8 : gate that the checkpoint was loaded in the right regime;
#                 it is a next-token accuracy at the probe position, not a claim.
#
# WHY H_eval=6 AND NOT 8 FOR s44 (2026-10-26, first attempt aborted in 3 s):
#   the gate fired at s44/H8 with next-token acc 0.646 < 0.8. Not a loading bug --
#   ouro-4x8_s44 trajectory.jsonl (step 100000) has per_hop_acc
#     h<=6 = 1.000,  h=7 = 0.7467,  h=8 = 0.6923
#   i.e. H=8 is outside its frontier (6) by construction. Probing a hop the model
#   cannot do measures the failure, not the mechanism. Both seeds are therefore
#   probed at H_eval=6, where BOTH are saturated (1.000):
#     s42: frontier 15, s44: frontier 6  -> same ruler, different iteration need.
#   s42 @H=8 is kept as an extra (it passes the gate, acc 0.9925 at h=8); never
#   put it in the same table as the H=6 pair. s44 @H=8 is not run at all.
probe_one() {
    local run=$1 h_eval=$2
    local out="$PROBE_DIR/${run}_H${h_eval}"
    run_stage "probe_${run}_H${h_eval}" "[ -f $out/probe_results.json ]" \
        $PY src/probe_h2.py \
        --checkpoint "runs/slice/$run/checkpoint.pt" \
        --config_name ouro-4x8 \
        --out_dir "$out" \
        --N 16 --M 5 --H_eval "$h_eval" \
        --n_samples 5000 --batch_size 32 \
        --eval_seed 12345 --probe_seed 0 --train_frac 0.8 \
        --split_by_prompt --per_layer --probe_types linear \
        --complete_answer_eval --k_sweep \
        --sanity_min_acc 0.8 --device "${DEVICE:-cuda}"
}

probe_one ouro-4x8_s42 8      # done 11:14 (14 min); extra, gate passes
probe_one ouro-4x8_s42 6      # matched pair, the one to compare with s44
probe_one ouro-4x8_s44 6      # H=8 is out of frontier, see note above

# ---------------------------------------------------------------------------
# C1: ouro-4x10, seed 44, everything else exactly as the ouro-4x8 run
# ---------------------------------------------------------------------------
# Eval keeps the FULL 15-hop grid on purpose: the prediction is stated in H units
# ("frontier 6 -> 15"), so the run has to be readable in the same units as the
# older ones. --eval_hops half would save 47% of eval time but move the ruler.
C1_OUT="runs/slice/ouro-4x10_s44"
c1_done() {
    [ -f "$C1_OUT/checkpoint.pt" ] && \
    tail -1 "$C1_OUT/trajectory.jsonl" 2>/dev/null | grep -q "\"step\": $STEPS"
}
c1_running() { pgrep -f "train_uniform_h.py.*$C1_OUT" >/dev/null 2>&1; }

if [ -n "$PROBE_ONLY" ]; then
    log "PROBE_ONLY=1 -> C1 training not touched"
elif c1_running; then
    log "SKIP  train_ouro-4x10_s44 (already running, pid: $(pgrep -f "train_uniform_h.py.*$C1_OUT" | tr '\n' ' '))"
else
    run_stage "train_ouro-4x10_s44" "c1_done" \
        $PY src/train_uniform_h.py \
        --config_name ouro-4x10 --seed 44 \
        --N 16 --M 5 --H_min 1 --H_max 15 --total_batch "$TOTAL_BATCH" \
    --grad_accum_steps "$ACCUM" --max_len "$MAXLEN" \
        --wd 0.01 --warmup_steps 1000 --grad_clip 1.0 --total_steps "$STEPS" \
        --curriculum_floor_steps 7000 --curriculum_ramp_steps 23000 \
        --eval_interval 4000 --eval_n_samples 8000 --ckpt_interval 5000 \
        --zero_init --node_format plain --device "${DEVICE:-cuda}" \
        --save_dir "$C1_OUT"
fi

log "queue finished. Results:"
log "  C2 probes : $PROBE_DIR/ouro-4x8_s42_H6/  $PROBE_DIR/ouro-4x8_s44_H6/  (extra: ouro-4x8_s42_H8/)"
log "  C1 train  : $C1_OUT/  (batch $TOTAL_BATCH x $ACCUM = $((TOTAL_BATCH * ACCUM)) samples/step, max_len $MAXLEN)"
