# Shared settings for the recipes. Source this file; do not run it.
#
#   RUNS    where checkpoints are written (default: ./runs). A run writes one rolling
#           checkpoint.pt (about 40 MB for a looped 4x8 model, 300 MB for an untied 32-layer one).
#   PROBES  where probe, lens and patching outputs are written (default: ./probe_outputs).
#   GPU     CUDA device index for this process (default: 0).
#   DRY_RUN=1 prints each command instead of running it.
#   STEPS   training budget (default: 100000, the paper's). Lower it for a quick check.
#
# Every recipe runs its jobs one after another. To use several GPUs, run the printed
# commands (DRY_RUN=1) through your own launcher, one per GPU.
#
# The base tokenizer (hf-internal-testing/llama-tokenizer) is downloaded from the Hugging Face
# Hub on first use and cached. On machines without network access, copy the cache entry
# (~/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer) over and set
# HF_HUB_OFFLINE=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUNS="${RUNS:-$ROOT/runs}"
PROBES="${PROBES:-$ROOT/probe_outputs}"
GPU="${GPU:-0}"
STEPS="${STEPS:-100000}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY="uv run --no-sync python"

# The recipe every N=16 training run shares (App. "Training details"): 100k steps, AdamW with
# weight decay 0.01, 1,000 warm-up steps, cosine decay, clipping at 1, 120 prompts per step
# left-padded to 200 tokens, the hop curriculum (H=1 for 7k steps, then a linear ramp to H=15
# by step 30k), and the evaluation during training every 4k steps on 8,000 prompts.
RECIPE="--N 16 --M 5 --H_min 1 --H_max 15 --total_batch 120 --max_len 200 \
--wd 0.01 --warmup_steps 1000 --grad_clip 1.0 --total_steps $STEPS \
--curriculum_floor_steps 7000 --curriculum_ramp_steps 23000 \
--eval_interval 4000 --eval_n_samples 8000"

# N=32: a 384-token context and 2,000 warm-up steps; trained at lr 3e-4.
RECIPE_N32="--N 32 --M 5 --H_min 1 --H_max 15 --total_batch 120 --max_len 384 \
--wd 0.01 --warmup_steps 2000 --grad_clip 1.0 --total_steps $STEPS \
--curriculum_floor_steps 7000 --curriculum_ramp_steps 23000 \
--eval_interval 4000 --eval_n_samples 8000"

# Prompt-disjoint probes (Methods, "Linear probes"): 5,000 fresh prompts, split by prompt into
# 4,000 for fitting and 1,000 held out, one linear probe after every layer, with full-answer scoring.
PROBE="--N 16 --M 5 --n_samples 5000 --batch_size 32 --eval_seed 12345 --probe_seed 0 \
--train_frac 0.8 --split_by_prompt --per_layer --probe_types linear --complete_answer_eval \
--sanity_min_acc 0.8"

# The earlier split by query row (App. "Probes"): 80:20 over query rows, probes after every
# iteration of a looped model and after every layer of an untied one.
PROBE_ROW="--N 16 --M 5 --eval_seed 12345 --probe_seed 0 --train_frac 0.8"

run() {
    echo "+ $*"
    if [ -z "${DRY_RUN:-}" ]; then eval "$@"; fi
}

# Looped runs zero-initialize their output projections and untied runs do not. The trainer
# uses the same default; the recipes pass it explicitly so every command states it.
zero_init_flag() {
    case "$1" in deep-*) echo "--no_zero_init" ;; *) echo "--zero_init" ;; esac
}
