#!/usr/bin/env bash
# Shared GRPO launcher. ReMind selects its config through run_grpo_remind.sh.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: MODEL_PATH=... TRAIN_DATA=... VAL_DATA=... bash examples/remind/run_grpo.sh [--dry-run] [key=value ...]

MODEL_PATH accepts a local model directory or a Hugging Face model ID.
TRAIN_DATA and VAL_DATA accept JSON/JSONL/Parquet files or Hugging Face datasets.
Optional: CONFIG_PATH, IMAGE_DIR, PYTHON_BIN, NPROC_PER_NODE, NNODES,
          EXPERIMENT_NAME, SAVE_CHECKPOINT_PATH, FIND_LAST_CHECKPOINT.
Extra arguments override the YAML config using OmegaConf key=value syntax.
For multiple nodes, start a Ray cluster first and set RAY_ADDRESS and NNODES.
Run this launcher once on the head node. --dry-run prints the command only.
HELP
    exit 0
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or Hugging Face model ID.}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the training data.}"
: "${VAL_DATA:?Set VAL_DATA to the validation data.}"

CONFIG_PATH=${CONFIG_PATH:-"${SCRIPT_DIR}/configs/vanilla_grpo.yaml"}
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 1; }
PYTHON_BIN=${PYTHON_BIN:-${ENV_DIR:+${ENV_DIR}/bin/python}}
PYTHON_BIN=${PYTHON_BIN:-python}

cmd=("$PYTHON_BIN" -m verl.trainer.main "config=$CONFIG_PATH"
    "worker.actor.model.model_path=$MODEL_PATH" "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_DATA")

for pair in \
    IMAGE_DIR:data.image_dir \
    FORMAT_PROMPT:data.format_prompt \
    REWARD_FUNCTION:worker.reward.reward_function \
    NPROC_PER_NODE:trainer.n_gpus_per_node \
    NNODES:trainer.nnodes \
    EXPERIMENT_NAME:trainer.experiment_name \
    SAVE_CHECKPOINT_PATH:trainer.save_checkpoint_path \
    FIND_LAST_CHECKPOINT:trainer.find_last_checkpoint \
    GLOBAL_BATCH_SIZE:worker.actor.global_batch_size \
    ROLLOUT_BATCH_SIZE:data.rollout_batch_size \
    ROLLOUT_N:worker.rollout.n \
    VAL_FREQ:trainer.val_freq \
    SAVE_FREQ:trainer.save_freq \
    ONLINE_FILTERING:algorithm.online_filtering \
    REVIEW_QUEUE_ENABLED:algorithm.review_queue_enabled \
    REVIEW_REPLACE_RATIO:algorithm.review_replace_ratio \
    REVIEW_FREQ:algorithm.review_freq \
    REVIEW_QUEUE_MAX_SIZE:algorithm.review_queue_max_size \
    REVIEW_QUEUE_SAMPLE_RATE:algorithm.review_queue_sample_rate \
    REVIEW_START_STEP:algorithm.review_start_step; do
    var=${pair%%:*}
    key=${pair#*:}
    if [[ -n "${!var:-}" ]]; then
        cmd+=("${key}=${!var}")
    fi
done
cmd+=("$@")

cd "$PROJECT_DIR"
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
