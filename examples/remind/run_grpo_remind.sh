#!/bin/bash
# GRPO + ReMind image-text training launcher.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export CONFIG_PATH=${CONFIG_PATH:-"${SCRIPT_DIR}/configs/grpo_remind.yaml"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo-remind"}
export REVIEW_QUEUE_ENABLED=${REVIEW_QUEUE_ENABLED:-True}
export REVIEW_REPLACE_RATIO=${REVIEW_REPLACE_RATIO:-0.10}
export REVIEW_FREQ=${REVIEW_FREQ:-5}
export REVIEW_QUEUE_MAX_SIZE=${REVIEW_QUEUE_MAX_SIZE:-20000}
export REVIEW_QUEUE_SAMPLE_RATE=${REVIEW_QUEUE_SAMPLE_RATE:-0.25}
export REVIEW_START_STEP=${REVIEW_START_STEP:-50}
export ONLINE_FILTERING=${ONLINE_FILTERING:-False}

exec bash "${SCRIPT_DIR}/run_grpo.sh" "$@"
