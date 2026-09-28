#!/bin/bash
set -euo pipefail

# =============================================================================
# Model Merger Script
# Merge FSDP sharded checkpoint into a single HuggingFace model.
#
# Usage:
#   bash scripts/model_merger.sh
# =============================================================================

# Path configuration (modify these)
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
: "${CKPT_DIR:?Set CKPT_DIR to the actor checkpoint directory.}"
: "${DST_MODEL_DIR:?Set DST_MODEL_DIR to the output model directory.}"

# =============================================================================
cd "$PROJECT_DIR" || exit

# Merge model
python scripts/model_merger.py --local_dir "$CKPT_DIR"

# Create destination directory
mkdir -p "$DST_MODEL_DIR"

# Copy huggingface model
cp -a "$CKPT_DIR/huggingface/." "$DST_MODEL_DIR/"

echo "Done. Model copied to: $DST_MODEL_DIR"
