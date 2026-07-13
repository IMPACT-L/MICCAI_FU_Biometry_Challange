#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/vitlarge_hidden_context_local_a4c_hcivc_v1}"
SUBMISSION_DIR="${2:-output/submissions/vitlarge_hidden_context_local_a4c_hcivc_v1}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --batch-size 8 \
  --num-workers 4 \
  --model-profile vitlarge_hidden_context_local_a4c_hcivc_v1
