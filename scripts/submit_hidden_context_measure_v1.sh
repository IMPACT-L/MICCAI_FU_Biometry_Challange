#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_measure_v1}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_taskfpn_hidden_context_measure_v1}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --batch-size 8 \
  --num-workers 4 \
  --model-profile hidden_context_measure_v1
