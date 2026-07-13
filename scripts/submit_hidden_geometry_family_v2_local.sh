#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_geometry_family_v2_local}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_geometry_family_v2_local}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hidden_geometry_family_v2 \
  --batch-size 8 \
  --num-workers 4
