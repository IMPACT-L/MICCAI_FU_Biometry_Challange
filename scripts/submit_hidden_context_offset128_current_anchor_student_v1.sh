#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_current_anchor_student_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_offset128_current_anchor_student_v1_seed42}"
REFERENCE_JSON="${3:-output/submissions/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42/regression_predictions.json}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hidden_context_local_offset128_v1 \
  --batch-size 8 \
  --num-workers 4 && \
bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}" "${REFERENCE_JSON}"
