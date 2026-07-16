#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/hrnet_w32_context_offset128_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/hrnet_w32_context_offset128_v1_seed42}"
REFERENCE_JSON="${3:-output/submissions/current_anchor_student_blends_v1/all_hard_light/regression_predictions.json}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hrnet_w32_context_offset128_v1 \
  --batch-size 4

python scripts/audit_submission_before_upload.py \
  --candidate-json "${SUBMISSION_DIR}/regression_predictions.json" \
  --reference-json "${REFERENCE_JSON}" \
  --output-json "${SUBMISSION_DIR}/pre_submit_audit.json"
