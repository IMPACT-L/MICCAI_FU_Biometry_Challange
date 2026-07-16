#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_target_equivariance_safe_v2_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_target_equivariance_safe_v2_seed42}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hidden_context_local_offset128_v1 \
  --batch-size 8 \
  --num-workers 4 && \
python scripts/audit_submission_before_upload.py \
  --candidate-json "${SUBMISSION_DIR}/regression_predictions.json" \
  --reference-json output/submissions/dinov3_vitb_hidden_context_offset128_femurclean_v10_probe_aop850_locked_seed42/regression_predictions.json \
  --output-json "${SUBMISSION_DIR}/pre_submit_audit_vs_24_21_best.json"
