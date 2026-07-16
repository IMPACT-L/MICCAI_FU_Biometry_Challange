#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_input768_offset192_headadapt_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_input768_offset192_headadapt_v1_seed42}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hidden_context_input768_offset192_v1 \
  --batch-size 2 \
  --num-workers 4 && \
bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}" \
  output/submissions/dinov3_vitb_hidden_context_offset128_femurclean_v10_probe_aop850_locked_seed42/regression_predictions.json
