#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hidden_context_local_offset128_v2 \
  --batch-size 8 \
  --num-workers 4 && \
bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}"
