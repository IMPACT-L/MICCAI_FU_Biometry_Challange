#!/usr/bin/env bash
set -euo pipefail

ROI_RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42}"
ANCHOR_JSON="${3:-output/submissions/dinov3_vitb_hidden_context_offset128_femurclean_v10_probe_aop850_locked_seed42/regression_predictions.json}"

python scripts/roi_refine_submission_from_anchor.py \
  --anchor-json "${ANCHOR_JSON}" \
  --roi-checkpoint "${ROI_RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --roi-model-profile hidden_context_local_offset128_v1 \
  --roi-task-ids HC,IVC,PLAX \
  --roi-context 1.80 \
  --roi-min-size 112 \
  --roi-gate-task-thresholds HC=6,IVC=5,PLAX=5 \
  --batch-size 8 && \
bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}" "${ANCHOR_JSON}"
