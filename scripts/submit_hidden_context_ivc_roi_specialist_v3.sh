#!/usr/bin/env bash
set -euo pipefail

ROI_RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_ivc_roi_specialist_v3_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_ivc_roi_specialist_v3_seed42}"
ANCHOR_JSON="${3:-output/submissions/dinov3_vitb_hidden_context_ivc_anatomy_v2_strong_seed42/regression_predictions.json}"

python scripts/roi_refine_submission_from_anchor.py \
  --anchor-json "${ANCHOR_JSON}" \
  --roi-checkpoint "${ROI_RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --roi-model-profile hidden_context_local_offset128_v2 \
  --roi-task-ids IVC \
  --roi-context 2.20 \
  --roi-min-size 112 \
  --roi-gate-task-thresholds IVC=5 \
  --batch-size 8

python scripts/audit_submission_before_upload.py \
  --candidate-json "${SUBMISSION_DIR}/regression_predictions.json" \
  --reference-json "${ANCHOR_JSON}" \
  --output-json "${SUBMISSION_DIR}/pre_submit_audit.json" \
  --fail-on-risk

echo "Candidate submission: ${SUBMISSION_DIR}/submission.zip"
