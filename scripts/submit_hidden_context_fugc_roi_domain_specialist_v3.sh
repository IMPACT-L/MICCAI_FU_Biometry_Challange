#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_fugc_roi_domain_specialist_v3_seed42}"
OUTPUT_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_fugc_roi_domain_specialist_v3_seed42}"
ANCHOR_JSON="${3:-output/submissions/dinov3_vitb_hidden_context_ivc_anatomy_v2_strong_seed42/regression_predictions.json}"
REFERENCE_JSON="${4:-${ANCHOR_JSON}}"

python scripts/roi_refine_submission_from_anchor.py \
  --manifest data/manifests/validation_manifest.csv \
  --anchor-json "${ANCHOR_JSON}" \
  --roi-checkpoint "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${OUTPUT_DIR}" \
  --roi-task-ids FUGC \
  --roi-context 1.70 \
  --roi-min-size 240 \
  --roi-gate-max-shift-px 5.0 \
  --roi-gate-task-thresholds FUGC=5.0 \
  --batch-size 8

python scripts/audit_submission_before_upload.py \
  --candidate-json "${OUTPUT_DIR}/regression_predictions.json" \
  --reference-json "${REFERENCE_JSON}" \
  --output-json "${OUTPUT_DIR}/pre_submit_audit.json" \
  --fail-on-risk

echo "Candidate submission: ${OUTPUT_DIR}/submission.zip"
