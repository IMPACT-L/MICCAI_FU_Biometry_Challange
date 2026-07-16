#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/submissions/dinov3_vitb_hidden_context_ivc_anatomy_v2_strong_seed42}"
ANCHOR_JSON="${2:-output/submissions/current_anchor_student_blends_v1/all_hard_light/regression_predictions.json}"
REFERENCE_JSON="${3:-$ANCHOR_JSON}"

python scripts/refine_submission_ivc_anatomy.py \
  --input-json "${ANCHOR_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --search-px 10.0 \
  --center-search-px 4.0 \
  --max-mean-shift-px 7.0 \
  --max-point-shift-px 10.0 \
  --min-score-delta 0.025 \
  --distance-penalty 0.014 \
  --length-penalty 0.040 \
  --length-ratio-min 0.60 \
  --length-ratio-max 1.55 \
  --radius-samples 31 \
  --center-samples 5 \
  --zip-submission

python scripts/audit_submission_before_upload.py \
  --candidate-json "${OUTPUT_DIR}/regression_predictions.json" \
  --reference-json "${REFERENCE_JSON}" \
  --output-json "${OUTPUT_DIR}/pre_submit_audit.json" \
  --fail-on-risk

echo "Candidate submission: ${OUTPUT_DIR}/submission.zip"
