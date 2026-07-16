#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/submissions/dinov3_vitb_hidden_context_fugc_axis_anatomy_v1_seed42}"
ANCHOR_JSON="${2:-output/submissions/dinov3_vitb_hidden_context_ivc_anatomy_v2_strong_seed42/regression_predictions.json}"
REFERENCE_JSON="${3:-$ANCHOR_JSON}"

python scripts/refine_submission_fugc_axis_anatomy.py \
  --input-json "${ANCHOR_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --center-search-px 3.0 \
  --length-search-px 5.0 \
  --angle-search-deg 4.0 \
  --max-mean-shift-px 4.5 \
  --max-point-shift-px 7.0 \
  --min-score-delta 0.020 \
  --distance-penalty 0.016 \
  --length-penalty 0.060 \
  --angle-penalty 0.035 \
  --length-ratio-min 0.88 \
  --length-ratio-max 1.12 \
  --center-samples 5 \
  --length-samples 21 \
  --angle-samples 5 \
  --zip-submission

python scripts/audit_submission_before_upload.py \
  --candidate-json "${OUTPUT_DIR}/regression_predictions.json" \
  --reference-json "${REFERENCE_JSON}" \
  --output-json "${OUTPUT_DIR}/pre_submit_audit.json" \
  --fail-on-risk

echo "Candidate submission: ${OUTPUT_DIR}/submission.zip"
