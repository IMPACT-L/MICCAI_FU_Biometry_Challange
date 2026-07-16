#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/submissions/dinov3_vitb_hidden_context_plax_pair_anatomy_v2_medium_seed42}"
ANCHOR_JSON="${2:-output/submissions/dinov3_vitb_hidden_context_ivc_anatomy_v2_strong_seed42/regression_predictions.json}"
REFERENCE_JSON="${3:-$ANCHOR_JSON}"

python scripts/refine_submission_plax_pair_anatomy.py \
  --input-json "${ANCHOR_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --search-px 6.0 \
  --center-search-px 3.0 \
  --max-pair-mean-shift-px 4.5 \
  --max-pair-point-shift-px 7.0 \
  --max-global-mean-shift-px 3.8 \
  --max-global-point-shift-px 8.0 \
  --min-score-delta 0.018 \
  --distance-penalty 0.018 \
  --length-penalty 0.075 \
  --length-ratio-min 0.82 \
  --length-ratio-max 1.18 \
  --radius-samples 23 \
  --center-samples 5 \
  --zip-submission

python scripts/audit_submission_before_upload.py \
  --candidate-json "${OUTPUT_DIR}/regression_predictions.json" \
  --reference-json "${REFERENCE_JSON}" \
  --output-json "${OUTPUT_DIR}/pre_submit_audit.json" \
  --fail-on-risk

echo "Candidate submission: ${OUTPUT_DIR}/submission.zip"
