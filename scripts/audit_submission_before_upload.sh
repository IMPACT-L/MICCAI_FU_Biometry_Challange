#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_DIR="${1:?Usage: bash scripts/audit_submission_before_upload.sh <candidate_submission_dir> [reference_json]}"
REFERENCE_JSON="${2:-output/submissions/dinov3_vitb_hidden_context_offset128_v1_seed42/regression_predictions.json}"
CANDIDATE_JSON="${CANDIDATE_DIR%/}/regression_predictions.json"

python scripts/audit_submission_before_upload.py \
  --candidate-json "${CANDIDATE_JSON}" \
  --reference-json "${REFERENCE_JSON}" \
  --fail-on-risk \
  --output-json "${CANDIDATE_DIR%/}/pre_submit_audit.json"
