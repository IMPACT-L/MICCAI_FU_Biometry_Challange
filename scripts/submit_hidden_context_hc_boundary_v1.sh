#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-mid}"
INPUT_JSON="${2:-output/submissions/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42/regression_predictions.json}"
OUTPUT_DIR="${3:-output/submissions/dinov3_vitb_hidden_context_hc_boundary_v1_${MODE}_seed42}"

case "${MODE}" in
  soft)
    SEARCH_PX=8
    MAX_SHIFT_PX=2.5
    BLEND=0.35
    DISTANCE_PENALTY=0.35
    CENTER_STRENGTH=0.10
    SAMPLES=33
    ;;
  mid)
    SEARCH_PX=10
    MAX_SHIFT_PX=4.0
    BLEND=0.45
    DISTANCE_PENALTY=0.28
    CENTER_STRENGTH=0.12
    SAMPLES=41
    ;;
  strong)
    SEARCH_PX=12
    MAX_SHIFT_PX=5.5
    BLEND=0.55
    DISTANCE_PENALTY=0.22
    CENTER_STRENGTH=0.15
    SAMPLES=45
    ;;
  *)
    echo "Unsupported mode: ${MODE}. Use soft, mid, or strong." >&2
    exit 2
    ;;
esac

python scripts/refine_submission_hc_ellipse_boundary.py \
  --input-json "${INPUT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --search-px "${SEARCH_PX}" \
  --max-shift-px "${MAX_SHIFT_PX}" \
  --blend "${BLEND}" \
  --distance-penalty "${DISTANCE_PENALTY}" \
  --center-strength "${CENTER_STRENGTH}" \
  --samples "${SAMPLES}" \
  --zip-submission && \
bash scripts/audit_submission_before_upload.sh "${OUTPUT_DIR}" "${INPUT_JSON}"
