#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_fugc_strip_axis_offset128_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_offset128_v10_fugc_strip_axis_only_v1_seed42}"
RAW_DIR="${3:-output/submissions/dinov3_vitb_hidden_context_fugc_strip_axis_offset128_v1_seed42_raw}"
ANCHOR_JSON="${4:-output/submissions/dinov3_vitb_hidden_context_offset128_femurclean_v10_probe_aop850_locked_seed42/regression_predictions.json}"
REFERENCE_JSON="${5:-${ANCHOR_JSON}}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${RAW_DIR}" \
  --model-profile hidden_context_fugc_strip_axis_offset128_v1 \
  --batch-size 8 \
  --num-workers 4

mkdir -p "${SUBMISSION_DIR}"

python scripts/blend_submission_predictions_by_task.py \
  --anchor-json "${ANCHOR_JSON}" \
  --task-blend FUGC=1.0:"${RAW_DIR}/regression_predictions.json" \
  --output-json "${SUBMISSION_DIR}/regression_predictions.json"

python - <<'PY' "${SUBMISSION_DIR}"
import sys
import zipfile
from pathlib import Path

out = Path(sys.argv[1])
with zipfile.ZipFile(out / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(out / "regression_predictions.json", arcname="regression_predictions.json")
PY

cat > "${SUBMISSION_DIR}/blend_sources.txt" <<EOF
anchor: ${ANCHOR_JSON}
FUGC alpha=1.0 source=${RAW_DIR}/regression_predictions.json
all_other_tasks: anchor unchanged
EOF

bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}" "${REFERENCE_JSON}"
