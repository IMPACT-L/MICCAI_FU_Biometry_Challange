#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_offset128_edge_snap_safe_v1_seed42}"
RAW_DIR="${SUBMISSION_DIR}_raw"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${RAW_DIR}" \
  --model-profile hidden_context_local_offset128_v1 \
  --batch-size 8 \
  --num-workers 4

python scripts/snap_predictions_to_ultrasound_edges.py \
  --input-json "${RAW_DIR}/regression_predictions.json" \
  --output-dir "${SUBMISSION_DIR}" \
  --data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --task-ids HC,IVC,PLAX,PSAX,fetal_femur \
  --zip-submission

bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}"
