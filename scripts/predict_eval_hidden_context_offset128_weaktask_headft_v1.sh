#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_weaktask_headft_v1_seed42}"
ANCHOR_SPLIT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/splits/val_split.csv}"

python baseline/model.py \
  --data-root data \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${RUN_DIR}/predictions_valsplit" \
  --split-csv "${ANCHOR_SPLIT}" \
  --output-filename regression_predictions.json \
  --model-profile hidden_context_local_offset128_v1 \
  --batch-size 8 && \
python baseline/evaluate.py \
  --data-root data \
  --pred-root "${RUN_DIR}/predictions_valsplit" \
  --split-csv "${ANCHOR_SPLIT}" \
  --output-file "${RUN_DIR}/evaluation_results_valsplit.json" \
  --summary-file "${RUN_DIR}/evaluation_summary_valsplit.txt"
