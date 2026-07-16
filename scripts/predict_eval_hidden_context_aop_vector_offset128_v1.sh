#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_aop_vector_offset128_v1_seed42}"

python baseline/model.py \
  --data-root data \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${RUN_DIR}/predictions" \
  --model-profile hidden_context_aop_vector_offset128_v1 \
  --batch-size 8 && \
python baseline/evaluate.py \
  --data-root data \
  --pred-root "${RUN_DIR}/predictions" \
  --output-file "${RUN_DIR}/evaluation_results.json" \
  --summary-file "${RUN_DIR}/evaluation_summary.txt"
