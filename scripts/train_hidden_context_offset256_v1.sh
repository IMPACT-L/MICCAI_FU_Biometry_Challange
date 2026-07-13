#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset256_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_offset256_v1 \
  --seed "${SEED}" \
  --epochs 300 \
  --early-stopping-patience 12 \
  --early-stopping-min-delta 0.0 \
  --batch-size 3 \
  --num-workers 4 \
  --learning-rate 5e-6 \
  --init-checkpoint "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}"
