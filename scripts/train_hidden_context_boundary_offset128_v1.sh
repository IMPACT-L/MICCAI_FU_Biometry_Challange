#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_boundary_offset128_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_boundary_offset128_v1 \
  --seed "${SEED}" \
  --epochs 80 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-5 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-boundary-branch-only \
  --no-ema \
  --output-dir "${OUTPUT_DIR}"
