#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/hrnet_w32_context_offset128_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hrnet_w32_context_offset128_v1 \
  --seed "${SEED}" \
  --epochs 160 \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 0.01 \
  --batch-size 2 \
  --num-workers 4 \
  --grad-accum-steps 2 \
  --learning-rate 3e-5 \
  --init-checkpoint "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}"
