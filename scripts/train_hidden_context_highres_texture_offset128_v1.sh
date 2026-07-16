#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_highres_texture_offset128_v1_seed42}"
ANCHOR_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_highres_texture_offset128_v1 \
  --seed "${SEED}" \
  --epochs 80 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --grad-accum-steps 2 \
  --learning-rate 1e-5 \
  --init-checkpoint "${ANCHOR_CKPT}" \
  --train-highres-texture-only \
  --output-dir "${OUTPUT_DIR}"
