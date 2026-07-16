#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_input768_offset192_headadapt_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_input768_offset192_v1 \
  --seed "${SEED}" \
  --epochs 80 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 0.01 \
  --batch-size 1 \
  --num-workers 4 \
  --grad-accum-steps 8 \
  --learning-rate 1e-6 \
  --ema-decay 0.999 \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --init-checkpoint "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}"
