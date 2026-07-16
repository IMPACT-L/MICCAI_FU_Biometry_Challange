#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_pixel_unet_offset128_v1_seed42}"
ANCHOR_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"
ANCHOR_JSON="${ANCHOR_JSON:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/predictions/regression_predictions.json}"

python baseline/train.py \
  --model-profile hidden_context_pixel_unet_offset128_v1 \
  --seed "${SEED}" \
  --epochs 100 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --grad-accum-steps 2 \
  --learning-rate 2e-5 \
  --init-checkpoint "${ANCHOR_CKPT}" \
  --train-highres-texture-only \
  --roi-anchor-json "${ANCHOR_JSON}" \
  --anchor-consistency-weight 0.02 \
  --anchor-consistency-tasks A4C,AOP,FUGC,HC,IVC,PLAX,PSAX,fetal_femur \
  --output-dir "${OUTPUT_DIR}"
