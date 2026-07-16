#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_weaktask_headft_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_offset128_v1 \
  --seed "${SEED}" \
  --epochs 160 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.02 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-6 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-task-ids A4C,HC,IVC,PSAX \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --no-ema \
  --task-loss-weights A4C=1.15,HC=1.00,IVC=1.25,PSAX=1.25 \
  --sampler-task-weights A4C=1.25,IVC=1.35,PSAX=1.35 \
  --output-dir "${OUTPUT_DIR}"
