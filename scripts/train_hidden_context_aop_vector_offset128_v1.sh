#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_aop_vector_offset128_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_aop_vector_offset128_v1 \
  --seed "${SEED}" \
  --epochs 80 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 3e-6 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-task-ids AOP \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --aop-angle-loss-weight 0.02 \
  --output-dir "${OUTPUT_DIR}"
