#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_teacher_anchor_v1_seed42}"
ANCHOR_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_offset128_v1 \
  --seed "${SEED}" \
  --epochs 120 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 8e-7 \
  --init-checkpoint "${ANCHOR_CKPT}" \
  --teacher-checkpoint "${ANCHOR_CKPT}" \
  --teacher-consistency-weight 0.08 \
  --freeze-encoder \
  --output-dir "${OUTPUT_DIR}"
