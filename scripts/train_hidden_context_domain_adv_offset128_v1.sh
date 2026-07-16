#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_domain_adv_offset128_v1_seed42}"
INIT_CHECKPOINT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_domain_adv_offset128_v1 \
  --epochs 80 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-6 \
  --seed "${SEED}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --grad-accum-steps 2 \
  --ema-decay 0.999 \
  --output-dir "${RUN_DIR}"
