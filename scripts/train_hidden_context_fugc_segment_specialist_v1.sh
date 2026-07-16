#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_fugc_segment_specialist_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_fugc_segment_specialist_v1 \
  --seed "${SEED}" \
  --epochs 260 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-5 \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --train-task-ids FUGC \
  --init-checkpoint "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}"
