#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_offset128_v1 \
  --seed "${SEED}" \
  --epochs 160 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-5 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-task-ids HC,IVC,PLAX \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --train-roi-crop \
  --roi-crop-tasks HC,IVC,PLAX \
  --roi-context-min 1.35 \
  --roi-context-max 2.10 \
  --roi-center-jitter 0.08 \
  --measurement-loss-weight 0.008 \
  --measurement-loss-tasks HC,IVC,PLAX \
  --measurement-task-weights HC=1.25,IVC=1.60,PLAX=1.00 \
  --ivc-band-loss-weight 0.10 \
  --output-dir "${OUTPUT_DIR}"
