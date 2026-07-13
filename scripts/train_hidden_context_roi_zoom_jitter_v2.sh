#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_roi_zoom_jitter_v2_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-5 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-roi-crop \
  --roi-crop-tasks A4C,AOP,FA,FUGC,HC,IVC,PLAX,PSAX,fetal_femur \
  --roi-context-min 1.45 \
  --roi-context-max 2.50 \
  --roi-center-jitter 0.25 \
  --output-dir "${OUTPUT_DIR}"
