#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_roi_anchor_aopfa_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"
LOCAL_ANCHOR_JSON="${4:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/predictions/regression_predictions.json}"

python baseline/train.py \
  --encoder-name vit_base_patch16_dinov3 \
  --input-size 512 \
  --heatmap-size 128 \
  --heatmap-sigma 2.8 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile hidden_a4c_hc_ivc_fugc_offset_v1 \
  --task-adapter-profile context_local_v1 \
  --task-loss-family-profile uniform \
  --split-mode grouped \
  --augmentation-profile baseline \
  --checkpoint-score-mode server_proxy_v1 \
  --cardiac-split-screen-mode keep \
  --seed "${SEED}" \
  --epochs 120 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 8e-6 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-task-ids AOP,FA \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --train-roi-crop \
  --roi-crop-tasks AOP,FA \
  --roi-anchor-json "${LOCAL_ANCHOR_JSON}" \
  --roi-context-min 1.50 \
  --roi-context-max 2.20 \
  --roi-center-jitter 0.06 \
  --anchor-consistency-weight 0.004 \
  --anchor-consistency-tasks AOP,FA \
  --measurement-loss-weight 0.004 \
  --measurement-loss-tasks AOP,FA \
  --measurement-task-weights AOP=1.20,FA=1.10 \
  --aop-angle-loss-weight 0.003 \
  --checkpoint-task-weights AOP=1.70,FA=1.50 \
  --output-dir "${OUTPUT_DIR}"
