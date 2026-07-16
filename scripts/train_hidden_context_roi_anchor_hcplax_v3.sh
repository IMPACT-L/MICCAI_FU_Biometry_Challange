#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_roi_anchor_hcplax_v3_seed42}"
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
  --train-task-ids HC,PLAX \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --train-roi-crop \
  --roi-crop-tasks HC,PLAX \
  --roi-anchor-json "${LOCAL_ANCHOR_JSON}" \
  --roi-context-min 1.45 \
  --roi-context-max 2.15 \
  --roi-center-jitter 0.06 \
  --anchor-consistency-weight 0.005 \
  --anchor-consistency-tasks HC,PLAX \
  --measurement-loss-weight 0.006 \
  --measurement-loss-tasks HC,PLAX \
  --measurement-task-weights HC=1.35,PLAX=1.10 \
  --checkpoint-task-weights HC=1.80,PLAX=1.60 \
  --output-dir "${OUTPUT_DIR}"
