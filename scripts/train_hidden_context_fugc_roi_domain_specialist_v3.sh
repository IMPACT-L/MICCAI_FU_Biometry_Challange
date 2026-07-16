#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_fugc_roi_domain_specialist_v3_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --epochs 140 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.006 \
  --batch-size 8 \
  --num-workers 4 \
  --learning-rate 2e-5 \
  --seed "${SEED}" \
  --encoder-name vit_base_patch16_dinov3 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile hidden_a4c_hc_ivc_fugc_segment_specialist_v1 \
  --task-adapter-profile context_local_v1 \
  --input-size 512 \
  --heatmap-size 128 \
  --heatmap-sigma 2.8 \
  --train-task-ids FUGC \
  --train-roi-crop \
  --roi-crop-tasks FUGC \
  --roi-context-min 1.25 \
  --roi-context-max 2.15 \
  --roi-center-jitter 0.12 \
  --split-mode grouped \
  --augmentation-profile ultrasound_robust_v1 \
  --checkpoint-score-mode server_proxy_v1 \
  --measurement-loss-weight 0.06 \
  --measurement-loss-tasks FUGC \
  --measurement-task-weights FUGC=3.0 \
  --fugc-segment-loss-weight 0.25 \
  --freeze-encoder \
  --freeze-other-heads \
  --init-checkpoint "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}"
