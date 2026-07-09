#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_taskfilm_seed7_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_cluster_ft_seed7/checkpoints/best_model.pth}"
SEED="${3:-7}"

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --seed "${SEED}" \
  --learning-rate 2e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --encoder-name vit_base_patch16_dinov3 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile uniform \
  --task-adapter-profile taskfilm_v1 \
  --split-mode grouped \
  --augmentation-profile baseline \
  --checkpoint-score-mode server_proxy_v1 \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.0 \
  --task-loss-weights A4C=1.25,AOP=1.00,FA=1.15,FUGC=1.00,HC=1.35,IVC=1.35,PLAX=1.10,PSAX=1.10,fetal_femur=1.30 \
  --sampler-task-weights AOP=1.20,FA=1.20,HC=1.40,IVC=1.40,fetal_femur=1.25 \
  --output-dir "${OUTPUT_DIR}"
