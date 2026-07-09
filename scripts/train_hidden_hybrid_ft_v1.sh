#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_hybrid_ft_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_grouped_serverproxy_v1/checkpoints/best_model.pth}"

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 5e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --encoder-name vit_base_patch16_dinov3 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile uniform \
  --task-adapter-profile uniform \
  --split-mode grouped \
  --augmentation-profile baseline \
  --checkpoint-score-mode server_proxy_v2 \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.0 \
  --task-loss-weights A4C=1.30,AOP=1.00,FA=1.10,FUGC=1.00,HC=1.30,IVC=1.35,PLAX=1.10,PSAX=1.10,fetal_femur=1.20 \
  --checkpoint-task-weights A4C=2.30,AOP=1.00,FA=1.05,FUGC=1.00,HC=1.70,IVC=1.60,PLAX=1.00,PSAX=1.20,fetal_femur=1.15 \
  --sampler-task-weights AOP=1.15,FA=1.15,HC=1.35,IVC=1.40,fetal_femur=1.20 \
  --output-dir "${OUTPUT_DIR}"
