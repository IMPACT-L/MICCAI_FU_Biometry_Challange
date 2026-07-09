#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_proxy_v2_conservative_v1}"

python baseline/train.py \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
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
  --task-loss-weights A4C=1.35,AOP=1.00,FA=1.10,FUGC=1.00,HC=1.25,IVC=1.25,PLAX=1.10,PSAX=1.10,fetal_femur=1.15 \
  --checkpoint-task-weights A4C=2.40,AOP=1.10,FA=1.10,FUGC=1.00,HC=1.80,IVC=1.60,PLAX=1.00,PSAX=1.20,fetal_femur=1.10 \
  --output-dir "${OUTPUT_DIR}"
