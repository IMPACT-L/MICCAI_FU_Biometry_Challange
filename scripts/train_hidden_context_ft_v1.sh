#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_ft_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_grouped_serverproxy_v1/checkpoints/best_model.pth}"

python baseline/train.py \
  --model-profile hidden_context_ft_v1 \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 5e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --task-loss-weights A4C=1.25,AOP=1.00,FA=1.15,FUGC=1.00,HC=1.35,IVC=1.35,PLAX=1.10,PSAX=1.10,fetal_femur=1.30 \
  --sampler-task-weights AOP=1.20,FA=1.20,HC=1.40,IVC=1.40,fetal_femur=1.25 \
  --output-dir "${OUTPUT_DIR}"
