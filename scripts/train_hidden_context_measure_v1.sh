#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_measure_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_ft_seed7/checkpoints/best_model.pth}"
SEED="${3:-7}"

python baseline/train.py \
  --model-profile hidden_context_measure_v1 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --task-loss-weights A4C=1.20,AOP=1.00,FA=1.15,FUGC=1.00,HC=1.40,IVC=1.40,PLAX=1.10,PSAX=1.10,fetal_femur=1.35 \
  --sampler-task-weights AOP=1.15,FA=1.20,HC=1.45,IVC=1.45,fetal_femur=1.30 \
  --output-dir "${OUTPUT_DIR}"
