#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_hcivc_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_measure_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_hcivc_v1 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1.5e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --task-loss-weights A4C=1.20,AOP=1.00,FA=1.15,FUGC=1.00,HC=1.45,IVC=1.55,PLAX=1.10,PSAX=1.10,fetal_femur=1.35 \
  --sampler-task-weights AOP=1.15,FA=1.20,HC=1.55,IVC=1.65,fetal_femur=1.30 \
  --output-dir "${OUTPUT_DIR}"
