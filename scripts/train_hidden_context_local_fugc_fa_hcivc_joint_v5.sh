#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_hcivc_joint_v5_seed42}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_headonly_v3_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 2e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --train-task-ids HC,IVC \
  --freeze-encoder \
  --freeze-other-heads \
  --measurement-loss-weight 0.010 \
  --measurement-task-weights FA=1.00,HC=1.45,IVC=1.65,PLAX=0.75,fetal_femur=0.75 \
  --task-loss-weights A4C=1.20,AOP=1.00,FA=1.00,FUGC=1.00,HC=1.55,IVC=1.75,PLAX=1.00,PSAX=1.00,fetal_femur=1.00 \
  --sampler-task-weights HC=1.60,IVC=1.85 \
  --output-dir "${OUTPUT_DIR}"
