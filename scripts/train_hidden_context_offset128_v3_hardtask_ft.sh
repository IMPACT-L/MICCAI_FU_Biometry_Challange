#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_offset128_v2 \
  --seed "${SEED}" \
  --epochs 320 \
  --early-stopping-patience 14 \
  --early-stopping-min-delta 0.01 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 5e-6 \
  --init-checkpoint "${INIT_CKPT}" \
  --train-task-ids A4C,AOP,HC,IVC,PLAX \
  --freeze-encoder \
  --freeze-other-heads \
  --task-loss-weights A4C=1.35,AOP=1.25,FA=1.00,FUGC=1.00,HC=1.70,IVC=1.90,PLAX=1.45,PSAX=1.00,fetal_femur=1.00 \
  --checkpoint-task-weights A4C=1.35,AOP=1.30,FA=0.80,FUGC=0.90,HC=1.90,IVC=2.20,PLAX=1.50,PSAX=1.00,fetal_femur=0.90 \
  --measurement-task-weights HC=1.45,IVC=1.75,PLAX=0.95 \
  --output-dir "${OUTPUT_DIR}"
