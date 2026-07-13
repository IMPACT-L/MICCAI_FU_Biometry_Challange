#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_aop_headonly_seed42}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fa_headonly_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_a4c_hcivc_v1 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-4 \
  --init-checkpoint "${BASE_CKPT}" \
  --train-task-ids AOP \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --output-dir "${OUTPUT_DIR}"
