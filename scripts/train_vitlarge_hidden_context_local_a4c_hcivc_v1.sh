#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/vitlarge_hidden_context_local_a4c_hcivc_v1}"
BASE_CKPT="${2:-output/runs/vitlarge_dinov3_taskfpn_v1/checkpoints/best_model.pth}"
SEED="${3:-42}"

python baseline/train.py \
  --model-profile vitlarge_hidden_context_local_a4c_hcivc_v1 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 2 \
  --num-workers 4 \
  --grad-accum-steps 4 \
  --learning-rate 2e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --output-dir "${OUTPUT_DIR}"
