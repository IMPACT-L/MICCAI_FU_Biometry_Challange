#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_headonly_v3_seed73}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_headonly_seed42/checkpoints/best_model.pth}"

python baseline/train.py \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --seed 73 \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-4 \
  --init-checkpoint "${BASE_CKPT}" \
  --train-task-ids FA \
  --freeze-encoder \
  --freeze-fpn \
  --freeze-adapters \
  --freeze-other-heads \
  --output-dir "${OUTPUT_DIR}"
