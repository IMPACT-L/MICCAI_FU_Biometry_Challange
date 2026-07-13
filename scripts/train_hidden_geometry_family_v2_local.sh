#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_geometry_family_v2_local}"
SEED="${2:-42}"
INIT_CHECKPOINT="${3:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42/checkpoints/best_model.pth}"

python baseline/train.py \
  --model-profile hidden_geometry_family_v2 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}"
