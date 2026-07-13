#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_headonly_v3_seed42/checkpoints/best_model.pth}"

mkdir -p "${OUTPUT_DIR}/checkpoints"

python scripts/recalibrate_bn_on_manifest.py \
  --manifest data/manifests/validation_manifest.csv \
  --checkpoint-path "${BASE_CKPT}" \
  --output-path "${OUTPUT_DIR}/checkpoints/best_model.pth" \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --batch-size 8 \
  --num-workers 4 \
  --num-passes 2
