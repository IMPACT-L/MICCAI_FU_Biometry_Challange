#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_proxy_v3_seed42}"
SEED="${2:-42}"

python baseline/train.py \
  --model-profile hidden_context_local_a4c_hcivc_proxy_v3 \
  --seed "${SEED}" \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-4 \
  --output-dir "${OUTPUT_DIR}"
