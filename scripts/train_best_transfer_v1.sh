#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/best_stable_v1}"

python baseline/train.py \
  --model-profile best_stable_v1 \
  --epochs 2000 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --output-dir "${OUTPUT_DIR}"
