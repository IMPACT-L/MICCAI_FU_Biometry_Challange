#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_hidden_context_domain_adv_offset128_v1_seed42}"
OUT_DIR="${2:-output/submissions/dinov3_vitb_hidden_context_domain_adv_offset128_v1_seed42}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${OUT_DIR}" \
  --batch-size 8 \
  --num-workers 4
