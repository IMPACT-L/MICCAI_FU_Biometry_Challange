#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_combo_hc_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_combo_light_v1/checkpoints/best_model.pth}"

mkdir -p "${OUTPUT_DIR}/checkpoints"

python scripts/compose_task_heads.py \
  --base-checkpoint "${BASE_CKPT}" \
  --source HC=output/runs/dinov3_vitb_taskfpn_hidden_context_local_hc_headonly_seed42/checkpoints/best_model.pth \
  --output-path "${OUTPUT_DIR}/checkpoints/best_model.pth"
