#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_combo_light_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_headonly_v3_seed42/checkpoints/best_model.pth}"

mkdir -p "${OUTPUT_DIR}/checkpoints"

python scripts/compose_task_heads.py \
  --base-checkpoint "${BASE_CKPT}" \
  --source AOP=output/runs/dinov3_vitb_taskfpn_hidden_context_local_aop_headonly_seed42/checkpoints/best_model.pth \
  --source PLAX=output/runs/dinov3_vitb_taskfpn_hidden_context_local_plax_headonly_seed42_v2/checkpoints/best_model.pth \
  --source PSAX=output/runs/dinov3_vitb_taskfpn_hidden_context_local_psax_headonly_seed42/checkpoints/best_model.pth \
  --output-path "${OUTPUT_DIR}/checkpoints/best_model.pth"
