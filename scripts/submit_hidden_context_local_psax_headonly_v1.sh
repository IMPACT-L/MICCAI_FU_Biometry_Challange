#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_psax_headonly_seed42}"
SUBMIT_DIR="${2:-output/submissions/dinov3_vitb_taskfpn_hidden_context_local_psax_headonly_seed42}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMIT_DIR}" \
  --model-profile hidden_context_local_a4c_hcivc_v1 \
  --batch-size 8 \
  --num-workers 4
