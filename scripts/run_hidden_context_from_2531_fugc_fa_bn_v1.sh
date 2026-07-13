#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_from_2531_fugc_fa_bn_v1_seed42}"
SEED="${2:-42}"
BASE_CKPT="${3:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local/checkpoints/best_model.pth}"

FUGC_DIR="${ROOT_DIR}_fugc"
FA_DIR="${ROOT_DIR}_fugc_fa"
BN_DIR="${ROOT_DIR}_fugc_fa_bn"

bash scripts/train_hidden_context_local_fugc_headonly_v2.sh \
  "${FUGC_DIR}" \
  "${BASE_CKPT}" \
  "${SEED}"

bash scripts/train_hidden_context_local_fugc_fa_headonly_v3.sh \
  "${FA_DIR}" \
  "${FUGC_DIR}/checkpoints/best_model.pth" \
  "${SEED}"

bash scripts/recalibrate_hidden_context_local_fugc_fa_bn_v1.sh \
  "${BN_DIR}" \
  "${FA_DIR}/checkpoints/best_model.pth"

echo "Final recalibrated checkpoint: ${BN_DIR}/checkpoints/best_model.pth"
