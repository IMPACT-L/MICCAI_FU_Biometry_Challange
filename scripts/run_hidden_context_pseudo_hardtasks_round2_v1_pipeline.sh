#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
BASE_RUN_DIR="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42}"
BASE_SUBMISSION_DIR="${3:-output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42}"
RUN_DIR="${4:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed${SEED}}"
SUBMISSION_DIR="${5:-output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed${SEED}}"
PSEUDO_ROOT="${6:-output/pseudo_datasets/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed${SEED}}"

if [ ! -f "${BASE_SUBMISSION_DIR}/regression_predictions.json" ]; then
  bash scripts/submit_hidden_context_pseudo_hardtasks_v1.sh \
    "${BASE_RUN_DIR}" \
    "${BASE_SUBMISSION_DIR}"
fi

bash scripts/train_hidden_context_pseudo_hardtasks_round2_v1.sh \
  "${RUN_DIR}" \
  "${BASE_RUN_DIR}/checkpoints/best_model.pth" \
  "${BASE_SUBMISSION_DIR}/regression_predictions.json" \
  "${PSEUDO_ROOT}" \
  "${SEED}"

bash scripts/submit_hidden_context_pseudo_hardtasks_round2_v1.sh \
  "${RUN_DIR}" \
  "${SUBMISSION_DIR}"
