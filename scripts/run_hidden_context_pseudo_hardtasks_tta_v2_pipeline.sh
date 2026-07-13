#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
BASE_RUN_DIR="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42}"
ROOT_RUN_DIR="${3:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_tta_v2_seed${SEED}}"
ROOT_SUBMISSION_DIR="${4:-output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_tta_v2_seed${SEED}}"

BN_RUN_DIR="${ROOT_RUN_DIR}_bn"
TEACHER_SUBMISSION_DIR="${ROOT_SUBMISSION_DIR}_teacher"
PSEUDO_ROOT="output/pseudo_datasets/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_tta_v2_seed${SEED}"
FINAL_RUN_DIR="${ROOT_RUN_DIR}"
FINAL_SUBMISSION_DIR="${ROOT_SUBMISSION_DIR}"

bash scripts/recalibrate_hidden_context_pseudo_hardtasks_bn_v1.sh \
  "${BN_RUN_DIR}" \
  "${BASE_RUN_DIR}/checkpoints/best_model.pth"

bash scripts/submit_hidden_context_pseudo_hardtasks_bn_tta_teacher_v2.sh \
  "${BN_RUN_DIR}" \
  "${TEACHER_SUBMISSION_DIR}"

bash scripts/train_hidden_context_pseudo_hardtasks_tta_v2.sh \
  "${FINAL_RUN_DIR}" \
  "${BN_RUN_DIR}/checkpoints/best_model.pth" \
  "${TEACHER_SUBMISSION_DIR}/regression_predictions.json" \
  "${PSEUDO_ROOT}" \
  "${SEED}"

bash scripts/submit_hidden_context_pseudo_hardtasks_tta_v2.sh \
  "${FINAL_RUN_DIR}" \
  "${FINAL_SUBMISSION_DIR}"

echo "Pipeline complete."
echo "Final run: ${FINAL_RUN_DIR}"
echo "Final submission: ${FINAL_SUBMISSION_DIR}"
