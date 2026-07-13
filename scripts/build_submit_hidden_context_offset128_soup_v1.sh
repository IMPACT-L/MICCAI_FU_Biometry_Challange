#!/usr/bin/env bash
set -euo pipefail

ALPHA="${1:-0.85}"
RUN_DIR="${2:-output/runs/dinov3_vitb_hidden_context_offset128_soup_v1_a85}"
SUBMISSION_DIR="${3:-output/submissions/dinov3_vitb_hidden_context_offset128_soup_v1_a85}"
BASE_CKPT="${4:-output/runs/dinov3_vitb_hidden_context_offset128_v2_hcivc_seed42/checkpoints/best_model.pth}"
ADAPT_CKPT="${5:-output/runs/dinov3_vitb_hidden_context_offset128_v3_hardtask_ft_seed42/checkpoints/best_model.pth}"

mkdir -p "${RUN_DIR}/checkpoints"

python scripts/merge_checkpoints.py \
  --checkpoint-a "${BASE_CKPT}" \
  --checkpoint-b "${ADAPT_CKPT}" \
  --alpha "${ALPHA}" \
  --output-path "${RUN_DIR}/checkpoints/best_model.pth"

python baseline/model.py \
  --data-root data \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${RUN_DIR}/predictions" \
  --model-profile hidden_context_local_offset128_v2 \
  --batch-size 8

python baseline/evaluate.py \
  --data-root data \
  --pred-root "${RUN_DIR}/predictions" \
  --output-file "${RUN_DIR}/evaluation_results.json" \
  --summary-file "${RUN_DIR}/evaluation_summary.txt"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --model-profile hidden_context_local_offset128_v2 \
  --batch-size 8 \
  --num-workers 4

bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}"
