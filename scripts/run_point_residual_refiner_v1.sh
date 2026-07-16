#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_point_residual_refiner_v1_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_point_residual_refiner_v1_seed42}"
SEED="${3:-42}"

TRAIN_PRED_JSON="${TRAIN_PRED_JSON:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/predictions/regression_predictions.json}"
ANCHOR_JSON="${ANCHOR_JSON:-output/submissions/current_anchor_student_blends_v1/all_hard_light/regression_predictions.json}"

python scripts/train_apply_point_residual_refiner_v1.py train \
  --data-root data \
  --train-pred-json "${TRAIN_PRED_JSON}" \
  --output-dir "${RUN_DIR}" \
  --task-ids A4C,AOP,FUGC,HC,IVC,PLAX,PSAX,fetal_femur \
  --epochs 80 \
  --patience 8 \
  --batch-size 256 \
  --num-workers 4 \
  --patch-size 64 \
  --max-delta-px 18.0 \
  --learning-rate 2e-4 \
  --seed "${SEED}"

python scripts/train_apply_point_residual_refiner_v1.py apply \
  --checkpoint-path "${RUN_DIR}/point_residual_refiner_v1.pth" \
  --input-json "${ANCHOR_JSON}" \
  --manifest data/manifests/validation_manifest.csv \
  --output-dir "${SUBMISSION_DIR}" \
  --task-ids A4C,AOP,FUGC,HC,IVC,PLAX,PSAX,fetal_femur \
  --strength 0.35 \
  --max-apply-px 5.0 \
  --row-gate-px 5.0

python scripts/audit_submission_before_upload.py \
  --candidate-json "${SUBMISSION_DIR}/regression_predictions.json" \
  --reference-json "${ANCHOR_JSON}" \
  --output-json "${SUBMISSION_DIR}/pre_submit_audit.json"
