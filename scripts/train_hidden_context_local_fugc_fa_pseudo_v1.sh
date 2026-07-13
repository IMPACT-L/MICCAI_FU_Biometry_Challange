#!/usr/bin/env bash
set -euo pipefail

PSEUDO_ROOT="${1:-output/pseudo_datasets/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1/checkpoints/best_model.pth}"
OUTPUT_DIR="${3:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_pseudo_v1}"

python scripts/build_pseudo_dataset_from_submission.py \
  --base-data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --submission-json output/submissions/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1/regression_predictions.json \
  --output-root "${PSEUDO_ROOT}"

python baseline/train.py \
  --data-root "${PSEUDO_ROOT}" \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --seed 42 \
  --epochs 200 \
  --early-stopping-patience 8 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --freeze-encoder \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.0 \
  --output-dir "${OUTPUT_DIR}"
