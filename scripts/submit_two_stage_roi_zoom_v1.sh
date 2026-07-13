#!/usr/bin/env bash
set -euo pipefail

COARSE_CKPT="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local/checkpoints/best_model.pth}"
ROI_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_roi_zoom_v1_seed42/checkpoints/best_model.pth}"
SUBMIT_DIR="${3:-output/submissions/dinov3_vitb_hidden_context_two_stage_roi_zoom_v1_seed42}"

python scripts/two_stage_roi_submit.py \
  --manifest data/manifests/validation_manifest.csv \
  --coarse-checkpoint "${COARSE_CKPT}" \
  --roi-checkpoint "${ROI_CKPT}" \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --roi-task-ids A4C,AOP,FA,FUGC,HC,IVC,PLAX,PSAX,fetal_femur \
  --roi-context 1.80 \
  --roi-min-size 112 \
  --batch-size 8 \
  --num-workers 4 \
  --output-dir "${SUBMIT_DIR}"
