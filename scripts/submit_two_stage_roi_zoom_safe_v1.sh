#!/usr/bin/env bash
set -euo pipefail

COARSE_CKPT="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local/checkpoints/best_model.pth}"
ROI_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_roi_zoom_v1_seed42/checkpoints/best_model.pth}"
SUBMIT_DIR="${3:-output/submissions/dinov3_vitb_hidden_context_two_stage_roi_zoom_safe_v1_seed42}"

python scripts/two_stage_roi_submit.py \
  --manifest data/manifests/validation_manifest.csv \
  --coarse-checkpoint "${COARSE_CKPT}" \
  --roi-checkpoint "${ROI_CKPT}" \
  --model-profile hidden_context_local_fugc_headonly_v2 \
  --roi-task-ids A4C,AOP,FA,FUGC,HC,IVC,PLAX,PSAX,fetal_femur \
  --roi-context 1.80 \
  --roi-min-size 112 \
  --roi-gate-max-shift-px 10 \
  --roi-gate-task-thresholds A4C=14,AOP=8,FA=8,FUGC=5,HC=8,IVC=8,PLAX=10,PSAX=8,fetal_femur=6 \
  --batch-size 8 \
  --num-workers 4 \
  --output-dir "${SUBMIT_DIR}"
