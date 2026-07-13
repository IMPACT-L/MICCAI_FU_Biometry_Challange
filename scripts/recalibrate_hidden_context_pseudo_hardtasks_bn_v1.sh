#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_bn_v1_seed42}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42/checkpoints/best_model.pth}"

mkdir -p "${OUTPUT_DIR}/checkpoints"

python scripts/recalibrate_bn_on_manifest.py \
  --manifest data/manifests/validation_manifest.csv \
  --checkpoint-path "${BASE_CKPT}" \
  --output-path "${OUTPUT_DIR}/checkpoints/best_model.pth" \
  --encoder-name vit_base_patch16_dinov3 \
  --task-head-profile challenge_v1 \
  --task-decoder-profile hidden_a4c_hc_ivc_fugc_refine_v1 \
  --task-adapter-profile context_local_v1 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --input-size 512 \
  --batch-size 8 \
  --num-workers 4 \
  --num-passes 2
