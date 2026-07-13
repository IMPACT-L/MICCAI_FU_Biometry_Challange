#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_tta_v2_seed42}"
SUBMISSION_DIR="${2:-output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_tta_v2_seed42}"

python submit.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/best_model.pth" \
  --output-dir "${SUBMISSION_DIR}" \
  --encoder-name vit_base_patch16_dinov3 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile hidden_a4c_hc_ivc_fugc_refine_v1 \
  --task-adapter-profile context_local_v1 \
  --batch-size 8 \
  --num-workers 4
