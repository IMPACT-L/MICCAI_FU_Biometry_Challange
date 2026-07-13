#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1/checkpoints/best_model.pth}"
SUBMISSION_JSON="${3:-output/submissions/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1/regression_predictions.json}"
PSEUDO_ROOT="${4:-output/pseudo_datasets/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_bn_v1_hardtasks}"
SEED="${5:-42}"

python scripts/build_pseudo_dataset_from_submission.py \
  --base-data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --submission-json "${SUBMISSION_JSON}" \
  --output-root "${PSEUDO_ROOT}" \
  --task-ids A4C,FA,HC,IVC,PLAX

python baseline/train.py \
  --data-root "${PSEUDO_ROOT}" \
  --seed "${SEED}" \
  --epochs 80 \
  --val-split 0.15 \
  --early-stopping-patience 8 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 5e-6 \
  --init-checkpoint "${BASE_CKPT}" \
  --encoder-name vit_base_patch16_dinov3 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile hidden_a4c_hc_ivc_fugc_refine_v1 \
  --task-adapter-profile context_local_v1 \
  --split-mode pseudo_domain_grouped \
  --augmentation-profile baseline \
  --checkpoint-score-mode server_proxy_v1 \
  --train-task-ids A4C,FA,HC,IVC,PLAX \
  --freeze-encoder \
  --freeze-other-heads \
  --measurement-loss-weight 0.008 \
  --measurement-loss-tasks FA,HC,IVC,PLAX \
  --measurement-task-weights FA=1.00,HC=1.25,IVC=1.60,PLAX=0.85 \
  --dataset-loss-weight 0.0 \
  --femur-shaft-loss-weight 0.0 \
  --fugc-segment-loss-weight 0.0 \
  --ivc-band-loss-weight 0.08 \
  --task-loss-weights A4C=1.10,FA=1.10,HC=1.35,IVC=1.45,PLAX=1.15 \
  --sampler-task-weights A4C=1.10,FA=1.10,HC=1.35,IVC=1.45,PLAX=1.15 \
  --ema-decay 0.999 \
  --output-dir "${OUTPUT_DIR}"
