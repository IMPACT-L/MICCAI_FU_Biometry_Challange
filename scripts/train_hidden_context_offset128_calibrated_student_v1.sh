#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_offset128_calibrated_student_v1_seed42}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"
TEACHER_JSON="${3:-output/submissions/dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_focused_v6_top2_seed42/regression_predictions.json}"
PSEUDO_ROOT="${4:-output/pseudo_datasets/dinov3_vitb_hidden_context_offset128_calibrated_student_v1}"
SEED="${5:-42}"

python scripts/build_pseudo_dataset_from_submission.py \
  --base-data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --submission-json "${TEACHER_JSON}" \
  --output-root "${PSEUDO_ROOT}" \
  --pseudo-only

python baseline/train.py \
  --data-root "${PSEUDO_ROOT}" \
  --seed "${SEED}" \
  --epochs 80 \
  --val-split 0.05 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.0005 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 1e-5 \
  --init-checkpoint "${BASE_CKPT}" \
  --encoder-name vit_base_patch16_dinov3 \
  --input-size 512 \
  --heatmap-size 128 \
  --heatmap-sigma 2.8 \
  --fpn-mode task_specific \
  --fpn-type fpn \
  --task-head-profile challenge_v1 \
  --task-decoder-profile hidden_a4c_hc_ivc_fugc_offset_v1 \
  --task-adapter-profile context_local_v1 \
  --split-mode pseudo_domain_grouped \
  --augmentation-profile baseline \
  --checkpoint-score-mode server_proxy_v1 \
  --freeze-encoder \
  --measurement-loss-weight 0.0 \
  --dataset-loss-weight 0.0 \
  --structure-loss-weight 0.0 \
  --femur-shaft-loss-weight 0.0 \
  --fugc-segment-loss-weight 0.0 \
  --ivc-band-loss-weight 0.0 \
  --task-loss-weights A4C=1.2,AOP=1.4,FA=1.0,FUGC=1.0,HC=1.0,IVC=1.0,PLAX=1.0,PSAX=1.25,fetal_femur=1.0 \
  --sampler-task-weights A4C=1.2,AOP=1.4,FUGC=1.0,PSAX=1.25 \
  --ema-decay 0.999 \
  --output-dir "${OUTPUT_DIR}"
