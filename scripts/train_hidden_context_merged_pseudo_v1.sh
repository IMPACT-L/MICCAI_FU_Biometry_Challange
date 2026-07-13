#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_taskfpn_hidden_context_merged_pseudo_v1_seed42}"
BASE_CKPT="${2:-output/runs/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42/checkpoints/best_model.pth}"
MERGED_JSON="${3:-output/submissions/dinov3_vitb_taskfpn_hidden_context_merged_teacher_v1/regression_predictions.json}"
PSEUDO_ROOT="${4:-output/pseudo_datasets/dinov3_vitb_taskfpn_hidden_context_merged_teacher_v1}"
SEED="${5:-42}"

mkdir -p "$(dirname "${MERGED_JSON}")"

python scripts/merge_submission_predictions_by_task.py \
  --fallback-json output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42/regression_predictions.json \
  --task-source A4C=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_fugc_a4c_headonly_v3_seed42/regression_predictions.json \
  --task-source AOP=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_aop_headonly_seed42/regression_predictions.json \
  --task-source FA=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_hcivcfa_joint_seed42/regression_predictions.json \
  --task-source FUGC=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_fugc_headonly_seed42/regression_predictions.json,output/submissions/dinov3_vitb_taskfpn_hidden_context_local_fugc_fa_headonly_v3_seed42/regression_predictions.json \
  --task-source HC=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_hcivcfa_joint_seed42/regression_predictions.json \
  --task-source IVC=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_hcivcfa_joint_seed42/regression_predictions.json \
  --task-source PLAX=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_plax_headonly_seed42_v2/regression_predictions.json \
  --task-source PSAX=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_psax_headonly_seed42/regression_predictions.json \
  --task-source fetal_femur=output/submissions/dinov3_vitb_taskfpn_hidden_context_local_femur_headonly_seed42_v2/regression_predictions.json \
  --output-json "${MERGED_JSON}"

python scripts/build_pseudo_dataset_from_submission.py \
  --base-data-root data \
  --manifest data/manifests/validation_manifest.csv \
  --submission-json "${MERGED_JSON}" \
  --output-root "${PSEUDO_ROOT}"

python baseline/train.py \
  --data-root "${PSEUDO_ROOT}" \
  --seed "${SEED}" \
  --epochs 120 \
  --val-split 0.15 \
  --early-stopping-patience 10 \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 3e-6 \
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
  --freeze-encoder \
  --measurement-loss-weight 0.008 \
  --measurement-loss-tasks FA,HC,IVC,PLAX,fetal_femur \
  --measurement-task-weights FA=1.00,HC=1.20,IVC=1.45,PLAX=0.80,fetal_femur=0.75 \
  --dataset-loss-weight 0.0 \
  --femur-shaft-loss-weight 0.15 \
  --fugc-segment-loss-weight 0.08 \
  --ivc-band-loss-weight 0.08 \
  --task-loss-weights A4C=1.10,AOP=1.05,FA=1.15,FUGC=1.10,HC=1.35,IVC=1.45,PLAX=1.10,PSAX=1.10,fetal_femur=1.25 \
  --sampler-task-weights A4C=1.10,AOP=1.05,FA=1.15,FUGC=1.10,HC=1.35,IVC=1.45,PLAX=1.10,PSAX=1.10,fetal_femur=1.25 \
  --ema-decay 0.999 \
  --output-dir "${OUTPUT_DIR}"
