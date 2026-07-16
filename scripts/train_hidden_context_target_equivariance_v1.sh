#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_target_equivariance_v1_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_femurclean_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python scripts/train_target_equivariance_adapt_v1.py \
  --manifest data/manifests/validation_manifest.csv \
  --checkpoint-path "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-profile hidden_context_local_offset128_v1 \
  --task-ids A4C,AOP,HC,IVC,PSAX,fetal_femur \
  --trainable-scope heads_adapters \
  --epochs 8 \
  --batch-size 8 \
  --num-workers 4 \
  --learning-rate 2e-6 \
  --anchor-weight 0.35 \
  --max-rotation-deg 3.0 \
  --scale-jitter 0.035 \
  --anisotropic-jitter 0.015 \
  --max-translation 0.025 \
  --brightness-jitter 0.035 \
  --contrast-jitter 0.08 \
  --gamma-jitter 1.08 \
  --noise-std 0.003 \
  --seed "${SEED}"
