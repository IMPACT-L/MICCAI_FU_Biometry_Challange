#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-output/runs/dinov3_vitb_hidden_context_target_equivariance_safe_v2_seed42}"
INIT_CKPT="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_femurclean_seed42/checkpoints/best_model.pth}"
SEED="${3:-42}"

python scripts/train_target_equivariance_adapt_v1.py \
  --manifest data/manifests/validation_manifest.csv \
  --checkpoint-path "${INIT_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-profile hidden_context_local_offset128_v1 \
  --task-ids FUGC,HC,IVC,fetal_femur \
  --trainable-scope heads_adapters \
  --epochs 7 \
  --batch-size 8 \
  --num-workers 4 \
  --learning-rate 1.5e-6 \
  --anchor-weight 0.50 \
  --max-rotation-deg 2.5 \
  --scale-jitter 0.030 \
  --anisotropic-jitter 0.012 \
  --max-translation 0.020 \
  --brightness-jitter 0.030 \
  --contrast-jitter 0.070 \
  --gamma-jitter 1.07 \
  --noise-std 0.002 \
  --seed "${SEED}"
