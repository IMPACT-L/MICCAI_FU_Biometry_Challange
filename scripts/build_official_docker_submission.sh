#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DOCKER_IMAGE_REF [CHECKPOINT_PATH]" >&2
  echo "Example: $0 your-account/fub2026-final:v1 /path/to/best_model.pth" >&2
  exit 2
fi

IMAGE_REF="$1"
CHECKPOINT_PATH="${2:-output/runs/dinov3_vitb_hidden_context_offset128_v1_seed42/checkpoints/best_model.pth}"

if [[ "$IMAGE_REF" != "${IMAGE_REF,,}" ]]; then
  echo "Docker image reference must be lowercase: $IMAGE_REF" >&2
  exit 2
fi
if [[ "$IMAGE_REF" == your-* || "$IMAGE_REF" == your_* || "$IMAGE_REF" == *your_dockerhub* || "$IMAGE_REF" == *your-dockerhub* ]]; then
  echo "Docker image reference still looks like a placeholder: $IMAGE_REF" >&2
  echo "Use your real lowercase Docker Hub username, for example: your-account/fub2026-final:v1" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 2
fi

mkdir -p .docker_build
CONTEXT_DIR="$(mktemp -d .docker_build/official.XXXXXX)"

cleanup() {
  if [[ "${KEEP_DOCKER_CONTEXT:-0}" == "1" ]]; then
    echo "Kept Docker build context: $CONTEXT_DIR"
  else
    rm -rf "$CONTEXT_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$CONTEXT_DIR/baseline"
cp -a baseline/. "$CONTEXT_DIR/baseline/"
cp baseline/requirements.txt "$CONTEXT_DIR/requirements.txt"
cp submit.py "$CONTEXT_DIR/submit.py"
cp docker/official/Dockerfile "$CONTEXT_DIR/Dockerfile"
cp docker/official/model.py "$CONTEXT_DIR/model.py"
cp docker/official/predict.py "$CONTEXT_DIR/predict.py"
cp "$CHECKPOINT_PATH" "$CONTEXT_DIR/best_model.pth"

find "$CONTEXT_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$CONTEXT_DIR" -type f -name '*.pyc' -delete

echo "Building official challenge Docker image: $IMAGE_REF"
echo "Bundled checkpoint: $CHECKPOINT_PATH"
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  --build-arg "INFERENCE_BATCH_SIZE=${DOCKER_INFERENCE_BATCH_SIZE:-8}" \
  -t "$IMAGE_REF" "$CONTEXT_DIR"

SAFE_REF="$(printf '%s' "$IMAGE_REF" | tr '/:' '__')"
PACKAGE_DIR="${DOCKER_SUBMISSION_DIR:-output/docker_submissions/$SAFE_REF}"
if [[ -e "$PACKAGE_DIR" && "${OVERWRITE_DOCKER_SUBMISSION:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing Docker submission package: $PACKAGE_DIR" >&2
  echo "Use a new image tag, remove that directory, or set OVERWRITE_DOCKER_SUBMISSION=1." >&2
  exit 2
fi

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/submission"
printf '{"image": "%s"}\n' "$IMAGE_REF" > "$PACKAGE_DIR/submission/submission.json"
(cd "$PACKAGE_DIR" && zip -qr submission.zip submission)

echo "Built image: $IMAGE_REF"
echo "Wrote upload package: $PACKAGE_DIR/submission.zip"
unzip -l "$PACKAGE_DIR/submission.zip"
