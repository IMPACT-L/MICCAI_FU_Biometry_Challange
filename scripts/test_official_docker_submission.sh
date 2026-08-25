#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 DOCKER_IMAGE_REF" >&2
  exit 2
fi

IMAGE_REF="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="/tmp/fu_biometry_official_input"
OUTPUT_DIR="/tmp/fu_biometry_official_output"
TEST_LIMIT="${OFFICIAL_DOCKER_TEST_LIMIT:-0}"
TEST_PER_TASK="${OFFICIAL_DOCKER_TEST_PER_TASK:-0}"
GPU_ARGS="${OFFICIAL_DOCKER_GPU_ARGS---gpus all}"

rm -rf "$INPUT_DIR" "$OUTPUT_DIR"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

REPO_ROOT="$REPO_ROOT" OFFICIAL_DOCKER_TEST_LIMIT="$TEST_LIMIT" OFFICIAL_DOCKER_TEST_PER_TASK="$TEST_PER_TASK" python - <<'PY'
import csv
import os
import shutil
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
manifest_path = root / "data/manifests/validation_manifest.csv"
input_dir = Path("/tmp/fu_biometry_official_input")
limit = int(os.environ.get("OFFICIAL_DOCKER_TEST_LIMIT", "0") or "0")
per_task_limit = int(os.environ.get("OFFICIAL_DOCKER_TEST_PER_TASK", "0") or "0")
task_counts = {}

task_dirs = {}
rows_out = []
with manifest_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        task_id = row["task_id"]
        if per_task_limit > 0 and task_counts.get(task_id, 0) >= per_task_limit:
            continue
        if limit > 0 and len(rows_out) >= limit:
            break
        image_path = row["image_path"]
        src = (manifest_path.parent / row["abs_path"]).resolve()
        dst_dir = input_dir / f"{task_id}_test"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(image_path).name
        if not dst.exists():
            shutil.copy2(src, dst)
        rows_out.append(
            {
                "image_path": f"{task_id}/{Path(image_path).name}",
                "task_name": "Regression",
                "task_id": task_id,
                "num_classes": row["num_points"],
                "height": row["height"],
                "width": row["width"],
            }
        )
        task_counts[task_id] = task_counts.get(task_id, 0) + 1

with (input_dir / "test_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["image_path", "task_name", "task_id", "num_classes", "height", "width"],
    )
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Prepared {len(rows_out)} rows at {input_dir}")
PY

GPU_ARG_ARRAY=()
if [[ -n "$GPU_ARGS" ]]; then
  # shellcheck disable=SC2206
  GPU_ARG_ARRAY=($GPU_ARGS)
fi

docker run --rm "${GPU_ARG_ARRAY[@]}" \
  --network none \
  --memory 7g --cpus 4 --shm-size 2g \
  -v "$INPUT_DIR:/input:ro" \
  -v "$OUTPUT_DIR:/output:rw" \
  -e GU_INPUT_DIR=/input \
  -e GU_OUTPUT_DIR=/output \
  "$IMAGE_REF"

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("/tmp/fu_biometry_official_output/regression_predictions.json")
data = json.loads(path.read_text())
print(f"{len(data)} predictions")
print(dict(sorted(Counter(item["task_id"] for item in data).items())))
bad = [item for item in data if set(item.keys()) != {"image_path", "task_id", "predicted_points_pixels"}]
if bad:
    raise SystemExit(f"Unexpected keys in output: {bad[0].keys()}")
print(f"Output OK: {path}")
PY
