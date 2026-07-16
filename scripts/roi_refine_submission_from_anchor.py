#!/usr/bin/env python3
"""Apply a trained ROI model to selected tasks on top of an anchor submission.

This keeps all non-target tasks exactly from the anchor JSON. For selected tasks,
it crops around the anchor landmarks, runs a ROI-trained model on the crop, maps
the crop prediction back to the original image, and accepts it only if the mean
pixel shift remains below the configured safety gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from submit import (  # noqa: E402
    EXPECTED_VALIDATION_COUNTS,
    build_task_configs,
    predict_task_transformed_coords,
    round_float_list,
    validate_predictions,
)
from two_stage_roi_submit import (  # noqa: E402
    crop_normalized_to_original_normalized,
    load_submission_model,
    make_roi_box,
    mean_pixel_shift,
    parse_task_float_map,
    parse_task_ids,
    read_manifest_rows,
    resolve_manifest_image_path,
)
from baseline.utils import (  # noqa: E402
    canonicalize_task_coords,
    letterbox_image_and_points,
    transformed_coords_to_original_normalized,
)


def normalize_key(task_id: str, image_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return task_id, "/".join(parts)
    return task_id, f"{task_id}/{os.path.basename(normalized)}"


def load_anchor(anchor_json: str) -> dict[tuple[str, str], dict]:
    with open(anchor_json, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    mapping = {}
    for item in predictions:
        key = normalize_key(str(item["task_id"]), str(item["image_path"]))
        mapping[key] = item
    return mapping


def normalized_to_pixel_points(norm_coords: list[float] | np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(norm_coords, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] *= max(float(width) - 1.0, 1.0)
    points[:, 1] *= max(float(height) - 1.0, 1.0)
    return points


def item_from_normalized(row: dict, task_id: str, norm_coords: np.ndarray, original_size: tuple[int, int]) -> dict:
    pred = round_float_list(np.asarray(norm_coords, dtype=np.float32).reshape(-1).tolist())
    original_height, original_width = original_size
    pixel_coords = []
    for idx in range(0, len(pred), 2):
        pixel_coords.extend([float(pred[idx]) * original_width, float(pred[idx + 1]) * original_height])
    return {
        "image_path": row["image_path"],
        "task_id": task_id,
        "predicted_points_normalized": pred,
        "predicted_points_pixels": round_float_list(pixel_coords),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI-refine selected tasks on top of an existing anchor submission.")
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--anchor-json", required=True)
    parser.add_argument("--roi-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-model-profile", default=None)
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--fpn-mode", choices=("shared", "task_specific"), default=None)
    parser.add_argument("--fpn-type", choices=("fpn", "bifpn"), default=None)
    parser.add_argument("--task-head-profile", choices=("uniform", "challenge_legacy_v1", "challenge_v1"), default=None)
    parser.add_argument("--task-decoder-profile", default=None)
    parser.add_argument("--task-adapter-profile", default=None)
    parser.add_argument("--roi-task-ids", default="HC,IVC,PLAX")
    parser.add_argument("--roi-context", type=float, default=1.8)
    parser.add_argument("--roi-min-size", type=float, default=112.0)
    parser.add_argument("--roi-gate-max-shift-px", type=float, default=8.0)
    parser.add_argument("--roi-gate-task-thresholds", default="HC=7,IVC=6,PLAX=6")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    anchor_predictions = load_anchor(args.anchor_json)
    rows = read_manifest_rows(manifest_path)
    task_configs = build_task_configs(manifest_path)
    roi_task_ids = parse_task_ids(args.roi_task_ids) or set()
    task_gate_thresholds = parse_task_float_map(args.roi_gate_task_thresholds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_model, roi_config = load_submission_model(
        os.path.abspath(args.roi_checkpoint),
        task_configs,
        device,
        args.roi_model_profile,
        args.encoder_name,
        args.fpn_mode,
        args.fpn_type,
        args.task_head_profile,
        args.task_decoder_profile,
        args.task_adapter_profile,
    )
    print(f"ROI config: {roi_config}")
    print(f"Anchor JSON: {args.anchor_json}")
    print(f"ROI tasks: {sorted(roi_task_ids)}")
    print(f"ROI gate: default={args.roi_gate_max_shift_px}, task_thresholds={task_gate_thresholds}")

    transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    output_items: list[dict | None] = [None] * len(rows)
    roi_batches: dict[str, list[dict]] = {}
    accepted = Counter()
    rejected = Counter()

    for idx, row in enumerate(rows):
        task_id = str(row["task_id"])
        key = normalize_key(task_id, row["image_path"])
        anchor_item = anchor_predictions.get(key)
        if anchor_item is None:
            raise KeyError(f"Missing anchor prediction for {key}")
        output_items[idx] = dict(anchor_item)
        if task_id not in roi_task_ids:
            continue

        image_path = resolve_manifest_image_path(row, manifest_path)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        anchor_norm = np.asarray(anchor_item["predicted_points_normalized"], dtype=np.float32)
        points_px = normalized_to_pixel_points(anchor_norm, image_width, image_height)
        x0, y0, x1, y1 = make_roi_box(
            points_px,
            image_width=image_width,
            image_height=image_height,
            context=float(args.roi_context),
            min_size=float(args.roi_min_size),
        )
        crop = image[y0:y1, x0:x1]
        dummy = np.zeros((int(row["num_points"]), 2), dtype=np.float32)
        crop_letterboxed, _, crop_meta = letterbox_image_and_points(crop, dummy, int(roi_config["input_size"]))
        transformed = transforms(image=crop_letterboxed)
        roi_batches.setdefault(task_id, []).append(
            {
                "row_idx": idx,
                "image": transformed["image"],
                "meta": crop_meta,
                "crop_box": (x0, y0, x1, y1),
                "original_size": (image_height, image_width),
                "anchor_norm": anchor_norm,
                "row": row,
            }
        )

    with torch.no_grad():
        for task_id, items in tqdm(sorted(roi_batches.items()), desc="ROI refinement"):
            for start in range(0, len(items), int(args.batch_size)):
                chunk = items[start : start + int(args.batch_size)]
                images = torch.stack([item["image"] for item in chunk], 0).to(device)
                outputs_transformed = predict_task_transformed_coords(roi_model, images, task_id, "none", None)
                crop_local_norm = transformed_coords_to_original_normalized(
                    outputs_transformed,
                    [item["meta"] for item in chunk],
                )
                crop_norm = canonicalize_task_coords(
                    crop_normalized_to_original_normalized(
                        crop_local_norm,
                        [item["crop_box"] for item in chunk],
                        [item["original_size"] for item in chunk],
                    ),
                    task_id,
                )
                for local_idx, item in enumerate(chunk):
                    row_idx = item["row_idx"]
                    candidate = crop_norm[local_idx].cpu().numpy()
                    threshold = task_gate_thresholds.get(task_id, float(args.roi_gate_max_shift_px))
                    shift_px = mean_pixel_shift(item["anchor_norm"], candidate, item["original_size"])
                    if shift_px <= float(threshold):
                        output_items[row_idx] = item_from_normalized(
                            item["row"],
                            task_id,
                            candidate,
                            item["original_size"],
                        )
                        accepted[task_id] += 1
                    else:
                        rejected[task_id] += 1

    predictions = [item for item in output_items if item is not None]
    validate_predictions(predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "regression_predictions.json"
    json_path.write_text(json.dumps(predictions, indent=2))
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="regression_predictions.json")

    summary = {
        "anchor_json": args.anchor_json,
        "roi_checkpoint": args.roi_checkpoint,
        "roi_tasks": sorted(roi_task_ids),
        "accepted": dict(sorted(accepted.items())),
        "rejected": dict(sorted(rejected.items())),
        "gate_default": float(args.roi_gate_max_shift_px),
        "gate_task_thresholds": task_gate_thresholds,
    }
    (output_dir / "roi_refine_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {json_path}")
    print(f"Wrote {zip_path}")
    print(f"Validated counts: {EXPECTED_VALIDATION_COUNTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
