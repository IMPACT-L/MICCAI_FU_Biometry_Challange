#!/usr/bin/env python3
"""Build FUGC length-prior submission variants from an existing anchor.

This is intentionally conservative: it changes only FUGC, keeps each segment
center and angle fixed, and scales endpoint distance around the segment center.
Use it to test whether the public FUGC convention prefers shorter/longer
segments without changing all other tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

import numpy as np


def normalize_key(task_id: str, image_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return task_id, "/".join(parts)
    return task_id, f"{task_id}/{os.path.basename(normalized)}"


def load_manifest_sizes(manifest_path: str) -> dict[tuple[str, str], tuple[int, int]]:
    sizes: dict[tuple[str, str], tuple[int, int]] = {}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = normalize_key(str(row["task_id"]), str(row["image_path"]))
            sizes[key] = (int(row["height"]), int(row["width"]))
    return sizes


def update_points(record: dict, points: np.ndarray, height: int, width: int) -> dict:
    clipped = points.astype(np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, max(float(width - 1), 1.0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, max(float(height - 1), 1.0))
    normalized = clipped.copy()
    normalized[:, 0] /= max(float(width), 1.0)
    normalized[:, 1] /= max(float(height), 1.0)
    updated = dict(record)
    updated["predicted_points_pixels"] = [round(float(v), 6) for v in clipped.reshape(-1)]
    updated["predicted_points_normalized"] = [round(float(v), 8) for v in normalized.reshape(-1)]
    return updated


def scale_segment(points: np.ndarray, scale: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    return center + (points - center) * float(scale)


def safe_name(scale: float) -> str:
    return f"s{scale:.3f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--scales", default="0.98,0.96,0.94")
    parser.add_argument("--zip-submission", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Output root already exists; refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    with open(args.input_json, "r", encoding="utf-8") as handle:
        anchor_predictions = json.load(handle)
    sizes = load_manifest_sizes(args.manifest)
    scales = [float(item) for item in args.scales.split(",") if item.strip()]

    for scale in scales:
        output_dir = output_root / safe_name(scale)
        output_dir.mkdir(parents=True, exist_ok=False)
        output_predictions = []
        shifts: list[float] = []
        length_deltas: list[float] = []
        changed = 0

        for record in anchor_predictions:
            task_id = str(record["task_id"])
            if task_id != "FUGC":
                output_predictions.append(record)
                continue
            key = normalize_key(task_id, str(record["image_path"]))
            if key not in sizes:
                raise KeyError(f"Missing manifest size for {key}")
            height, width = sizes[key]
            points = np.asarray(record["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
            refined = scale_segment(points, scale)
            old_length = float(np.linalg.norm(points[1] - points[0]))
            new_length = float(np.linalg.norm(refined[1] - refined[0]))
            shifts.append(float(np.linalg.norm(refined - points, axis=1).mean()))
            length_deltas.append(new_length - old_length)
            changed += 1
            output_predictions.append(update_points(record, refined, height=height, width=width))

        output_json = output_dir / "regression_predictions.json"
        output_json.write_text(json.dumps(output_predictions, indent=2), encoding="utf-8")
        if args.zip_submission:
            with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(output_json, arcname="regression_predictions.json")

        summary = {
            "input_json": args.input_json,
            "scale": scale,
            "changed_fugc": changed,
            "mean_shift_px": float(np.mean(shifts)) if shifts else 0.0,
            "max_shift_px": float(np.max(shifts)) if shifts else 0.0,
            "mean_length_delta_px": float(np.mean(length_deltas)) if length_deltas else 0.0,
            "min_length_delta_px": float(np.min(length_deltas)) if length_deltas else 0.0,
            "max_length_delta_px": float(np.max(length_deltas)) if length_deltas else 0.0,
        }
        (output_dir / "fugc_length_prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
