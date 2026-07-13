#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from utils import MEASUREMENT_PAIRS, compute_measurements_from_points  # noqa: E402


def normalize_image_path(image_path: str, task_id: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return "/".join(parts)
    return f"{task_id}/{os.path.basename(normalized)}"


def parse_task_values(values: str | None, default: float) -> dict[str, float]:
    parsed: dict[str, float] = {}
    if not values:
        return parsed
    for item in values.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected TASK=VALUE, got: {item}")
        task_id, value = item.split("=", 1)
        parsed[task_id.strip()] = float(value)
    return parsed


def load_manifest(manifest_path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    dataframe = pd.read_csv(manifest_path)
    sizes = {}
    for row in dataframe.itertuples(index=False):
        task_id = str(row.task_id)
        image_path = normalize_image_path(str(row.image_path), task_id)
        sizes[(task_id, image_path)] = (float(row.width), float(row.height))
    return sizes


def row_points(row: pd.Series, num_points: int) -> np.ndarray:
    coords: list[float] = []
    for point_idx in range(1, num_points + 1):
        value = row.get(f"point_{point_idx}_xy")
        if pd.notna(value):
            coords.extend(json.loads(value))
        else:
            coords.extend([0.0, 0.0])
    return np.asarray(coords, dtype=np.float32).reshape(num_points, 2)


def load_train_measurements(data_root: Path) -> dict[str, np.ndarray]:
    csv_dir = data_root / "csv"
    csv_files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    measurements: dict[str, list[np.ndarray]] = {}
    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        for task_id, task_df in dataframe.groupby("task_id", sort=True):
            task_id = str(task_id)
            if task_id not in MEASUREMENT_PAIRS:
                continue
            num_points = int(task_df["num_classes"].iloc[0])
            for _, row in task_df.iterrows():
                points = row_points(row, num_points)
                measures = compute_measurements_from_points(points[None, ...], task_id)[0]
                measurements.setdefault(task_id, []).append(measures)
    return {task_id: np.stack(values, axis=0) for task_id, values in measurements.items()}


def load_prediction_measurements(predictions: list[dict]) -> dict[str, np.ndarray]:
    measurements: dict[str, list[np.ndarray]] = {}
    for item in predictions:
        task_id = str(item["task_id"])
        if task_id not in MEASUREMENT_PAIRS:
            continue
        points = np.asarray(item["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
        measures = compute_measurements_from_points(points[None, ...], task_id)[0]
        measurements.setdefault(task_id, []).append(measures)
    return {task_id: np.stack(values, axis=0) for task_id, values in measurements.items()}


def calibrate_points(
    points: np.ndarray,
    task_id: str,
    factors: np.ndarray,
    alpha: float,
    max_point_shift: float,
) -> tuple[np.ndarray, float]:
    output = points.copy()
    max_shift_seen = 0.0
    for pair_idx, (start_idx, end_idx) in enumerate(MEASUREMENT_PAIRS.get(task_id, [])):
        if start_idx >= len(output) or end_idx >= len(output) or pair_idx >= len(factors):
            continue
        start = output[start_idx].copy()
        end = output[end_idx].copy()
        center = 0.5 * (start + end)
        factor = 1.0 + alpha * (float(factors[pair_idx]) - 1.0)
        new_start = center + (start - center) * factor
        new_end = center + (end - center) * factor

        shift = max(float(np.linalg.norm(new_start - start)), float(np.linalg.norm(new_end - end)))
        if max_point_shift > 0.0 and shift > max_point_shift:
            clip = max_point_shift / max(shift, 1e-8)
            new_start = start + (new_start - start) * clip
            new_end = end + (new_end - end) * clip
            shift = max_point_shift

        output[start_idx] = new_start
        output[end_idx] = new_end
        max_shift_seen = max(max_shift_seen, shift)
    return output, max_shift_seen


def normalized_from_pixels(points: np.ndarray, width: float, height: float) -> list[float]:
    denom = np.asarray([max(width, 1.0), max(height, 1.0)], dtype=np.float32)
    normalized = points / denom[None, :]
    normalized = np.clip(normalized, 0.0, 1.0)
    return [round(float(value), 6) for value in normalized.reshape(-1)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservatively calibrate submission measurement lengths.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--tasks", default="FUGC,HC,IVC,FA,PLAX,fetal_femur")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--task-alpha", default=None, help="Optional TASK=ALPHA overrides, comma-separated.")
    parser.add_argument("--min-factor", type=float, default=0.80)
    parser.add_argument("--max-factor", type=float, default=1.20)
    parser.add_argument("--max-point-shift", type=float, default=6.0)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = json.loads(input_path.read_text())
    sizes = load_manifest(Path(args.manifest))
    train_measurements = load_train_measurements(Path(args.data_root))
    pred_measurements = load_prediction_measurements(predictions)
    task_ids = {task.strip() for task in args.tasks.split(",") if task.strip()}
    task_alpha = parse_task_values(args.task_alpha, args.alpha)

    factors_by_task: dict[str, np.ndarray] = {}
    summary = {
        "input_json": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "alpha": float(args.alpha),
        "task_alpha": task_alpha,
        "min_factor": float(args.min_factor),
        "max_factor": float(args.max_factor),
        "max_point_shift": float(args.max_point_shift),
        "tasks": {},
    }

    for task_id in sorted(task_ids):
        if task_id not in train_measurements or task_id not in pred_measurements:
            continue
        train_median = np.median(train_measurements[task_id], axis=0)
        pred_median = np.median(pred_measurements[task_id], axis=0)
        raw_factors = train_median / np.clip(pred_median, 1e-6, None)
        factors = np.clip(raw_factors, float(args.min_factor), float(args.max_factor))
        factors_by_task[task_id] = factors.astype(np.float32)
        summary["tasks"][task_id] = {
            "train_median": [float(value) for value in train_median],
            "prediction_median": [float(value) for value in pred_median],
            "raw_factor": [float(value) for value in raw_factors],
            "clipped_factor": [float(value) for value in factors],
            "alpha": float(task_alpha.get(task_id, args.alpha)),
        }

    calibrated = []
    shift_values: dict[str, list[float]] = {}
    for item in predictions:
        task_id = str(item["task_id"])
        output_item = dict(item)
        if task_id in factors_by_task:
            image_path = normalize_image_path(str(item["image_path"]), task_id)
            width, height = sizes.get((task_id, image_path), (None, None))
            if width is None or height is None:
                # Fall back to the existing normalized vector when size metadata is absent.
                normalized = np.asarray(item["predicted_points_normalized"], dtype=np.float32).reshape(-1, 2)
                pixels = np.asarray(item["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
                inferred = pixels / np.clip(normalized, 1e-6, None)
                width = float(np.nanmedian(inferred[:, 0]))
                height = float(np.nanmedian(inferred[:, 1]))
            points = np.asarray(item["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
            new_points, max_shift = calibrate_points(
                points,
                task_id,
                factors_by_task[task_id],
                alpha=float(task_alpha.get(task_id, args.alpha)),
                max_point_shift=float(args.max_point_shift),
            )
            shift_values.setdefault(task_id, []).append(max_shift)
            output_item["predicted_points_pixels"] = [round(float(value), 6) for value in new_points.reshape(-1)]
            output_item["predicted_points_normalized"] = normalized_from_pixels(new_points, width, height)
        calibrated.append(output_item)

    for task_id, values in sorted(shift_values.items()):
        summary["tasks"].setdefault(task_id, {})
        summary["tasks"][task_id]["num_adjusted"] = len(values)
        summary["tasks"][task_id]["mean_max_point_shift"] = float(np.mean(values))
        summary["tasks"][task_id]["p90_max_point_shift"] = float(np.percentile(values, 90))

    json_path = output_dir / "regression_predictions.json"
    json_path.write_text(json.dumps(calibrated, indent=2))
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="regression_predictions.json")
    summary_path = output_dir / "measurement_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {json_path}")
    print(f"Wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
