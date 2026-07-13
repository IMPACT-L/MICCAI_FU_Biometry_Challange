#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from utils import (  # noqa: E402
    DEFAULT_NORMALIZER_EPS,
    MEASUREMENT_PAIRS,
    canonicalize_task_coords,
    compute_measurements_from_points,
    compute_normalization_stats_from_dataframe,
)


TASK_ORDER = ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"]


def normalize_eval_image_path(image_path: str, task_id: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return "/".join(parts)
    return f"{task_id}/{os.path.basename(normalized)}"


def load_dataframe(data_root: Path) -> pd.DataFrame:
    csv_dir = data_root / "csv"
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    csv_files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")
    dataframe = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
    is_regression = dataframe["task_name"].astype(str).eq("Regression")
    is_known = dataframe["task_id"].astype(str).isin(TASK_ORDER)
    return dataframe[is_regression | is_known].reset_index(drop=True)


def load_prediction_map(prediction_json: Path) -> dict[tuple[str, str], list[float]]:
    predictions = json.loads(prediction_json.read_text())
    pred_map: dict[tuple[str, str], list[float]] = {}
    for pred in predictions:
        task_id = str(pred["task_id"])
        image_path = normalize_eval_image_path(str(pred["image_path"]), task_id)
        pred_map[(task_id, image_path)] = list(map(float, pred["predicted_points_pixels"]))
    return pred_map


def row_points(row: pd.Series, num_points: int) -> np.ndarray:
    coords: list[float] = []
    for point_idx in range(1, num_points + 1):
        value = row.get(f"point_{point_idx}_xy")
        if pd.notna(value):
            coords.extend(json.loads(value))
        else:
            coords.extend([0.0, 0.0])
    return np.asarray(coords, dtype=np.float32)


def compute_mre(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    pred = pred_coords.reshape(-1, 2)
    gt = gt_coords.reshape(-1, 2)
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def compute_measurement_mae(pred_coords: np.ndarray, gt_coords: np.ndarray, task_id: str) -> float:
    if task_id not in MEASUREMENT_PAIRS:
        return float("nan")
    pred_measures = compute_measurements_from_points(pred_coords.reshape(1, -1, 2), task_id)
    gt_measures = compute_measurements_from_points(gt_coords.reshape(1, -1, 2), task_id)
    if pred_measures.shape[1] == 0:
        return float("nan")
    return float(np.abs(pred_measures - gt_measures).mean())


def evaluate(data_root: Path, prediction_json: Path) -> pd.DataFrame:
    dataframe = load_dataframe(data_root)
    pred_map = load_prediction_map(prediction_json)
    normalizers = compute_normalization_stats_from_dataframe(dataframe)
    rows = []

    for task_id in TASK_ORDER:
        task_df = dataframe[dataframe["task_id"].astype(str).eq(task_id)]
        if task_df.empty:
            continue
        num_points = int(task_df["num_classes"].iloc[0])
        mre_values = []
        measurement_values = []
        missing = 0
        for _, row in task_df.iterrows():
            image_path = normalize_eval_image_path(str(row["image_path"]), task_id)
            key = (task_id, image_path)
            if key not in pred_map:
                missing += 1
                continue
            gt = canonicalize_task_coords(row_points(row, num_points), task_id)
            pred = canonicalize_task_coords(np.asarray(pred_map[key], dtype=np.float32), task_id)
            mre_values.append(compute_mre(pred, gt))
            measurement_values.append(compute_measurement_mae(pred, gt, task_id))

        task_stats = normalizers.get(task_id, {})
        mre_iqr = max(float(task_stats.get("mre_iqr", DEFAULT_NORMALIZER_EPS)), DEFAULT_NORMALIZER_EPS)
        measurement_iqr_values = task_stats.get("measurement_iqr", [])
        measurement_iqr = (
            max(float(np.mean(measurement_iqr_values)), DEFAULT_NORMALIZER_EPS)
            if measurement_iqr_values
            else DEFAULT_NORMALIZER_EPS
        )
        mre = float(np.mean(mre_values)) if mre_values else float("nan")
        measurement = float(np.nanmean(measurement_values)) if measurement_values else float("nan")
        norm_mre = mre / mre_iqr if np.isfinite(mre) else float("nan")
        norm_measurement = measurement / measurement_iqr if np.isfinite(measurement) else float("nan")
        combined = (
            0.5 * norm_mre + 0.5 * norm_measurement
            if np.isfinite(norm_measurement)
            else norm_mre
        )
        rows.append(
            {
                "task_id": task_id,
                "samples": len(mre_values),
                "missing": missing,
                "mre_px": mre,
                "measurement_mae_px": measurement,
                "normalized_mre": norm_mre,
                "normalized_measurement_mae": norm_measurement,
                "combined_proxy": combined,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a submission JSON with local server-style measurement proxy.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--output-file")
    args = parser.parse_args()

    results = evaluate(Path(args.data_root), Path(args.prediction_json))
    if results.empty:
        raise ValueError("No evaluated tasks found.")

    metric_cols = ["mre_px", "measurement_mae_px", "normalized_mre", "normalized_measurement_mae", "combined_proxy"]
    summary = {column: float(results[column].mean()) for column in metric_cols}
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "tasks": results.to_dict(orient="records"),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
