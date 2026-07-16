#!/usr/bin/env python3
"""Diagnose the FUGC public-validation gap.

This script overlays FUGC submission predictions on the released validation
images and compares their geometry against the released FUGC training labels.
It is intentionally diagnostic only: it does not alter predictions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (80, 255, 80),
    (60, 170, 255),
    (255, 120, 60),
    (220, 80, 255),
    (255, 255, 80),
    (80, 255, 255),
]


def normalize_key(task_id: str, image_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return task_id, "/".join(parts)
    return task_id, f"{task_id}/{os.path.basename(normalized)}"


def parse_points(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    points = np.asarray(value, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    if points.shape != (2, 2):
        raise ValueError(f"Expected two 2D points, got shape {points.shape}")
    return points


def load_manifest_fugc(manifest_path: str) -> list[dict[str, object]]:
    manifest_path = os.path.abspath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    rows: list[dict[str, object]] = []
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["task_id"] != "FUGC":
                continue
            abs_path = row.get("abs_path", "")
            if abs_path and not os.path.isabs(abs_path):
                abs_path = os.path.normpath(os.path.join(manifest_dir, abs_path))
            rows.append(
                {
                    "task_id": "FUGC",
                    "image_path": row["image_path"],
                    "abs_path": abs_path,
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                }
            )
    return rows


def load_submission_fugc(json_path: str) -> dict[tuple[str, str], np.ndarray]:
    with open(json_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    preds: dict[tuple[str, str], np.ndarray] = {}
    for item in payload:
        task_id = str(item.get("task_id", ""))
        image_path = str(item.get("image_path", ""))
        if task_id != "FUGC" and not image_path.startswith("FUGC/"):
            continue
        points = item.get("predicted_points_pixels")
        if points is None:
            points = item.get("points") or item.get("coords") or item.get("landmarks")
        preds[normalize_key("FUGC", image_path)] = parse_points(points)
    return preds


def segment_features(points: np.ndarray, width: float, height: float) -> dict[str, float]:
    vector = points[1] - points[0]
    length_px = float(np.linalg.norm(vector))
    return {
        "length_px": length_px,
        "length_norm_xy": float(np.linalg.norm(vector / np.asarray([width, height], dtype=np.float32))),
        "angle_deg": float(math.degrees(math.atan2(float(vector[1]), float(vector[0])))),
        "center_x_norm": float(points[:, 0].mean() / width),
        "center_y_norm": float(points[:, 1].mean() / height),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def load_training_geometry(train_csv: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(train_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            p1 = parse_points([ast.literal_eval(row["point_1_xy"]), ast.literal_eval(row["point_2_xy"])])
            rows.append(segment_features(p1, float(row["width"]), float(row["height"])))
    return rows


def draw_panel(
    image_path: str,
    image_label: str,
    candidate_points: list[tuple[str, np.ndarray]],
    panel_width: int,
) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    scale = panel_width / float(width)
    panel = cv2.resize(image, (panel_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    for index, (name, points) in enumerate(candidate_points):
        color = COLORS[index % len(COLORS)]
        scaled = np.round(points * scale).astype(np.int32)
        cv2.line(panel, tuple(scaled[0]), tuple(scaled[1]), color, 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(scaled[0]), 4, color, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(scaled[1]), 4, color, -1, cv2.LINE_AA)
        feat = segment_features(points, width, height)
        text = f"{name}: L={feat['length_px']:.1f} A={feat['angle_deg']:.1f}"
        cv2.putText(panel, text, (8, 22 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(panel, image_label, (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def make_montage(panels: list[np.ndarray], cols: int) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to draw")
    max_h = max(panel.shape[0] for panel in panels)
    max_w = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(canvas)
    rows = []
    for start in range(0, len(padded), cols):
        chunk = padded[start : start + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(padded[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--train-csv", default="data/csv/FUGC_train.csv")
    parser.add_argument("--submission", action="append", required=True, help="NAME=path/to/regression_predictions.json")
    parser.add_argument("--output-dir", default="output/diagnostics/fugc_public_gap")
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest_fugc(args.manifest)
    train_geometry = load_training_geometry(args.train_csv)
    submissions: list[tuple[str, dict[tuple[str, str], np.ndarray]]] = []
    for spec in args.submission:
        if "=" not in spec:
            raise ValueError("--submission must be NAME=path")
        name, path = spec.split("=", 1)
        submissions.append((name, load_submission_fugc(path)))

    per_submission: dict[str, dict[str, dict[str, float]]] = {}
    panels: list[np.ndarray] = []
    for row in manifest_rows:
        key = normalize_key("FUGC", str(row["image_path"]))
        candidate_points = []
        for name, preds in submissions:
            if key in preds:
                candidate_points.append((name, preds[key]))
                feat = segment_features(preds[key], float(row["width"]), float(row["height"]))
                for metric_name, value in feat.items():
                    per_submission.setdefault(name, {}).setdefault(metric_name, []).append(value)
        panels.append(draw_panel(str(row["abs_path"]), str(row["image_path"]), candidate_points, args.panel_width))

    montage = make_montage(panels, cols=args.cols)
    montage_path = output_dir / "fugc_public_predictions_montage.jpg"
    cv2.imwrite(str(montage_path), montage)

    stats: dict[str, object] = {
        "train": {name: summarize([row[name] for row in train_geometry]) for name in train_geometry[0].keys()},
        "submissions": {
            name: {metric: summarize(values) for metric, values in metrics.items()}
            for name, metrics in per_submission.items()
        },
        "montage": str(montage_path),
    }
    with open(output_dir / "fugc_public_gap_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    summary_lines = [f"Montage: {montage_path}", "", "Training geometry:"]
    for metric, values in stats["train"].items():
        summary_lines.append(
            f"  {metric}: mean={values['mean']:.4f}, p10={values['p10']:.4f}, p50={values['p50']:.4f}, p90={values['p90']:.4f}"
        )
    for name, metrics in stats["submissions"].items():
        summary_lines.append("")
        summary_lines.append(f"Submission: {name}")
        for metric, values in metrics.items():
            summary_lines.append(
                f"  {metric}: mean={values['mean']:.4f}, p10={values['p10']:.4f}, p50={values['p50']:.4f}, p90={values['p90']:.4f}"
            )
    summary = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
