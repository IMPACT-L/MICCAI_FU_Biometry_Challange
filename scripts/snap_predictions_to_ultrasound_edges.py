#!/usr/bin/env python3
import argparse
import csv
import json
import os
import zipfile
from collections import defaultdict

import cv2
import numpy as np


TASK_DEFAULT_RADII = {
    "A4C": 3,
    "AOP": 3,
    "FA": 3,
    "FUGC": 2,
    "HC": 2,
    "IVC": 3,
    "PLAX": 2,
    "PSAX": 2,
    "fetal_femur": 3,
}

TASK_MAX_SHIFT = {
    "A4C": 2.5,
    "AOP": 2.5,
    "FA": 2.0,
    "FUGC": 1.5,
    "HC": 1.5,
    "IVC": 2.5,
    "PLAX": 1.5,
    "PSAX": 1.5,
    "fetal_femur": 2.5,
}


def normalize_key(task_id: str, image_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return task_id, "/".join(parts)
    return task_id, f"{task_id}/{os.path.basename(normalized)}"


def load_manifest_paths(manifest_path: str | None) -> dict[tuple[str, str], str]:
    if not manifest_path:
        return {}
    manifest_path = os.path.abspath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    mapping = {}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = str(row["task_id"])
            image_path = str(row["image_path"])
            abs_path = str(row.get("abs_path", ""))
            if abs_path and not os.path.isabs(abs_path):
                abs_path = os.path.normpath(os.path.join(manifest_dir, abs_path))
            mapping[normalize_key(task_id, image_path)] = abs_path
    return mapping


def resolve_image_path(data_root: str, task_id: str, image_path: str, manifest_paths: dict[tuple[str, str], str]) -> str | None:
    key = normalize_key(task_id, image_path)
    manifest_path = manifest_paths.get(key)
    if manifest_path and os.path.isfile(manifest_path):
        return manifest_path

    normalized = key[1]
    candidates = [
        os.path.join(data_root, "images", normalized),
        os.path.join(data_root, normalized),
        os.path.join(data_root, "images", task_id, os.path.basename(normalized)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def build_edge_score(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray_f = gray.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(gray_f, (0, 0), 1.1)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / max(float(np.percentile(grad, 99.5)), 1e-6)
    grad = np.clip(grad, 0.0, 1.0)

    bright = gray_f / max(float(np.percentile(gray_f, 99.5)), 1e-6)
    bright = np.clip(bright, 0.0, 1.0)

    # Ultrasound landmarks usually sit on bright tissue boundaries, not just any edge.
    score = 0.75 * grad + 0.25 * bright
    return cv2.GaussianBlur(score.astype(np.float32), (0, 0), 0.6)


def snap_point(point: np.ndarray, score: np.ndarray, radius: int, max_shift: float) -> np.ndarray:
    height, width = score.shape[:2]
    x, y = float(point[0]), float(point[1])
    cx = int(round(np.clip(x, 0, width - 1)))
    cy = int(round(np.clip(y, 0, height - 1)))
    x0, x1 = max(0, cx - radius), min(width - 1, cx + radius)
    y0, y1 = max(0, cy - radius), min(height - 1, cy + radius)
    if x1 < x0 or y1 < y0:
        return point.copy()

    patch = score[y0 : y1 + 1, x0 : x1 + 1]
    if patch.size == 0:
        return point.copy()

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    sigma2 = max((radius * 0.55) ** 2, 1e-6)
    local_score = patch - 0.08 * dist2 / sigma2
    best_idx = np.unravel_index(int(np.argmax(local_score)), local_score.shape)
    best = np.array([xx[best_idx], yy[best_idx]], dtype=np.float32)

    delta = best - point.astype(np.float32)
    shift = float(np.linalg.norm(delta))
    if shift <= 1e-6:
        return point.copy()
    if shift > max_shift:
        delta *= float(max_shift / shift)
    return point.astype(np.float32) + delta


def smooth_closed_curve(points: np.ndarray, strength: float = 0.15) -> np.ndarray:
    if len(points) < 4:
        return points
    previous_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    return (1.0 - strength) * points + strength * 0.5 * (previous_points + next_points)


def postprocess_points(points: np.ndarray, task_id: str, score: np.ndarray, shape_smoothing: bool = False) -> np.ndarray:
    radius = TASK_DEFAULT_RADII.get(task_id, 2)
    max_shift = TASK_MAX_SHIFT.get(task_id, 1.5)
    snapped = np.stack([snap_point(point, score, radius=radius, max_shift=max_shift) for point in points], axis=0)

    if shape_smoothing and task_id in {"HC", "PSAX"}:
        snapped = smooth_closed_curve(snapped, strength=0.08)
    elif shape_smoothing and task_id == "A4C" and len(snapped) >= 8:
        # Keep contour-like cardiac landmarks locally smooth while allowing valve/apex points to move.
        snapped = 0.92 * snapped + 0.08 * np.stack(
            [
                0.5 * (snapped[max(idx - 1, 0)] + snapped[min(idx + 1, len(snapped) - 1)])
                for idx in range(len(snapped))
            ],
            axis=0,
        )
    return snapped.astype(np.float32)


def update_prediction(record: dict, points_px: np.ndarray, width: int, height: int) -> dict:
    points_px[:, 0] = np.clip(points_px[:, 0], 0.0, max(float(width - 1), 1.0))
    points_px[:, 1] = np.clip(points_px[:, 1], 0.0, max(float(height - 1), 1.0))
    points_norm = points_px.copy()
    points_norm[:, 0] /= max(float(width), 1.0)
    points_norm[:, 1] /= max(float(height), 1.0)

    updated = dict(record)
    updated["predicted_points_pixels"] = [round(float(v), 6) for v in points_px.reshape(-1).tolist()]
    updated["predicted_points_normalized"] = [round(float(v), 6) for v in points_norm.reshape(-1).tolist()]
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservative ultrasound edge snapping post-process for landmark predictions."
    )
    parser.add_argument("--input-json", required=True, help="Input regression_predictions.json.")
    parser.add_argument("--output-dir", required=True, help="Output directory for post-processed predictions.")
    parser.add_argument("--data-root", default="data", help="Dataset root for resolving local images.")
    parser.add_argument("--manifest", default=None, help="Optional validation manifest for resolving official images.")
    parser.add_argument(
        "--task-ids",
        default="A4C,AOP,FA,FUGC,HC,IVC,PLAX,PSAX,fetal_femur",
        help="Comma-separated task IDs to post-process.",
    )
    parser.add_argument("--zip-submission", action="store_true", help="Also write submission.zip.")
    parser.add_argument(
        "--shape-smoothing",
        action="store_true",
        help="Apply contour smoothing after edge snapping. Disabled by default because sparse landmark order can be task-specific.",
    )
    args = parser.parse_args()

    enabled_tasks = {item.strip() for item in str(args.task_ids).split(",") if item.strip()}
    manifest_paths = load_manifest_paths(args.manifest)
    with open(args.input_json, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    image_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    output_predictions = []
    stats = defaultdict(list)
    missing = []

    for record in predictions:
        task_id = str(record["task_id"])
        if task_id not in enabled_tasks:
            output_predictions.append(record)
            continue

        image_path = resolve_image_path(args.data_root, task_id, record["image_path"], manifest_paths)
        if image_path is None:
            missing.append((task_id, record["image_path"]))
            output_predictions.append(record)
            continue

        if image_path not in image_cache:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                missing.append((task_id, record["image_path"]))
                output_predictions.append(record)
                continue
            image_cache[image_path] = (image, build_edge_score(image))
        image, score = image_cache[image_path]
        height, width = image.shape[:2]

        points = np.asarray(record["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
        snapped = postprocess_points(points, task_id, score, shape_smoothing=bool(args.shape_smoothing))
        shifts = np.linalg.norm(snapped - points, axis=1)
        stats[task_id].append(float(np.mean(shifts)))
        output_predictions.append(update_prediction(record, snapped, width=width, height=height))

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "regression_predictions.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(output_predictions, handle, indent=2)

    summary_path = os.path.join(args.output_dir, "edge_snap_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Edge snapping summary\n")
        for task_id in sorted(stats):
            values = np.asarray(stats[task_id], dtype=np.float32)
            line = f"{task_id}: n={len(values)}, mean_shift_px={float(values.mean()):.6f}, max_mean_shift_px={float(values.max()):.6f}"
            print(line)
            handle.write(line + "\n")
        if missing:
            handle.write(f"missing_images={len(missing)}\n")
            print(f"Warning: missing images for {len(missing)} predictions")

    if args.zip_submission:
        zip_path = os.path.join(args.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_json, arcname="regression_predictions.json")
        print(f"Wrote {zip_path}")

    print(f"Wrote {output_json}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
