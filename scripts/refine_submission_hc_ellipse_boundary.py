#!/usr/bin/env python3
"""HC-specific ellipse/radial boundary refinement for prediction JSON files.

The HC task uses two diameter pairs. This post-process keeps the current point
ordering but searches along each radial diameter direction for a stronger skull
boundary response, then lightly enforces that opposite points share one center.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


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
    mapping: dict[tuple[str, str], str] = {}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = str(row["task_id"])
            image_path = str(row["image_path"])
            abs_path = str(row.get("abs_path", ""))
            if abs_path and not os.path.isabs(abs_path):
                abs_path = os.path.normpath(os.path.join(manifest_dir, abs_path))
            mapping[normalize_key(task_id, image_path)] = abs_path
    return mapping


def resolve_image_path(
    data_root: str,
    task_id: str,
    image_path: str,
    manifest_paths: dict[tuple[str, str], str],
) -> str | None:
    key = normalize_key(task_id, image_path)
    manifest_path = manifest_paths.get(key)
    if manifest_path and os.path.isfile(manifest_path):
        return manifest_path

    normalized = key[1]
    candidates = [
        os.path.join(data_root, "images", normalized),
        os.path.join(data_root, "validation_ready", normalized),
        os.path.join(data_root, normalized),
        os.path.join(data_root, "images", task_id, os.path.basename(normalized)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def build_skull_boundary_score(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray_f = gray.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / max(float(np.percentile(grad, 99.5)), 1e-6)
    grad = np.clip(grad, 0.0, 1.0)

    bright = gray_f / max(float(np.percentile(gray_f, 99.5)), 1e-6)
    bright = np.clip(bright, 0.0, 1.0)
    score = 0.72 * grad + 0.28 * bright

    # Suppress pure black background and isolated speckle away from tissue.
    tissue = (cv2.GaussianBlur(gray_f, (0, 0), 2.0) > 0.025).astype(np.float32)
    score *= tissue
    return cv2.GaussianBlur(score.astype(np.float32), (0, 0), 0.7)


def bilinear_score(score: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = score.shape[:2]
    x = np.clip(points[:, 0], 0.0, width - 1.001)
    y = np.clip(points[:, 1], 0.0, height - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = x - x0
    dy = y - y0
    top = (1.0 - dx) * score[y0, x0] + dx * score[y0, x1]
    bottom = (1.0 - dx) * score[y1, x0] + dx * score[y1, x1]
    return ((1.0 - dy) * top + dy * bottom).astype(np.float32)


def refine_radial_point(
    point: np.ndarray,
    center: np.ndarray,
    score: np.ndarray,
    search_px: float,
    max_shift_px: float,
    blend: float,
    distance_penalty: float,
    samples: int,
) -> np.ndarray:
    vector = point.astype(np.float32) - center.astype(np.float32)
    radius = float(np.linalg.norm(vector))
    if radius < 2.0:
        return point.astype(np.float32).copy()

    direction = vector / radius
    offsets = np.linspace(-search_px, search_px, samples, dtype=np.float32)
    candidates = center.astype(np.float32)[None, :] + direction[None, :] * (radius + offsets)[:, None]
    values = bilinear_score(score, candidates)
    values -= distance_penalty * np.abs(offsets) / max(float(search_px), 1e-6)
    best = candidates[int(np.argmax(values))]

    delta = best - point.astype(np.float32)
    shift = float(np.linalg.norm(delta))
    if shift > max_shift_px:
        delta *= max_shift_px / max(shift, 1e-6)
    return point.astype(np.float32) + float(blend) * delta


def enforce_diameter_center(points: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0 or points.shape[0] != 4:
        return points
    refined = points.astype(np.float32).copy()
    center_v = 0.5 * (refined[0] + refined[1])
    center_h = 0.5 * (refined[2] + refined[3])
    center = 0.5 * (center_v + center_h)

    symmetric = refined.copy()
    v_half = 0.5 * (refined[0] - refined[1])
    h_half = 0.5 * (refined[2] - refined[3])
    symmetric[0] = center + v_half
    symmetric[1] = center - v_half
    symmetric[2] = center + h_half
    symmetric[3] = center - h_half
    return (1.0 - strength) * refined + strength * symmetric


def refine_hc_points(
    points: np.ndarray,
    score: np.ndarray,
    search_px: float,
    max_shift_px: float,
    blend: float,
    distance_penalty: float,
    center_strength: float,
    samples: int,
) -> np.ndarray:
    if points.shape[0] != 4:
        return points.astype(np.float32)
    center = 0.25 * (points[0] + points[1] + points[2] + points[3])
    refined = np.stack(
        [
            refine_radial_point(
                point,
                center,
                score,
                search_px=search_px,
                max_shift_px=max_shift_px,
                blend=blend,
                distance_penalty=distance_penalty,
                samples=samples,
            )
            for point in points
        ],
        axis=0,
    )
    refined = enforce_diameter_center(refined, strength=center_strength)
    shifts = refined - points.astype(np.float32)
    norms = np.linalg.norm(shifts, axis=1)
    too_far = norms > max_shift_px
    if too_far.any():
        shifts[too_far] *= (max_shift_px / np.maximum(norms[too_far], 1e-6))[:, None]
        refined = points.astype(np.float32) + shifts
    return refined.astype(np.float32)


def update_prediction(record: dict, points_px: np.ndarray, width: int, height: int) -> dict:
    points_px[:, 0] = np.clip(points_px[:, 0], 0.0, max(float(width - 1), 1.0))
    points_px[:, 1] = np.clip(points_px[:, 1], 0.0, max(float(height - 1), 1.0))
    points_norm = points_px.copy()
    points_norm[:, 0] /= max(float(width), 1.0)
    points_norm[:, 1] /= max(float(height), 1.0)
    updated = dict(record)
    updated["predicted_points_pixels"] = [round(float(v), 6) for v in points_px.reshape(-1)]
    updated["predicted_points_normalized"] = [round(float(v), 6) for v in points_norm.reshape(-1)]
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine HC predictions using skull-boundary radial ellipse constraints.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--search-px", type=float, default=10.0)
    parser.add_argument("--max-shift-px", type=float, default=5.0)
    parser.add_argument("--blend", type=float, default=0.55)
    parser.add_argument("--distance-penalty", type=float, default=0.20)
    parser.add_argument("--center-strength", type=float, default=0.15)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--zip-submission", action="store_true")
    args = parser.parse_args()

    manifest_paths = load_manifest_paths(args.manifest)
    with open(args.input_json, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    image_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    output_predictions = []
    stats = defaultdict(list)
    missing = []

    for record in predictions:
        task_id = str(record["task_id"])
        if task_id != "HC":
            output_predictions.append(record)
            continue

        image_path = resolve_image_path(args.data_root, task_id, record["image_path"], manifest_paths)
        if image_path is None:
            missing.append(record["image_path"])
            output_predictions.append(record)
            continue
        if image_path not in image_cache:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                missing.append(record["image_path"])
                output_predictions.append(record)
                continue
            image_cache[image_path] = (image, build_skull_boundary_score(image))

        image, score = image_cache[image_path]
        height, width = image.shape[:2]
        points = np.asarray(record["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
        refined = refine_hc_points(
            points,
            score,
            search_px=args.search_px,
            max_shift_px=args.max_shift_px,
            blend=args.blend,
            distance_penalty=args.distance_penalty,
            center_strength=args.center_strength,
            samples=args.samples,
        )
        shifts = np.linalg.norm(refined - points, axis=1)
        stats["HC"].append(float(np.mean(shifts)))
        output_predictions.append(update_prediction(record, refined, width=width, height=height))

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "regression_predictions.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(output_predictions, handle, indent=2)

    summary_path = os.path.join(args.output_dir, "hc_ellipse_boundary_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        values = np.asarray(stats["HC"], dtype=np.float32)
        line = (
            f"HC: n={len(values)}, mean_shift_px={float(values.mean()) if len(values) else 0.0:.6f}, "
            f"max_mean_shift_px={float(values.max()) if len(values) else 0.0:.6f}"
        )
        print(line)
        handle.write(line + "\n")
        handle.write(
            "params="
            + json.dumps(
                {
                    "search_px": args.search_px,
                    "max_shift_px": args.max_shift_px,
                    "blend": args.blend,
                    "distance_penalty": args.distance_penalty,
                    "center_strength": args.center_strength,
                    "samples": args.samples,
                },
                sort_keys=True,
            )
            + "\n"
        )
        if missing:
            handle.write(f"missing_images={len(missing)}\n")
            print(f"Warning: missing images for {len(missing)} HC predictions")

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
