#!/usr/bin/env python3
"""PLAX pair-wise anatomy refiner for submission JSON files.

PLAX contains 22 landmarks arranged as 11 measurement pairs. This script keeps
all non-PLAX tasks unchanged and refines each PLAX pair by searching for nearby
bright ultrasound wall evidence while preserving the pair direction and length.
The goal is to improve coordinate placement without destabilizing the derived
measurement lengths.
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


def robust_normalize(value: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    denom = max(float(np.percentile(value, percentile)), 1e-6)
    return np.clip(value / denom, 0.0, 1.0).astype(np.float32)


def build_plax_wall_score(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray_f = gray.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = robust_normalize(cv2.magnitude(gx, gy), percentile=99.5)
    bright = robust_normalize(gray_f, percentile=99.0)
    local_mean = cv2.GaussianBlur(gray_f, (0, 0), 4.0)
    bright_local = np.clip(gray_f - 0.55 * local_mean, 0.0, 1.0)
    bright_local = robust_normalize(bright_local, percentile=99.0)
    tissue = (cv2.GaussianBlur(gray_f, (0, 0), 2.0) > 0.025).astype(np.float32)

    score = (0.62 * grad + 0.25 * bright + 0.13 * bright_local) * tissue
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 0.55)
    return gray_f.astype(np.float32), score.astype(np.float32)


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


def pair_score(
    candidate: np.ndarray,
    anchor: np.ndarray,
    gray: np.ndarray,
    wall_score: np.ndarray,
    distance_penalty: float,
    length_penalty: float,
) -> float:
    vector = candidate[0] - candidate[1]
    length = float(np.linalg.norm(vector))
    if length < 3.0:
        return -1e6
    direction = vector / max(length, 1e-6)
    endpoint_score = float(np.mean(bilinear_score(wall_score, candidate)))

    # Reward local contrast across the wall direction. This is intentionally
    # weak because PLAX pairs include both chamber and wall-thickness measures.
    inside = np.stack([candidate[0] - direction * 2.0, candidate[1] + direction * 2.0], axis=0)
    outside = np.stack([candidate[0] + direction * 2.0, candidate[1] - direction * 2.0], axis=0)
    contrast = float(np.mean(np.abs(bilinear_score(gray, outside) - bilinear_score(gray, inside))))

    anchor_length = float(np.linalg.norm(anchor[0] - anchor[1]))
    mean_shift = float(np.linalg.norm(candidate - anchor, axis=1).mean())
    length_delta = abs(length - anchor_length) / max(anchor_length, 1.0)
    return endpoint_score + 0.08 * contrast - distance_penalty * mean_shift - length_penalty * length_delta


def refine_pair(
    pair: np.ndarray,
    gray: np.ndarray,
    wall_score: np.ndarray,
    search_px: float,
    center_search_px: float,
    max_mean_shift_px: float,
    max_point_shift_px: float,
    min_score_delta: float,
    distance_penalty: float,
    length_penalty: float,
    length_ratio_min: float,
    length_ratio_max: float,
    radius_samples: int,
    center_samples: int,
) -> tuple[np.ndarray, dict]:
    anchor = pair.astype(np.float32)
    center = 0.5 * (anchor[0] + anchor[1])
    vector = anchor[0] - anchor[1]
    length = float(np.linalg.norm(vector))
    if length < 3.0:
        return anchor, {"accepted": False, "reason": "short_anchor"}

    direction = vector / max(length, 1e-6)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    radius = 0.5 * length

    base_score = pair_score(
        anchor,
        anchor,
        gray,
        wall_score,
        distance_penalty=distance_penalty,
        length_penalty=length_penalty,
    )
    best = anchor.copy()
    best_score = base_score

    radius_offsets = np.linspace(-search_px, search_px, radius_samples, dtype=np.float32)
    center_offsets = np.linspace(-center_search_px, center_search_px, center_samples, dtype=np.float32)
    for dr in radius_offsets:
        candidate_radius = max(radius + float(dr), 1.5)
        candidate_length = 2.0 * candidate_radius
        ratio = candidate_length / max(length, 1e-6)
        if ratio < length_ratio_min or ratio > length_ratio_max:
            continue
        for da in center_offsets:
            for dp in center_offsets:
                candidate_center = center + direction * float(da) + perpendicular * float(dp)
                candidate = np.stack(
                    [
                        candidate_center + direction * candidate_radius,
                        candidate_center - direction * candidate_radius,
                    ],
                    axis=0,
                ).astype(np.float32)
                shifts = np.linalg.norm(candidate - anchor, axis=1)
                if float(shifts.mean()) > max_mean_shift_px or float(shifts.max()) > max_point_shift_px:
                    continue
                score = pair_score(
                    candidate,
                    anchor,
                    gray,
                    wall_score,
                    distance_penalty=distance_penalty,
                    length_penalty=length_penalty,
                )
                if score > best_score:
                    best_score = score
                    best = candidate

    score_delta = float(best_score - base_score)
    shifts = np.linalg.norm(best - anchor, axis=1)
    accepted = score_delta >= min_score_delta and float(shifts.mean()) > 1e-5
    return (
        best.astype(np.float32) if accepted else anchor,
        {
            "accepted": bool(accepted),
            "reason": "accepted" if accepted else "score_gate",
            "base_score": float(base_score),
            "best_score": float(best_score),
            "score_delta": score_delta,
            "mean_shift_px": float(shifts.mean()) if accepted else 0.0,
            "max_shift_px": float(shifts.max()) if accepted else 0.0,
            "anchor_length_px": float(length),
            "candidate_length_px": float(np.linalg.norm(best[0] - best[1])),
        },
    )


def refine_plax_points(points: np.ndarray, gray: np.ndarray, wall_score: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[dict]]:
    refined = points.astype(np.float32).copy()
    details = []
    if refined.shape[0] != 22:
        return refined, [{"accepted": False, "reason": "point_count"}]

    for pair_idx in range(11):
        start = pair_idx * 2
        pair = refined[start : start + 2]
        candidate, info = refine_pair(
            pair,
            gray,
            wall_score,
            search_px=args.search_px,
            center_search_px=args.center_search_px,
            max_mean_shift_px=args.max_pair_mean_shift_px,
            max_point_shift_px=args.max_pair_point_shift_px,
            min_score_delta=args.min_score_delta,
            distance_penalty=args.distance_penalty,
            length_penalty=args.length_penalty,
            length_ratio_min=args.length_ratio_min,
            length_ratio_max=args.length_ratio_max,
            radius_samples=args.radius_samples,
            center_samples=args.center_samples,
        )
        info["pair_idx"] = pair_idx
        details.append(info)
        refined[start : start + 2] = candidate

    original = points.astype(np.float32)
    shifts = refined - original
    point_norms = np.linalg.norm(shifts, axis=1)
    too_far = point_norms > args.max_global_point_shift_px
    if too_far.any():
        shifts[too_far] *= (args.max_global_point_shift_px / np.maximum(point_norms[too_far], 1e-6))[:, None]
        refined = original + shifts
    mean_shift = float(np.linalg.norm(refined - original, axis=1).mean())
    if mean_shift > args.max_global_mean_shift_px:
        scale = args.max_global_mean_shift_px / max(mean_shift, 1e-6)
        refined = original + (refined - original) * scale
    return refined.astype(np.float32), details


def update_prediction(record: dict, points_px: np.ndarray, width: int, height: int) -> dict:
    points_px = points_px.astype(np.float32).copy()
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
    parser = argparse.ArgumentParser(description="Refine PLAX predictions with conservative pair-wise wall search.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--search-px", type=float, default=4.0)
    parser.add_argument("--center-search-px", type=float, default=2.0)
    parser.add_argument("--max-pair-mean-shift-px", type=float, default=3.0)
    parser.add_argument("--max-pair-point-shift-px", type=float, default=4.5)
    parser.add_argument("--max-global-mean-shift-px", type=float, default=2.5)
    parser.add_argument("--max-global-point-shift-px", type=float, default=5.0)
    parser.add_argument("--min-score-delta", type=float, default=0.012)
    parser.add_argument("--distance-penalty", type=float, default=0.022)
    parser.add_argument("--length-penalty", type=float, default=0.095)
    parser.add_argument("--length-ratio-min", type=float, default=0.88)
    parser.add_argument("--length-ratio-max", type=float, default=1.12)
    parser.add_argument("--radius-samples", type=int, default=17)
    parser.add_argument("--center-samples", type=int, default=5)
    parser.add_argument("--zip-submission", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")

    manifest_paths = load_manifest_paths(args.manifest)
    with open(args.input_json, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    output_predictions = []
    image_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    details = []
    stats = defaultdict(list)
    missing = []

    for record in predictions:
        task_id = str(record["task_id"])
        if task_id != "PLAX":
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
            gray, wall_score = build_plax_wall_score(image)
            image_cache[image_path] = (image, gray, wall_score)

        image, gray, wall_score = image_cache[image_path]
        height, width = image.shape[:2]
        points = np.asarray(record["predicted_points_pixels"], dtype=np.float32).reshape(-1, 2)
        refined, image_details = refine_plax_points(points, gray, wall_score, args)
        shifts = np.linalg.norm(refined - points, axis=1)
        accepted_pairs = sum(1 for item in image_details if item.get("accepted"))
        stats["mean_shift_px"].append(float(shifts.mean()))
        stats["accepted_pairs"].append(float(accepted_pairs))
        details.append(
            {
                "image_path": record["image_path"],
                "mean_shift_px": float(shifts.mean()),
                "max_shift_px": float(shifts.max()) if len(shifts) else 0.0,
                "accepted_pairs": accepted_pairs,
                "pairs": image_details,
            }
        )
        output_predictions.append(update_prediction(record, refined, width=width, height=height))

    output_dir.mkdir(parents=True, exist_ok=False)
    output_json = output_dir / "regression_predictions.json"
    output_json.write_text(json.dumps(output_predictions, indent=2))

    if args.zip_submission:
        zip_path = output_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_json, arcname="regression_predictions.json")
        print(f"Wrote {zip_path}")

    summary = {
        "input_json": args.input_json,
        "n_plax": len(details),
        "missing_images": missing,
        "mean_image_shift_px": float(np.mean(stats["mean_shift_px"])) if stats["mean_shift_px"] else 0.0,
        "max_image_shift_px": float(np.max(stats["mean_shift_px"])) if stats["mean_shift_px"] else 0.0,
        "mean_accepted_pairs": float(np.mean(stats["accepted_pairs"])) if stats["accepted_pairs"] else 0.0,
        "params": vars(args),
        "details": details,
    }
    summary_path = output_dir / "plax_pair_anatomy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2, sort_keys=True))
    print(f"Wrote {output_json}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
