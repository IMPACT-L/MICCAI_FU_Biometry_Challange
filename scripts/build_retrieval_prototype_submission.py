#!/usr/bin/env python3
"""Build retrieval-augmented landmark submission variants.

This is a non-parametric model layer: for selected tasks, retrieve visually
similar labelled training images and use their normalized landmark geometry as
an anatomical prototype. The prototype is blended with a trusted anchor only
when the shift stays inside a task-specific gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np


TASK_CSV = {
    "A4C": "A4C_train.csv",
    "AOP": "AOP_train.csv",
    "FA": "FA_train.csv",
    "FUGC": "FUGC_train.csv",
    "HC": "HC_train.csv",
    "IVC": "IVC_train.csv",
    "PLAX": "PLAX_train.csv",
    "PSAX": "PSAX_train.csv",
    "fetal_femur": "Reg-Two_3.fetal_femur.csv",
}

DEFAULT_SHIFT_GATES = {
    "A4C": 8.0,
    "AOP": 5.0,
    "FA": 5.0,
    "FUGC": 3.0,
    "HC": 5.0,
    "IVC": 5.0,
    "PLAX": 5.0,
    "PSAX": 5.0,
    "fetal_femur": 4.0,
}


def parse_map(value: str | None, default: dict[str, float]) -> dict[str, float]:
    out = dict(default)
    if not value:
        return out
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        key, raw = item.split("=", 1)
        out[key.strip()] = float(raw)
    return out


def normalize_key(task_id: str, image_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return task_id, "/".join(parts)
    return task_id, f"{task_id}/{os.path.basename(normalized)}"


def parse_points_from_row(row: dict[str, str]) -> np.ndarray:
    point_cols = sorted(
        [key for key in row if key.startswith("point_") and key.endswith("_xy")],
        key=lambda key: int(key.split("_")[1]),
    )
    points = [ast.literal_eval(row[key]) for key in point_cols]
    return np.asarray(points, dtype=np.float32)


def resolve_image(data_root: str, image_path: str, task_id: str) -> str | None:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    basename = os.path.basename(normalized)
    candidates = [
        os.path.join(data_root, normalized),
        os.path.join(data_root, "images", normalized),
        os.path.join(data_root, "images", task_id, basename),
        os.path.join(data_root, "validation_ready", task_id, basename),
        os.path.join(data_root, "extracted", task_id, task_id, basename),
        os.path.join(data_root, "extracted", task_id, "train", basename),
        os.path.join(data_root, "extracted", task_id, "images", basename),
        os.path.join(data_root, "extracted", task_id, "images_4000", basename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def image_feature(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image_f = image.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(image_f, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad /= max(float(np.percentile(grad, 99.0)), 1e-6)
    grad = np.clip(grad, 0.0, 1.0)
    hist = cv2.calcHist([(image_f * 255).astype(np.uint8)], [0], None, [32], [0, 256]).reshape(-1)
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    feat = np.concatenate([image_f.reshape(-1), 0.75 * grad.reshape(-1), 8.0 * hist], axis=0).astype(np.float32)
    feat -= float(feat.mean())
    feat /= max(float(np.linalg.norm(feat)), 1e-6)
    return feat


def load_manifest_sizes(manifest_path: str) -> dict[tuple[str, str], tuple[int, int]]:
    sizes = {}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = normalize_key(str(row["task_id"]), str(row["image_path"]))
            sizes[key] = (int(row["height"]), int(row["width"]))
    return sizes


def load_task_memory(data_root: str, csv_root: str, task_id: str, feature_size: int) -> dict[str, np.ndarray]:
    csv_path = os.path.join(csv_root, TASK_CSV[task_id])
    features = []
    labels = []
    paths = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = str(row["image_path"])
            resolved = resolve_image(data_root, image_path, task_id)
            if resolved is None:
                continue
            points = parse_points_from_row(row)
            width = float(row["width"])
            height = float(row["height"])
            norm = points.copy()
            norm[:, 0] /= max(width, 1.0)
            norm[:, 1] /= max(height, 1.0)
            try:
                features.append(image_feature(resolved, feature_size))
            except FileNotFoundError:
                continue
            labels.append(norm)
            paths.append(image_path)
    if not features:
        raise RuntimeError(f"No retrieval memory loaded for {task_id}")
    return {
        "features": np.stack(features, axis=0).astype(np.float32),
        "labels": np.stack(labels, axis=0).astype(np.float32),
        "paths": np.asarray(paths),
    }


def update_prediction(record: dict, norm_points: np.ndarray, height: int, width: int) -> dict:
    norm = np.clip(norm_points.astype(np.float32), 0.0, 1.0)
    pixels = norm.copy()
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)
    pixels[:, 0] = np.clip(pixels[:, 0], 0.0, max(float(width - 1), 1.0))
    pixels[:, 1] = np.clip(pixels[:, 1], 0.0, max(float(height - 1), 1.0))
    updated = dict(record)
    updated["predicted_points_normalized"] = [round(float(v), 8) for v in norm.reshape(-1)]
    updated["predicted_points_pixels"] = [round(float(v), 6) for v in pixels.reshape(-1)]
    return updated


def output_name(alpha: float) -> str:
    return f"a{alpha:.3f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--csv-root", default="data/csv")
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--tasks", default="IVC,PLAX,fetal_femur")
    parser.add_argument("--alphas", default="0.05,0.10,0.15")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.035)
    parser.add_argument("--feature-size", type=int, default=96)
    parser.add_argument("--shift-gates", default=None)
    parser.add_argument("--zip-submission", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Output root already exists; refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    task_ids = [item.strip() for item in args.tasks.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    gates = parse_map(args.shift_gates, DEFAULT_SHIFT_GATES)

    with open(args.anchor_json, "r", encoding="utf-8") as handle:
        anchor_predictions = json.load(handle)
    sizes = load_manifest_sizes(args.manifest)
    memories = {task_id: load_task_memory(args.data_root, args.csv_root, task_id, args.feature_size) for task_id in task_ids}

    retrieval_cache: dict[tuple[str, str], dict] = {}
    for record in anchor_predictions:
        task_id = str(record["task_id"])
        if task_id not in memories:
            continue
        key = normalize_key(task_id, str(record["image_path"]))
        image_path = resolve_image(args.data_root, key[1], task_id)
        if image_path is None:
            continue
        query = image_feature(image_path, args.feature_size)
        memory = memories[task_id]
        sims = memory["features"] @ query
        top_idx = np.argsort(-sims)[: args.top_k]
        top_sims = sims[top_idx]
        weights = np.exp((top_sims - float(top_sims.max())) / max(args.temperature, 1e-6))
        weights /= max(float(weights.sum()), 1e-6)
        proto = np.sum(memory["labels"][top_idx] * weights[:, None, None], axis=0)
        retrieval_cache[key] = {
            "prototype": proto.astype(np.float32),
            "top_similarity": float(top_sims[0]),
            "mean_similarity": float(np.mean(top_sims)),
            "top_paths": [str(memory["paths"][idx]) for idx in top_idx],
        }

    for alpha in alphas:
        output_dir = output_root / output_name(alpha)
        output_dir.mkdir(parents=True, exist_ok=False)
        output_predictions = []
        details = []
        changed_by_task = {task_id: 0 for task_id in task_ids}
        rejected_by_task = {task_id: 0 for task_id in task_ids}

        for record in anchor_predictions:
            task_id = str(record["task_id"])
            key = normalize_key(task_id, str(record["image_path"]))
            if task_id not in memories or key not in retrieval_cache or key not in sizes:
                output_predictions.append(record)
                continue
            height, width = sizes[key]
            anchor_norm = np.asarray(record["predicted_points_normalized"], dtype=np.float32).reshape(-1, 2)
            proto = retrieval_cache[key]["prototype"]
            if proto.shape != anchor_norm.shape:
                output_predictions.append(record)
                rejected_by_task[task_id] += 1
                continue
            blended = anchor_norm * (1.0 - alpha) + proto * alpha
            shift_px = blended - anchor_norm
            shift_px[:, 0] *= float(width)
            shift_px[:, 1] *= float(height)
            mean_shift = float(np.linalg.norm(shift_px, axis=1).mean())
            if mean_shift > gates.get(task_id, 5.0):
                output_predictions.append(record)
                rejected_by_task[task_id] += 1
                accepted = False
            else:
                output_predictions.append(update_prediction(record, blended, height, width))
                changed_by_task[task_id] += 1
                accepted = True
            details.append(
                {
                    "task_id": task_id,
                    "image_path": record["image_path"],
                    "accepted": accepted,
                    "mean_shift_px": mean_shift,
                    "top_similarity": retrieval_cache[key]["top_similarity"],
                    "mean_similarity": retrieval_cache[key]["mean_similarity"],
                    "top_paths": retrieval_cache[key]["top_paths"],
                }
            )

        output_json = output_dir / "regression_predictions.json"
        output_json.write_text(json.dumps(output_predictions, indent=2), encoding="utf-8")
        if args.zip_submission:
            with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(output_json, arcname="regression_predictions.json")
        summary = {
            "anchor_json": args.anchor_json,
            "alpha": alpha,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "feature_size": args.feature_size,
            "tasks": task_ids,
            "shift_gates": gates,
            "changed_by_task": changed_by_task,
            "rejected_by_task": rejected_by_task,
            "details": details,
        }
        (output_dir / "retrieval_prototype_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "alpha": alpha, "changed_by_task": changed_by_task, "rejected_by_task": rejected_by_task}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
