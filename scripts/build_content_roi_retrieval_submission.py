#!/usr/bin/env python3
"""Content-ROI retrieval prototype submission builder.

This adds an image-preprocessing model layer before retrieval:
1. detect the real ultrasound content box;
2. extract features only from that box, suppressing black borders and text;
3. store training landmarks in crop-normalized coordinates;
4. retrieve similar crop-normalized prototypes and map them back to the
   validation image.

The output remains gated against a trusted anchor so unsafe shifts are rejected.
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
    "A4C": 7.0,
    "AOP": 5.0,
    "FA": 5.0,
    "FUGC": 3.0,
    "HC": 5.0,
    "IVC": 5.0,
    "PLAX": 5.0,
    "PSAX": 5.0,
    "fetal_femur": 4.0,
}


def parse_float_map(value: str | None, default: dict[str, float]) -> dict[str, float]:
    parsed = dict(default)
    if not value:
        return parsed
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        key, raw = item.split("=", 1)
        parsed[key.strip()] = float(raw)
    return parsed


def normalize_key(task_id: str, image_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return task_id, "/".join(parts)
    return task_id, f"{task_id}/{os.path.basename(normalized)}"


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


def parse_points_from_row(row: dict[str, str]) -> np.ndarray:
    point_cols = sorted(
        [key for key in row if key.startswith("point_") and key.endswith("_xy")],
        key=lambda key: int(key.split("_")[1]),
    )
    return np.asarray([ast.literal_eval(row[key]) for key in point_cols], dtype=np.float32)


def detect_content_box(image_bgr: np.ndarray, pad_ratio: float) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    threshold = max(6, int(np.percentile(blur, 72) * 0.18))
    mask = (blur > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return 0, 0, width, height

    min_area = 0.015 * width * height
    best_label = None
    best_area = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area and area >= min_area:
            best_label = label
            best_area = area

    if best_label is None:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return 0, 0, width, height
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
    else:
        x0 = int(stats[best_label, cv2.CC_STAT_LEFT])
        y0 = int(stats[best_label, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[best_label, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[best_label, cv2.CC_STAT_HEIGHT])

    pad = int(round(pad_ratio * max(x1 - x0, y1 - y0)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)

    if (x1 - x0) < 0.25 * width or (y1 - y0) < 0.25 * height:
        return 0, 0, width, height
    return x0, y0, x1, y1


def crop_normalize_points(points_px: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = points_px.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - float(x0)) / max(float(x1 - x0), 1.0)
    out[:, 1] = (out[:, 1] - float(y0)) / max(float(y1 - y0), 1.0)
    return np.clip(out, 0.0, 1.0)


def crop_norm_to_image_norm(points_crop: np.ndarray, box: tuple[int, int, int, int], height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = points_crop.astype(np.float32).copy()
    out[:, 0] = (float(x0) + out[:, 0] * float(x1 - x0)) / max(float(width), 1.0)
    out[:, 1] = (float(y0) + out[:, 1] * float(y1 - y0)) / max(float(height), 1.0)
    return np.clip(out, 0.0, 1.0)


def image_feature(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    size: int,
    box_feature_weight: float = 0.0,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    image_h, image_w = image_bgr.shape[:2]
    crop = image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray_f = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad /= max(float(np.percentile(grad, 99.0)), 1e-6)
    grad = np.clip(grad, 0.0, 1.0)

    # Low-resolution anatomy grid preserves gross layout without overfitting to
    # text overlays or black borders.
    pooled = cv2.resize(gray_f, (24, 24), interpolation=cv2.INTER_AREA)
    pooled_grad = cv2.resize(grad, (24, 24), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([(gray_f * 255).astype(np.uint8)], [0], None, [32], [0, 256]).reshape(-1)
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    parts = [pooled.reshape(-1), 0.85 * pooled_grad.reshape(-1), 5.0 * hist]
    if box_feature_weight > 0.0:
        box_w = max(float(x1 - x0), 1.0)
        box_h = max(float(y1 - y0), 1.0)
        geom = np.asarray(
            [
                (float(x0) + 0.5 * box_w) / max(float(image_w), 1.0),
                (float(y0) + 0.5 * box_h) / max(float(image_h), 1.0),
                box_w / max(float(image_w), 1.0),
                box_h / max(float(image_h), 1.0),
                np.log(box_w / box_h),
                (box_w * box_h) / max(float(image_w * image_h), 1.0),
            ],
            dtype=np.float32,
        )
        parts.append(box_feature_weight * geom)
    feat = np.concatenate(parts, axis=0).astype(np.float32)
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


def load_task_memory(
    data_root: str,
    csv_root: str,
    task_id: str,
    feature_size: int,
    pad_ratio: float,
    box_feature_weight: float,
) -> dict[str, np.ndarray]:
    csv_path = os.path.join(csv_root, TASK_CSV[task_id])
    features = []
    labels = []
    paths = []
    boxes = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = str(row["image_path"])
            resolved = resolve_image(data_root, image_path, task_id)
            if resolved is None:
                continue
            image = cv2.imread(resolved, cv2.IMREAD_COLOR)
            if image is None:
                continue
            box = detect_content_box(image, pad_ratio=pad_ratio)
            points = parse_points_from_row(row)
            features.append(image_feature(image, box, feature_size, box_feature_weight=box_feature_weight))
            labels.append(crop_normalize_points(points, box))
            paths.append(image_path)
            boxes.append(box)
    if not features:
        raise RuntimeError(f"No memory loaded for {task_id}")
    return {
        "features": np.stack(features, axis=0).astype(np.float32),
        "labels": np.stack(labels, axis=0).astype(np.float32),
        "paths": np.asarray(paths),
        "boxes": np.asarray(boxes, dtype=np.int32),
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
    parser.add_argument("--alphas", default="0.03,0.06,0.10")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--feature-size", type=int, default=128)
    parser.add_argument("--pad-ratio", type=float, default=0.06)
    parser.add_argument("--box-feature-weight", type=float, default=0.0)
    parser.add_argument("--shift-gates", default=None)
    parser.add_argument("--zip-submission", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Output root already exists; refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    gates = parse_float_map(args.shift_gates, DEFAULT_SHIFT_GATES)

    with open(args.anchor_json, "r", encoding="utf-8") as handle:
        anchor_predictions = json.load(handle)
    sizes = load_manifest_sizes(args.manifest)
    memories = {
        task: load_task_memory(
            args.data_root,
            args.csv_root,
            task,
            args.feature_size,
            args.pad_ratio,
            args.box_feature_weight,
        )
        for task in tasks
    }

    retrieval = {}
    for record in anchor_predictions:
        task_id = str(record["task_id"])
        if task_id not in memories:
            continue
        key = normalize_key(task_id, str(record["image_path"]))
        image_path = resolve_image(args.data_root, key[1], task_id)
        if image_path is None or key not in sizes:
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = sizes[key]
        box = detect_content_box(image, pad_ratio=args.pad_ratio)
        query = image_feature(image, box, args.feature_size, box_feature_weight=args.box_feature_weight)
        memory = memories[task_id]
        sims = memory["features"] @ query
        top_idx = np.argsort(-sims)[: args.top_k]
        top_sims = sims[top_idx]
        weights = np.exp((top_sims - float(top_sims.max())) / max(args.temperature, 1e-6))
        weights /= max(float(weights.sum()), 1e-6)
        proto_crop = np.sum(memory["labels"][top_idx] * weights[:, None, None], axis=0)
        proto_image = crop_norm_to_image_norm(proto_crop, box, height=height, width=width)
        retrieval[key] = {
            "prototype": proto_image.astype(np.float32),
            "box": box,
            "top_similarity": float(top_sims[0]),
            "mean_similarity": float(np.mean(top_sims)),
            "top_paths": [str(memory["paths"][idx]) for idx in top_idx],
        }

    for alpha in alphas:
        output_dir = output_root / output_name(alpha)
        output_dir.mkdir(parents=True, exist_ok=False)
        changed = {task: 0 for task in tasks}
        rejected = {task: 0 for task in tasks}
        details = []
        output_predictions = []
        for record in anchor_predictions:
            task_id = str(record["task_id"])
            key = normalize_key(task_id, str(record["image_path"]))
            if task_id not in memories or key not in retrieval or key not in sizes:
                output_predictions.append(record)
                continue
            height, width = sizes[key]
            anchor = np.asarray(record["predicted_points_normalized"], dtype=np.float32).reshape(-1, 2)
            proto = retrieval[key]["prototype"]
            if proto.shape != anchor.shape:
                rejected[task_id] += 1
                output_predictions.append(record)
                continue
            candidate = anchor * (1.0 - alpha) + proto * alpha
            shift = candidate - anchor
            shift[:, 0] *= float(width)
            shift[:, 1] *= float(height)
            mean_shift = float(np.linalg.norm(shift, axis=1).mean())
            accepted = mean_shift <= gates.get(task_id, 5.0)
            if accepted:
                changed[task_id] += 1
                output_predictions.append(update_prediction(record, candidate, height=height, width=width))
            else:
                rejected[task_id] += 1
                output_predictions.append(record)
            details.append(
                {
                    "task_id": task_id,
                    "image_path": record["image_path"],
                    "accepted": accepted,
                    "mean_shift_px": mean_shift,
                    "box": list(map(int, retrieval[key]["box"])),
                    "top_similarity": retrieval[key]["top_similarity"],
                    "mean_similarity": retrieval[key]["mean_similarity"],
                    "top_paths": retrieval[key]["top_paths"],
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
            "tasks": tasks,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "feature_size": args.feature_size,
            "pad_ratio": args.pad_ratio,
            "box_feature_weight": args.box_feature_weight,
            "shift_gates": gates,
            "changed_by_task": changed,
            "rejected_by_task": rejected,
            "details": details,
        }
        (output_dir / "content_roi_retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "alpha": alpha, "changed_by_task": changed, "rejected_by_task": rejected}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
