#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def load_predictions(path: Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text())
    return {(str(item["task_id"]), str(item["image_path"])): item for item in data}


def parse_task_ids(value: str | None) -> set[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def coords(item: dict, field: str = "predicted_points_normalized") -> np.ndarray:
    values = np.asarray(item[field], dtype=np.float64)
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError(f"Invalid {field} vector for {item.get('task_id')}/{item.get('image_path')}")
    return values.reshape(-1, 2)


def infer_image_size(item: dict) -> tuple[float, float]:
    norm = coords(item, "predicted_points_normalized")
    pix = coords(item, "predicted_points_pixels")
    valid_x = np.abs(norm[:, 0]) > 1e-6
    valid_y = np.abs(norm[:, 1]) > 1e-6
    if not np.any(valid_x) or not np.any(valid_y):
        raise ValueError(f"Could not infer image size for {item.get('task_id')}/{item.get('image_path')}")
    width = float(np.median(pix[valid_x, 0] / norm[valid_x, 0]))
    height = float(np.median(pix[valid_y, 1] / norm[valid_y, 1]))
    return max(width, 1.0), max(height, 1.0)


def update_item(item: dict, normalized: np.ndarray) -> dict:
    normalized = np.clip(normalized.astype(np.float64), 0.0, 1.0)
    width, height = infer_image_size(item)
    pixels = normalized.copy()
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    updated = dict(item)
    updated["predicted_points_normalized"] = [round(float(v), 6) for v in normalized.reshape(-1)]
    updated["predicted_points_pixels"] = [round(float(v), 6) for v in pixels.reshape(-1)]
    return updated


def fit_point_bias(x: np.ndarray, y: np.ndarray) -> dict:
    return {"bias": np.mean(y - x, axis=0)}


def apply_point_bias(x: np.ndarray, params: dict) -> np.ndarray:
    return x + params["bias"]


def fit_point_affine(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
    slope_min: float,
    slope_max: float,
) -> dict:
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = np.sum(x_centered * y_centered, axis=0)
    denominator = np.sum(x_centered * x_centered, axis=0) + float(ridge)
    slope = numerator / np.maximum(denominator, 1e-12)
    slope = np.clip(slope, slope_min, slope_max)
    intercept = y_mean - slope * x_mean
    return {"slope": slope, "intercept": intercept}


def apply_point_affine(x: np.ndarray, params: dict) -> np.ndarray:
    return x * params["slope"] + params["intercept"]


def fit_calibrators(
    fit_input: dict[tuple[str, str], dict],
    target: dict[tuple[str, str], dict],
    task_ids: set[str] | None,
    mode: str,
    ridge: float,
    slope_min: float,
    slope_max: float,
) -> dict[str, dict]:
    by_task_x: dict[str, list[np.ndarray]] = {}
    by_task_y: dict[str, list[np.ndarray]] = {}
    for key, input_item in fit_input.items():
        task_id, _ = key
        if task_ids is not None and task_id not in task_ids:
            continue
        if key not in target:
            raise KeyError(f"Missing target prediction for {key}")
        x = coords(input_item).reshape(-1)
        y = coords(target[key]).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"Coordinate shape mismatch for {key}: {x.shape} != {y.shape}")
        by_task_x.setdefault(task_id, []).append(x)
        by_task_y.setdefault(task_id, []).append(y)

    calibrators = {}
    for task_id in sorted(by_task_x):
        x_task = np.stack(by_task_x[task_id], axis=0)
        y_task = np.stack(by_task_y[task_id], axis=0)
        if mode == "point_bias":
            params = fit_point_bias(x_task, y_task)
        elif mode == "point_affine":
            params = fit_point_affine(
                x_task,
                y_task,
                ridge=ridge,
                slope_min=slope_min,
                slope_max=slope_max,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        calibrators[task_id] = params
    return calibrators


def apply_calibrators(
    apply_input: dict[tuple[str, str], dict],
    calibrators: dict[str, dict],
    mode: str,
    strength: float,
) -> list[dict]:
    output = []
    shifts_by_task: dict[str, list[float]] = {}
    for key in sorted(apply_input, key=lambda item: (item[0], item[1])):
        task_id, _ = key
        item = apply_input[key]
        if task_id not in calibrators:
            output.append(dict(item))
            continue
        x = coords(item).reshape(-1)
        if mode == "point_bias":
            y_hat = apply_point_bias(x, calibrators[task_id])
        elif mode == "point_affine":
            y_hat = apply_point_affine(x, calibrators[task_id])
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        calibrated = x + float(strength) * (y_hat - x)
        before = x.reshape(-1, 2)
        after = calibrated.reshape(-1, 2)
        shifts_by_task.setdefault(task_id, []).append(float(np.mean(np.linalg.norm(after - before, axis=1))))
        output.append(update_item(item, after))

    summary = {
        task_id: {
            "count": len(values),
            "mean_normalized_shift": float(np.mean(values)),
            "max_normalized_shift": float(np.max(values)),
        }
        for task_id, values in sorted(shifts_by_task.items())
    }
    return output, summary


def serializable_calibrators(calibrators: dict[str, dict]) -> dict:
    payload = {}
    for task_id, params in calibrators.items():
        payload[task_id] = {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in params.items()
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit tiny residual coordinate calibrators between two submissions and apply them to a submission."
    )
    parser.add_argument("--fit-input-json", required=True)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--apply-input-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--mode", choices=("point_bias", "point_affine"), default="point_affine")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--slope-min", type=float, default=0.75)
    parser.add_argument("--slope-max", type=float, default=1.25)
    args = parser.parse_args()

    fit_input_path = Path(args.fit_input_json).resolve()
    target_path = Path(args.target_json).resolve()
    apply_input_path = Path(args.apply_input_json).resolve() if args.apply_input_json else fit_input_path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = parse_task_ids(args.task_ids)
    fit_input = load_predictions(fit_input_path)
    target = load_predictions(target_path)
    apply_input = load_predictions(apply_input_path)

    calibrators = fit_calibrators(
        fit_input=fit_input,
        target=target,
        task_ids=task_ids,
        mode=args.mode,
        ridge=float(args.ridge),
        slope_min=float(args.slope_min),
        slope_max=float(args.slope_max),
    )
    predictions, shift_summary = apply_calibrators(
        apply_input=apply_input,
        calibrators=calibrators,
        mode=args.mode,
        strength=float(args.strength),
    )

    output_json = output_dir / "regression_predictions.json"
    output_json.write_text(json.dumps(predictions, indent=2))
    with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_json, arcname="regression_predictions.json")

    summary = {
        "fit_input_json": str(fit_input_path),
        "target_json": str(target_path),
        "apply_input_json": str(apply_input_path),
        "mode": args.mode,
        "strength": float(args.strength),
        "ridge": float(args.ridge),
        "slope_min": float(args.slope_min),
        "slope_max": float(args.slope_max),
        "task_ids": sorted(calibrators),
        "calibrators": serializable_calibrators(calibrators),
        "shift_summary_normalized": shift_summary,
    }
    (output_dir / "residual_calibration_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "calibrators"}, indent=2, sort_keys=True))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_dir / 'submission.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
