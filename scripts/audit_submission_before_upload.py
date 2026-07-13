#!/usr/bin/env python
"""Pre-submit risk audit for FU-Biometry validation submissions.

This does not estimate hidden-label accuracy. It protects limited submissions by
rejecting JSONs that diverge too far from a trusted reference submission, have
bad counts, missing rows, point-count mismatches, or invalid coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from utils import MEASUREMENT_PAIRS, compute_measurements_from_points  # noqa: E402


EXPECTED_COUNTS = {
    "A4C": 20,
    "AOP": 60,
    "FA": 188,
    "FUGC": 20,
    "HC": 215,
    "IVC": 10,
    "PLAX": 26,
    "PSAX": 18,
    "fetal_femur": 62,
}
DEFAULT_TASK_MEAN_THRESHOLDS = {
    "A4C": 14.0,
    "AOP": 8.0,
    "FA": 8.0,
    "FUGC": 5.0,
    "HC": 8.0,
    "IVC": 8.0,
    "PLAX": 10.0,
    "PSAX": 8.0,
    "fetal_femur": 6.0,
}


def parse_float_map(value: str | None) -> dict[str, float]:
    if value is None or str(value).strip() == "":
        return {}
    parsed = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected TASK=value entry, got: {item}")
        key, raw = item.split("=", 1)
        parsed[key.strip()] = float(raw)
    return parsed


def normalize_image_path(image_path: str, task_id: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return "/".join(parts)
    return f"{task_id}/{os.path.basename(normalized)}"


def load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Prediction JSON must be a list: {path}")
    return data


def prediction_map(predictions: list[dict]) -> dict[tuple[str, str], dict]:
    output = {}
    duplicates = []
    for item in predictions:
        task_id = str(item.get("task_id", ""))
        image_path = normalize_image_path(str(item.get("image_path", "")), task_id)
        key = (task_id, image_path)
        if key in output:
            duplicates.append(key)
        output[key] = item
    if duplicates:
        preview = ", ".join(f"{task}/{path}" for task, path in duplicates[:5])
        raise ValueError(f"Duplicate prediction keys found: {preview}")
    return output


def points_array(item: dict, field: str) -> np.ndarray:
    if field not in item:
        raise ValueError(f"Missing field '{field}' in prediction for {item.get('task_id')}:{item.get('image_path')}")
    values = np.asarray(item[field], dtype=np.float32)
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError(f"Invalid coordinate vector in field '{field}' for {item.get('image_path')}")
    return values.reshape(-1, 2)


def measurement_delta(candidate_px: np.ndarray, reference_px: np.ndarray, task_id: str) -> float:
    pairs = MEASUREMENT_PAIRS.get(task_id, [])
    if not pairs:
        return float("nan")
    candidate = compute_measurements_from_points(candidate_px[None, ...], task_id)
    reference = compute_measurements_from_points(reference_px[None, ...], task_id)
    if candidate.shape[1] == 0:
        return float("nan")
    return float(np.abs(candidate - reference).mean())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a candidate submission before spending a CodaBench submission.")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument(
        "--reference-json",
        default="output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_round2_v1_seed42_local/regression_predictions.json",
        help="Trusted reference JSON, normally the current 25.31 best.",
    )
    parser.add_argument(
        "--task-mean-thresholds",
        default=",".join(f"{task}={value:g}" for task, value in DEFAULT_TASK_MEAN_THRESHOLDS.items()),
        help="Per-task mean pixel-shift thresholds versus the reference.",
    )
    parser.add_argument("--max-row-mean-shift-px", type=float, default=28.0)
    parser.add_argument("--max-task-p90-shift-px", type=float, default=24.0)
    parser.add_argument("--max-invalid-normalized", type=int, default=0)
    parser.add_argument("--fail-on-risk", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    candidate_path = Path(args.candidate_json)
    reference_path = Path(args.reference_json)
    candidate = load_predictions(candidate_path)
    reference = load_predictions(reference_path)
    candidate_map = prediction_map(candidate)
    reference_map = prediction_map(reference)

    issues = []
    candidate_counts = Counter(str(item.get("task_id", "")) for item in candidate)
    if dict(candidate_counts) != EXPECTED_COUNTS:
        issues.append(f"task counts mismatch: got {dict(sorted(candidate_counts.items()))}, expected {EXPECTED_COUNTS}")
    if len(candidate) != sum(EXPECTED_COUNTS.values()):
        issues.append(f"row count mismatch: got {len(candidate)}, expected {sum(EXPECTED_COUNTS.values())}")

    missing_from_candidate = sorted(set(reference_map) - set(candidate_map))
    extra_in_candidate = sorted(set(candidate_map) - set(reference_map))
    if missing_from_candidate:
        preview = ", ".join(f"{task}/{path}" for task, path in missing_from_candidate[:5])
        issues.append(f"candidate missing {len(missing_from_candidate)} reference rows: {preview}")
    if extra_in_candidate:
        preview = ", ".join(f"{task}/{path}" for task, path in extra_in_candidate[:5])
        issues.append(f"candidate has {len(extra_in_candidate)} extra rows: {preview}")

    invalid_normalized = 0
    point_mismatches = 0
    task_shifts = defaultdict(list)
    task_measure_deltas = defaultdict(list)
    row_shift_outliers = []
    for key in sorted(set(reference_map) & set(candidate_map)):
        task_id, image_path = key
        cand_item = candidate_map[key]
        ref_item = reference_map[key]
        cand_norm = points_array(cand_item, "predicted_points_normalized")
        cand_px = points_array(cand_item, "predicted_points_pixels")
        ref_px = points_array(ref_item, "predicted_points_pixels")
        if cand_px.shape != ref_px.shape:
            point_mismatches += 1
            continue
        invalid_normalized += int(np.sum((cand_norm < -1e-5) | (cand_norm > 1.0 + 1e-5)))
        distances = np.linalg.norm(cand_px - ref_px, axis=1)
        mean_shift = float(np.mean(distances))
        task_shifts[task_id].append(mean_shift)
        measure_delta = measurement_delta(cand_px, ref_px, task_id)
        if np.isfinite(measure_delta):
            task_measure_deltas[task_id].append(measure_delta)
        if mean_shift > args.max_row_mean_shift_px:
            row_shift_outliers.append((task_id, image_path, mean_shift))

    if point_mismatches:
        issues.append(f"point-count mismatches versus reference: {point_mismatches}")
    if invalid_normalized > args.max_invalid_normalized:
        issues.append(f"invalid normalized coordinate values: {invalid_normalized}")
    if row_shift_outliers:
        worst = sorted(row_shift_outliers, key=lambda item: item[2], reverse=True)[:5]
        preview = ", ".join(f"{task}/{path}={shift:.2f}px" for task, path, shift in worst)
        issues.append(f"{len(row_shift_outliers)} rows exceed max mean shift {args.max_row_mean_shift_px:.2f}px: {preview}")

    thresholds = dict(DEFAULT_TASK_MEAN_THRESHOLDS)
    thresholds.update(parse_float_map(args.task_mean_thresholds))
    task_rows = []
    for task_id in EXPECTED_COUNTS:
        shifts = task_shifts.get(task_id, [])
        threshold = thresholds.get(task_id, float("inf"))
        mean_shift = float(np.mean(shifts)) if shifts else float("nan")
        median_shift = percentile(shifts, 50)
        p90_shift = percentile(shifts, 90)
        max_shift = float(np.max(shifts)) if shifts else float("nan")
        measure_delta = float(np.mean(task_measure_deltas[task_id])) if task_measure_deltas.get(task_id) else float("nan")
        over_threshold = int(np.sum(np.asarray(shifts, dtype=np.float32) > threshold)) if shifts else 0
        task_rows.append(
            {
                "task_id": task_id,
                "n": len(shifts),
                "mean_shift_px": mean_shift,
                "median_shift_px": median_shift,
                "p90_shift_px": p90_shift,
                "max_shift_px": max_shift,
                "measurement_delta_px": measure_delta,
                "mean_threshold_px": threshold,
                "rows_over_threshold": over_threshold,
            }
        )
        if np.isfinite(mean_shift) and mean_shift > threshold:
            issues.append(f"{task_id} mean shift {mean_shift:.2f}px exceeds threshold {threshold:.2f}px")
        if np.isfinite(p90_shift) and p90_shift > args.max_task_p90_shift_px:
            issues.append(f"{task_id} p90 shift {p90_shift:.2f}px exceeds threshold {args.max_task_p90_shift_px:.2f}px")

    overall_shift = float(np.mean([row["mean_shift_px"] for row in task_rows if np.isfinite(row["mean_shift_px"])]))
    status = "FAIL" if issues else "PASS"

    print("=" * 80)
    print("Pre-Submission Risk Audit")
    print("=" * 80)
    print(f"Candidate: {candidate_path}")
    print(f"Reference: {reference_path}")
    print(f"Status: {status}")
    print(f"Overall mean task shift: {overall_shift:.4f}px")
    print("\nTask shift summary versus reference:")
    print(
        "task_id      n   mean_px  median_px  p90_px  max_px  meas_delta_px  threshold  over"
    )
    for row in task_rows:
        print(
            f"{row['task_id']:<12} {row['n']:>3} "
            f"{row['mean_shift_px']:>8.3f} {row['median_shift_px']:>10.3f} "
            f"{row['p90_shift_px']:>7.3f} {row['max_shift_px']:>7.3f} "
            f"{row['measurement_delta_px']:>13.3f} {row['mean_threshold_px']:>10.3f} "
            f"{row['rows_over_threshold']:>4}"
        )

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo structural or shift-risk issues detected.")

    payload = {
        "status": status,
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "overall_mean_task_shift_px": overall_shift,
        "tasks": task_rows,
        "issues": issues,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {output_path}")

    if args.fail_on_risk and issues:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
