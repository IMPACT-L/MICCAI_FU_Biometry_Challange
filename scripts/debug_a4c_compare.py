#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from utils import canonicalize_task_coords  # noqa: E402


TASK_ID = "A4C"


def normalize_image_key(image_path: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == TASK_ID:
        return "/".join(parts)
    return f"{TASK_ID}/{os.path.basename(normalized)}"


def resolve_image_path(data_root: str, rel_path: str) -> str | None:
    rel_norm = os.path.normpath(rel_path)
    cleaned_rel = rel_norm
    while cleaned_rel.startswith(".." + os.sep):
        cleaned_rel = cleaned_rel[3:]
    for root in [os.path.join(data_root, "images"), data_root]:
        candidate = os.path.normpath(os.path.join(root, cleaned_rel))
        if os.path.isfile(candidate):
            return candidate
    return None


def load_a4c_dataframe(data_root: str) -> pd.DataFrame:
    csv_path = os.path.join(data_root, "csv", "A4C_train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing A4C CSV: {csv_path}")
    dataframe = pd.read_csv(csv_path).reset_index(drop=True)
    dataframe["image_key"] = dataframe["image_path"].astype(str).map(normalize_image_key)
    return dataframe


def load_predictions(pred_root: str) -> Dict[str, List[float]]:
    pred_file = pred_root
    if os.path.isdir(pred_root):
        pred_file = os.path.join(pred_root, "regression_predictions.json")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    with open(pred_file, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    pred_map: Dict[str, List[float]] = {}
    for row in predictions:
        if str(row.get("task_id")) != TASK_ID:
            continue
        pred_map[normalize_image_key(row["image_path"])] = row["predicted_points_pixels"]
    if not pred_map:
        raise ValueError(f"No {TASK_ID} predictions found in {pred_file}")
    return pred_map


def extract_gt(row: pd.Series, num_points: int) -> np.ndarray:
    coords: List[float] = []
    for idx in range(1, num_points + 1):
        coords.extend(json.loads(row[f"point_{idx}_xy"]))
    coords = canonicalize_task_coords(coords, TASK_ID)
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2)


def extract_pred(coords: List[float]) -> np.ndarray:
    coords = canonicalize_task_coords(coords, TASK_ID)
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2)


def compute_mre(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    distances = np.linalg.norm(pred_points - gt_points, axis=-1)
    return float(np.mean(distances))


def bbox_area(points: np.ndarray) -> float:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    wh = np.maximum(max_xy - min_xy, 1e-6)
    return float(wh[0] * wh[1])


def center_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    return float(np.linalg.norm(points_a.mean(axis=0) - points_b.mean(axis=0)))


def draw_panel(image: np.ndarray, gt_points: np.ndarray, pred_points: np.ndarray, title: str, mre: float) -> np.ndarray:
    canvas = image.copy()
    gt_i = gt_points.astype(np.int32)
    pred_i = pred_points.astype(np.int32)
    for idx, (gt_pt, pred_pt) in enumerate(zip(gt_i, pred_i), start=1):
        gt_xy = tuple(gt_pt.tolist())
        pred_xy = tuple(pred_pt.tolist())
        cv2.circle(canvas, gt_xy, 5, (0, 255, 0), -1)
        cv2.circle(canvas, pred_xy, 5, (255, 64, 64), -1)
        cv2.line(canvas, gt_xy, pred_xy, (255, 220, 0), 2)
        cv2.putText(canvas, str(idx), gt_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), gt_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    banner = np.full((64, canvas.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(banner, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (24, 24, 24), 2, cv2.LINE_AA)
    cv2.putText(
        banner,
        f"MRE(px): {mre:.3f} | GT=green Pred=red Error=yellow",
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (48, 48, 48),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, canvas])


def build_side_by_side(image: np.ndarray, gt_points: np.ndarray, good_points: np.ndarray, bad_points: np.ndarray, good_mre: float, bad_mre: float) -> np.ndarray:
    left = draw_panel(image, gt_points, good_points, "Reference / Best branch", good_mre)
    right = draw_panel(image, gt_points, bad_points, "Candidate branch", bad_mre)
    divider = np.full((left.shape[0], 14, 3), 230, dtype=np.uint8)
    return np.hstack([left, divider, right])


def main():
    parser = argparse.ArgumentParser(description="Compare A4C predictions from a good and bad branch with overlays.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--good-pred-root", required=True, help="Directory or JSON for the reference predictions.")
    parser.add_argument("--bad-pred-root", required=True, help="Directory or JSON for the candidate predictions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataframe = load_a4c_dataframe(args.data_root)
    good_pred_map = load_predictions(args.good_pred_root)
    bad_pred_map = load_predictions(args.bad_pred_root)

    records = []
    for _, row in dataframe.iterrows():
        image_key = row["image_key"]
        if image_key not in good_pred_map or image_key not in bad_pred_map:
            continue
        gt_points = extract_gt(row, int(row["num_classes"]))
        good_points = extract_pred(good_pred_map[image_key])
        bad_points = extract_pred(bad_pred_map[image_key])

        good_mre = compute_mre(good_points, gt_points)
        bad_mre = compute_mre(bad_points, gt_points)
        gt_area = bbox_area(gt_points)
        good_area = bbox_area(good_points)
        bad_area = bbox_area(bad_points)
        records.append(
            {
                "image_path": row["image_path"],
                "image_key": image_key,
                "good_mre": good_mre,
                "bad_mre": bad_mre,
                "delta_mre": bad_mre - good_mre,
                "gt_bbox_area": gt_area,
                "good_bbox_area": good_area,
                "bad_bbox_area": bad_area,
                "good_area_ratio": good_area / max(gt_area, 1e-6),
                "bad_area_ratio": bad_area / max(gt_area, 1e-6),
                "good_center_dist": center_distance(good_points, gt_points),
                "bad_center_dist": center_distance(bad_points, gt_points),
            }
        )

    summary_df = pd.DataFrame.from_records(records)
    if summary_df.empty:
        raise ValueError("No overlapping A4C predictions found.")

    summary_df = summary_df.sort_values(["delta_mre", "bad_mre"], ascending=[False, False]).reset_index(drop=True)
    summary_path = os.path.join(args.output_dir, "a4c_compare_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    indexed_df = dataframe.set_index("image_key")
    for rank, row in enumerate(summary_df.head(int(args.top_k)).itertuples(index=False), start=1):
        source_row = indexed_df.loc[row.image_key]
        if isinstance(source_row, pd.DataFrame):
            source_row = source_row.iloc[0]
        image_abs = resolve_image_path(args.data_root, str(source_row["image_path"]))
        if image_abs is None:
            continue
        image = cv2.imread(image_abs)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_points = extract_gt(source_row, int(source_row["num_classes"]))
        good_points = extract_pred(good_pred_map[row.image_key])
        bad_points = extract_pred(bad_pred_map[row.image_key])
        overlay = build_side_by_side(image, gt_points, good_points, bad_points, float(row.good_mre), float(row.bad_mre))
        name = f"{rank:02d}_delta_{row.delta_mre:.2f}_{os.path.basename(str(source_row['image_path']))}"
        save_path = os.path.join(overlay_dir, os.path.splitext(name)[0] + ".png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    stats = {
        "num_compared_samples": int(len(summary_df)),
        "mean_good_mre": float(summary_df["good_mre"].mean()),
        "mean_bad_mre": float(summary_df["bad_mre"].mean()),
        "mean_delta_mre": float(summary_df["delta_mre"].mean()),
        "mean_good_area_ratio": float(summary_df["good_area_ratio"].mean()),
        "mean_bad_area_ratio": float(summary_df["bad_area_ratio"].mean()),
        "median_bad_area_ratio": float(summary_df["bad_area_ratio"].median()),
        "median_good_area_ratio": float(summary_df["good_area_ratio"].median()),
    }
    with open(os.path.join(args.output_dir, "a4c_compare_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Saved summary: {summary_path}")
    print(f"Saved overlays: {overlay_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
