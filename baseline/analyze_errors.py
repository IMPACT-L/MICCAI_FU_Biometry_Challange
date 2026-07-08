import argparse
import glob
import json
import os
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from utils import canonicalize_task_coords


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


def normalize_eval_image_path(image_path: str, task_id: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return "/".join(parts)
    return f"{task_id}/{os.path.basename(normalized)}"


def resolve_image_path(data_root: str, rel_path: str) -> str | None:
    rel_norm = os.path.normpath(rel_path)
    cleaned_rel = rel_norm
    while cleaned_rel.startswith(".." + os.sep):
        cleaned_rel = cleaned_rel[3:]

    for root in [os.path.join(data_root, "images"), data_root]:
        direct = os.path.normpath(os.path.join(root, cleaned_rel))
        if os.path.isfile(direct):
            return direct
    return None


def load_dataframe(data_root: str) -> pd.DataFrame:
    csv_path = os.path.join(data_root, "csv")
    if not os.path.isdir(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    all_csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_path}")

    df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
    dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
    is_regression = dataframe["task_name"].astype(str).eq("Regression")
    is_extra_task = dataframe["task_id"].astype(str).isin(EXTRA_REGRESSION_TASK_IDS)
    dataframe = dataframe[is_regression | is_extra_task].reset_index(drop=True)
    if dataframe.empty:
        raise ValueError("No keypoint records found in local CSV files.")
    return dataframe


def load_predictions(pred_root: str) -> Dict[str, Dict[str, List[float]]]:
    pred_file = os.path.join(pred_root, "regression_predictions.json")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    with open(pred_file, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    pred_dict: Dict[str, Dict[str, List[float]]] = {}
    for pred in predictions:
        task_id = str(pred["task_id"])
        image_key = normalize_eval_image_path(pred["image_path"], task_id)
        pred_dict.setdefault(task_id, {})[image_key] = pred["predicted_points_pixels"]
    return pred_dict


def extract_gt_coords(row: pd.Series, num_points: int, task_id: str) -> np.ndarray:
    coords: List[float] = []
    for idx in range(1, num_points + 1):
        col = f"point_{idx}_xy"
        if col in row and pd.notna(row[col]):
            coords.extend(json.loads(row[col]))
        else:
            coords.extend([0.0, 0.0])
    coords = canonicalize_task_coords(coords, task_id)
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2)


def extract_pred_coords(pred_coords: List[float], task_id: str) -> np.ndarray:
    coords = canonicalize_task_coords(pred_coords, task_id)
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2)


def compute_mre(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=-1))
    return float(np.mean(distances))


def draw_overlay(image: np.ndarray, gt_points: np.ndarray, pred_points: np.ndarray, task_id: str, mre: float) -> np.ndarray:
    canvas = image.copy()
    gt_points = gt_points.astype(np.int32)
    pred_points = pred_points.astype(np.int32)

    for idx, (gt_pt, pred_pt) in enumerate(zip(gt_points, pred_points)):
        gt_xy = tuple(gt_pt.tolist())
        pred_xy = tuple(pred_pt.tolist())
        cv2.circle(canvas, gt_xy, 5, (0, 255, 0), -1)
        cv2.circle(canvas, pred_xy, 5, (255, 0, 0), -1)
        cv2.line(canvas, gt_xy, pred_xy, (0, 255, 255), 2)
        cv2.putText(canvas, str(idx + 1), gt_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), gt_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    banner = np.full((72, canvas.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        banner,
        f"Task: {task_id}  MRE(px): {mre:.3f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "GT=green  Pred=blue  Error=yellow",
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, canvas])


def main():
    parser = argparse.ArgumentParser(description="Analyze per-sample keypoint errors and save worst-case overlays.")
    parser.add_argument("--data-root", default="data", help="Dataset root containing csv/ and images/")
    parser.add_argument("--pred-root", default="predictions", help="Prediction directory containing regression_predictions.json")
    parser.add_argument("--output-dir", default="error_analysis", help="Directory to save CSV summaries and overlays")
    parser.add_argument("--top-k", type=int, default=10, help="Worst samples to save per task")
    parser.add_argument(
        "--task-ids",
        default="FUGC,IVC,HC,PSAX,A4C,fetal_femur,PLAX,AOP,FA",
        help="Comma-separated task IDs to analyze",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataframe = load_dataframe(args.data_root)
    pred_dict = load_predictions(args.pred_root)
    task_ids = [item.strip() for item in str(args.task_ids).split(",") if item.strip()]

    records: List[dict] = []
    for task_id in task_ids:
        task_df = dataframe[dataframe["task_id"].astype(str) == task_id]
        if task_df.empty:
            continue
        num_points = int(task_df["num_classes"].iloc[0])
        for _, row in task_df.iterrows():
            image_key = normalize_eval_image_path(row["image_path"], task_id)
            pred_coords = pred_dict.get(task_id, {}).get(image_key)
            if pred_coords is None:
                continue
            gt_points = extract_gt_coords(row, num_points, task_id)
            pred_points = extract_pred_coords(pred_coords, task_id)
            mre = compute_mre(pred_points, gt_points)
            records.append(
                {
                    "task_id": task_id,
                    "image_path": str(row["image_path"]),
                    "normalized_image_path": image_key,
                    "mre_pixels": mre,
                    "num_points": num_points,
                }
            )

    summary_df = pd.DataFrame.from_records(records)
    if summary_df.empty:
        raise ValueError("No comparable samples found between local labels and predictions.")
    summary_df = summary_df.sort_values(["task_id", "mre_pixels"], ascending=[True, False]).reset_index(drop=True)
    summary_path = os.path.join(args.output_dir, "worst_samples_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    for task_id in task_ids:
        task_rows = summary_df[summary_df["task_id"] == task_id].head(int(args.top_k))
        if task_rows.empty:
            continue
        task_dir = os.path.join(args.output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        task_df = dataframe[dataframe["task_id"].astype(str) == task_id]
        task_df = task_df.set_index(task_df["image_path"].astype(str))
        for rank, row in enumerate(task_rows.itertuples(index=False), start=1):
            image_path = str(row.image_path)
            image_abs = resolve_image_path(args.data_root, image_path)
            if image_abs is None:
                continue
            image = cv2.imread(image_abs)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            source_row = task_df.loc[image_path]
            if isinstance(source_row, pd.DataFrame):
                source_row = source_row.iloc[0]
            gt_points = extract_gt_coords(source_row, int(row.num_points), task_id)
            pred_points = extract_pred_coords(pred_dict[task_id][row.normalized_image_path], task_id)
            overlay = draw_overlay(image, gt_points, pred_points, task_id, float(row.mre_pixels))
            save_path = os.path.join(task_dir, f"{rank:02d}_{os.path.basename(image_path)}")
            save_path = os.path.splitext(save_path)[0] + ".png"
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    task_means = summary_df.groupby("task_id")["mre_pixels"].agg(["mean", "max", "count"]).reset_index()
    task_means.to_csv(os.path.join(args.output_dir, "task_error_summary.csv"), index=False)
    print(f"Saved summary: {summary_path}")
    print(f"Saved task summary: {os.path.join(args.output_dir, 'task_error_summary.csv')}")
    print(f"Saved overlays under: {args.output_dir}")


if __name__ == "__main__":
    main()
