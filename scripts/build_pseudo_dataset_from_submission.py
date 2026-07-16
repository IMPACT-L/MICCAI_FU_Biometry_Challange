#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.path.realpath(dst) == str(src.resolve()):
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def _point_column_names(num_points: int) -> list[str]:
    return [f"point_{idx}_xy" for idx in range(1, num_points + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone pseudo-labeled dataset root from a submission JSON and the official validation manifest."
    )
    parser.add_argument("--base-data-root", default="data", help="Original dataset root containing csv/, images/, manifests/, validation_ready/.")
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv", help="Official validation manifest.")
    parser.add_argument("--submission-json", required=True, help="Submission regression_predictions.json used as pseudo labels.")
    parser.add_argument("--output-root", required=True, help="New dataset root to create.")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated subset of task IDs to pseudo-label. Default: all tasks in manifest.",
    )
    parser.add_argument(
        "--copy-train-csv",
        action="store_true",
        help="Copy training CSVs into the pseudo root instead of symlinking them.",
    )
    parser.add_argument(
        "--pseudo-only",
        action="store_true",
        help="Do not include original training CSVs; build a dataset root containing only pseudo-labeled validation rows.",
    )
    args = parser.parse_args()

    base_data_root = Path(args.base_data_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    submission_json = Path(args.submission_json).resolve()
    output_root = Path(args.output_root).resolve()
    selected_task_ids = None
    if args.task_ids:
        selected_task_ids = {item.strip() for item in str(args.task_ids).split(",") if item.strip()}

    manifest_df = pd.read_csv(manifest_path)
    predictions = json.loads(submission_json.read_text())
    prediction_map = {(item["task_id"], item["image_path"]): item for item in predictions}

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "csv").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    if args.pseudo_only:
        pass
    elif args.copy_train_csv:
        shutil.copytree(base_data_root / "csv", output_root / "csv", dirs_exist_ok=True)
    else:
        for csv_file in sorted((base_data_root / "csv").glob("*.csv")):
            _safe_symlink(csv_file, output_root / "csv" / csv_file.name)

    shutil.copy2(manifest_path, output_root / "manifests" / manifest_path.name)
    _safe_symlink(base_data_root / "images", output_root / "images")
    _safe_symlink(base_data_root / "validation_ready", output_root / "validation_ready")

    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_df.to_dict("records"):
        task_id = str(row["task_id"])
        if selected_task_ids is not None and task_id not in selected_task_ids:
            continue
        image_path = str(row["image_path"])
        pred = prediction_map.get((task_id, image_path))
        if pred is None:
            raise KeyError(f"Missing prediction for {(task_id, image_path)} in {submission_json}")

        num_points = int(row["num_points"])
        pred_points = pred["predicted_points_pixels"]
        if len(pred_points) != num_points * 2:
            raise ValueError(
                f"Prediction length mismatch for {(task_id, image_path)}: "
                f"got {len(pred_points)}, expected {num_points * 2}"
            )

        pseudo_row = {
            "image_path": f"validation_ready/{image_path}",
            "height": int(row["height"]),
            "width": int(row["width"]),
            "task_name": "Regression",
            "num_classes": num_points,
            "task_id": task_id,
        }
        points = [pred_points[i : i + 2] for i in range(0, len(pred_points), 2)]
        for col_name, point_xy in zip(_point_column_names(num_points), points):
            pseudo_row[col_name] = json.dumps([round(float(point_xy[0]), 6), round(float(point_xy[1]), 6)])
        rows_by_task[task_id].append(pseudo_row)

    task_file_names = {
        "A4C": "pseudo_A4C_val.csv",
        "AOP": "pseudo_AOP_val.csv",
        "FA": "pseudo_FA_val.csv",
        "FUGC": "pseudo_FUGC_val.csv",
        "HC": "pseudo_HC_val.csv",
        "IVC": "pseudo_IVC_val.csv",
        "PLAX": "pseudo_PLAX_val.csv",
        "PSAX": "pseudo_PSAX_val.csv",
        "fetal_femur": "pseudo_fetal_femur_val.csv",
    }

    summary = {}
    for task_id, rows in sorted(rows_by_task.items()):
        out_name = task_file_names.get(task_id, f"pseudo_{task_id}_val.csv")
        out_path = output_root / "csv" / out_name
        pd.DataFrame(rows).to_csv(out_path, index=False)
        summary[task_id] = len(rows)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "submission_json": str(submission_json),
                "selected_task_ids": sorted(selected_task_ids) if selected_task_ids else "all",
                "pseudo_counts": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
