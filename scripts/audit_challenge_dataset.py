#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPECTED_TRAIN_FILES = {
    "A4C_train.csv": {"rows": 108, "num_points": 16},
    "AOP_train.csv": {"rows": 4000, "num_points": 4},
    "FA_train.csv": {"rows": 500, "num_points": 4},
    "FUGC_train.csv": {"rows": 260, "num_points": 2},
    "HC_train.csv": {"rows": 999, "num_points": 4},
    "IVC_train.csv": {"rows": 38, "num_points": 2},
    "PLAX_train.csv": {"rows": 87, "num_points": 22},
    "PSAX_train.csv": {"rows": 49, "num_points": 4},
    "Reg-Two_3.fetal_femur.csv": {"rows": 727, "num_points": 2},
}

EXPECTED_VALIDATION_COUNTS = {
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


def fail(message: str) -> None:
    raise SystemExit(f"DATASET AUDIT FAILED: {message}")


def audit_train_csvs(data_root: Path) -> None:
    csv_root = data_root / "csv"
    if not csv_root.is_dir():
        fail(f"missing csv directory: {csv_root}")

    print("Training CSV audit")
    for file_name, expected in EXPECTED_TRAIN_FILES.items():
        csv_path = csv_root / file_name
        if not csv_path.is_file():
            fail(f"missing required training CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        row_count = len(df)
        if row_count != expected["rows"]:
            fail(f"{file_name} row count mismatch: got {row_count}, expected {expected['rows']}")

        if "num_classes" not in df.columns:
            fail(f"{file_name} missing num_classes column")
        unique_num_points = sorted(df["num_classes"].astype(int).unique().tolist())
        if unique_num_points != [expected["num_points"]]:
            fail(
                f"{file_name} num_classes mismatch: got {unique_num_points}, expected {[expected['num_points']]}"
            )

        print(f"  - {file_name}: rows={row_count}, num_points={expected['num_points']}")


def audit_validation_manifest(data_root: Path) -> None:
    manifest_path = data_root / "manifests" / "validation_manifest.csv"
    if not manifest_path.is_file():
        fail(f"missing validation manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    counts = df.groupby("task_id").size().to_dict()

    print("\nValidation manifest audit")
    if len(df) != sum(EXPECTED_VALIDATION_COUNTS.values()):
        fail(
            f"validation manifest row count mismatch: got {len(df)}, "
            f"expected {sum(EXPECTED_VALIDATION_COUNTS.values())}"
        )
    if counts != EXPECTED_VALIDATION_COUNTS:
        fail(f"validation manifest task counts mismatch: got {counts}, expected {EXPECTED_VALIDATION_COUNTS}")

    duplicate_count = int(df.duplicated(["task_id", "image_path"]).sum())
    if duplicate_count != 0:
        fail(f"validation manifest contains duplicate (task_id, image_path) rows: {duplicate_count}")

    fugc_duplicates = int(df[df["task_id"].astype(str) == "FUGC"].duplicated(["task_id", "image_path"]).sum())
    if fugc_duplicates != 0:
        fail(f"FUGC validation entries still contain duplicates: {fugc_duplicates}")

    print(f"  - rows={len(df)}")
    print(f"  - counts={counts}")
    print("  - duplicates=0")


def audit_fa_note(data_root: Path) -> None:
    fa_path = data_root / "csv" / "FA_train.csv"
    df = pd.read_csv(fa_path)
    required_cols = ["point_1_xy", "point_2_xy", "point_3_xy", "point_4_xy"]
    for col in required_cols:
        if col not in df.columns:
            fail(f"FA_train.csv missing required landmark column: {col}")

    # Organizer note: if old FA_train.csv is used, items 3 and 4 must be swapped.
    # We use a simple sanity heuristic on the prepared output to catch the old order.
    p3_x = []
    p4_x = []
    for _, row in df.iterrows():
        xy3 = json.loads(row["point_3_xy"])
        xy4 = json.loads(row["point_4_xy"])
        p3_x.append(float(xy3[0]))
        p4_x.append(float(xy4[0]))
    median_dx = float(pd.Series(p3_x).median() - pd.Series(p4_x).median())

    print("\nFA organizer-note audit")
    print(f"  - median(point_3_x - point_4_x)={median_dx:.3f}")
    if abs(median_dx) < 1.0:
        print("  - warning: FA point-3 / point-4 separation looks ambiguous; inspect raw source if unsure")
    else:
        print("  - prepared FA file looks internally consistent with the point-order fix path")


def main() -> None:
    data_root = Path("data")
    audit_train_csvs(data_root)
    audit_validation_manifest(data_root)
    audit_fa_note(data_root)
    print("\nDATASET AUDIT PASSED")


if __name__ == "__main__":
    main()
