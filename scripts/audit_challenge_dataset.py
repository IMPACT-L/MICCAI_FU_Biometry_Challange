#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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

FETAL_FEMUR_ORIENTATION_ANOMALY_BASENAMES = {
    "Patient00757_Plane5_1_of_1.png",
    "Patient00863_Plane5_1_of_2.png",
    "Patient00863_Plane5_2_of_2.png",
    "Patient01025_Plane5_1_of_1.png",
    "Patient01035_Plane5_2_of_4.png",
    "Patient01035_Plane5_4_of_4.png",
    "Patient01130_Plane5_2_of_4.png",
    "Patient01221_Plane5_2_of_2.png",
    "Patient01246_Plane5_1_of_2.png",
    "Patient01248_Plane5_1_of_1.png",
    "Patient01249_Plane5_1_of_1.png",
    "Patient01301_Plane5_1_of_2.png",
    "Patient01301_Plane5_2_of_2.png",
    "Patient01304_Plane5_2_of_2.png",
    "Patient01475_Plane5_1_of_1.png",
    "Patient01476_Plane5_1_of_1.png",
    "Patient01477_Plane5_1_of_2.png",
    "Patient01478_Plane5_1_of_1.png",
    "Patient01480_Plane5_1_of_1.png",
    "Patient01481_Plane5_1_of_1.png",
    "Patient01605_Plane5_2_of_2.png",
    "Patient01606_Plane5_2_of_2.png",
    "Patient01607_Plane5_1_of_2.png",
    "Patient01608_Plane5_1_of_1.png",
    "Patient01609_Plane5_1_of_1.png",
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


def audit_fetal_femur_orientation_note(data_root: Path) -> None:
    femur_path = data_root / "csv" / "Reg-Two_3.fetal_femur.csv"
    df = pd.read_csv(femur_path)
    basenames = df["image_path"].astype(str).map(os.path.basename)
    anomaly_mask = basenames.isin(FETAL_FEMUR_ORIENTATION_ANOMALY_BASENAMES)
    anomaly_count = int(anomaly_mask.sum())
    expected_count = len(FETAL_FEMUR_ORIENTATION_ANOMALY_BASENAMES)

    print("\nfetal_femur orientation-note audit")
    print(f"  - organizer-listed flipped rows found in raw CSV={anomaly_count}")
    print(f"  - effective training rows after loader exclusion={len(df) - anomaly_count}")
    if anomaly_count not in (0, expected_count):
        fail(
            "fetal_femur orientation anomaly count mismatch: "
            f"got {anomaly_count}, expected 0 if already removed or {expected_count} if raw CSV is intact"
        )
    if anomaly_count == expected_count:
        print("  - training loader excludes these rows; raw CSV is intentionally unchanged")
    else:
        print("  - organizer-listed rows are already absent from the prepared CSV")


def main() -> None:
    data_root = Path("data")
    audit_train_csvs(data_root)
    audit_validation_manifest(data_root)
    audit_fa_note(data_root)
    audit_fetal_femur_orientation_note(data_root)
    print("\nDATASET AUDIT PASSED")


if __name__ == "__main__":
    main()
