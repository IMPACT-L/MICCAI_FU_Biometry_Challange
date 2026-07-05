#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PREFERRED_CSV_ORDER = [
    "A4C_train.csv",
    "AOP_train.csv",
    "FA_train_new.csv",
    "FA_train.csv",
    "FUGC_train.csv",
    "HC_train.csv",
    "IVC_train.csv",
    "PLAX_train.csv",
    "PSAX_train.csv",
    "Reg-Two_3.fetal_femur.csv",
]
RESERVED_COLUMNS = {"image_path", "file_path", "height", "width", "task_name", "num_classes", "task_id"}
CSV_ALIASES = {
    "A4C_train.csv": ["A4C_train.csv"],
    "AOP_train.csv": ["AOP_train.csv", "key_points_xy.csv"],
    "FA_train_new.csv": ["FA_train_new.csv"],
    "FA_train.csv": ["FA_train.csv"],
    "FUGC_train.csv": ["FUGC_train.csv", "FUGC.csv"],
    "HC_train.csv": ["HC_train.csv", "HC.csv"],
    "IVC_train.csv": ["IVC_train.csv", "train.csv"],
    "PLAX_train.csv": ["PLAX_train.csv", "train.csv"],
    "PSAX_train.csv": ["PSAX_train.csv", "train.csv"],
    "Reg-Two_3.fetal_femur.csv": ["Reg-Two_3.fetal_femur.csv"],
}
TASK_FOLDER_ALIASES = {
    "PSAX": ["PSAX", "PSAK"],
    "PSAK": ["PSAX", "PSAK"],
}
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FU_Biometry raw data for the baseline code.")
    parser.add_argument("--raw-root", required=True, help="Path to the extracted official dataset root.")
    parser.add_argument("--output-root", default="data", help="Prepared dataset output directory.")
    parser.add_argument(
        "--images-root",
        default=None,
        help="Optional separate root for extracted images when CSVs and images live in different trees.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize images into output-root/images.",
    )
    return parser.parse_args()


def find_candidate_csvs(raw_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in raw_root.rglob("*.csv"):
        name = path.name
        for canonical_name, aliases in CSV_ALIASES.items():
            if canonical_name in found:
                continue
            if name not in aliases:
                continue
            expected_task = infer_task_folder(canonical_name)
            parent_parts = {part.lower() for part in path.parts}
            alias_ok = expected_task.lower() in parent_parts
            if expected_task == "PSAX":
                alias_ok = alias_ok or "psak" in parent_parts
            if alias_ok or name == canonical_name or canonical_name == "Reg-Two_3.fetal_femur.csv":
                found[canonical_name] = path
                break
    return found


def select_csvs(candidates: dict[str, Path]) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    skip_fa_old = "FA_train_new.csv" in candidates
    for csv_name in PREFERRED_CSV_ORDER:
        if csv_name == "FA_train.csv" and skip_fa_old:
            continue
        path = candidates.get(csv_name)
        if path is not None:
            selected.append((csv_name, path))
    return selected


def build_image_index(images_root: Path) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_rel: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel_key = normalize_relpath(path.relative_to(images_root).as_posix())
        by_name[path.name].append(path)
        by_rel[rel_key] = path
    return by_name, by_rel


def normalize_relpath(value: str) -> str:
    value = value.replace("\\", "/").strip()
    while value.startswith("../"):
        value = value[3:]
    while value.startswith("./"):
        value = value[2:]
    return value


def infer_task_folder(csv_name: str) -> str:
    if csv_name == "Reg-Two_3.fetal_femur.csv":
        return "fetal_femur"
    stem = csv_name.removesuffix(".csv")
    if stem.endswith("_train_new"):
        stem = stem[: -len("_train_new")]
    elif stem.endswith("_train"):
        stem = stem[: -len("_train")]
    elif stem.endswith("_val"):
        stem = stem[: -len("_val")]
    return stem


def iter_task_aliases(task_folder: str) -> list[str]:
    return TASK_FOLDER_ALIASES.get(task_folder, [task_folder])


def infer_task_id(csv_name: str) -> str:
    folder = infer_task_folder(csv_name)
    if folder == "fetal_femur":
        return "fetal_femur"
    return folder


def collect_xy_columns(df: pd.DataFrame) -> list[str]:
    if any(col.startswith("point_") and col.endswith("_xy") for col in df.columns):
        return sorted(
            [col for col in df.columns if col.startswith("point_") and col.endswith("_xy")],
            key=lambda name: int(name.split("_")[1]),
        )

    xy_columns: list[str] = []
    for col in df.columns:
        if col in RESERVED_COLUMNS:
            continue
        if col.endswith("_xy"):
            xy_columns.append(col)
    return xy_columns


def maybe_fix_fa_order(csv_name: str, xy_columns: list[str], using_old_fa: bool) -> list[str]:
    if csv_name != "FA_train.csv" or not using_old_fa:
        return xy_columns
    if len(xy_columns) >= 4:
        fixed = list(xy_columns)
        fixed[2], fixed[3] = fixed[3], fixed[2]
        return fixed
    return xy_columns


def resolve_image_source(
    image_value: str,
    raw_root: Path,
    image_index_by_name: dict[str, list[Path]],
    image_index_by_rel: dict[str, Path],
    task_folder: str | None = None,
) -> Path:
    raw_value = str(image_value).strip()
    if not raw_value:
        raise FileNotFoundError("Empty image path in CSV row.")

    as_path = Path(raw_value)
    if as_path.is_absolute() and as_path.is_file():
        return as_path

    normalized = normalize_relpath(raw_value)
    direct = raw_root / normalized
    if direct.is_file():
        return direct

    rel_match = image_index_by_rel.get(normalized)
    if rel_match is not None:
        return rel_match

    basename = Path(normalized).name
    name_matches = image_index_by_name.get(basename, [])
    if len(name_matches) == 1:
        return name_matches[0]

    if task_folder is not None and name_matches:
        task_aliases = {alias.lower() for alias in iter_task_aliases(task_folder)}
        filtered = []
        for path in name_matches:
            rel_parts = {part.lower() for part in path.relative_to(raw_root).parts}
            if rel_parts & task_aliases:
                filtered.append(path)
        if len(filtered) == 1:
            return filtered[0]

    suffix_matches = [path for key, path in image_index_by_rel.items() if key.endswith(normalized)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    if task_folder is not None:
        task_aliases = {alias.lower() for alias in iter_task_aliases(task_folder)}
        filtered_suffix = []
        for key, path in image_index_by_rel.items():
            rel_parts = {part.lower() for part in path.relative_to(raw_root).parts}
            if key.endswith("/" + basename) and rel_parts & task_aliases:
                filtered_suffix.append(path)
        if len(filtered_suffix) == 1:
            return filtered_suffix[0]

    raise FileNotFoundError(f"Could not resolve image path: {image_value}")


def make_unique_destination(dest_dir: Path, file_name: str) -> Path:
    candidate = dest_dir / file_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        retry = dest_dir / f"{stem}_{index}{suffix}"
        if not retry.exists():
            return retry
        index += 1


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src.resolve(), dst)


def normalize_point_value(value) -> str:
    if pd.isna(value):
        return json.dumps([0.0, 0.0])
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, Iterable):
        parsed = list(value)
    else:
        raise ValueError(f"Unsupported point value: {value!r}")
    if len(parsed) != 2:
        raise ValueError(f"Expected point with 2 values, got {parsed!r}")
    return json.dumps([float(parsed[0]), float(parsed[1])])


def read_csv_with_fallbacks(csv_path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed reading CSV {csv_path} with fallback encodings.") from last_error


def prepare_csv(
    csv_name: str,
    csv_path: Path,
    output_root: Path,
    image_mode: str,
    image_index_by_name: dict[str, list[Path]],
    image_index_by_rel: dict[str, Path],
    raw_root: Path,
) -> dict[str, int]:
    df = read_csv_with_fallbacks(csv_path)
    if "image_path" not in df.columns and "file_path" in df.columns:
        df = df.rename(columns={"file_path": "image_path"})
    if "image_path" not in df.columns:
        raise ValueError(f"{csv_path} is missing image_path/file_path.")

    task_folder = infer_task_folder(csv_name)
    task_id = infer_task_id(csv_name)
    xy_columns = collect_xy_columns(df)
    xy_columns = maybe_fix_fa_order(csv_name, xy_columns, using_old_fa=(csv_name == "FA_train.csv"))
    if not xy_columns:
        raise ValueError(f"{csv_path} has no landmark columns ending with _xy.")

    image_out_dir = output_root / "images" / task_folder
    rows = []
    image_cache: dict[Path, str] = {}

    for _, row in df.iterrows():
        src_image = resolve_image_source(
            row["image_path"],
            raw_root=raw_root,
            image_index_by_name=image_index_by_name,
            image_index_by_rel=image_index_by_rel,
            task_folder=task_folder,
        )
        if src_image not in image_cache:
            dst_image = make_unique_destination(image_out_dir, src_image.name)
            materialize_image(src_image, dst_image, image_mode)
            image_cache[src_image] = str(dst_image.relative_to(output_root).as_posix())
        rel_output_image = image_cache[src_image]

        new_row = {
            "image_path": rel_output_image,
            "height": row["height"] if "height" in df.columns else "",
            "width": row["width"] if "width" in df.columns else "",
            "task_name": row["task_name"] if "task_name" in df.columns else "Regression",
            "num_classes": int(row["num_classes"]) if "num_classes" in df.columns else len(xy_columns),
            "task_id": row["task_id"] if "task_id" in df.columns else task_id,
        }
        for point_index, col_name in enumerate(xy_columns, start=1):
            new_row[f"point_{point_index}_xy"] = normalize_point_value(row[col_name])
        rows.append(new_row)

    out_csv_name = "FA_train.csv" if csv_name == "FA_train_new.csv" else csv_name
    out_csv_path = output_root / "csv" / out_csv_name
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv_path, index=False)

    return {
        "rows": len(rows),
        "points_per_image": len(xy_columns),
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve() if args.images_root else raw_root
    output_root = Path(args.output_root).expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")

    candidates = find_candidate_csvs(raw_root)
    selected_csvs = select_csvs(candidates)
    if not selected_csvs:
        raise FileNotFoundError(
            "No recognized challenge CSV files were found under the raw root."
        )

    image_index_by_name, image_index_by_rel = build_image_index(images_root)
    summary = {}
    for csv_name, csv_path in selected_csvs:
        summary[csv_name] = prepare_csv(
            csv_name=csv_name,
            csv_path=csv_path,
            output_root=output_root,
            image_mode=args.image_mode,
            image_index_by_name=image_index_by_name,
            image_index_by_rel=image_index_by_rel,
            raw_root=images_root,
        )

    print("Prepared FU_Biometry dataset:")
    print(f"  raw root: {raw_root}")
    print(f"  images root: {images_root}")
    print(f"  output root: {output_root}")
    for csv_name, stats in summary.items():
        print(
            f"  - {csv_name}: {stats['rows']} rows, {stats['points_per_image']} landmarks/image"
        )


if __name__ == "__main__":
    main()
