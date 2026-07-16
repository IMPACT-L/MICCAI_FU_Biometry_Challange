#!/usr/bin/env python
import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline")
sys.path.insert(0, BASELINE_DIR)

from dataset import KeypointDataset  # noqa: E402
from train import (  # noqa: E402
    CARDIAC_SPLIT_SCREEN_MODE,
    CARDIAC_SPLIT_SCREEN_VDARK_THRESHOLD,
    DATA_ROOT_PATH,
    RANDOM_SEED,
    SPLIT_MODE,
    VAL_SPLIT,
    _assign_cardiac_split_screen_flags,
    _assign_pseudo_domains,
    _grouped_stratified_split_indices,
    _pseudo_domain_grouped_split_indices,
    _stratified_split_indices,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the train/validation split used by baseline/train.py.")
    parser.add_argument("--data-root", default=DATA_ROOT_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--split-mode", choices=("row", "grouped", "pseudo_domain_grouped"), default=SPLIT_MODE)
    parser.add_argument(
        "--cardiac-split-screen-mode",
        choices=("keep", "exclude", "crop_panel"),
        default=CARDIAC_SPLIT_SCREEN_MODE,
    )
    parser.add_argument(
        "--cardiac-split-screen-vdark-threshold",
        type=float,
        default=CARDIAC_SPLIT_SCREEN_VDARK_THRESHOLD,
    )
    args = parser.parse_args()

    dataset = KeypointDataset(data_root=args.data_root, transforms=None)
    dataframe = _assign_cardiac_split_screen_flags(
        dataset.dataframe,
        args.data_root,
        vdark_threshold=args.cardiac_split_screen_vdark_threshold,
    )
    if args.cardiac_split_screen_mode == "exclude":
        dataframe = dataframe[~dataframe["is_split_screen_cardiac"].astype(bool)].reset_index(drop=True)
    elif args.cardiac_split_screen_mode not in {"keep", "crop_panel"}:
        raise ValueError(f"Unsupported cardiac_split_screen_mode: {args.cardiac_split_screen_mode}")
    dataframe = _assign_pseudo_domains(dataframe, args.data_root)

    if args.split_mode == "row":
        train_indices, val_indices = _stratified_split_indices(dataframe, args.val_split, args.seed)
    elif args.split_mode == "grouped":
        train_indices, val_indices = _grouped_stratified_split_indices(dataframe, args.val_split, args.seed)
    elif args.split_mode == "pseudo_domain_grouped":
        train_indices, val_indices = _pseudo_domain_grouped_split_indices(dataframe, args.val_split, args.seed)
    else:
        raise ValueError(f"Unsupported split_mode: {args.split_mode}")

    splits_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_path = os.path.join(splits_dir, "train_split.csv")
    val_path = os.path.join(splits_dir, "val_split.csv")
    dataframe.iloc[train_indices].reset_index(drop=True).to_csv(train_path, index=False)
    dataframe.iloc[val_indices].reset_index(drop=True).to_csv(val_path, index=False)

    print(f"Wrote {train_path} ({len(train_indices)} rows)")
    print(f"Wrote {val_path} ({len(val_indices)} rows)")
    print("Validation rows by task:")
    print(dataframe.iloc[val_indices].groupby("task_id").size().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
