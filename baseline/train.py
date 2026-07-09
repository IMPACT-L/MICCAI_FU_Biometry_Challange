import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import KeypointDataset, KeypointUniformSampler
from model_factory import MultiTaskModelFactory
from model_profiles import MODEL_PROFILE_NAMES, apply_model_profile
from utils import (
    build_line_mask_from_transformed_coords,
    compute_combined_score,
    compute_dataset_specific_loss,
    compute_femur_shaft_loss,
    compute_fugc_segment_loss,
    compute_ivc_band_loss,
    compute_normalization_stats_from_dataframe,
    compute_measurement_loss,
    compute_robust_domain_breakdown,
    compute_server_proxy_breakdown,
    evaluate_keypoint,
    keypoint_collate_fn,
    set_seed,
    softargmax_heatmaps_to_transformed_coords,
    transformed_coords_to_original_normalized,
)


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS =35
DATA_ROOT_PATH = "data"
OUTPUT_DIR = "output"
ENCODER = "vit_base_patch16_dinov3"
ENCODER_WEIGHTS = "pretrained"
RANDOM_SEED = 42
VAL_SPLIT = 0.2
CHECKPOINT_SCORE_NAME = "Combined score"
CHECKPOINT_SCORE_MODE = "combined"
SERVER_PROXY_HARD_TASK_IDS = ("A4C", "IVC", "PSAX", "fetal_femur", "HC")
SERVER_PROXY_V2_HARD_TASK_WEIGHT_OVERRIDES = {
    "A4C": 3.0,
    "IVC": 1.4,
    "AOP": 1.3,
    "FA": 1.2,
    "PSAX": 1.2,
    "HC": 1.1,
    "fetal_femur": 1.0,
}
INPUT_SIZE = 512
EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
USE_FPN = True  # ← 开关：设为 True 启用 FPN 特征金字塔
FPN_MODE = "shared"
FPN_TYPE = "fpn"
HEAD_TYPE = "deep"
TASK_HEAD_PROFILE = "challenge_v1"
TASK_DECODER_PROFILE = "uniform"
TASK_ADAPTER_PROFILE = "uniform"
TASK_LOSS_FAMILY_PROFILE = "uniform"
SPLIT_MODE = "row"
AUGMENTATION_PROFILE = "baseline"
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
FEMUR_SHAFT_LOSS_WEIGHT = 0.15
FUGC_SEGMENT_LOSS_WEIGHT = 0.08
IVC_BAND_LOSS_WEIGHT = 0.08
ROI_CROP_TASKS = ("FUGC", "IVC", "fetal_femur")
ROI_CONTEXT_RANGE = (1.2, 1.8)
TASK_LOSS_WEIGHTS = {
    "A4C": 1.35,
    "AOP": 0.80,
    "FA": 1.25,
    "FUGC": 1.00,
    "HC": 1.20,
    "IVC": 1.10,
    "PLAX": 1.35,
    "PSAX": 1.10,
    "fetal_femur": 1.25,
}
CHECKPOINT_TASK_WEIGHTS = {
    "A4C": 1.35,
    "AOP": 0.90,
    "FA": 0.90,
    "FUGC": 1.10,
    "HC": 1.25,
    "IVC": 1.50,
    "PLAX": 1.00,
    "PSAX": 1.20,
    "fetal_femur": 1.20,
}
CARDIAC_SPLIT_SCREEN_TASK_IDS = ("A4C", "PSAX", "PLAX", "IVC")
CARDIAC_SPLIT_SCREEN_MODE = "keep"
CARDIAC_SPLIT_SCREEN_VDARK_THRESHOLD = 12.0
CARDIAC_SPLIT_SCREEN_CENTER_MAX = 0.42
CARDIAC_SPLIT_SCREEN_RIGHT_EXTENT_MAX = 0.72


def _resolve_image_path_for_split(data_root: str, rel_path: str) -> str | None:
    rel_norm = os.path.normpath(str(rel_path))
    cleaned_rel = rel_norm
    while cleaned_rel.startswith(".." + os.sep):
        cleaned_rel = cleaned_rel[3:]
    for root in [os.path.join(data_root, "images"), data_root]:
        direct = os.path.normpath(os.path.join(root, cleaned_rel))
        if os.path.isfile(direct):
            return direct
    return None


def _assign_pseudo_domains(dataframe, data_root: str):
    feature_rows = []
    for idx, row in dataframe.iterrows():
        image_path = _resolve_image_path_for_split(data_root, row["image_path"])
        if image_path is None:
            continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        mean_val = float(image.mean())
        std_val = float(image.std())
        sharpness = float(cv2.Laplacian(image, cv2.CV_32F).var())
        height, width = image.shape[:2]
        feature_rows.append((idx, mean_val, std_val, sharpness, float(width) / max(float(height), 1.0)))

    if not feature_rows:
        dataframe = dataframe.copy()
        dataframe["pseudo_domain"] = "unknown"
        return dataframe

    feature_array = np.array([[row[1], row[2], row[3], row[4]] for row in feature_rows], dtype=np.float32)
    medians = np.median(feature_array, axis=0)

    domain_labels = {}
    for idx, mean_val, std_val, sharpness, aspect_ratio in feature_rows:
        brightness_bin = int(mean_val > medians[0])
        contrast_bin = int(std_val > medians[1])
        sharpness_bin = int(sharpness > medians[2])
        aspect_bin = int(aspect_ratio > medians[3])
        domain_labels[idx] = f"b{brightness_bin}_c{contrast_bin}_s{sharpness_bin}_a{aspect_bin}"

    dataframe = dataframe.copy()
    dataframe["pseudo_domain"] = [
        domain_labels.get(idx, "unknown")
        for idx in dataframe.index
    ]
    return dataframe


def _extract_row_x_extent(row) -> tuple[float, float, float] | None:
    try:
        width = max(float(row["width"]), 1.0)
        num_points = int(row["num_classes"])
    except Exception:
        return None

    xs = []
    for point_idx in range(1, num_points + 1):
        col = f"point_{point_idx}_xy"
        if col not in row or row[col] is None:
            continue
        try:
            point_xy = json.loads(row[col])
            xs.append(float(point_xy[0]))
        except Exception:
            continue
    if not xs:
        return None
    xs_np = np.array(xs, dtype=np.float32)
    return (
        float(xs_np.min() / width),
        float(xs_np.max() / width),
        float(xs_np.mean() / width),
    )


def _assign_cardiac_split_screen_flags(
    dataframe,
    data_root: str,
    vdark_threshold: float = CARDIAC_SPLIT_SCREEN_VDARK_THRESHOLD,
    center_max: float = CARDIAC_SPLIT_SCREEN_CENTER_MAX,
    right_extent_max: float = CARDIAC_SPLIT_SCREEN_RIGHT_EXTENT_MAX,
):
    dataframe = dataframe.copy()
    dataframe["split_screen_score"] = 0.0
    dataframe["is_split_screen_cardiac"] = False

    candidate_task_ids = set(CARDIAC_SPLIT_SCREEN_TASK_IDS)
    for idx, row in dataframe.iterrows():
        task_id = str(row["task_id"])
        if task_id not in candidate_task_ids:
            continue

        image_path = _resolve_image_path_for_split(data_root, row["image_path"])
        if image_path is None:
            continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        height, width = image.shape[:2]
        mid_left = max(0, width // 2 - 3)
        mid_right = min(width, width // 2 + 3)
        center_strip = image[:, mid_left:mid_right]
        if center_strip.size == 0:
            continue

        background_mean = float(image.mean())
        vdark = max(0.0, background_mean - float(center_strip.mean()))
        dataframe.at[idx, "split_screen_score"] = float(vdark)

        extent = _extract_row_x_extent(row)
        if extent is None:
            continue
        _, x_max_norm, x_center_norm = extent
        is_split_screen = (
            vdark >= float(vdark_threshold)
            and x_center_norm <= float(center_max)
            and x_max_norm <= float(right_extent_max)
        )
        dataframe.at[idx, "is_split_screen_cardiac"] = bool(is_split_screen)

    return dataframe


def _parse_weight_csv(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    result: dict[str, float] = {}
    for chunk in str(value).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight override entry: {item}. Expected TASK=VALUE.")
        key, raw_value = item.split("=", 1)
        task_id = key.strip()
        if not task_id:
            raise ValueError(f"Invalid empty task ID in weight override: {item}")
        result[task_id] = float(raw_value.strip())
    return result or None


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        for name, value in state_dict.items():
            if torch.is_floating_point(value):
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def _parse_task_id_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    task_ids = [item.strip() for item in str(value).split(",") if item.strip()]
    return task_ids or None


def _parse_task_id_set_csv(value: str | None) -> set[str] | None:
    task_ids = _parse_task_id_csv(value)
    if task_ids is None:
        return None
    return set(task_ids)


def _build_group_key(task_id: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(str(image_path)))[0]
    if task_id in {"A4C", "PLAX", "PSAX", "IVC"}:
        stem = re.sub(r"_frame\d+$", "", stem)
    if task_id == "fetal_femur":
        match = re.match(r"(Patient\d+)", stem)
        if match:
            return match.group(1)
    if task_id == "HC":
        stem = re.sub(r"_HC$", "", stem)
    return stem


def _filter_dataframe_by_task_ids(dataframe, task_ids: Iterable[str] | None):
    if not task_ids:
        return dataframe.reset_index(drop=True)
    task_id_set = {str(task_id) for task_id in task_ids}
    filtered = dataframe[dataframe["task_id"].astype(str).isin(task_id_set)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No samples found for train task IDs: {sorted(task_id_set)}")
    return filtered


def _load_checkpoint_payload(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("meta", {})
    return checkpoint, {}


def _load_matching_state_dict(model: torch.nn.Module, checkpoint_state_dict: dict[str, torch.Tensor]):
    model_state = model.state_dict()
    matched_state = {}
    skipped = []
    for name, value in checkpoint_state_dict.items():
        if name not in model_state:
            skipped.append((name, "missing_in_model"))
            continue
        if model_state[name].shape != value.shape:
            skipped.append((name, f"shape_mismatch:{tuple(value.shape)}->{tuple(model_state[name].shape)}"))
            continue
        matched_state[name] = value

    missing_after_match = sorted(set(model_state.keys()) - set(matched_state.keys()))
    load_result = model.load_state_dict(matched_state, strict=False)
    return load_result, skipped, missing_after_match


def _build_task_configs(dataframe):
    configs = []
    seen = set()
    for _, row in dataframe.iterrows():
        task_name = str(row["task_name"])
        task_id = str(row["task_id"])
        if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        configs.append(
            {
                "task_id": task_id,
                "task_name": "Regression",
                "num_classes": int(row["num_classes"]),
            }
        )
    if not configs:
        raise ValueError("No keypoint tasks found in dataset.")
    return configs


def _stratified_split_indices(dataframe, val_split: float, seed: int):
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError("val_split must be in (0, 1).")

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for _, group in dataframe.groupby("task_id", sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)

        total = len(indices)
        # Per-task split count (rounded) to keep each task close to the requested ratio.
        val_count = int(round(total * float(val_split)))
        if total >= 2:
            val_count = max(1, min(total - 1, val_count))
        else:
            val_count = 0

        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _grouped_stratified_split_indices(dataframe, val_split: float, seed: int):
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError("val_split must be in (0, 1).")

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for task_id, group in dataframe.groupby("task_id", sort=True):
        group = group.copy()
        group["__split_group_key"] = group["image_path"].astype(str).map(
            lambda path: _build_group_key(str(task_id), path)
        )
        unique_group_keys = group["__split_group_key"].drop_duplicates().tolist()
        rng.shuffle(unique_group_keys)

        total_groups = len(unique_group_keys)
        val_group_count = int(round(total_groups * float(val_split)))
        if total_groups >= 2:
            val_group_count = max(1, min(total_groups - 1, val_group_count))
        else:
            val_group_count = 0

        val_group_keys = set(unique_group_keys[:val_group_count])
        val_mask = group["__split_group_key"].isin(val_group_keys)
        val_indices.extend(group.index[val_mask].tolist())
        train_indices.extend(group.index[~val_mask].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _pseudo_domain_grouped_split_indices(dataframe, val_split: float, seed: int):
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError("val_split must be in (0, 1).")

    if "pseudo_domain" not in dataframe.columns:
        raise ValueError("pseudo_domain column is required for pseudo_domain_grouped split.")

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for task_id, task_group in dataframe.groupby("task_id", sort=True):
        task_group = task_group.copy()
        task_group["__split_group_key"] = task_group["image_path"].astype(str).map(
            lambda path: _build_group_key(str(task_id), path)
        )
        for _, domain_group in task_group.groupby("pseudo_domain", sort=True):
            unique_group_keys = domain_group["__split_group_key"].drop_duplicates().tolist()
            rng.shuffle(unique_group_keys)

            total_groups = len(unique_group_keys)
            val_group_count = int(round(total_groups * float(val_split)))
            if total_groups >= 2:
                val_group_count = max(1, min(total_groups - 1, val_group_count))
            else:
                val_group_count = 0

            val_group_keys = set(unique_group_keys[:val_group_count])
            val_mask = domain_group["__split_group_key"].isin(val_group_keys)
            val_indices.extend(domain_group.index[val_mask].tolist())
            train_indices.extend(domain_group.index[~val_mask].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _build_train_transforms(profile: str):
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    if profile == "baseline":
        return A.Compose(
            [
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    if profile == "strong_ultrasound_v1":
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=5, p=1.0),
                    ],
                    p=0.30,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.20,
                            contrast_limit=0.20,
                            p=1.0,
                        ),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.40,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    if profile == "ultrasound_robust_v1":
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(88, 112), p=1.0),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.12,
                            contrast_limit=0.12,
                            p=1.0,
                        ),
                        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.35,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                        A.GaussNoise(p=1.0),
                        A.ImageCompression(quality_range=(85, 98), p=1.0),
                    ],
                    p=0.20,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    if profile == "ultrasound_domain_shift_v1":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.Affine(
                    scale={"x": (0.92, 1.10), "y": (0.92, 1.10)},
                    translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                    rotate=(-12, 12),
                    shear={"x": (-8, 8), "y": (-4, 4)},
                    fit_output=False,
                    keep_ratio=False,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.55,
                ),
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(88, 112), p=1.0),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.14,
                            contrast_limit=0.14,
                            p=1.0,
                        ),
                        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.35,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                        A.GaussNoise(p=1.0),
                        A.ImageCompression(quality_range=(85, 98), p=1.0),
                    ],
                    p=0.20,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    if profile == "ultrasound_domain_shift_v2":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.30),
                A.Affine(
                    scale={"x": (0.88, 1.16), "y": (0.88, 1.16)},
                    translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                    rotate=(-15, 15),
                    shear={"x": (-10, 10), "y": (-6, 6)},
                    fit_output=False,
                    keep_ratio=False,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.65,
                ),
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(84, 116), p=1.0),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.18,
                            contrast_limit=0.18,
                            p=1.0,
                        ),
                        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.45,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=5, p=1.0),
                        A.GaussNoise(p=1.0),
                        A.ImageCompression(quality_range=(82, 97), p=1.0),
                    ],
                    p=0.28,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    if profile == "ultrasound_mixed_v1":
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.NoOp(p=1.0),
                        A.Affine(
                            scale={"x": (0.94, 1.08), "y": (0.94, 1.08)},
                            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                            rotate=(-8, 8),
                            shear={"x": (-5, 5), "y": (-3, 3)},
                            fit_output=False,
                            keep_ratio=False,
                            border_mode=cv2.BORDER_CONSTANT,
                            fill=0,
                            p=1.0,
                        ),
                    ],
                    p=0.35,
                ),
                A.HorizontalFlip(p=0.12),
                A.OneOf(
                    [
                        A.NoOp(p=1.0),
                        A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.10,
                            contrast_limit=0.10,
                            p=1.0,
                        ),
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.35,
                ),
                A.OneOf(
                    [
                        A.NoOp(p=1.0),
                        A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                        A.GaussNoise(p=1.0),
                        A.ImageCompression(quality_range=(88, 98), p=1.0),
                    ],
                    p=0.15,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    raise ValueError(f"Unsupported augmentation profile: {profile}")


def setup_logger(log_path: str) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main(
    val_split: float = VAL_SPLIT,
    model_profile: str | None = None,
    random_seed: int = RANDOM_SEED,
    use_fpn: bool = USE_FPN,
    fpn_mode: str = FPN_MODE,
    fpn_type: str = FPN_TYPE,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    steps_per_epoch: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    output_dir: str = OUTPUT_DIR,
    encoder_name: str = ENCODER,
    input_size: int = INPUT_SIZE,
    head_type: str = HEAD_TYPE,
    task_head_profile: str = TASK_HEAD_PROFILE,
    task_decoder_profile: str = TASK_DECODER_PROFILE,
    task_adapter_profile: str = TASK_ADAPTER_PROFILE,
    task_loss_family_profile: str = TASK_LOSS_FAMILY_PROFILE,
    learning_rate: float = LEARNING_RATE,
    init_checkpoint: str | None = None,
    train_task_ids: list[str] | None = None,
    measurement_loss_weight: float = 0.0,
    dataset_loss_weight: float = 0.0,
    femur_shaft_loss_weight: float = FEMUR_SHAFT_LOSS_WEIGHT,
    fugc_segment_loss_weight: float = FUGC_SEGMENT_LOSS_WEIGHT,
    ivc_band_loss_weight: float = IVC_BAND_LOSS_WEIGHT,
    task_loss_weight_overrides: dict[str, float] | None = None,
    sampler_task_weight_overrides: dict[str, float] | None = None,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    train_roi_crop: bool = False,
    roi_crop_tasks: set[str] | None = None,
    roi_context_min: float = ROI_CONTEXT_RANGE[0],
    roi_context_max: float = ROI_CONTEXT_RANGE[1],
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    split_mode: str = SPLIT_MODE,
    augmentation_profile: str = AUGMENTATION_PROFILE,
    checkpoint_task_weight_overrides: dict[str, float] | None = None,
    checkpoint_score_mode: str = CHECKPOINT_SCORE_MODE,
    cardiac_split_screen_mode: str = CARDIAC_SPLIT_SCREEN_MODE,
    cardiac_split_screen_vdark_threshold: float = CARDIAC_SPLIT_SCREEN_VDARK_THRESHOLD,
):
    if model_profile is not None:
        profile_config = apply_model_profile(
            model_profile,
            "train",
            {
                "encoder_name": encoder_name,
                "input_size": input_size,
                "use_fpn": use_fpn,
                "fpn_mode": fpn_mode,
                "fpn_type": fpn_type,
                "head_type": head_type,
                "task_head_profile": task_head_profile,
                "task_decoder_profile": task_decoder_profile,
                "task_adapter_profile": task_adapter_profile,
                "task_loss_family_profile": task_loss_family_profile,
                "split_mode": split_mode,
                "augmentation_profile": augmentation_profile,
                "checkpoint_score_mode": checkpoint_score_mode,
                "cardiac_split_screen_mode": cardiac_split_screen_mode,
                "measurement_loss_weight": measurement_loss_weight,
                "dataset_loss_weight": dataset_loss_weight,
                "femur_shaft_loss_weight": femur_shaft_loss_weight,
                "fugc_segment_loss_weight": fugc_segment_loss_weight,
                "ivc_band_loss_weight": ivc_band_loss_weight,
            },
        )
        encoder_name = str(profile_config["encoder_name"])
        input_size = int(profile_config["input_size"])
        use_fpn = bool(profile_config["use_fpn"])
        fpn_mode = str(profile_config["fpn_mode"])
        fpn_type = str(profile_config["fpn_type"])
        head_type = str(profile_config["head_type"])
        task_head_profile = str(profile_config["task_head_profile"])
        task_decoder_profile = str(profile_config["task_decoder_profile"])
        task_adapter_profile = str(profile_config["task_adapter_profile"])
        task_loss_family_profile = str(profile_config["task_loss_family_profile"])
        split_mode = str(profile_config["split_mode"])
        augmentation_profile = str(profile_config["augmentation_profile"])
        checkpoint_score_mode = str(profile_config["checkpoint_score_mode"])
        cardiac_split_screen_mode = str(profile_config["cardiac_split_screen_mode"])
        measurement_loss_weight = float(profile_config["measurement_loss_weight"])
        dataset_loss_weight = float(profile_config["dataset_loss_weight"])
        femur_shaft_loss_weight = float(profile_config["femur_shaft_loss_weight"])
        fugc_segment_loss_weight = float(profile_config["fugc_segment_loss_weight"])
        ivc_band_loss_weight = float(profile_config["ivc_band_loss_weight"])

    metric_column = "MRE (pixels)"
    metric_label_map = {
        "combined": CHECKPOINT_SCORE_NAME,
        "server_proxy_v1": "Server proxy score",
        "server_proxy_v2": "Server proxy score",
        "robust_domain_v1": "Robust domain score",
    }
    metric_label = metric_label_map.get(checkpoint_score_mode, CHECKPOINT_SCORE_NAME)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(output_dir)
    log_dir = os.path.join(output_dir, "log")
    metrics_dir = os.path.join(output_dir, "metrics")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, "tensorboard", run_id)
    model_save_path = os.path.join(checkpoints_dir, "best_model.pth")
    log_save_path = os.path.join(log_dir, f"training_{run_id}.log")
    metrics_save_path = os.path.join(metrics_dir, f"validation_metrics_{run_id}.csv")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(log_save_path)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training log saved to: {log_save_path}")
    logger.info(f"TensorBoard directory: {tensorboard_dir}")
    writer = SummaryWriter(log_dir=tensorboard_dir)

    set_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_checkpoint_meta = {}
    if init_checkpoint is not None:
        init_checkpoint = os.path.abspath(init_checkpoint)
        _, init_checkpoint_meta = _load_checkpoint_payload(init_checkpoint, device)
        if init_checkpoint_meta:
            if encoder_name == ENCODER and init_checkpoint_meta.get("encoder_name"):
                encoder_name = str(init_checkpoint_meta["encoder_name"])
            if head_type == HEAD_TYPE and init_checkpoint_meta.get("head_type"):
                head_type = str(init_checkpoint_meta["head_type"])
            if task_head_profile == TASK_HEAD_PROFILE and init_checkpoint_meta.get("task_head_profile"):
                task_head_profile = str(init_checkpoint_meta["task_head_profile"])
            elif task_head_profile == TASK_HEAD_PROFILE and "task_head_profile" not in init_checkpoint_meta:
                task_head_profile = "uniform"
            if (
                task_decoder_profile == TASK_DECODER_PROFILE
                and init_checkpoint_meta.get("task_decoder_profile")
            ):
                task_decoder_profile = str(init_checkpoint_meta["task_decoder_profile"])
            if (
                task_adapter_profile == TASK_ADAPTER_PROFILE
                and init_checkpoint_meta.get("task_adapter_profile")
            ):
                task_adapter_profile = str(init_checkpoint_meta["task_adapter_profile"])
            if (
                task_loss_family_profile == TASK_LOSS_FAMILY_PROFILE
                and init_checkpoint_meta.get("task_loss_family_profile")
            ):
                task_loss_family_profile = str(init_checkpoint_meta["task_loss_family_profile"])
            if input_size == INPUT_SIZE and init_checkpoint_meta.get("input_size"):
                input_size = int(init_checkpoint_meta["input_size"])
            if use_fpn == USE_FPN and "use_fpn" in init_checkpoint_meta:
                use_fpn = bool(init_checkpoint_meta["use_fpn"])
            if fpn_mode == FPN_MODE and init_checkpoint_meta.get("fpn_mode"):
                fpn_mode = str(init_checkpoint_meta["fpn_mode"])
            if fpn_type == FPN_TYPE and init_checkpoint_meta.get("fpn_type"):
                fpn_type = str(init_checkpoint_meta["fpn_type"])
        elif task_head_profile == TASK_HEAD_PROFILE:
            task_head_profile = "uniform"
    logger.info(f"Device used: {device}")
    logger.info(f"Model profile: {model_profile if model_profile is not None else 'manual'}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Encoder: {encoder_name}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Head type: {head_type}")
    logger.info(f"Task head profile: {task_head_profile}")
    logger.info(f"Task decoder profile: {task_decoder_profile}")
    logger.info(f"Task adapter profile: {task_adapter_profile}")
    logger.info(f"Task loss family profile: {task_loss_family_profile}")
    logger.info(f"FPN mode: {fpn_mode}")
    logger.info(f"FPN type: {fpn_type}")
    logger.info(f"Learning rate: {learning_rate:.8f}")
    logger.info(f"Init checkpoint: {init_checkpoint if init_checkpoint else 'none'}")
    logger.info(f"Train task IDs: {train_task_ids if train_task_ids else 'all'}")
    logger.info(f"Measurement loss weight: {measurement_loss_weight:.6f}")
    logger.info(f"Dataset-specific loss weight: {dataset_loss_weight:.6f}")
    logger.info(f"Femur shaft loss weight: {femur_shaft_loss_weight:.6f}")
    logger.info(f"FUGC segment loss weight: {fugc_segment_loss_weight:.6f}")
    logger.info(f"IVC band loss weight: {ivc_band_loss_weight:.6f}")
    logger.info(f"EMA: {'ENABLED' if use_ema else 'DISABLED'}")
    logger.info(f"EMA decay: {ema_decay:.6f}")
    logger.info(f"Train ROI crop: {'ENABLED' if train_roi_crop else 'DISABLED'}")
    logger.info(f"ROI crop tasks: {sorted(roi_crop_tasks) if roi_crop_tasks else 'all-enabled-tasks'}")
    logger.info(f"ROI context range: ({roi_context_min:.3f}, {roi_context_max:.3f})")
    logger.info(f"Grad accumulation steps: {grad_accum_steps}")
    logger.info(f"AMP: {'ENABLED' if (use_amp and device.type == 'cuda') else 'DISABLED'}")
    logger.info(f"Split mode: {split_mode}")
    logger.info(f"Augmentation profile: {augmentation_profile}")
    logger.info(f"Checkpoint score mode: {checkpoint_score_mode}")
    logger.info(f"Cardiac split-screen mode: {cardiac_split_screen_mode}")
    logger.info(f"Cardiac split-screen vertical-darkness threshold: {cardiac_split_screen_vdark_threshold:.6f}")
    logger.info(f"Heatmap size: {HEATMAP_SIZE}")
    effective_task_loss_weights = dict(TASK_LOSS_WEIGHTS)
    if task_loss_weight_overrides:
        effective_task_loss_weights.update(task_loss_weight_overrides)
    effective_checkpoint_task_weights = dict(CHECKPOINT_TASK_WEIGHTS)
    if checkpoint_task_weight_overrides:
        effective_checkpoint_task_weights.update(checkpoint_task_weight_overrides)
    logger.info(f"Task loss weights: {effective_task_loss_weights}")
    logger.info(f"Checkpoint task weights: {effective_checkpoint_task_weights}")
    logger.info(f"Sampler task weights override: {sampler_task_weight_overrides or {}}")

    logger.info(
        f"FPN neck: {'ENABLED' if use_fpn else 'DISABLED'} "
        f"(mode={fpn_mode}, type={fpn_type}, set USE_FPN={use_fpn} at top of train.py)"
    )

    train_transforms = _build_train_transforms(augmentation_profile)

    val_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    temp_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        input_size=input_size,
    )
    temp_dataset.dataframe = _assign_cardiac_split_screen_flags(
        temp_dataset.dataframe,
        DATA_ROOT_PATH,
        vdark_threshold=cardiac_split_screen_vdark_threshold,
    )
    split_screen_counts = (
        temp_dataset.dataframe[temp_dataset.dataframe["is_split_screen_cardiac"]]
        .groupby("task_id")
        .size()
        .to_dict()
    )
    logger.info(
        "Detected split-screen cardiac training rows: "
        f"{int(temp_dataset.dataframe['is_split_screen_cardiac'].sum())} total, by task={split_screen_counts}"
    )
    if cardiac_split_screen_mode == "exclude":
        before_count = len(temp_dataset.dataframe)
        temp_dataset.dataframe = temp_dataset.dataframe[
            ~temp_dataset.dataframe["is_split_screen_cardiac"].astype(bool)
        ].reset_index(drop=True)
        removed_count = before_count - len(temp_dataset.dataframe)
        logger.info(f"Excluded split-screen cardiac rows before split: {removed_count}")
    elif cardiac_split_screen_mode != "keep":
        raise ValueError(f"Unsupported cardiac_split_screen_mode: {cardiac_split_screen_mode}")
    temp_dataset.dataframe = _assign_pseudo_domains(temp_dataset.dataframe, DATA_ROOT_PATH)

    task_configs = _build_task_configs(temp_dataset.dataframe)
    task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in task_configs}
    effective_sampler_task_weights = {cfg["task_id"]: 1.0 for cfg in task_configs}
    if sampler_task_weight_overrides:
        effective_sampler_task_weights.update(sampler_task_weight_overrides)

    if split_mode == "row":
        train_indices, val_indices = _stratified_split_indices(
            temp_dataset.dataframe,
            val_split=val_split,
            seed=random_seed,
        )
    elif split_mode == "grouped":
        train_indices, val_indices = _grouped_stratified_split_indices(
            temp_dataset.dataframe,
            val_split=val_split,
            seed=random_seed,
        )
    elif split_mode == "pseudo_domain_grouped":
        train_indices, val_indices = _pseudo_domain_grouped_split_indices(
            temp_dataset.dataframe,
            val_split=val_split,
            seed=random_seed,
        )
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    train_size = len(train_indices)
    val_size = len(val_indices)
    normalization_stats = compute_normalization_stats_from_dataframe(
        temp_dataset.dataframe.iloc[train_indices].reset_index(drop=True)
    )

    train_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        input_size=input_size,
        roi_crop=train_roi_crop,
        roi_crop_tasks=roi_crop_tasks,
        roi_context_range=(roi_context_min, roi_context_max),
    )
    train_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)
    if train_task_ids:
        allowed_train_mask = train_dataset.dataframe["task_id"].astype(str).isin(set(train_task_ids))
        allowed_train_indices = set(train_dataset.dataframe.index[allowed_train_mask].tolist())
    else:
        allowed_train_indices = set(train_dataset.dataframe.index.tolist())
    train_indices = [idx for idx in train_indices if idx in allowed_train_indices]
    if not train_indices:
        raise ValueError(
            "No training samples remain after applying train task filter. "
            f"Requested task IDs: {train_task_ids}"
        )

    val_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=val_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        input_size=input_size,
        roi_crop=False,
    )
    val_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    logger.info(
        f"Dataset split ({split_mode}, val_split={val_split:.6f}): "
        f"{train_size} training samples, {val_size} validation samples"
    )

    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)

    train_sampler = KeypointUniformSampler(
        train_subset,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        task_sampling_weights=effective_sampler_task_weights,
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=keypoint_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=keypoint_collate_fn,
    )

    model = MultiTaskModelFactory(
        encoder_name=encoder_name,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=task_configs,
        heatmap_size=HEATMAP_SIZE,
        use_fpn=use_fpn,
        fpn_mode=fpn_mode,
        fpn_type=fpn_type,
        head_type=head_type,
        task_head_profile=task_head_profile,
        task_decoder_profile=task_decoder_profile,
        task_adapter_profile=task_adapter_profile,
    ).to(device)
    if init_checkpoint is not None:
        state_dict, checkpoint_meta = _load_checkpoint_payload(init_checkpoint, device)
        load_result, skipped_keys, missing_after_match = _load_matching_state_dict(model, state_dict)
        logger.info(f"Loaded initialization checkpoint with compatible-key warm start: {init_checkpoint}")
        logger.info(
            f"Warm-start summary: matched={len(model.state_dict()) - len(missing_after_match)}, "
            f"missing={len(missing_after_match)}, skipped={len(skipped_keys)}"
        )
        if skipped_keys:
            preview = ", ".join(f"{name} [{reason}]" for name, reason in skipped_keys[:12])
            logger.info(f"Skipped checkpoint keys (first 12): {preview}")
        if load_result.missing_keys:
            logger.info(f"Missing model keys after warm start (first 12): {load_result.missing_keys[:12]}")
        if load_result.unexpected_keys:
            logger.info(f"Unexpected checkpoint keys after warm start (first 12): {load_result.unexpected_keys[:12]}")
        if checkpoint_meta:
            logger.info(f"Init checkpoint meta: {checkpoint_meta}")

    param_groups = [{"params": model.encoder.parameters(), "lr": learning_rate * 0.2}]
    if model.fpn is not None:
        param_groups.append({"params": model.fpn.parameters(), "lr": learning_rate * 2.0})
    if getattr(model, "task_fpns", None) is not None:
        for task_id, task_fpn in model.task_fpns.items():
            param_groups.append({"params": task_fpn.parameters(), "lr": learning_rate * 2.0})
    if getattr(model, "soft_adapters", None) is not None:
        param_groups.append({"params": model.soft_adapters.parameters(), "lr": learning_rate * 5.0})
    if getattr(model, "local_refine_adapters", None) is not None:
        param_groups.append({"params": model.local_refine_adapters.parameters(), "lr": learning_rate * 5.0})
    if getattr(model, "context_adapters", None) is not None:
        param_groups.append({"params": model.context_adapters.parameters(), "lr": learning_rate * 5.0})
    if getattr(model, "task_film_adapters", None) is not None:
        param_groups.append({"params": model.task_film_adapters.parameters(), "lr": learning_rate * 5.0})
    for task_id, head in model.heads.items():
        param_groups.append({"params": head.parameters(), "lr": learning_rate * 10.0})

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    ema = ModelEma(model, decay=ema_decay) if use_ema else None
    scaler = GradScaler(enabled=bool(use_amp and device.type == "cuda"))

    best_val_score = float("inf")
    epochs_without_improvement = 0
    best_val_results_df = None
    logger.info(f"Best-checkpoint metric: {metric_label} (lower is better)")
    if early_stopping_patience is None:
        logger.info("Early stopping: DISABLED")
    else:
        logger.info(
            "Early stopping: ENABLED "
            f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta:.6f})"
        )
    logger.info("=" * 50)
    logger.info("--- Start Keypoint Training ---")

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_losses = defaultdict(list)
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

            for batch in loop:
                images = batch["image"].to(device)
                task_ids = batch["task_id"]
                batch_loss_values = []

                # Handle mixed-task batches safely (different tasks can have different keypoint counts).
                for current_task_id in sorted(set(task_ids)):
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == current_task_id]
                    task_images = images[task_indices]
                    task_heatmaps = torch.stack([batch["heatmap"][i] for i in task_indices], 0).to(device)

                    # Forward pass (FPN applied inside model if enabled)
                    with autocast(enabled=bool(use_amp and device.type == "cuda")):
                        model_output = model(task_images, task_id=current_task_id, return_prior=True)
                        aux_outputs = {}
                        if len(model_output) == 3:
                            pred_logits, pred_heatmaps, aux_outputs = model_output
                        else:
                            pred_logits, pred_heatmaps = model_output
                        target_coords = torch.stack([batch["train_label"][i] for i in task_indices], 0).to(device)
                        pred_coords_transformed = softargmax_heatmaps_to_transformed_coords(pred_logits)
                        refined_coords_transformed = aux_outputs.get("refined_coords_transformed")
                        target_coords_transformed = target_coords.clone()
                        coord_loss = F.l1_loss(pred_coords_transformed, target_coords_transformed)
                        if refined_coords_transformed is not None:
                            refined_coord_loss = F.l1_loss(refined_coords_transformed, target_coords_transformed)
                            coord_loss = 0.4 * coord_loss + 0.6 * refined_coord_loss
                        heatmap_loss = F.mse_loss(pred_heatmaps, task_heatmaps)
                        pred_coords_original = transformed_coords_to_original_normalized(
                            refined_coords_transformed if refined_coords_transformed is not None else pred_coords_transformed,
                            [batch["meta"][i] for i in task_indices],
                        ).to(device)
                        target_coords_original = torch.stack([batch["label"][i] for i in task_indices], 0).to(device)
                        measurement_loss = compute_measurement_loss(
                            pred_coords_original,
                            target_coords_original,
                            [batch["meta"][i] for i in task_indices],
                            current_task_id,
                        )
                        dataset_specific_loss = compute_dataset_specific_loss(
                            pred_coords_original,
                            target_coords_original,
                            [batch["meta"][i] for i in task_indices],
                            current_task_id,
                            profile=task_loss_family_profile,
                        )
                        femur_shaft_loss = pred_logits.new_tensor(0.0)
                        if current_task_id == "fetal_femur":
                            femur_shaft_loss = compute_femur_shaft_loss(
                                aux_outputs.get("shaft_logits"),
                                target_coords_transformed,
                                heatmap_size=HEATMAP_SIZE,
                            )
                        fugc_segment_loss = pred_logits.new_tensor(0.0)
                        if current_task_id == "FUGC":
                            fugc_segment_loss = compute_fugc_segment_loss(
                                aux_outputs.get("segment_logits"),
                                target_coords_transformed,
                                heatmap_size=HEATMAP_SIZE,
                            )
                        ivc_band_loss = pred_logits.new_tensor(0.0)
                        if current_task_id == "IVC":
                            ivc_band_loss = compute_ivc_band_loss(
                                aux_outputs.get("band_logits"),
                                target_coords_transformed,
                                heatmap_size=HEATMAP_SIZE,
                            )
                        base_loss = (
                            heatmap_loss
                            + 0.2 * coord_loss
                            + measurement_loss_weight * measurement_loss
                            + dataset_loss_weight * dataset_specific_loss
                            + femur_shaft_loss_weight * femur_shaft_loss
                            + fugc_segment_loss_weight * fugc_segment_loss
                            + ivc_band_loss_weight * ivc_band_loss
                        )
                        task_weight = float(effective_task_loss_weights.get(current_task_id, 1.0))
                        loss = base_loss * task_weight

                    scaled_loss = loss / float(max(grad_accum_steps, 1))
                    scaler.scale(scaled_loss).backward()
                    accumulation_counter += 1
                    if accumulation_counter >= max(grad_accum_steps, 1):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accumulation_counter = 0
                        if ema is not None:
                            ema.update(model)

                    loss_value = float(loss.item())
                    batch_loss_values.append(loss_value)
                    epoch_train_losses[current_task_id].append(loss_value)

                if accumulation_counter >= max(grad_accum_steps, 1):
                    accumulation_counter = 0

                mean_batch_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
                loop.set_postfix(
                    loss=f"{mean_batch_loss:.6f}",
                    groups=len(set(task_ids)),
                )

            if accumulation_counter > 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            logger.info(f"\n--- Epoch {epoch + 1} Average Train Loss ---")
            train_epoch_mean = float(np.mean([v for values in epoch_train_losses.values() for v in values]))
            writer.add_scalar("train/loss_epoch_mean", train_epoch_mean, epoch + 1)
            writer.add_scalar("train/lr_encoder", float(optimizer.param_groups[0]["lr"]), epoch + 1)
            for task_id in sorted(epoch_train_losses.keys()):
                avg_loss = float(np.mean(epoch_train_losses[task_id]))
                logger.info(f"  - {task_id}: {avg_loss:.6f}")
                writer.add_scalar(f"train/task_loss/{task_id}", avg_loss, epoch + 1)

            restore_state = None
            if ema is not None:
                restore_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                ema.copy_to(model)
            if checkpoint_score_mode == "robust_domain_v1":
                val_results_df, val_case_df = evaluate_keypoint(
                    model,
                    val_loader,
                    device,
                    task_id_to_name,
                    normalization_stats=normalization_stats,
                    return_case_df=True,
                )
            else:
                val_results_df = evaluate_keypoint(
                    model,
                    val_loader,
                    device,
                    task_id_to_name,
                    normalization_stats=normalization_stats,
                )
                val_case_df = None
            if restore_state is not None:
                model.load_state_dict(restore_state, strict=True)
            proxy_breakdown = None
            robust_domain_breakdown = None
            if checkpoint_score_mode == "server_proxy_v1":
                proxy_breakdown = compute_server_proxy_breakdown(
                    val_results_df,
                    normalization_stats=normalization_stats,
                    task_weights=effective_checkpoint_task_weights,
                    hard_task_ids=SERVER_PROXY_HARD_TASK_IDS,
                )
                selected_val_score = float(proxy_breakdown["proxy_score"])
            elif checkpoint_score_mode == "server_proxy_v2":
                proxy_breakdown = compute_server_proxy_breakdown(
                    val_results_df,
                    normalization_stats=normalization_stats,
                    task_weights=effective_checkpoint_task_weights,
                    hard_task_ids=tuple(SERVER_PROXY_V2_HARD_TASK_WEIGHT_OVERRIDES.keys()),
                    hard_task_weight_overrides=SERVER_PROXY_V2_HARD_TASK_WEIGHT_OVERRIDES,
                    base_weight=0.45,
                    hard_task_weight=0.35,
                    worst_task_weight=0.20,
                    worst_k=3,
                )
                selected_val_score = float(proxy_breakdown["proxy_score"])
            elif checkpoint_score_mode == "robust_domain_v1":
                robust_domain_breakdown = compute_robust_domain_breakdown(
                    val_results_df,
                    val_case_df,
                    normalization_stats=normalization_stats,
                    task_weights=effective_checkpoint_task_weights,
                )
                selected_val_score = float(robust_domain_breakdown["robust_score"])
            else:
                selected_val_score = compute_combined_score(
                    val_results_df,
                    normalization_stats=normalization_stats,
                    task_weights=effective_checkpoint_task_weights,
                )
            selected_val_mre = float("inf")
            selected_val_measurement = float("inf")
            selected_val_normalized_mre = float("inf")
            selected_val_normalized_measurement = float("inf")
            if not val_results_df.empty and metric_column in val_results_df.columns:
                selected_val_mre = float(val_results_df[metric_column].mean())
            if not val_results_df.empty and "Normalized MRE" in val_results_df.columns:
                selected_val_normalized_mre = float(val_results_df["Normalized MRE"].mean())
            if not val_results_df.empty and "Measurement MAE (pixels)" in val_results_df.columns:
                measurement_values = val_results_df["Measurement MAE (pixels)"].dropna()
                if len(measurement_values) > 0:
                    selected_val_measurement = float(measurement_values.mean())
            if not val_results_df.empty and "Normalized Measurement MAE" in val_results_df.columns:
                measurement_values = val_results_df["Normalized Measurement MAE"].dropna()
                if len(measurement_values) > 0:
                    selected_val_normalized_measurement = float(measurement_values.mean())

            logger.info(f"\n--- Epoch {epoch + 1} Validation Report ---")
            if not val_results_df.empty:
                logger.info(val_results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
                for _, row in val_results_df.iterrows():
                    writer.add_scalar(
                        f"val/{metric_column}/{row['Task ID']}",
                        float(row[metric_column]),
                        epoch + 1,
                    )
                    if "Normalized MRE" in row and not np.isnan(row["Normalized MRE"]):
                        writer.add_scalar(
                            f"val/normalized_mre/{row['Task ID']}",
                            float(row["Normalized MRE"]),
                            epoch + 1,
                        )
                    if "Measurement MAE (pixels)" in row and not np.isnan(row["Measurement MAE (pixels)"]):
                        writer.add_scalar(
                            f"val/measurement_mae_pixels/{row['Task ID']}",
                            float(row["Measurement MAE (pixels)"]),
                            epoch + 1,
                        )
                    if "Normalized Measurement MAE" in row and not np.isnan(row["Normalized Measurement MAE"]):
                        writer.add_scalar(
                            f"val/normalized_measurement_mae/{row['Task ID']}",
                            float(row["Normalized Measurement MAE"]),
                            epoch + 1,
                        )
            writer.add_scalar("val/mre_pixels_mean", selected_val_mre, epoch + 1)
            writer.add_scalar("val/combined_score", selected_val_score, epoch + 1)
            if np.isfinite(selected_val_normalized_mre):
                writer.add_scalar("val/normalized_mre_mean", selected_val_normalized_mre, epoch + 1)
            if np.isfinite(selected_val_normalized_measurement):
                writer.add_scalar(
                    "val/normalized_measurement_mae_mean",
                    selected_val_normalized_measurement,
                    epoch + 1,
                )
            if np.isfinite(selected_val_mre):
                writer.add_scalar("val/mre_pixels_mean_only", selected_val_mre, epoch + 1)
            if np.isfinite(selected_val_measurement):
                writer.add_scalar("val/measurement_mae_pixels_mean", selected_val_measurement, epoch + 1)
            logger.info(f"--- Average Val MRE (pixels): {selected_val_mre:.6f} ---")
            if np.isfinite(selected_val_measurement):
                logger.info(f"--- Average Val Measurement MAE (pixels): {selected_val_measurement:.6f} ---")
            if np.isfinite(selected_val_normalized_mre):
                logger.info(f"--- Average Val Normalized MRE: {selected_val_normalized_mre:.6f} ---")
            if np.isfinite(selected_val_normalized_measurement):
                logger.info(
                    f"--- Average Val Normalized Measurement MAE: {selected_val_normalized_measurement:.6f} ---"
                )
            if proxy_breakdown is not None:
                logger.info(
                    "--- Server proxy breakdown: "
                    f"base={proxy_breakdown['base_score']:.6f}, "
                    f"hard={proxy_breakdown['hard_task_score']:.6f}, "
                    f"worst={proxy_breakdown['worst_task_score']:.6f} ---"
                )
            if robust_domain_breakdown is not None:
                logger.info(
                    "--- Robust domain breakdown: "
                    f"base={robust_domain_breakdown['base_score']:.6f}, "
                    f"mean_domain={robust_domain_breakdown['mean_domain_score']:.6f}, "
                    f"worst_domain={robust_domain_breakdown['worst_domain_score']:.6f} ---"
                )
            logger.info(f"--- Average Val {metric_label} (Lower is better): {selected_val_score:.6f} ---")

            improved = selected_val_score < (best_val_score - early_stopping_min_delta)
            if improved:
                best_val_score = selected_val_score
                epochs_without_improvement = 0
                best_val_results_df = val_results_df.copy()
                checkpoint_payload = {
                    "state_dict": ema.shadow if ema is not None else model.state_dict(),
                        "meta": {
                            "encoder_name": encoder_name,
                            "use_fpn": use_fpn,
                            "fpn_mode": fpn_mode,
                            "fpn_type": fpn_type,
                            "head_type": head_type,
                            "task_head_profile": task_head_profile,
                            "task_decoder_profile": task_decoder_profile,
                            "task_adapter_profile": task_adapter_profile,
                            "task_loss_family_profile": task_loss_family_profile,
                            "input_size": input_size,
                            "heatmap_size": list(HEATMAP_SIZE),
                            "checkpoint_metric": metric_label,
                            "measurement_loss_weight": measurement_loss_weight,
                            "dataset_loss_weight": dataset_loss_weight,
                            "femur_shaft_loss_weight": femur_shaft_loss_weight,
                            "fugc_segment_loss_weight": fugc_segment_loss_weight,
                            "ivc_band_loss_weight": ivc_band_loss_weight,
                            "use_ema": use_ema,
                            "ema_decay": ema_decay,
                            "task_loss_weight_overrides": task_loss_weight_overrides or {},
                            "sampler_task_weight_overrides": sampler_task_weight_overrides or {},
                            "normalization_scheme": "train_iqr_proxy",
                            "train_roi_crop": train_roi_crop,
                            "roi_crop_tasks": sorted(roi_crop_tasks) if roi_crop_tasks else [],
                            "roi_context_range": [roi_context_min, roi_context_max],
                            "split_mode": split_mode,
                            "augmentation_profile": augmentation_profile,
                            "checkpoint_task_weight_overrides": checkpoint_task_weight_overrides or {},
                            "checkpoint_score_mode": checkpoint_score_mode,
                            "cardiac_split_screen_mode": cardiac_split_screen_mode,
                            "cardiac_split_screen_vdark_threshold": cardiac_split_screen_vdark_threshold,
                            "random_seed": random_seed,
                            "server_proxy_hard_task_ids": list(SERVER_PROXY_HARD_TASK_IDS),
                            "server_proxy_v2_hard_task_weight_overrides": SERVER_PROXY_V2_HARD_TASK_WEIGHT_OVERRIDES,
                        },
                    }
                torch.save(checkpoint_payload, model_save_path)
                logger.info(f"-> New best model saved! {metric_label} improved to: {best_val_score:.6f}")
            else:
                epochs_without_improvement += 1
                if early_stopping_patience is not None:
                    logger.info(
                        "-> No significant improvement. "
                        f"Early stopping counter: {epochs_without_improvement}/{early_stopping_patience}"
                    )

            scheduler.step()

            if (
                early_stopping_patience is not None
                and epochs_without_improvement > early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered after "
                    f"{epoch + 1} epochs. Best {metric_label}: {best_val_score:.6f}"
                )
                break

        if best_val_results_df is not None and not best_val_results_df.empty:
            best_val_results_df.to_csv(metrics_save_path, index=False)
            logger.info(f"Best validation metrics saved at: {metrics_save_path}")

        logger.info(f"\n--- Training Finished ---")
        logger.info(f"Best model saved at: {model_save_path}")
        logger.info(f"Training log saved at: {log_save_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint training script")
    parser.add_argument(
        "--model-profile",
        type=str,
        choices=MODEL_PROFILE_NAMES,
        default=None,
        help="Named architecture/training recipe preset. When provided, it overrides the matching model-shape and split settings.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help="Validation ratio per task (0~1), e.g. 0.2 means each task uses 20%% for validation.",
    )
    parser.add_argument(
        "--fpn",
        action="store_true",
        default=USE_FPN,
        help=f"Enable FPN neck (default: {USE_FPN}).",
    )
    parser.add_argument(
        "--fpn-mode",
        type=str,
        choices=("shared", "task_specific"),
        default=FPN_MODE,
        help=f"FPN layout when enabled (default: {FPN_MODE}).",
    )
    parser.add_argument(
        "--fpn-type",
        type=str,
        choices=("fpn", "bifpn"),
        default=FPN_TYPE,
        help=f"FPN neck type when enabled (default: {FPN_TYPE}).",
    )
    parser.add_argument(
        "--no-fpn",
        action="store_true",
        help="Force disable FPN neck, overriding default.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size per sampled task batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional override for the number of sampled training batches per epoch.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many consecutive non-improving epochs. Omit to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum decrease in validation metric required to reset early stopping.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to store logs, checkpoints, and metrics (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default=ENCODER,
        help=f"Backbone model name (default: {ENCODER}).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=INPUT_SIZE,
        help=f"Square letterbox input size (default: {INPUT_SIZE}).",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        choices=("basic", "deep"),
        default=HEAD_TYPE,
        help=f"Decoder head type (default: {HEAD_TYPE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Base learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for initialization, split generation, and sampling (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to load before training/fine-tuning.",
    )
    parser.add_argument(
        "--train-task-ids",
        type=str,
        default=None,
        help="Optional comma-separated task IDs to train on, e.g. HC,PLAX,A4C,fetal_femur.",
    )
    parser.add_argument(
        "--task-head-profile",
        type=str,
        choices=("uniform", "challenge_legacy_v1", "challenge_v1"),
        default=TASK_HEAD_PROFILE,
        help=f"Task-specific head sizing profile (default: {TASK_HEAD_PROFILE}).",
    )
    parser.add_argument(
        "--task-decoder-profile",
        type=str,
        choices=("uniform", "cardiac_graph_v1", "coarse_refine_v1", "ivc_refine_v1", "ivc_refine_v2", "fugc_refine_v1", "hc_refine_v1", "geometry_v1", "weak_tasks_v1", "dedicated_legacy_v1", "dedicated_v1"),
        default=TASK_DECODER_PROFILE,
        help=f"Task-specific decoder family profile (default: {TASK_DECODER_PROFILE}).",
    )
    parser.add_argument(
        "--task-adapter-profile",
        type=str,
        choices=("uniform", "softsharing_v1", "localrefine_v1", "coarse_refine_v1", "taskfilm_v1"),
        default=TASK_ADAPTER_PROFILE,
        help=f"Task-specific feature adapter profile (default: {TASK_ADAPTER_PROFILE}).",
    )
    parser.add_argument(
        "--task-loss-family-profile",
        type=str,
        choices=("uniform", "dataset_v1", "weak_tasks_v1"),
        default=TASK_LOSS_FAMILY_PROFILE,
        help=f"Dataset-family auxiliary loss profile (default: {TASK_LOSS_FAMILY_PROFILE}).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=("row", "grouped", "pseudo_domain_grouped"),
        default=SPLIT_MODE,
        help=f"Validation split strategy (default: {SPLIT_MODE}).",
    )
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        choices=("baseline", "strong_ultrasound_v1", "ultrasound_robust_v1", "ultrasound_domain_shift_v1", "ultrasound_domain_shift_v2", "ultrasound_mixed_v1"),
        default=AUGMENTATION_PROFILE,
        help=f"Training augmentation profile (default: {AUGMENTATION_PROFILE}).",
    )
    parser.add_argument(
        "--checkpoint-score-mode",
        type=str,
        choices=("combined", "server_proxy_v1", "server_proxy_v2", "robust_domain_v1"),
        default=CHECKPOINT_SCORE_MODE,
        help=f"Validation score used for best-checkpoint selection (default: {CHECKPOINT_SCORE_MODE}).",
    )
    parser.add_argument(
        "--cardiac-split-screen-mode",
        type=str,
        choices=("keep", "exclude"),
        default=CARDIAC_SPLIT_SCREEN_MODE,
        help=f"How to handle detected split-screen cardiac training rows (default: {CARDIAC_SPLIT_SCREEN_MODE}).",
    )
    parser.add_argument(
        "--cardiac-split-screen-vdark-threshold",
        type=float,
        default=CARDIAC_SPLIT_SCREEN_VDARK_THRESHOLD,
        help="Vertical center-strip darkness threshold used to detect split-screen cardiac images.",
    )
    parser.add_argument(
        "--fugc-segment-loss-weight",
        type=float,
        default=FUGC_SEGMENT_LOSS_WEIGHT,
        help="Auxiliary short-segment mask loss for FUGC. Use 0.0 to disable.",
    )
    parser.add_argument(
        "--ivc-band-loss-weight",
        type=float,
        default=IVC_BAND_LOSS_WEIGHT,
        help="Auxiliary diameter-band mask loss for IVC. Use 0.0 to disable.",
    )
    parser.add_argument(
        "--measurement-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary measurement loss weight. Use 0.0 for the task-weighted baseline.",
    )
    parser.add_argument(
        "--dataset-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary dataset-family loss weight. Use 0.0 to disable.",
    )
    parser.add_argument(
        "--femur-shaft-loss-weight",
        type=float,
        default=FEMUR_SHAFT_LOSS_WEIGHT,
        help="Auxiliary shaft mask loss for fetal_femur. Use 0.0 to disable.",
    )
    parser.add_argument(
        "--task-loss-weights",
        type=str,
        default=None,
        help="Optional comma-separated TASK=VALUE loss-weight overrides.",
    )
    parser.add_argument(
        "--checkpoint-task-weights",
        type=str,
        default=None,
        help="Optional comma-separated TASK=VALUE checkpoint-selection weight overrides.",
    )
    parser.add_argument(
        "--sampler-task-weights",
        type=str,
        default=None,
        help="Optional comma-separated TASK=VALUE sampler-weight overrides.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay for validation/checkpoint model.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA validation/checkpoint smoothing.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many sampled batches before optimizer step.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP mixed-precision training.",
    )
    parser.add_argument(
        "--train-roi-crop",
        action="store_true",
        help="Enable GT-centered ROI crop augmentation for selected training tasks.",
    )
    parser.add_argument(
        "--roi-crop-tasks",
        type=str,
        default=",".join(ROI_CROP_TASKS),
        help="Comma-separated task IDs for ROI crop training augmentation.",
    )
    parser.add_argument(
        "--roi-context-min",
        type=float,
        default=ROI_CONTEXT_RANGE[0],
        help="Minimum context multiplier for ROI crop box.",
    )
    parser.add_argument(
        "--roi-context-max",
        type=float,
        default=ROI_CONTEXT_RANGE[1],
        help="Maximum context multiplier for ROI crop box.",
    )
    args = parser.parse_args()

    # --no-fpn takes precedence over --fpn
    use_fpn = USE_FPN
    if args.no_fpn:
        use_fpn = False
    elif args.fpn:
        use_fpn = True

    main(
        val_split=float(args.val_split),
        model_profile=args.model_profile,
        random_seed=int(args.seed),
        use_fpn=use_fpn,
        fpn_mode=str(args.fpn_mode),
        fpn_type=str(args.fpn_type),
        num_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        steps_per_epoch=args.steps_per_epoch,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        output_dir=str(args.output_dir),
        encoder_name=str(args.encoder_name),
        input_size=int(args.input_size),
        head_type=str(args.head_type),
        task_head_profile=str(args.task_head_profile),
        task_decoder_profile=str(args.task_decoder_profile),
        task_adapter_profile=str(args.task_adapter_profile),
        task_loss_family_profile=str(args.task_loss_family_profile),
        split_mode=str(args.split_mode),
        augmentation_profile=str(args.augmentation_profile),
        checkpoint_score_mode=str(args.checkpoint_score_mode),
        cardiac_split_screen_mode=str(args.cardiac_split_screen_mode),
        cardiac_split_screen_vdark_threshold=float(args.cardiac_split_screen_vdark_threshold),
        learning_rate=float(args.learning_rate),
        init_checkpoint=args.init_checkpoint,
        train_task_ids=_parse_task_id_csv(args.train_task_ids),
        measurement_loss_weight=float(args.measurement_loss_weight),
        dataset_loss_weight=float(args.dataset_loss_weight),
        femur_shaft_loss_weight=float(args.femur_shaft_loss_weight),
        fugc_segment_loss_weight=float(args.fugc_segment_loss_weight),
        ivc_band_loss_weight=float(args.ivc_band_loss_weight),
        task_loss_weight_overrides=_parse_weight_csv(args.task_loss_weights),
        checkpoint_task_weight_overrides=_parse_weight_csv(args.checkpoint_task_weights),
        sampler_task_weight_overrides=_parse_weight_csv(args.sampler_task_weights),
        use_ema=not bool(args.no_ema),
        ema_decay=float(args.ema_decay),
        grad_accum_steps=int(args.grad_accum_steps),
        use_amp=not bool(args.no_amp),
        train_roi_crop=bool(args.train_roi_crop),
        roi_crop_tasks=_parse_task_id_set_csv(args.roi_crop_tasks),
        roi_context_min=float(args.roi_context_min),
        roi_context_max=float(args.roi_context_max),
    )
