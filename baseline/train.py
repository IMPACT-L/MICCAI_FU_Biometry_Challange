import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import KeypointDataset, KeypointUniformSampler
from model_factory import MultiTaskModelFactory
from utils import (
    build_line_mask_from_transformed_coords,
    compute_combined_score,
    compute_dataset_specific_loss,
    compute_femur_shaft_loss,
    compute_fugc_segment_loss,
    compute_normalization_stats_from_dataframe,
    compute_measurement_loss,
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
INPUT_SIZE = 512
EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
USE_FPN = True  # ← 开关：设为 True 启用 FPN 特征金字塔
FPN_MODE = "shared"
HEAD_TYPE = "deep"
TASK_HEAD_PROFILE = "challenge_v1"
TASK_DECODER_PROFILE = "uniform"
TASK_LOSS_FAMILY_PROFILE = "uniform"
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
FEMUR_SHAFT_LOSS_WEIGHT = 0.15
FUGC_SEGMENT_LOSS_WEIGHT = 0.08
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
    use_fpn: bool = USE_FPN,
    fpn_mode: str = FPN_MODE,
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
    task_loss_family_profile: str = TASK_LOSS_FAMILY_PROFILE,
    learning_rate: float = LEARNING_RATE,
    init_checkpoint: str | None = None,
    train_task_ids: list[str] | None = None,
    measurement_loss_weight: float = 0.0,
    dataset_loss_weight: float = 0.0,
    femur_shaft_loss_weight: float = FEMUR_SHAFT_LOSS_WEIGHT,
    fugc_segment_loss_weight: float = FUGC_SEGMENT_LOSS_WEIGHT,
    task_loss_weight_overrides: dict[str, float] | None = None,
    sampler_task_weight_overrides: dict[str, float] | None = None,
    use_ema: bool = True,
    ema_decay: float = 0.999,
):
    metric_column = "MRE (pixels)"
    metric_label = CHECKPOINT_SCORE_NAME
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

    set_seed(RANDOM_SEED)
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
        elif task_head_profile == TASK_HEAD_PROFILE:
            task_head_profile = "uniform"
    logger.info(f"Device used: {device}")
    logger.info(f"Encoder: {encoder_name}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Head type: {head_type}")
    logger.info(f"Task head profile: {task_head_profile}")
    logger.info(f"Task decoder profile: {task_decoder_profile}")
    logger.info(f"Task loss family profile: {task_loss_family_profile}")
    logger.info(f"FPN mode: {fpn_mode}")
    logger.info(f"Learning rate: {learning_rate:.8f}")
    logger.info(f"Init checkpoint: {init_checkpoint if init_checkpoint else 'none'}")
    logger.info(f"Train task IDs: {train_task_ids if train_task_ids else 'all'}")
    logger.info(f"Measurement loss weight: {measurement_loss_weight:.6f}")
    logger.info(f"Dataset-specific loss weight: {dataset_loss_weight:.6f}")
    logger.info(f"Femur shaft loss weight: {femur_shaft_loss_weight:.6f}")
    logger.info(f"FUGC segment loss weight: {fugc_segment_loss_weight:.6f}")
    logger.info(f"EMA: {'ENABLED' if use_ema else 'DISABLED'}")
    logger.info(f"EMA decay: {ema_decay:.6f}")
    logger.info(f"Heatmap size: {HEATMAP_SIZE}")
    effective_task_loss_weights = dict(TASK_LOSS_WEIGHTS)
    if task_loss_weight_overrides:
        effective_task_loss_weights.update(task_loss_weight_overrides)
    logger.info(f"Task loss weights: {effective_task_loss_weights}")
    logger.info(f"Sampler task weights override: {sampler_task_weight_overrides or {}}")

    logger.info(
        f"FPN neck: {'ENABLED' if use_fpn else 'DISABLED'} "
        f"(mode={fpn_mode}, set USE_FPN={use_fpn} at top of train.py)"
    )

    train_transforms = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

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

    task_configs = _build_task_configs(temp_dataset.dataframe)
    task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in task_configs}
    effective_sampler_task_weights = {cfg["task_id"]: 1.0 for cfg in task_configs}
    if sampler_task_weight_overrides:
        effective_sampler_task_weights.update(sampler_task_weight_overrides)

    train_indices, val_indices = _stratified_split_indices(
        temp_dataset.dataframe,
        val_split=val_split,
        seed=RANDOM_SEED,
    )
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
    )
    val_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    logger.info(
        f"Dataset split (per-task stratified, val_split={val_split:.6f}): "
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
        head_type=head_type,
        task_head_profile=task_head_profile,
        task_decoder_profile=task_decoder_profile,
    ).to(device)
    if init_checkpoint is not None:
        state_dict, checkpoint_meta = _load_checkpoint_payload(init_checkpoint, device)
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded initialization checkpoint: {init_checkpoint}")
        if checkpoint_meta:
            logger.info(f"Init checkpoint meta: {checkpoint_meta}")

    param_groups = [{"params": model.encoder.parameters(), "lr": learning_rate * 0.2}]
    if model.fpn is not None:
        param_groups.append({"params": model.fpn.parameters(), "lr": learning_rate * 2.0})
    if getattr(model, "task_fpns", None) is not None:
        for task_id, task_fpn in model.task_fpns.items():
            param_groups.append({"params": task_fpn.parameters(), "lr": learning_rate * 2.0})
    for task_id, head in model.heads.items():
        param_groups.append({"params": head.parameters(), "lr": learning_rate * 10.0})

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    ema = ModelEma(model, decay=ema_decay) if use_ema else None

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
                    model_output = model(task_images, task_id=current_task_id, return_prior=True)
                    aux_outputs = {}
                    if len(model_output) == 3:
                        pred_logits, pred_heatmaps, aux_outputs = model_output
                    else:
                        pred_logits, pred_heatmaps = model_output
                    target_coords = torch.stack([batch["train_label"][i] for i in task_indices], 0).to(device)
                    pred_coords_transformed = softargmax_heatmaps_to_transformed_coords(pred_logits)
                    target_coords_transformed = target_coords.clone()
                    coord_loss = F.l1_loss(pred_coords_transformed, target_coords_transformed)
                    heatmap_loss = F.mse_loss(pred_heatmaps, task_heatmaps)
                    pred_coords_original = transformed_coords_to_original_normalized(
                        pred_coords_transformed,
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
                    base_loss = (
                        heatmap_loss
                        + 0.2 * coord_loss
                        + measurement_loss_weight * measurement_loss
                        + dataset_loss_weight * dataset_specific_loss
                        + femur_shaft_loss_weight * femur_shaft_loss
                        + fugc_segment_loss_weight * fugc_segment_loss
                    )
                    task_weight = float(effective_task_loss_weights.get(current_task_id, 1.0))
                    loss = base_loss * task_weight

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if ema is not None:
                        ema.update(model)

                    loss_value = float(loss.item())
                    batch_loss_values.append(loss_value)
                    epoch_train_losses[current_task_id].append(loss_value)

                mean_batch_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
                loop.set_postfix(
                    loss=f"{mean_batch_loss:.6f}",
                    groups=len(set(task_ids)),
                )

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
            val_results_df = evaluate_keypoint(
                model,
                val_loader,
                device,
                task_id_to_name,
                normalization_stats=normalization_stats,
            )
            if restore_state is not None:
                model.load_state_dict(restore_state, strict=True)
            selected_val_score = compute_combined_score(val_results_df, normalization_stats=normalization_stats)
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
                            "head_type": head_type,
                            "task_head_profile": task_head_profile,
                            "task_decoder_profile": task_decoder_profile,
                            "task_loss_family_profile": task_loss_family_profile,
                            "input_size": input_size,
                            "heatmap_size": list(HEATMAP_SIZE),
                            "checkpoint_metric": metric_label,
                            "measurement_loss_weight": measurement_loss_weight,
                            "dataset_loss_weight": dataset_loss_weight,
                            "femur_shaft_loss_weight": femur_shaft_loss_weight,
                            "fugc_segment_loss_weight": fugc_segment_loss_weight,
                            "use_ema": use_ema,
                            "ema_decay": ema_decay,
                            "task_loss_weight_overrides": task_loss_weight_overrides or {},
                            "sampler_task_weight_overrides": sampler_task_weight_overrides or {},
                            "normalization_scheme": "train_iqr_proxy",
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
        choices=("uniform", "challenge_v1"),
        default=TASK_HEAD_PROFILE,
        help=f"Task-specific head sizing profile (default: {TASK_HEAD_PROFILE}).",
    )
    parser.add_argument(
        "--task-decoder-profile",
        type=str,
        choices=("uniform", "geometry_v1", "weak_tasks_v1", "dedicated_v1"),
        default=TASK_DECODER_PROFILE,
        help=f"Task-specific decoder family profile (default: {TASK_DECODER_PROFILE}).",
    )
    parser.add_argument(
        "--task-loss-family-profile",
        type=str,
        choices=("uniform", "dataset_v1", "weak_tasks_v1"),
        default=TASK_LOSS_FAMILY_PROFILE,
        help=f"Dataset-family auxiliary loss profile (default: {TASK_LOSS_FAMILY_PROFILE}).",
    )
    parser.add_argument(
        "--fugc-segment-loss-weight",
        type=float,
        default=FUGC_SEGMENT_LOSS_WEIGHT,
        help="Auxiliary short-segment mask loss for FUGC. Use 0.0 to disable.",
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
    args = parser.parse_args()

    # --no-fpn takes precedence over --fpn
    use_fpn = USE_FPN
    if args.no_fpn:
        use_fpn = False
    elif args.fpn:
        use_fpn = True

    main(
        val_split=float(args.val_split),
        use_fpn=use_fpn,
        fpn_mode=str(args.fpn_mode),
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
        task_loss_family_profile=str(args.task_loss_family_profile),
        learning_rate=float(args.learning_rate),
        init_checkpoint=args.init_checkpoint,
        train_task_ids=_parse_task_id_csv(args.train_task_ids),
        measurement_loss_weight=float(args.measurement_loss_weight),
        dataset_loss_weight=float(args.dataset_loss_weight),
        femur_shaft_loss_weight=float(args.femur_shaft_loss_weight),
        fugc_segment_loss_weight=float(args.fugc_segment_loss_weight),
        task_loss_weight_overrides=_parse_weight_csv(args.task_loss_weights),
        sampler_task_weight_overrides=_parse_weight_csv(args.sampler_task_weights),
        use_ema=not bool(args.no_ema),
        ema_decay=float(args.ema_decay),
    )
