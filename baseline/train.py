import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime

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
from utils import evaluate_keypoint, keypoint_collate_fn, set_seed, softargmax_heatmaps_to_transformed_coords


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS =35
DATA_ROOT_PATH = "data"
OUTPUT_DIR = "output"
ENCODER = "vit_base_patch16_dinov3"
ENCODER_WEIGHTS = "pretrained"
RANDOM_SEED = 42
VAL_SPLIT = 0.2
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 1.8
INPUT_SIZE = 512
EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
USE_FPN = True  # ← 开关：设为 True 启用 FPN 特征金字塔
HEAD_TYPE = "deep"


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
):
    metric_column = "MRE (pixels)"
    metric_label = "MRE (pixels)"
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
    logger.info(f"Device used: {device}")
    logger.info(f"Encoder: {encoder_name}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Head type: {head_type}")

    logger.info(f"FPN neck: {'ENABLED' if use_fpn else 'DISABLED'} (set USE_FPN={use_fpn} at top of train.py)")

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

    train_indices, val_indices = _stratified_split_indices(
        temp_dataset.dataframe,
        val_split=val_split,
        seed=RANDOM_SEED,
    )
    train_size = len(train_indices)
    val_size = len(val_indices)

    train_dataset = KeypointDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        input_size=input_size,
    )
    train_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

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
        head_type=head_type,
    ).to(device)

    param_groups = [{"params": model.encoder.parameters(), "lr": LEARNING_RATE * 0.2}]
    for task_id, head in model.heads.items():
        param_groups.append({"params": head.parameters(), "lr": LEARNING_RATE * 10.0})

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

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
                    pred_logits, pred_heatmaps = model(task_images, task_id=current_task_id, return_prior=True)
                    target_coords = torch.stack([batch["train_label"][i] for i in task_indices], 0).to(device)
                    pred_coords_transformed = softargmax_heatmaps_to_transformed_coords(pred_logits)
                    target_coords_transformed = target_coords.clone()
                    coord_loss = F.l1_loss(pred_coords_transformed, target_coords_transformed)
                    heatmap_loss = F.mse_loss(pred_heatmaps, task_heatmaps)
                    loss = heatmap_loss + 0.2 * coord_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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

            val_results_df = evaluate_keypoint(model, val_loader, device, task_id_to_name)
            selected_val_score = float("inf")
            if not val_results_df.empty and metric_column in val_results_df.columns:
                selected_val_score = float(val_results_df[metric_column].mean())

            logger.info(f"\n--- Epoch {epoch + 1} Validation Report ---")
            if not val_results_df.empty:
                logger.info(val_results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
                for _, row in val_results_df.iterrows():
                    writer.add_scalar(
                        f"val/{metric_column}/{row['Task ID']}",
                        float(row[metric_column]),
                        epoch + 1,
                    )
            writer.add_scalar("val/mre_pixels_mean", selected_val_score, epoch + 1)
            logger.info(f"--- Average Val {metric_label} (Lower is better): {selected_val_score:.6f} ---")

            improved = selected_val_score < (best_val_score - early_stopping_min_delta)
            if improved:
                best_val_score = selected_val_score
                epochs_without_improvement = 0
                best_val_results_df = val_results_df.copy()
                checkpoint_payload = {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "encoder_name": encoder_name,
                        "use_fpn": use_fpn,
                        "head_type": head_type,
                        "input_size": input_size,
                        "heatmap_size": list(HEATMAP_SIZE),
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
    )
