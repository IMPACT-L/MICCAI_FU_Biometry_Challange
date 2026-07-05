import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def keypoint_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], 0)
    labels = [item["label"] for item in batch]
    train_labels = [item["train_label"] for item in batch]
    heatmaps = [item["heatmap"] for item in batch]
    task_ids = [item["task_id"] for item in batch]
    meta = [item["meta"] for item in batch]
    return {
        "image": images,
        "label": labels,
        "train_label": train_labels,
        "heatmap": heatmaps,
        "task_id": task_ids,
        "meta": meta,
    }


def letterbox_image_and_points(
    image: np.ndarray,
    coords_px: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    original_height, original_width = image.shape[:2]
    scale = min(input_size / max(original_width, 1), input_size / max(original_height, 1))
    resized_width = max(int(round(original_width * scale)), 1)
    resized_height = max(int(round(original_height * scale)), 1)

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((input_size, input_size, 3), dtype=image.dtype)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    coords_px = coords_px.copy().reshape(-1, 2)
    coords_px[:, 0] = coords_px[:, 0] * scale + pad_x
    coords_px[:, 1] = coords_px[:, 1] * scale + pad_y

    meta = {
        "original_width": float(original_width),
        "original_height": float(original_height),
        "input_size": float(input_size),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }
    return canvas, coords_px, meta


def decode_heatmaps_to_transformed_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    bsz, num_points, h, w = heatmaps.shape
    flat_idx = heatmaps.view(bsz, num_points, -1).argmax(dim=-1)
    ys = torch.div(flat_idx, w, rounding_mode="floor").float()
    xs = (flat_idx % w).float()

    x = xs / max(float(w - 1), 1.0)
    y = ys / max(float(h - 1), 1.0)
    return torch.stack([x, y], dim=-1).reshape(bsz, num_points * 2)


def softargmax_heatmaps_to_transformed_coords(logits: torch.Tensor) -> torch.Tensor:
    bsz, num_points, h, w = logits.shape
    probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
    xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    grid_x = xs.view(1, 1, 1, w)
    grid_y = ys.view(1, 1, h, 1)
    expected_x = (probs * grid_x).sum(dim=(-2, -1))
    expected_y = (probs * grid_y).sum(dim=(-2, -1))
    return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)


def transformed_coords_to_original_normalized(
    coords: torch.Tensor,
    meta: list[dict],
) -> torch.Tensor:
    output = []
    for sample_idx, sample_meta in enumerate(meta):
        sample = coords[sample_idx].detach().cpu().numpy().copy()
        sample = sample.reshape(-1, 2)
        input_size = max(float(sample_meta["input_size"]) - 1.0, 1.0)
        sample[:, 0] = sample[:, 0] * input_size
        sample[:, 1] = sample[:, 1] * input_size
        sample[:, 0] = (sample[:, 0] - float(sample_meta["pad_x"])) / max(float(sample_meta["scale"]), 1e-8)
        sample[:, 1] = (sample[:, 1] - float(sample_meta["pad_y"])) / max(float(sample_meta["scale"]), 1e-8)
        sample[:, 0] /= max(float(sample_meta["original_width"]) - 1.0, 1.0)
        sample[:, 1] /= max(float(sample_meta["original_height"]) - 1.0, 1.0)
        sample = np.clip(sample, 0.0, 1.0)
        output.append(torch.from_numpy(sample.reshape(-1)).float())
    return torch.stack(output, 0)


def decode_heatmaps_to_normalized_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """Decode [B, K, H, W] heatmaps into transformed-image normalized coordinates [B, 2K]."""
    return decode_heatmaps_to_transformed_coords(heatmaps)


def calculate_mre(y_true: torch.Tensor, y_pred: torch.Tensor, image_size=(256, 256)) -> float:
    """Mean radial error in pixels (Euclidean distance per keypoint)."""
    h, w = image_size
    y_true_px = y_true.detach().cpu().numpy().copy()
    y_pred_px = y_pred.detach().cpu().numpy().copy()

    y_true_px[:, 0::2] *= w
    y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w
    y_pred_px[:, 1::2] *= h

    y_true_pts = y_true_px.reshape(y_true_px.shape[0], -1, 2)
    y_pred_pts = y_pred_px.reshape(y_pred_px.shape[0], -1, 2)
    distances = np.sqrt(np.sum((y_pred_pts - y_true_pts) ** 2, axis=-1))
    return float(np.mean(distances))


def calculate_mre_per_sample(y_true: torch.Tensor, y_pred: torch.Tensor, meta: list[dict]) -> float:
    y_true_np = y_true.detach().cpu().numpy().copy().reshape(y_true.shape[0], -1, 2)
    y_pred_np = y_pred.detach().cpu().numpy().copy().reshape(y_pred.shape[0], -1, 2)
    scores = []
    for idx, sample_meta in enumerate(meta):
        width = max(float(sample_meta["original_width"]) - 1.0, 1.0)
        height = max(float(sample_meta["original_height"]) - 1.0, 1.0)
        gt_pts = y_true_np[idx].copy()
        pred_pts = y_pred_np[idx].copy()
        gt_pts[:, 0] *= width
        gt_pts[:, 1] *= height
        pred_pts[:, 0] *= width
        pred_pts[:, 1] *= height
        distances = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=-1))
        scores.append(float(np.mean(distances)))
    return float(np.mean(scores))


def evaluate_keypoint(model, val_loader, device, task_id_to_name):
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"]
            task_ids = batch["task_id"]
            meta = batch["meta"]

            unique_tasks = set(task_ids)
            for task_id in unique_tasks:
                task_indices = [i for i, t in enumerate(task_ids) if t == task_id]
                task_images = images[task_indices]
                task_labels = torch.stack([labels[i] for i in task_indices], 0).to(device)
                task_meta = [meta[i] for i in task_indices]

                pred_logits = model(task_images, task_id=task_id)
                pred_coords_transformed = softargmax_heatmaps_to_transformed_coords(pred_logits)
                pred_coords = transformed_coords_to_original_normalized(pred_coords_transformed, task_meta).to(device)
                task_metrics[task_id]["MRE (pixels)"].append(
                    calculate_mre_per_sample(
                        task_labels,
                        pred_coords,
                        task_meta,
                    )
                )

    results = []
    for task_id in sorted(task_metrics.keys()):
        row = {"Task ID": task_id, "Task Name": task_id_to_name.get(task_id, "Regression")}
        for metric_name, values in task_metrics[task_id].items():
            row[metric_name] = float(np.mean(values)) if values else 0.0
        results.append(row)

    return pd.DataFrame(results)
