import random
from collections import defaultdict
import json

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


MEASUREMENT_PAIRS = {
    "A4C": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    "AOP": [(0, 1), (2, 3)],
    "FA": [(0, 1), (2, 3)],
    "FUGC": [(0, 1)],
    "HC": [(0, 1), (2, 3)],
    "IVC": [(0, 1)],
    "PLAX": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)],
    "PSAX": [(0, 1), (2, 3)],
    "fetal_femur": [(0, 1)],
}

TASK_LOSS_FAMILY_PRESETS = {
    "uniform": {},
    "dataset_v1": {
        "A4C": "dense",
        "PLAX": "dense",
        "HC": "compact",
        "AOP": "compact",
        "FA": "compact",
        "PSAX": "compact",
        "FUGC": "line",
        "IVC": "line",
        "fetal_femur": "line",
    },
}

DEFAULT_NORMALIZER_EPS = 1.0


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


def _coords_to_pixel_points(coords: torch.Tensor, meta: list[dict]) -> np.ndarray:
    coords_np = coords.detach().cpu().numpy().copy().reshape(coords.shape[0], -1, 2)
    output = []
    for idx, sample_meta in enumerate(meta):
        width = max(float(sample_meta["original_width"]) - 1.0, 1.0)
        height = max(float(sample_meta["original_height"]) - 1.0, 1.0)
        pts = coords_np[idx].copy()
        pts[:, 0] *= width
        pts[:, 1] *= height
        output.append(pts)
    return np.stack(output, axis=0)


def compute_measurements_from_points(points_px: np.ndarray, task_id: str) -> np.ndarray:
    pairs = MEASUREMENT_PAIRS.get(task_id, [])
    if not pairs:
        return np.zeros((points_px.shape[0], 0), dtype=np.float32)
    measures = []
    for start_idx, end_idx in pairs:
        start_pts = points_px[:, start_idx, :]
        end_pts = points_px[:, end_idx, :]
        dist = np.linalg.norm(end_pts - start_pts, axis=-1)
        measures.append(dist.astype(np.float32))
    return np.stack(measures, axis=1) if measures else np.zeros((points_px.shape[0], 0), dtype=np.float32)


def compute_normalization_stats_from_dataframe(dataframe: pd.DataFrame) -> dict:
    stats = {}
    for task_id, task_df in dataframe.groupby("task_id", sort=True):
        num_points = int(task_df["num_classes"].iloc[0])
        pairs = MEASUREMENT_PAIRS.get(task_id, [])
        measurement_values = []
        mre_reference_values = []

        for _, row in task_df.iterrows():
            coords = []
            for point_idx in range(1, num_points + 1):
                col = f"point_{point_idx}_xy"
                if col in row and pd.notna(row[col]):
                    coords.extend(json.loads(row[col]))
                else:
                    coords.extend([0.0, 0.0])
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            if pairs:
                measures = compute_measurements_from_points(points[None, ...], task_id)[0]
                measurement_values.append(measures)
                mre_reference_values.extend(measures.tolist())

        task_stats = {
            "mre_iqr": DEFAULT_NORMALIZER_EPS,
            "measurement_iqr": [],
        }

        if mre_reference_values:
            q1, q3 = np.percentile(np.array(mre_reference_values, dtype=np.float32), [25, 75])
            task_stats["mre_iqr"] = float(max(q3 - q1, DEFAULT_NORMALIZER_EPS))

        if measurement_values:
            measurement_array = np.stack(measurement_values, axis=0)
            for measure_idx in range(measurement_array.shape[1]):
                q1, q3 = np.percentile(measurement_array[:, measure_idx], [25, 75])
                task_stats["measurement_iqr"].append(float(max(q3 - q1, DEFAULT_NORMALIZER_EPS)))

        stats[task_id] = task_stats

    return stats


def calculate_measurement_mae_per_sample(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    meta: list[dict],
    task_id: str,
) -> float:
    gt_points = _coords_to_pixel_points(y_true, meta)
    pred_points = _coords_to_pixel_points(y_pred, meta)
    gt_measures = compute_measurements_from_points(gt_points, task_id)
    pred_measures = compute_measurements_from_points(pred_points, task_id)
    if gt_measures.shape[1] == 0:
        return 0.0
    return float(np.mean(np.abs(pred_measures - gt_measures)))


def compute_measurement_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    meta: list[dict],
    task_id: str,
) -> torch.Tensor:
    pairs = MEASUREMENT_PAIRS.get(task_id, [])
    if not pairs:
        return pred_coords.new_tensor(0.0)

    pred = pred_coords.reshape(pred_coords.shape[0], -1, 2)
    target = target_coords.reshape(target_coords.shape[0], -1, 2)
    losses = []
    for batch_idx, sample_meta in enumerate(meta):
        width = pred_coords.new_tensor(max(float(sample_meta["original_width"]) - 1.0, 1.0))
        height = pred_coords.new_tensor(max(float(sample_meta["original_height"]) - 1.0, 1.0))
        scale = pred_coords.new_tensor([width, height])
        pred_pts = pred[batch_idx] * scale
        target_pts = target[batch_idx] * scale
        for start_idx, end_idx in pairs:
            pred_dist = torch.norm(pred_pts[end_idx] - pred_pts[start_idx], p=2)
            target_dist = torch.norm(target_pts[end_idx] - target_pts[start_idx], p=2)
            losses.append(torch.abs(pred_dist - target_dist))
    if not losses:
        return pred_coords.new_tensor(0.0)
    return torch.stack(losses).mean()


def _coords_to_pixel_points_torch(coords: torch.Tensor, meta: list[dict]) -> torch.Tensor:
    points = coords.reshape(coords.shape[0], -1, 2)
    scales = []
    for sample_meta in meta:
        width = max(float(sample_meta["original_width"]) - 1.0, 1.0)
        height = max(float(sample_meta["original_height"]) - 1.0, 1.0)
        scales.append([width, height])
    scale_tensor = coords.new_tensor(scales).unsqueeze(1)
    return points * scale_tensor


def compute_line_direction_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    meta: list[dict],
    task_id: str,
) -> torch.Tensor:
    pairs = MEASUREMENT_PAIRS.get(task_id, [])
    if not pairs:
        return pred_coords.new_tensor(0.0)
    pred_pts = _coords_to_pixel_points_torch(pred_coords, meta)
    target_pts = _coords_to_pixel_points_torch(target_coords, meta)
    losses = []
    for start_idx, end_idx in pairs:
        pred_vec = pred_pts[:, end_idx] - pred_pts[:, start_idx]
        target_vec = target_pts[:, end_idx] - target_pts[:, start_idx]
        pred_unit = pred_vec / pred_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_unit = target_vec / target_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        losses.append((pred_unit - target_unit).abs().mean(dim=-1))
    return torch.stack(losses, dim=0).mean()


def compute_pairwise_distance_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    meta: list[dict],
) -> torch.Tensor:
    pred_pts = _coords_to_pixel_points_torch(pred_coords, meta)
    target_pts = _coords_to_pixel_points_torch(target_coords, meta)
    num_points = pred_pts.shape[1]
    if num_points < 2:
        return pred_coords.new_tensor(0.0)
    pred_dist = torch.cdist(pred_pts, pred_pts, p=2)
    target_dist = torch.cdist(target_pts, target_pts, p=2)
    mask = torch.triu(torch.ones(num_points, num_points, device=pred_coords.device, dtype=torch.bool), diagonal=1)
    return (pred_dist[:, mask] - target_dist[:, mask]).abs().mean()


def build_line_mask_from_transformed_coords(
    coords: torch.Tensor,
    heatmap_size: tuple[int, int],
    thickness: int = 3,
) -> torch.Tensor:
    height, width = int(heatmap_size[0]), int(heatmap_size[1])
    coords_np = coords.detach().cpu().numpy().reshape(coords.shape[0], -1, 2)
    masks = []
    for sample_points in coords_np:
        canvas = np.zeros((height, width), dtype=np.float32)
        if sample_points.shape[0] >= 2:
            start = sample_points[0]
            end = sample_points[1]
            start_xy = (
                int(round(float(start[0]) * max(width - 1, 1))),
                int(round(float(start[1]) * max(height - 1, 1))),
            )
            end_xy = (
                int(round(float(end[0]) * max(width - 1, 1))),
                int(round(float(end[1]) * max(height - 1, 1))),
            )
            cv2.line(canvas, start_xy, end_xy, color=1.0, thickness=thickness)
        masks.append(canvas)
    return torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1).to(coords.device, dtype=coords.dtype)


def compute_femur_shaft_loss(
    shaft_logits: torch.Tensor | None,
    target_coords_transformed: torch.Tensor,
    heatmap_size: tuple[int, int],
) -> torch.Tensor:
    if shaft_logits is None:
        return target_coords_transformed.new_tensor(0.0)
    target_mask = build_line_mask_from_transformed_coords(
        target_coords_transformed,
        heatmap_size=heatmap_size,
        thickness=3,
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(shaft_logits, target_mask)


def compute_fugc_segment_loss(
    segment_logits: torch.Tensor | None,
    target_coords_transformed: torch.Tensor,
    heatmap_size: tuple[int, int],
) -> torch.Tensor:
    if segment_logits is None:
        return target_coords_transformed.new_tensor(0.0)
    target_mask = build_line_mask_from_transformed_coords(
        target_coords_transformed,
        heatmap_size=heatmap_size,
        thickness=2,
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(segment_logits, target_mask)


def compute_dataset_specific_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    meta: list[dict],
    task_id: str,
    profile: str = "uniform",
) -> torch.Tensor:
    if profile not in TASK_LOSS_FAMILY_PRESETS:
        raise ValueError(f"Unsupported task loss family profile: {profile}")
    family = TASK_LOSS_FAMILY_PRESETS[profile].get(task_id)
    if family is None:
        return pred_coords.new_tensor(0.0)
    measurement_loss = compute_measurement_loss(pred_coords, target_coords, meta, task_id)
    if family == "line":
        direction_loss = compute_line_direction_loss(pred_coords, target_coords, meta, task_id)
        return 0.7 * measurement_loss + 0.3 * direction_loss
    if family == "compact":
        return measurement_loss
    if family == "dense":
        pairwise_loss = compute_pairwise_distance_loss(pred_coords, target_coords, meta)
        return 0.5 * measurement_loss + 0.5 * pairwise_loss
    raise ValueError(f"Unsupported task loss family: {family}")


def compute_combined_score(results_df: pd.DataFrame, normalization_stats: dict | None = None) -> float:
    if results_df.empty or "MRE (pixels)" not in results_df.columns:
        return float("inf")

    normalized_mre_values = []
    normalized_measurement_values = []

    for _, row in results_df.iterrows():
        task_id = row["Task ID"]
        task_stats = (normalization_stats or {}).get(task_id, {})
        mre_norm = float(task_stats.get("mre_iqr", DEFAULT_NORMALIZER_EPS))
        normalized_mre_values.append(float(row["MRE (pixels)"]) / max(mre_norm, DEFAULT_NORMALIZER_EPS))

        measurement_value = row.get("Measurement MAE (pixels)")
        if measurement_value is not None and np.isfinite(measurement_value):
            measurement_norms = task_stats.get("measurement_iqr", [])
            if measurement_norms:
                normalized_measurement_values.append(float(measurement_value) / max(float(np.mean(measurement_norms)), DEFAULT_NORMALIZER_EPS))

    normalized_mre = float(np.mean(normalized_mre_values)) if normalized_mre_values else float("inf")
    if not normalized_measurement_values:
        return normalized_mre
    normalized_measurement = float(np.mean(normalized_measurement_values))
    return 0.5 * normalized_mre + 0.5 * normalized_measurement


def evaluate_keypoint(model, val_loader, device, task_id_to_name, normalization_stats: dict | None = None):
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
                if task_id in MEASUREMENT_PAIRS:
                    task_metrics[task_id]["Measurement MAE (pixels)"].append(
                        calculate_measurement_mae_per_sample(
                            task_labels,
                            pred_coords,
                            task_meta,
                            task_id,
                        )
                    )

    results = []
    for task_id in sorted(task_metrics.keys()):
        row = {"Task ID": task_id, "Task Name": task_id_to_name.get(task_id, "Regression")}
        for metric_name, values in task_metrics[task_id].items():
            row[metric_name] = float(np.mean(values)) if values else 0.0
        task_stats = (normalization_stats or {}).get(task_id, {})
        row["Normalized MRE"] = row["MRE (pixels)"] / max(float(task_stats.get("mre_iqr", DEFAULT_NORMALIZER_EPS)), DEFAULT_NORMALIZER_EPS)
        if "Measurement MAE (pixels)" in row:
            measurement_norms = task_stats.get("measurement_iqr", [])
            if measurement_norms:
                row["Normalized Measurement MAE"] = row["Measurement MAE (pixels)"] / max(float(np.mean(measurement_norms)), DEFAULT_NORMALIZER_EPS)
        results.append(row)

    return pd.DataFrame(results)
