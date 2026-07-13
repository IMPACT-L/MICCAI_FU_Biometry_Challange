#!/usr/bin/env python
"""Two-stage coarse-to-ROI submission generation.

Stage 1 predicts landmarks on the full image. Stage 2 crops a task-specific ROI
around the coarse landmarks, runs a second ROI-trained model on the zoomed crop,
and maps the refined crop coordinates back to the original image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submit import (  # noqa: E402
    EXPECTED_VALIDATION_COUNTS,
    ValidationManifestDataset,
    build_task_configs,
    collate_fn,
    infer_model_config_from_checkpoint,
    load_checkpoint_payload,
    predict_task_transformed_coords,
    round_float_list,
    validate_predictions,
)
from baseline.model_factory import MultiTaskModelFactory  # noqa: E402
from baseline.model_profiles import MODEL_PROFILE_NAMES, apply_model_profile  # noqa: E402
from baseline.utils import (  # noqa: E402
    canonicalize_task_coords,
    letterbox_image_and_points,
    transformed_coords_to_original_normalized,
)


DEFAULT_ROI_TASK_IDS = "A4C,AOP,FA,FUGC,HC,IVC,PLAX,PSAX,fetal_femur"


def parse_task_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def parse_task_float_map(value: str | None) -> dict[str, float]:
    if value is None or str(value).strip() == "":
        return {}
    output = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected TASK=value entry, got: {item}")
        key, raw = item.split("=", 1)
        output[key.strip()] = float(raw)
    return output


def read_manifest_rows(manifest_path: str) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def resolve_manifest_image_path(row: dict, manifest_path: str) -> str:
    abs_path = row["abs_path"]
    if not os.path.isabs(abs_path):
        abs_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(manifest_path)), abs_path))
    return abs_path


def load_submission_model(
    checkpoint_path: str,
    task_configs: list[dict],
    device: torch.device,
    model_profile: str | None,
    encoder_name: str | None,
    fpn_mode: str | None,
    fpn_type: str | None,
    task_head_profile: str | None,
    task_decoder_profile: str | None,
    task_adapter_profile: str | None,
):
    checkpoint, checkpoint_meta = load_checkpoint_payload(checkpoint_path, device)
    (
        inferred_encoder_name,
        inferred_use_fpn,
        inferred_fpn_mode,
        inferred_fpn_type,
        inferred_head_type,
        inferred_task_head_profile,
        inferred_task_decoder_profile,
        inferred_task_adapter_profile,
        inferred_input_size,
        inferred_heatmap_size,
    ) = infer_model_config_from_checkpoint(checkpoint, checkpoint_meta)

    config = {
        "encoder_name": encoder_name or inferred_encoder_name,
        "use_fpn": inferred_use_fpn,
        "fpn_mode": fpn_mode or inferred_fpn_mode,
        "fpn_type": fpn_type or inferred_fpn_type,
        "task_head_profile": task_head_profile or inferred_task_head_profile,
        "task_decoder_profile": task_decoder_profile or inferred_task_decoder_profile,
        "task_adapter_profile": task_adapter_profile or inferred_task_adapter_profile,
    }
    if model_profile is not None:
        config = apply_model_profile(model_profile, "inference", config)

    model = MultiTaskModelFactory(
        encoder_name=str(config["encoder_name"]),
        encoder_weights="pretrained",
        task_configs=task_configs,
        heatmap_size=inferred_heatmap_size,
        use_fpn=bool(config["use_fpn"]),
        fpn_mode=str(config["fpn_mode"]),
        fpn_type=str(config["fpn_type"]),
        head_type=inferred_head_type,
        task_head_profile=str(config["task_head_profile"]),
        task_decoder_profile=str(config["task_decoder_profile"]),
        task_adapter_profile=str(config["task_adapter_profile"]),
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, {
        **config,
        "head_type": inferred_head_type,
        "input_size": int(inferred_input_size),
        "heatmap_size": inferred_heatmap_size,
    }


def normalized_to_pixel_points(norm_coords: list[float] | np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(norm_coords, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] *= max(float(width) - 1.0, 1.0)
    points[:, 1] *= max(float(height) - 1.0, 1.0)
    return points


def make_roi_box(
    points_px: np.ndarray,
    image_width: int,
    image_height: int,
    context: float,
    min_size: float,
) -> tuple[int, int, int, int]:
    finite = np.isfinite(points_px).all(axis=1)
    points = points_px[finite]
    if len(points) == 0:
        return 0, 0, int(image_width), int(image_height)

    x_min = float(points[:, 0].min())
    y_min = float(points[:, 1].min())
    x_max = float(points[:, 0].max())
    y_max = float(points[:, 1].max())
    box_w = max(x_max - x_min, 8.0)
    box_h = max(y_max - y_min, 8.0)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)

    crop_w = min(float(image_width), max(box_w * context, min_size))
    crop_h = min(float(image_height), max(box_h * context, min_size))
    x0 = int(np.floor(np.clip(center_x - crop_w * 0.5, 0.0, max(float(image_width) - crop_w, 0.0))))
    y0 = int(np.floor(np.clip(center_y - crop_h * 0.5, 0.0, max(float(image_height) - crop_h, 0.0))))
    x1 = int(np.ceil(min(float(image_width), float(x0) + crop_w)))
    y1 = int(np.ceil(min(float(image_height), float(y0) + crop_h)))

    if x1 <= x0 or y1 <= y0:
        return 0, 0, int(image_width), int(image_height)
    return x0, y0, x1, y1


def crop_normalized_to_original_normalized(
    crop_norm: torch.Tensor,
    crop_boxes: list[tuple[int, int, int, int]],
    original_sizes: list[tuple[int, int]],
) -> torch.Tensor:
    output = []
    for sample_idx, sample in enumerate(crop_norm.detach().cpu().numpy()):
        x0, y0, x1, y1 = crop_boxes[sample_idx]
        original_height, original_width = original_sizes[sample_idx]
        crop_width = max(float(x1 - x0) - 1.0, 1.0)
        crop_height = max(float(y1 - y0) - 1.0, 1.0)
        points = sample.reshape(-1, 2).copy()
        points[:, 0] = (points[:, 0] * crop_width + float(x0)) / max(float(original_width) - 1.0, 1.0)
        points[:, 1] = (points[:, 1] * crop_height + float(y0)) / max(float(original_height) - 1.0, 1.0)
        output.append(torch.from_numpy(np.clip(points.reshape(-1), 0.0, 1.0)).float())
    return torch.stack(output, 0)


def mean_pixel_shift(
    first_norm: np.ndarray,
    second_norm: np.ndarray,
    original_size: tuple[int, int],
) -> float:
    original_height, original_width = original_size
    first = first_norm.reshape(-1, 2).astype(np.float32).copy()
    second = second_norm.reshape(-1, 2).astype(np.float32).copy()
    scale = np.array(
        [max(float(original_width) - 1.0, 1.0), max(float(original_height) - 1.0, 1.0)],
        dtype=np.float32,
    )
    distances = np.linalg.norm((first - second) * scale[None, :], axis=1)
    return float(np.mean(distances))


def prediction_to_submission_item(row: dict, task_id: str, norm_coords: np.ndarray, original_size: tuple[int, int]) -> dict:
    pred = round_float_list(norm_coords.reshape(-1).tolist())
    original_height, original_width = original_size
    pixel_coords = []
    for idx in range(0, len(pred), 2):
        pixel_coords.extend([float(pred[idx]) * original_width, float(pred[idx + 1]) * original_height])
    return {
        "image_path": row["image_path"],
        "task_id": task_id,
        "predicted_points_normalized": pred,
        "predicted_points_pixels": round_float_list(pixel_coords),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate two-stage coarse-to-ROI submission archive.")
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--roi-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-profile", choices=MODEL_PROFILE_NAMES, default=None)
    parser.add_argument("--coarse-model-profile", choices=MODEL_PROFILE_NAMES, default=None)
    parser.add_argument("--roi-model-profile", choices=MODEL_PROFILE_NAMES, default=None)
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--fpn-mode", choices=("shared", "task_specific"), default=None)
    parser.add_argument("--fpn-type", choices=("fpn", "bifpn"), default=None)
    parser.add_argument("--task-head-profile", choices=("uniform", "challenge_legacy_v1", "challenge_v1"), default=None)
    parser.add_argument(
        "--task-decoder-profile",
        choices=(
            "uniform",
            "cardiac_graph_v1",
            "coarse_refine_v1",
            "ivc_refine_v1",
            "ivc_refine_v2",
            "fugc_refine_v1",
            "hc_refine_v1",
            "hidden_hc_ivc_refine_v1",
            "hidden_a4c_hc_ivc_refine_v1",
            "hidden_a4c_hc_ivc_fugc_refine_v1",
            "hidden_a4c_hc_ivc_plax_refine_v1",
            "hidden_a4c_hc_ivc_femur_refine_v1",
            "hidden_a4cv2_hc_ivc_fugc_refine_v1",
            "hidden_a4cv3_hc_ivc_fugc_refine_v1",
            "hidden_a4cv4_hc_ivc_fugc_refine_v1",
            "geometry_v1",
            "geometry_family_v2",
            "weak_tasks_v1",
            "dedicated_legacy_v1",
            "dedicated_v1",
        ),
        default=None,
    )
    parser.add_argument(
        "--task-adapter-profile",
        choices=("uniform", "softsharing_v1", "localrefine_v1", "coarse_refine_v1", "context_experts_v1", "context_local_v1", "taskfilm_v1"),
        default=None,
    )
    parser.add_argument("--roi-task-ids", default=DEFAULT_ROI_TASK_IDS)
    parser.add_argument("--roi-context", type=float, default=1.8)
    parser.add_argument("--roi-min-size", type=float, default=112.0)
    parser.add_argument(
        "--roi-gate-max-shift-px",
        type=float,
        default=None,
        help="Fallback to coarse prediction when ROI refinement moves landmarks by more than this mean pixel distance.",
    )
    parser.add_argument(
        "--roi-gate-task-thresholds",
        default=None,
        help="Optional task-specific gate thresholds, e.g. A4C=12,FUGC=8,HC=10.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test only; disables official count validation.")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    output_dir = os.path.abspath(args.output_dir)
    coarse_checkpoint = os.path.abspath(args.coarse_checkpoint)
    roi_checkpoint = os.path.abspath(args.roi_checkpoint)
    if not os.path.exists(coarse_checkpoint):
        raise FileNotFoundError(f"Coarse checkpoint not found: {coarse_checkpoint}")
    if not os.path.exists(roi_checkpoint):
        raise FileNotFoundError(f"ROI checkpoint not found: {roi_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_configs = build_task_configs(manifest_path)
    rows = read_manifest_rows(manifest_path)
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    coarse_profile = args.coarse_model_profile or args.model_profile
    roi_profile = args.roi_model_profile or args.model_profile
    coarse_model, coarse_config = load_submission_model(
        coarse_checkpoint,
        task_configs,
        device,
        coarse_profile,
        args.encoder_name,
        args.fpn_mode,
        args.fpn_type,
        args.task_head_profile,
        args.task_decoder_profile,
        args.task_adapter_profile,
    )
    roi_model, roi_config = load_submission_model(
        roi_checkpoint,
        task_configs,
        device,
        roi_profile,
        args.encoder_name,
        args.fpn_mode,
        args.fpn_type,
        args.task_head_profile,
        args.task_decoder_profile,
        args.task_adapter_profile,
    )
    print(f"Coarse config: {coarse_config}")
    print(f"ROI config: {roi_config}")
    print(f"ROI refinement tasks: {sorted(parse_task_ids(args.roi_task_ids) or [])}")
    task_gate_thresholds = parse_task_float_map(args.roi_gate_task_thresholds)
    if args.roi_gate_max_shift_px is not None or task_gate_thresholds:
        print(
            "ROI self-consistency gate: "
            f"default={args.roi_gate_max_shift_px}, task_thresholds={task_gate_thresholds}"
        )

    coarse_dataset = ValidationManifestDataset(manifest_path=manifest_path, input_size=int(coarse_config["input_size"]))
    if args.max_samples is not None:
        coarse_dataset.rows = coarse_dataset.rows[: int(args.max_samples)]
    coarse_loader = DataLoader(
        coarse_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    coarse_predictions: list[np.ndarray | None] = [None] * len(rows)
    original_sizes: list[tuple[int, int] | None] = [None] * len(rows)
    row_offset = 0
    with torch.no_grad():
        for batch in tqdm(coarse_loader, desc="Stage 1 coarse inference"):
            images = batch["image"].to(device)
            task_ids = batch["task_id"]
            batch_preds: list[np.ndarray | None] = [None] * len(task_ids)
            for task_id in sorted(set(task_ids)):
                task_indices = [idx for idx, value in enumerate(task_ids) if value == task_id]
                task_images = images[task_indices]
                outputs_transformed = predict_task_transformed_coords(coarse_model, task_images, task_id, "none", None)
                outputs = transformed_coords_to_original_normalized(
                    outputs_transformed,
                    [batch["meta"][idx] for idx in task_indices],
                )
                outputs = canonicalize_task_coords(outputs, task_id)
                for local_idx, batch_idx in enumerate(task_indices):
                    batch_preds[batch_idx] = outputs[local_idx].cpu().numpy()
            for batch_idx, pred in enumerate(batch_preds):
                coarse_predictions[row_offset + batch_idx] = pred
                original_sizes[row_offset + batch_idx] = tuple(batch["original_size"][batch_idx])
            row_offset += len(task_ids)

    roi_task_ids = parse_task_ids(args.roi_task_ids) or set()
    transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    roi_batches: dict[str, list[dict]] = {}
    for idx, row in enumerate(rows):
        task_id = str(row["task_id"])
        if task_id not in roi_task_ids:
            continue
        image_path = resolve_manifest_image_path(row, manifest_path)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read ROI image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        coarse = coarse_predictions[idx]
        if coarse is None:
            continue
        points_px = normalized_to_pixel_points(coarse, image_width, image_height)
        x0, y0, x1, y1 = make_roi_box(
            points_px,
            image_width=image_width,
            image_height=image_height,
            context=float(args.roi_context),
            min_size=float(args.roi_min_size),
        )
        crop = image[y0:y1, x0:x1]
        dummy = np.zeros((int(row["num_points"]), 2), dtype=np.float32)
        crop_letterboxed, _, crop_meta = letterbox_image_and_points(crop, dummy, int(roi_config["input_size"]))
        transformed = transforms(image=crop_letterboxed)
        roi_batches.setdefault(task_id, []).append(
            {
                "row_idx": idx,
                "image": transformed["image"],
                "meta": crop_meta,
                "crop_box": (x0, y0, x1, y1),
                "original_size": (image_height, image_width),
            }
        )

    refined_predictions: list[np.ndarray | None] = list(coarse_predictions)
    with torch.no_grad():
        for task_id, items in tqdm(sorted(roi_batches.items()), desc="Stage 2 ROI refinement"):
            for start in range(0, len(items), args.batch_size):
                chunk = items[start : start + args.batch_size]
                images = torch.stack([item["image"] for item in chunk], 0).to(device)
                outputs_transformed = predict_task_transformed_coords(roi_model, images, task_id, "none", None)
                crop_norm = transformed_coords_to_original_normalized(
                    outputs_transformed,
                    [item["meta"] for item in chunk],
                )
                orig_norm = crop_normalized_to_original_normalized(
                    crop_norm,
                    [item["crop_box"] for item in chunk],
                    [item["original_size"] for item in chunk],
                )
                orig_norm = canonicalize_task_coords(orig_norm, task_id)
                for local_idx, item in enumerate(chunk):
                    row_idx = item["row_idx"]
                    candidate = orig_norm[local_idx].cpu().numpy()
                    threshold = task_gate_thresholds.get(task_id, args.roi_gate_max_shift_px)
                    if threshold is None:
                        refined_predictions[row_idx] = candidate
                        continue

                    coarse = coarse_predictions[row_idx]
                    if coarse is None:
                        refined_predictions[row_idx] = candidate
                        continue
                    shift_px = mean_pixel_shift(coarse, candidate, item["original_size"])
                    if shift_px <= float(threshold):
                        refined_predictions[row_idx] = candidate

    predictions = []
    for idx, row in enumerate(rows):
        pred = refined_predictions[idx]
        if pred is None:
            raise RuntimeError(f"Missing prediction for row {idx}: {row}")
        original_size = original_sizes[idx]
        if original_size is None:
            image_path = resolve_manifest_image_path(row, manifest_path)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read final image size: {image_path}")
            original_size = image.shape[:2]
        predictions.append(prediction_to_submission_item(row, str(row["task_id"]), pred, original_size))

    if args.max_samples is None:
        validate_predictions(predictions)
    else:
        counts = Counter(item["task_id"] for item in predictions)
        print(f"Smoke mode: skipped official count validation. Counts={dict(sorted(counts.items()))}")

    os.makedirs(output_dir, exist_ok=False)
    json_path = os.path.join(output_dir, "regression_predictions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="regression_predictions.json")

    print(f"Wrote {json_path}")
    print(f"Wrote {zip_path}")
    if args.max_samples is None:
        print(f"Validated counts: {EXPECTED_VALIDATION_COUNTS}")


if __name__ == "__main__":
    main()
