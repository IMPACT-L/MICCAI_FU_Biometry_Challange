#!/usr/bin/env python3
"""Train/apply a local image-patch residual landmark refiner.

This is a bounded second-stage model. It learns how the current anchor model's
predicted points differ from released labels by looking at a small image patch
around each predicted point. At submission time it applies only small gated
corrections to a trusted anchor JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
TASKS = ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"]
TASK_TO_INDEX = {task: idx for idx, task in enumerate(TASKS)}
EXPECTED_COUNTS = {
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


def normalize_image_path(image_path: str, task_id: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return "/".join(parts)
    return f"{task_id}/{os.path.basename(normalized)}"


def parse_task_ids(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def read_json(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def pred_map(predictions: Iterable[dict]) -> dict[tuple[str, str], dict]:
    output = {}
    for item in predictions:
        task_id = str(item["task_id"])
        key = (task_id, normalize_image_path(str(item["image_path"]), task_id))
        output[key] = item
    return output


def coords(item: dict, field: str) -> np.ndarray:
    values = np.asarray(item[field], dtype=np.float32)
    return values.reshape(-1, 2)


def infer_size_from_prediction(item: dict) -> tuple[float, float]:
    norm = coords(item, "predicted_points_normalized")
    pix = coords(item, "predicted_points_pixels")
    x_valid = np.abs(norm[:, 0]) > 1e-6
    y_valid = np.abs(norm[:, 1]) > 1e-6
    width = float(np.median(pix[x_valid, 0] / norm[x_valid, 0])) if np.any(x_valid) else 1.0
    height = float(np.median(pix[y_valid, 1] / norm[y_valid, 1])) if np.any(y_valid) else 1.0
    return max(width, 1.0), max(height, 1.0)


def update_prediction_item(item: dict, points_px: np.ndarray) -> dict:
    width, height = infer_size_from_prediction(item)
    points_px = points_px.astype(np.float32).copy()
    points_px[:, 0] = np.clip(points_px[:, 0], 0.0, width - 1.0)
    points_px[:, 1] = np.clip(points_px[:, 1], 0.0, height - 1.0)
    points_norm = points_px.copy()
    points_norm[:, 0] /= width
    points_norm[:, 1] /= height
    updated = dict(item)
    updated["predicted_points_pixels"] = [round(float(v), 6) for v in points_px.reshape(-1)]
    updated["predicted_points_normalized"] = [round(float(v), 6) for v in points_norm.reshape(-1)]
    return updated


def load_train_rows(data_root: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for csv_path in sorted((data_root / "csv").glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "task_id" not in df.columns or "image_path" not in df.columns:
            continue
        for _, row in df.iterrows():
            task_id = str(row["task_id"])
            if task_id not in TASK_TO_INDEX:
                continue
            num_points = int(row["num_classes"])
            points = []
            for idx in range(1, num_points + 1):
                raw = row.get(f"point_{idx}_xy")
                points.append(json.loads(raw) if pd.notna(raw) else [0.0, 0.0])
            image_path = normalize_image_path(str(row["image_path"]), task_id)
            rows[(task_id, image_path)] = {
                "task_id": task_id,
                "image_path": image_path,
                "height": float(row["height"]),
                "width": float(row["width"]),
                "points_px": np.asarray(points, dtype=np.float32),
            }
    return rows


def load_validation_manifest(manifest_path: Path) -> dict[tuple[str, str], Path]:
    manifest = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent
    paths = {}
    for _, row in manifest.iterrows():
        task_id = str(row["task_id"])
        image_path = normalize_image_path(str(row["image_path"]), task_id)
        abs_path = Path(str(row["abs_path"]))
        if not abs_path.is_absolute():
            abs_path = (base_dir / abs_path).resolve()
        paths[(task_id, image_path)] = abs_path
    return paths


def resolve_train_image(data_root: Path, image_path: str) -> Path:
    candidate = data_root / "images" / image_path
    if candidate.is_file():
        return candidate
    candidate = data_root / image_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not resolve training image: {image_path}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def extract_patch(image: np.ndarray, x: float, y: float, patch_size: int) -> np.ndarray:
    half = patch_size / 2.0
    matrix = np.array([[1.0, 0.0, x - half], [0.0, 1.0, y - half]], dtype=np.float32)
    patch = cv2.warpAffine(
        image,
        matrix,
        (patch_size, patch_size),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    sobel_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    grad = grad / (float(np.percentile(grad, 99.0)) + 1e-6)
    patch = (patch - float(patch.mean())) / (float(patch.std()) + 1e-6)
    return np.stack([patch, np.clip(grad, 0.0, 1.0)], axis=0).astype(np.float32)


@dataclass
class PointSample:
    image_path: Path
    task_id: str
    point_index: int
    num_points: int
    pred_px: np.ndarray
    target_delta_px: np.ndarray
    image_size: tuple[float, float]


class PointResidualDataset(Dataset):
    def __init__(self, samples: list[PointSample], patch_size: int, max_delta_px: float, augment: bool):
        self.samples = samples
        self.patch_size = int(patch_size)
        self.max_delta_px = float(max_delta_px)
        self.augment = bool(augment)
        self._image_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: Path) -> np.ndarray:
        if path not in self._image_cache:
            self._image_cache[path] = read_gray(path)
        return self._image_cache[path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = self._image(sample.image_path)
        pred_px = sample.pred_px.copy()
        target_delta = sample.target_delta_px.copy()
        if self.augment:
            pred_px += np.random.normal(0.0, 2.0, size=2).astype(np.float32)
        patch = extract_patch(image, float(pred_px[0]), float(pred_px[1]), self.patch_size)
        width, height = sample.image_size
        meta = np.array(
            [
                pred_px[0] / max(width, 1.0),
                pred_px[1] / max(height, 1.0),
                float(sample.point_index) / max(float(sample.num_points - 1), 1.0),
                float(sample.num_points) / 24.0,
                float(width) / max(float(height), 1.0),
            ],
            dtype=np.float32,
        )
        task_index = TASK_TO_INDEX[sample.task_id]
        target = np.clip(target_delta / self.max_delta_px, -1.0, 1.0).astype(np.float32)
        return {
            "patch": torch.from_numpy(patch),
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "meta": torch.from_numpy(meta),
            "target": torch.from_numpy(target),
        }


class PointResidualRefiner(nn.Module):
    def __init__(self, num_tasks: int = len(TASKS), task_dim: int = 16, meta_dim: int = 5):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, task_dim)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 + task_dim + meta_dim, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, patch: torch.Tensor, task_index: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        encoded = self.patch_encoder(patch).flatten(1)
        task = self.task_embed(task_index)
        return self.regressor(torch.cat([encoded, task, meta], dim=1))


def make_samples(data_root: Path, prediction_json: Path, task_ids: set[str] | None) -> list[PointSample]:
    gt_rows = load_train_rows(data_root)
    predictions = pred_map(read_json(prediction_json))
    samples: list[PointSample] = []
    for key, pred_item in sorted(predictions.items()):
        task_id, image_path = key
        if task_ids is not None and task_id not in task_ids:
            continue
        gt = gt_rows.get(key)
        if gt is None:
            continue
        pred_points = coords(pred_item, "predicted_points_pixels")
        gt_points = gt["points_px"]
        if pred_points.shape != gt_points.shape:
            continue
        image_file = resolve_train_image(data_root, image_path)
        for point_index, (pred_px, gt_px) in enumerate(zip(pred_points, gt_points)):
            samples.append(
                PointSample(
                    image_path=image_file,
                    task_id=task_id,
                    point_index=point_index,
                    num_points=pred_points.shape[0],
                    pred_px=pred_px.astype(np.float32),
                    target_delta_px=(gt_px - pred_px).astype(np.float32),
                    image_size=(float(gt["width"]), float(gt["height"])),
                )
            )
    if not samples:
        raise ValueError("No point samples were built. Check prediction JSON and data paths.")
    return samples


def split_samples(samples: list[PointSample], val_fraction: float, seed: int) -> tuple[list[PointSample], list[PointSample]]:
    rng = random.Random(seed)
    by_task: dict[str, list[PointSample]] = {}
    for sample in samples:
        by_task.setdefault(sample.task_id, []).append(sample)
    train, val = [], []
    for task_samples in by_task.values():
        rng.shuffle(task_samples)
        n_val = max(1, int(round(len(task_samples) * val_fraction)))
        val.extend(task_samples[:n_val])
        train.extend(task_samples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = parse_task_ids(args.task_ids)
    samples = make_samples(data_root, Path(args.train_pred_json), task_ids)
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)

    train_loader = DataLoader(
        PointResidualDataset(train_samples, args.patch_size, args.max_delta_px, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PointResidualDataset(val_samples, args.patch_size, args.max_delta_px, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PointResidualRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = math.inf
    best_state = None
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            patch = batch["patch"].to(device, non_blocking=True)
            task_index = batch["task_index"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred = model(patch, task_index, meta)
            loss = F.smooth_l1_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_px = []
        with torch.no_grad():
            for batch in val_loader:
                patch = batch["patch"].to(device, non_blocking=True)
                task_index = batch["task_index"].to(device, non_blocking=True)
                meta = batch["meta"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                pred = model(patch, task_index, meta)
                val_losses.append(float(F.smooth_l1_loss(pred, target).detach().cpu()))
                val_px.append(float(torch.linalg.norm((pred - target) * args.max_delta_px, dim=1).mean().detach().cpu()))
        val_loss = float(np.mean(val_losses))
        val_err = float(np.mean(val_px))
        print(
            f"epoch={epoch:03d} train_loss={np.mean(train_losses):.6f} "
            f"val_loss={val_loss:.6f} val_residual_err_px={val_err:.3f}"
        )
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = {
        "state_dict": best_state,
        "meta": {
            "task_ids": sorted(task_ids) if task_ids is not None else TASKS,
            "patch_size": args.patch_size,
            "max_delta_px": args.max_delta_px,
            "train_pred_json": str(Path(args.train_pred_json).resolve()),
        },
    }
    torch.save(checkpoint, output_dir / "point_residual_refiner_v1.pth")
    print(f"Wrote {output_dir / 'point_residual_refiner_v1.pth'}")


def apply(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    meta = payload.get("meta", {})
    patch_size = int(args.patch_size or meta.get("patch_size", 64))
    max_delta_px = float(args.max_delta_px or meta.get("max_delta_px", 18.0))
    task_ids = parse_task_ids(args.task_ids)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PointResidualRefiner().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_json(Path(args.input_json))
    val_paths = load_validation_manifest(Path(args.manifest))
    data_root = Path(args.data_root)
    image_cache: dict[Path, np.ndarray] = {}

    refined = []
    shifts_by_task: dict[str, list[float]] = {}
    with torch.no_grad():
        for item in predictions:
            task_id = str(item["task_id"])
            image_path = normalize_image_path(str(item["image_path"]), task_id)
            if task_ids is not None and task_id not in task_ids:
                refined.append(dict(item))
                continue
            image_file = val_paths.get((task_id, image_path))
            if image_file is None:
                fallback = data_root / "images" / image_path
                if fallback.is_file():
                    image_file = fallback
                else:
                    refined.append(dict(item))
                    continue
            if image_file not in image_cache:
                image_cache[image_file] = read_gray(image_file)
            image = image_cache[image_file]
            points = coords(item, "predicted_points_pixels")
            width, height = infer_size_from_prediction(item)
            updated_points = points.copy()
            for point_index, pred_px in enumerate(points):
                patch_np = extract_patch(image, float(pred_px[0]), float(pred_px[1]), patch_size)
                meta_np = np.array(
                    [
                        pred_px[0] / max(width, 1.0),
                        pred_px[1] / max(height, 1.0),
                        float(point_index) / max(float(points.shape[0] - 1), 1.0),
                        float(points.shape[0]) / 24.0,
                        float(width) / max(float(height), 1.0),
                    ],
                    dtype=np.float32,
                )
                delta_norm = model(
                    torch.from_numpy(patch_np[None]).to(device),
                    torch.tensor([TASK_TO_INDEX[task_id]], dtype=torch.long, device=device),
                    torch.from_numpy(meta_np[None]).to(device),
                )[0].detach().cpu().numpy()
                delta_px = np.clip(delta_norm * max_delta_px * float(args.strength), -args.max_apply_px, args.max_apply_px)
                updated_points[point_index] = pred_px + delta_px.astype(np.float32)
            mean_shift = float(np.mean(np.linalg.norm(updated_points - points, axis=1)))
            if mean_shift > float(args.row_gate_px):
                updated_points = points
                mean_shift = 0.0
            shifts_by_task.setdefault(task_id, []).append(mean_shift)
            refined.append(update_prediction_item(item, updated_points))

    output_json = output_dir / "regression_predictions.json"
    output_json.write_text(json.dumps(refined, indent=2))
    with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_json, arcname="regression_predictions.json")
    summary = {
        task_id: {
            "count": len(values),
            "mean_shift_px": float(np.mean(values)),
            "max_shift_px": float(np.max(values)),
        }
        for task_id, values in sorted(shifts_by_task.items())
    }
    (output_dir / "point_residual_refiner_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_dir / 'submission.zip'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/apply point-wise image-patch residual refiner.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data-root", default="data")
    train_parser.add_argument("--train-pred-json", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--task-ids", default="A4C,AOP,FUGC,HC,IVC,PLAX,PSAX,fetal_femur")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--patch-size", type=int, default=64)
    train_parser.add_argument("--max-delta-px", type=float, default=18.0)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--val-fraction", type=float, default=0.15)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=train)

    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--checkpoint-path", required=True)
    apply_parser.add_argument("--input-json", required=True)
    apply_parser.add_argument("--data-root", default="data")
    apply_parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    apply_parser.add_argument("--output-dir", required=True)
    apply_parser.add_argument("--task-ids", default="A4C,AOP,FUGC,HC,IVC,PLAX,PSAX,fetal_femur")
    apply_parser.add_argument("--patch-size", type=int, default=None)
    apply_parser.add_argument("--max-delta-px", type=float, default=None)
    apply_parser.add_argument("--strength", type=float, default=0.35)
    apply_parser.add_argument("--max-apply-px", type=float, default=5.0)
    apply_parser.add_argument("--row-gate-px", type=float, default=5.0)
    apply_parser.add_argument("--cpu", action="store_true")
    apply_parser.set_defaults(func=apply)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
