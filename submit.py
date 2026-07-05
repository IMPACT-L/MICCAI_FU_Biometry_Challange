import argparse
import csv
import json
import os
import zipfile
from collections import Counter

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baseline.model_factory import MultiTaskModelFactory
from baseline.utils import (
    decode_heatmaps_to_normalized_coords,
    letterbox_image_and_points,
    transformed_coords_to_original_normalized,
)


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


class ValidationManifestDataset(Dataset):
    def __init__(self, manifest_path: str, input_size: int = 518):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        self.input_size = input_size
        with open(self.manifest_path, newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Validation manifest is empty: {manifest_path}")

        self.transforms = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        abs_path = row["abs_path"]
        if not os.path.isabs(abs_path):
            abs_path = os.path.normpath(os.path.join(self.manifest_dir, abs_path))

        image = cv2.imread(abs_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read validation image: {abs_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        dummy_points = np.zeros((int(row["num_points"]), 2), dtype=np.float32)
        image, _, meta = letterbox_image_and_points(image, dummy_points, self.input_size)
        transformed = self.transforms(image=image)

        return {
            "image": transformed["image"],
            "task_id": row["task_id"],
            "image_path": row["image_path"],
            "original_size": (original_height, original_width),
            "num_points": int(row["num_points"]),
            "meta": meta,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch], 0),
        "task_id": [item["task_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "num_points": [item["num_points"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def build_task_configs(manifest_path: str):
    df = pd.read_csv(manifest_path)
    configs = []
    seen = set()
    for _, row in df.iterrows():
        task_id = str(row["task_id"])
        if task_id in seen:
            continue
        seen.add(task_id)
        configs.append(
            {
                "task_id": task_id,
                "task_name": "Regression",
                "num_classes": int(row["num_points"]),
            }
        )
    return configs


def validate_predictions(predictions):
    counts = Counter(item["task_id"] for item in predictions)
    if len(predictions) != sum(EXPECTED_VALIDATION_COUNTS.values()):
        raise ValueError(
            f"Submission row count mismatch: got {len(predictions)}, "
            f"expected {sum(EXPECTED_VALIDATION_COUNTS.values())}"
        )
    if dict(counts) != EXPECTED_VALIDATION_COUNTS:
        raise ValueError(
            f"Submission task counts mismatch: got {dict(sorted(counts.items()))}, "
            f"expected {EXPECTED_VALIDATION_COUNTS}"
        )

    seen = set()
    for item in predictions:
        key = (item["task_id"], item["image_path"])
        if key in seen:
            raise ValueError(f"Duplicate prediction entry found: {key}")
        seen.add(key)


def infer_model_config_from_checkpoint(checkpoint: dict) -> tuple[str, bool]:
    checkpoint_has_fpn = any(key.startswith("fpn.") for key in checkpoint.keys())
    head_key = next((key for key in checkpoint.keys() if key.endswith("decoder.0.weight")), None)
    if head_key is None:
        raise ValueError("Could not infer model config from checkpoint: missing head weights.")
    in_channels = int(checkpoint[head_key].shape[1])
    if checkpoint_has_fpn:
        if in_channels != 256:
            raise ValueError(f"Unexpected FPN head width in checkpoint: {in_channels}")
        return "vit_base_patch14_dinov2.lvd142m", True
    if in_channels == 384:
        return "vit_small_patch14_dinov2.lvd142m", False
    if in_channels == 768:
        return "vit_base_patch14_dinov2.lvd142m", False
    raise ValueError(f"Unsupported checkpoint head width: {in_channels}")


def main():
    parser = argparse.ArgumentParser(description="Generate a challenge-compliant submission archive.")
    parser.add_argument(
        "--manifest",
        default="data/manifests/validation_manifest.csv",
        help="Validation manifest CSV with the official 619 challenge rows.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="output_real_run/checkpoints/best_model.pth",
        help="Trained checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="submission_output",
        help="Directory to write regression_predictions.json and submission.zip.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--encoder-name",
        default=None,
        help="Optional backbone override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--fpn",
        dest="use_fpn",
        action="store_true",
        help="Force FPN-enabled model construction.",
    )
    parser.add_argument(
        "--no-fpn",
        dest="use_fpn",
        action="store_false",
        help="Force no-FPN model construction.",
    )
    parser.set_defaults(use_fpn=None)
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = ValidationManifestDataset(manifest_path=manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    task_configs = build_task_configs(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    inferred_encoder_name, inferred_use_fpn = infer_model_config_from_checkpoint(checkpoint)
    encoder_name = args.encoder_name or inferred_encoder_name
    use_fpn = inferred_use_fpn if args.use_fpn is None else args.use_fpn

    print(
        f"Checkpoint architecture: backbone={inferred_encoder_name}, "
        f"FPN={'ENABLED' if inferred_use_fpn else 'DISABLED'}"
    )
    print(
        f"Submission model config: backbone={encoder_name}, "
        f"FPN={'ENABLED' if use_fpn else 'DISABLED'}"
    )

    model = MultiTaskModelFactory(
        encoder_name=encoder_name,
        encoder_weights="pretrained",
        task_configs=task_configs,
        heatmap_size=(64, 64),
        use_fpn=use_fpn,
    ).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Submission inference"):
            images = batch["image"].to(device)
            task_ids = batch["task_id"]
            meta = batch["meta"]

            for task_id in sorted(set(task_ids)):
                task_indices = [i for i, value in enumerate(task_ids) if value == task_id]
                task_images = images[task_indices]
                pred_logits = model(task_images, task_id=task_id)
                pred_heatmaps = torch.sigmoid(pred_logits)
                outputs_transformed = decode_heatmaps_to_normalized_coords(pred_heatmaps)
                outputs = transformed_coords_to_original_normalized(
                    outputs_transformed,
                    [meta[i] for i in task_indices],
                )

                for output_idx, batch_idx in enumerate(task_indices):
                    pred = outputs[output_idx].cpu().numpy().tolist()
                    original_height, original_width = batch["original_size"][batch_idx]
                    pixel_coords = []
                    for point_idx in range(0, len(pred), 2):
                        x_norm = float(pred[point_idx])
                        y_norm = float(pred[point_idx + 1])
                        pixel_coords.extend([x_norm * original_width, y_norm * original_height])

                    predictions.append(
                        {
                            "image_path": batch["image_path"][batch_idx],
                            "task_id": task_id,
                            "predicted_points_normalized": pred,
                            "predicted_points_pixels": pixel_coords,
                        }
                    )

    validate_predictions(predictions)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "regression_predictions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="regression_predictions.json")

    print(f"Wrote {json_path}")
    print(f"Wrote {zip_path}")
    print(f"Validated counts: {EXPECTED_VALIDATION_COUNTS}")


if __name__ == "__main__":
    main()
