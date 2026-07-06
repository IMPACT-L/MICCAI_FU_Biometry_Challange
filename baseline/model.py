import argparse
import glob
import json
import numpy as np
import os
import zipfile
from typing import Optional

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_factory import MultiTaskModelFactory
from utils import decode_heatmaps_to_normalized_coords, letterbox_image_and_points, transformed_coords_to_original_normalized


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
OUTPUT_DECIMALS = 6


def normalize_submission_image_path(image_path: str, task_id: str) -> str:
    normalized = os.path.normpath(str(image_path)).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if parts and parts[0] == task_id:
        return "/".join(parts)
    return f"{task_id}/{os.path.basename(normalized)}"


def round_float_list(values, decimals: int = OUTPUT_DECIMALS):
    return [round(float(value), decimals) for value in values]


def load_checkpoint_payload(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("meta", {})
    return checkpoint, {}


class InferenceDataset(Dataset):
    """Inference dataset for keypoint tasks only."""

    def __init__(
        self,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        split_csv: Optional[str] = None,
        input_size: int = 518,
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.input_size = input_size
        self.csv_path = os.path.join(self.data_root, "csv")
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        all_csv_files = glob.glob(os.path.join(self.csv_path, "*.csv"))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")

        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

        is_regression = dataframe["task_name"].astype(str).eq("Regression")
        is_extra_task = dataframe["task_id"].astype(str).isin(EXTRA_REGRESSION_TASK_IDS)
        self.dataframe = dataframe[is_regression | is_extra_task].reset_index(drop=True)
        if self.dataframe.empty:
            raise ValueError(
                "No keypoint records found. Expect task_name == 'Regression' or task_id in "
                f"{sorted(EXTRA_REGRESSION_TASK_IDS)}."
            )

        if split_csv is not None:
            if not os.path.exists(split_csv):
                raise FileNotFoundError(f"Split CSV not found: {split_csv}")

            split_df = pd.read_csv(split_csv)
            if "image_path" not in split_df.columns:
                raise ValueError(f"Split CSV must contain column 'image_path': {split_csv}")

            split_paths = set(split_df["image_path"].astype(str).tolist())
            self.dataframe = self.dataframe[self.dataframe["image_path"].astype(str).isin(split_paths)]
            self.dataframe = self.dataframe.reset_index(drop=True)

            if self.dataframe.empty:
                raise ValueError(
                    "No matching keypoint samples found after applying split CSV filter."
                )
            print(f"Applied split CSV filter: {split_csv}")

        print(f"Keypoint data loading complete. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, rel_path: str) -> Optional[str]:
        rel_norm = os.path.normpath(rel_path)
        cleaned_rel = rel_norm
        while cleaned_rel.startswith(".." + os.sep):
            cleaned_rel = cleaned_rel[3:]

        for root in [os.path.join(self.data_root, "images"), self.data_root]:
            direct = os.path.normpath(os.path.join(root, cleaned_rel))
            if os.path.isfile(direct):
                return direct

        return None

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record["task_id"]
        image_rel_path = record["image_path"]
        image_abs_path = self._resolve_image_path(image_rel_path)
        if image_abs_path is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.imread(image_abs_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        dummy_points = np.zeros((int(record["num_classes"]), 2), dtype=np.float32)
        image, _, meta = letterbox_image_and_points(image, dummy_points, self.input_size)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "task_id": task_id,
            "task_name": "Regression",
            "image_path": image_rel_path,
            "original_size": (original_height, original_width),
            "index": idx,
            "meta": meta,
        }


def inference_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], 0)
    return {
        "image": images,
        "task_id": [item["task_id"] for item in batch],
        "task_name": [item["task_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "index": [item["index"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


class Model:
    """Inference model for keypoint localization only."""

    def __init__(
        self,
        checkpoint_path: str = "best_model.pth",
        encoder_name: str = "vit_base_patch14_dinov2.lvd142m",
        use_fpn: Optional[bool] = None,
        fpn_mode: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.checkpoint_path = checkpoint_path
        self.encoder_name = encoder_name
        self.use_fpn = use_fpn
        self.fpn_mode = fpn_mode
        self.head_type = "basic"
        self.task_head_profile = "uniform"
        self.task_decoder_profile = "uniform"
        self.model = None
        self.task_configs = None
        self.task_id_to_name = None
        self.heatmap_size = (64, 64)
        self.input_size = 518

        self.transforms = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _build_task_configs(self, dataframe: pd.DataFrame):
        task_configs = []
        seen = set()
        for _, row in dataframe.iterrows():
            task_id = row["task_id"]
            if task_id in seen:
                continue
            seen.add(task_id)
            task_configs.append(
                {
                    "task_id": task_id,
                    "task_name": "Regression",
                    "num_classes": int(row["num_classes"]),
                }
            )
        return task_configs

    def _load_model(self):
        model_path = self.checkpoint_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint, checkpoint_meta = load_checkpoint_payload(model_path, self.device)
        checkpoint_has_shared_fpn = any(key.startswith("fpn.") for key in checkpoint.keys())
        checkpoint_has_task_fpns = any(key.startswith("task_fpns.") for key in checkpoint.keys())
        checkpoint_has_fpn = checkpoint_has_shared_fpn or checkpoint_has_task_fpns
        use_fpn = checkpoint_meta.get("use_fpn", checkpoint_has_fpn) if self.use_fpn is None else self.use_fpn
        inferred_fpn_mode = checkpoint_meta.get(
            "fpn_mode",
            "task_specific" if checkpoint_has_task_fpns else "shared",
        )
        self.fpn_mode = inferred_fpn_mode if self.fpn_mode is None else self.fpn_mode
        self.encoder_name = checkpoint_meta.get("encoder_name", self.encoder_name)
        self.head_type = checkpoint_meta.get("head_type", self.head_type)
        self.task_head_profile = checkpoint_meta.get("task_head_profile", self.task_head_profile)
        self.task_decoder_profile = checkpoint_meta.get(
            "task_decoder_profile",
            self.task_decoder_profile,
        )
        self.input_size = int(checkpoint_meta.get("input_size", self.input_size))
        self.heatmap_size = tuple(checkpoint_meta.get("heatmap_size", list(self.heatmap_size)))
        print(
            "Checkpoint architecture: "
            f"encoder={self.encoder_name}, "
            f"head={self.head_type}, "
            f"task_head_profile={self.task_head_profile}, "
            f"task_decoder_profile={self.task_decoder_profile}, "
            f"fpn_mode={self.fpn_mode}, "
            f"heatmap_size={self.heatmap_size}, "
            f"FPN {'ENABLED' if checkpoint_has_fpn else 'DISABLED'}; "
            f"loading model with FPN {'ENABLED' if use_fpn else 'DISABLED'}"
        )

        self.model = MultiTaskModelFactory(
            encoder_name=self.encoder_name,
            encoder_weights="pretrained",
            task_configs=self.task_configs,
            heatmap_size=self.heatmap_size,
            use_fpn=use_fpn,
            fpn_mode=self.fpn_mode,
            head_type=self.head_type,
            task_head_profile=self.task_head_profile,
            task_decoder_profile=self.task_decoder_profile,
        ).to(self.device)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(
        self,
        data_root: str,
        output_dir: str,
        batch_size: int = 8,
        split_csv: Optional[str] = None,
        output_filename: str = "landmark_predictions.json",
    ):
        print("=" * 60)
        print("Starting keypoint prediction...")
        print(f"Data directory: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model file not found: {self.checkpoint_path}")
        _, checkpoint_meta = load_checkpoint_payload(self.checkpoint_path, self.device)
        self.input_size = int(checkpoint_meta.get("input_size", self.input_size))
        dataset = InferenceDataset(
            data_root=data_root,
            transforms=self.transforms,
            split_csv=split_csv,
            input_size=self.input_size,
        )

        self.task_configs = self._build_task_configs(dataset.dataframe)
        self.task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in self.task_configs}
        self._load_model()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=inference_collate_fn,
        )

        regression_results = []
        task_counts = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch["image"].to(self.device)
                task_ids = batch["task_id"]
                image_paths = batch["image_path"]
                original_sizes = batch["original_size"]
                meta = batch["meta"]

                unique_tasks = list(set(task_ids))
                for task_id in unique_tasks:
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]
                    pred_logits = self.model(task_images, task_id=task_id)
                    pred_heatmaps = torch.sigmoid(pred_logits)
                    outputs_transformed = decode_heatmaps_to_normalized_coords(pred_heatmaps)
                    outputs = transformed_coords_to_original_normalized(
                        outputs_transformed,
                        [meta[i] for i in task_indices],
                    )

                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        original_size = original_sizes[batch_idx]
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                        regression_results.append(
                            self._process_regression(pred, task_id, image_path, original_size)
                        )

        json_path = os.path.join(output_dir, output_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(regression_results, f, indent=2, ensure_ascii=False)

        print(f"Saved keypoint predictions: {json_path} ({len(regression_results)} samples)")
        print("Prediction count by task:")
        for task_id in sorted(task_counts.keys()):
            print(f"  - {task_id}: {task_counts[task_id]} samples")
        return json_path

    def _process_regression(self, pred, task_id, image_path, original_size):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        coords = round_float_list(pred.flatten().tolist())
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i + 1]
            pixel_coords.extend([x_norm * w, y_norm * h])
        pixel_coords = round_float_list(pixel_coords)

        return {
            "image_path": normalize_submission_image_path(image_path, task_id),
            "task_id": task_id,
            "predicted_points_normalized": coords,
            "predicted_points_pixels": pixel_coords,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint prediction script")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="predictions/", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for full-dataset prediction")
    parser.add_argument("--split-csv", type=str, default=None, help="Optional split CSV path to restrict prediction set")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="best_model.pth",
        help="Path to the trained checkpoint to load.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="regression_predictions.json",
        help="Prediction JSON filename for challenge submission.",
    )
    parser.add_argument(
        "--zip-submission",
        action="store_true",
        help="Also create submission.zip containing the prediction JSON.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="vit_base_patch14_dinov2.lvd142m",
        help="Backbone name used for model construction.",
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
    parser.add_argument(
        "--fpn-mode",
        type=str,
        choices=("shared", "task_specific"),
        default=None,
        help="Optional FPN mode override. If omitted, inferred from checkpoint.",
    )
    args = parser.parse_args()

    model = Model(
        checkpoint_path=args.checkpoint_path,
        encoder_name=args.encoder_name,
        use_fpn=args.use_fpn,
        fpn_mode=args.fpn_mode,
    )
    json_path = model.predict(
        args.data_root,
        args.output_dir,
        batch_size=args.batch_size,
        split_csv=args.split_csv,
        output_filename=args.output_filename,
    )

    if args.zip_submission:
        zip_path = os.path.join(args.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=os.path.basename(json_path))
        print(f"Saved submission archive: {zip_path}")

    print("Inference complete!")
