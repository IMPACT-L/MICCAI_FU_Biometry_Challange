import argparse
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

from csv_utils import collect_effective_train_csvs
from model_factory import TASK_ADAPTER_PROFILE_PRESETS, TASK_DECODER_PROFILE_PRESETS, MultiTaskModelFactory
from model_profiles import MODEL_PROFILE_NAMES, apply_model_profile
from utils import (
    canonicalize_task_coords,
    CONTENT_CROP_MODES,
    content_crop_image_and_points,
    content_crop_enabled_for_task,
    content_crop_pad_ratio,
    decode_heatmaps_to_normalized_coords,
    letterbox_image_and_points,
    transformed_coords_to_original_normalized,
    update_letterbox_meta_for_crop,
)


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
OUTPUT_DECIMALS = 6
DEFAULT_TTA_TASK_IDS = {"AOP", "FA", "FUGC", "HC", "IVC", "PSAX", "fetal_femur"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)


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


def parse_task_id_csv(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    task_ids = {item.strip() for item in str(value).split(",") if item.strip()}
    return task_ids or None


def _unnormalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images * std + mean).clamp(0.0, 1.0)


def _normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images.clamp(0.0, 1.0) - mean) / std


def _apply_clahe_rgb(image_rgb_uint8: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _apply_gamma_rgb(image_rgb_uint8: np.ndarray, gamma: float) -> np.ndarray:
    image_float = image_rgb_uint8.astype(np.float32) / 255.0
    corrected = np.power(np.clip(image_float, 0.0, 1.0), gamma)
    return np.clip(corrected * 255.0, 0.0, 255.0).astype(np.uint8)


def build_photometric_variants(images: torch.Tensor) -> list[torch.Tensor]:
    base = _unnormalize_batch(images).detach().cpu()
    base_np = (base.permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

    variants_np = [
        base_np,
        np.stack([_apply_clahe_rgb(img, clip_limit=2.0) for img in base_np], axis=0),
        np.stack([_apply_gamma_rgb(img, gamma=0.85) for img in base_np], axis=0),
        np.stack([_apply_gamma_rgb(img, gamma=1.15) for img in base_np], axis=0),
    ]

    variants = []
    for variant_np in variants_np:
        variant = torch.from_numpy(variant_np).permute(0, 3, 1, 2).to(dtype=images.dtype, device=images.device) / 255.0
        variants.append(_normalize_batch(variant))
    return variants


def load_checkpoint_payload(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("meta", {})
    return checkpoint, {}


def infer_num_domain_classes(checkpoint: dict) -> int:
    for key, value in checkpoint.items():
        if key == "domain_classifier.classifier.5.weight" and getattr(value, "ndim", 0) == 2:
            return int(value.shape[0])
    return 0


class InferenceDataset(Dataset):
    """Inference dataset for keypoint tasks only."""

    def __init__(
        self,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        split_csv: Optional[str] = None,
        input_size: int = 518,
        input_crop_mode: str = "none",
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.input_size = input_size
        if str(input_crop_mode) not in {"none", *CONTENT_CROP_MODES}:
            raise ValueError(f"Unsupported input_crop_mode: {input_crop_mode}")
        self.input_crop_mode = str(input_crop_mode)
        self.csv_path = os.path.join(self.data_root, "csv")
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        all_csv_files = collect_effective_train_csvs(self.data_root, self.csv_path)
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")

        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

        is_regression = dataframe["task_name"].astype(str).eq("Regression")
        is_extra_task = dataframe["task_id"].astype(str).isin(EXTRA_REGRESSION_TASK_IDS)
        self.dataframe = dataframe[is_regression | is_extra_task].reset_index(drop=True)
        self.full_dataframe = self.dataframe.copy()
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
        content_crop_box = None
        if content_crop_enabled_for_task(self.input_crop_mode, task_id):
            image, _, content_crop_box = content_crop_image_and_points(
                image,
                None,
                pad_ratio=content_crop_pad_ratio(self.input_crop_mode),
            )
        dummy_points = np.zeros((int(record["num_classes"]), 2), dtype=np.float32)
        image, _, meta = letterbox_image_and_points(image, dummy_points, self.input_size)
        if content_crop_box is not None:
            meta = update_letterbox_meta_for_crop(
                meta,
                content_crop_box,
                original_size=(original_height, original_width),
            )

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
        encoder_feature_mode: Optional[str] = None,
        model_profile: Optional[str] = None,
        use_fpn: Optional[bool] = None,
        fpn_mode: Optional[str] = None,
        fpn_type: Optional[str] = None,
        task_head_profile: Optional[str] = None,
        task_decoder_profile: Optional[str] = None,
        task_adapter_profile: Optional[str] = None,
        input_crop_mode: Optional[str] = None,
        tta_mode: str = "none",
        tta_task_ids: Optional[set[str]] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.checkpoint_path = checkpoint_path
        self.model_profile = model_profile
        self.encoder_name = encoder_name
        self.encoder_feature_mode = encoder_feature_mode
        self.use_fpn = use_fpn
        self.fpn_mode = fpn_mode
        self.fpn_type = fpn_type
        self.head_type = "basic"
        self.task_head_profile = task_head_profile
        self.task_decoder_profile = task_decoder_profile
        self.task_adapter_profile = task_adapter_profile
        if input_crop_mode is not None and str(input_crop_mode) not in {"none", *CONTENT_CROP_MODES}:
            raise ValueError(f"Unsupported input_crop_mode: {input_crop_mode}")
        self.input_crop_mode = input_crop_mode
        self.tta_mode = str(tta_mode)
        self.tta_task_ids = tta_task_ids
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

    def _forward_task_coords(self, task_images: torch.Tensor, task_id: str) -> torch.Tensor:
        model_output = self.model(task_images, task_id=task_id, return_prior=True)
        aux_outputs = {}
        if len(model_output) == 3:
            _, pred_heatmaps, aux_outputs = model_output
        else:
            _, pred_heatmaps = model_output

        refined_coords = aux_outputs.get("refined_coords_transformed")
        return refined_coords if refined_coords is not None else decode_heatmaps_to_normalized_coords(pred_heatmaps)

    def _predict_task_transformed_coords(self, task_images: torch.Tensor, task_id: str) -> torch.Tensor:
        apply_tta = self.tta_mode != "none" and (self.tta_task_ids is None or task_id in self.tta_task_ids)
        use_hflip = self.tta_mode in {"hflip", "hflip_photometric"}
        use_photometric = self.tta_mode in {"photometric", "hflip_photometric"}

        def predict_variant(images_variant: torch.Tensor) -> torch.Tensor:
            if not use_photometric:
                return self._forward_task_coords(images_variant, task_id)
            variant_preds = [self._forward_task_coords(variant, task_id) for variant in build_photometric_variants(images_variant)]
            return torch.stack(variant_preds, dim=0).mean(dim=0)

        if not apply_tta:
            return self._forward_task_coords(task_images, task_id)

        base_coords = canonicalize_task_coords(predict_variant(task_images), task_id)
        if not use_hflip:
            return base_coords

        flipped_images = torch.flip(task_images, dims=[-1])
        flip_coords = predict_variant(flipped_images).clone()
        flip_coords[:, 0::2] = 1.0 - flip_coords[:, 0::2]
        flip_coords = canonicalize_task_coords(flip_coords, task_id)
        return 0.5 * (base_coords + flip_coords)

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
        checkpoint_has_domain_classifier = any(key.startswith("domain_classifier.") for key in checkpoint.keys())
        checkpoint_has_fpn = checkpoint_has_shared_fpn or checkpoint_has_task_fpns
        use_fpn = checkpoint_meta.get("use_fpn", checkpoint_has_fpn) if self.use_fpn is None else self.use_fpn
        domain_adversarial = bool(checkpoint_meta.get("domain_adversarial", checkpoint_has_domain_classifier))
        num_domain_classes = int(checkpoint_meta.get("num_domain_classes", infer_num_domain_classes(checkpoint)))
        inferred_fpn_mode = checkpoint_meta.get(
            "fpn_mode",
            "task_specific" if checkpoint_has_task_fpns else "shared",
        )
        inferred_fpn_type = checkpoint_meta.get("fpn_type", "fpn")
        self.fpn_mode = inferred_fpn_mode if self.fpn_mode is None else self.fpn_mode
        self.fpn_type = inferred_fpn_type if self.fpn_type is None else self.fpn_type
        self.encoder_name = checkpoint_meta.get("encoder_name", self.encoder_name)
        self.encoder_feature_mode = (
            self.encoder_feature_mode
            if self.encoder_feature_mode is not None
            else checkpoint_meta.get("encoder_feature_mode", "final")
        )
        self.head_type = checkpoint_meta.get("head_type", self.head_type)
        self.task_head_profile = (
            self.task_head_profile
            if self.task_head_profile is not None
            else checkpoint_meta.get("task_head_profile", "uniform")
        )
        self.task_decoder_profile = (
            self.task_decoder_profile
            if self.task_decoder_profile is not None
            else checkpoint_meta.get("task_decoder_profile", "uniform")
        )
        self.task_adapter_profile = (
            self.task_adapter_profile
            if self.task_adapter_profile is not None
            else checkpoint_meta.get("task_adapter_profile", "uniform")
        )
        self.input_crop_mode = (
            self.input_crop_mode
            if self.input_crop_mode is not None
            else checkpoint_meta.get("input_crop_mode", "none")
        )
        if self.model_profile is not None:
            profile_config = apply_model_profile(
                self.model_profile,
                "inference",
                {
                    "encoder_name": self.encoder_name,
                    "encoder_feature_mode": self.encoder_feature_mode,
                    "use_fpn": use_fpn,
                    "fpn_mode": self.fpn_mode,
                    "fpn_type": self.fpn_type,
                    "task_head_profile": self.task_head_profile,
                    "task_decoder_profile": self.task_decoder_profile,
                    "task_adapter_profile": self.task_adapter_profile,
                    "input_crop_mode": self.input_crop_mode,
                },
            )
            self.encoder_name = str(profile_config["encoder_name"])
            self.encoder_feature_mode = str(profile_config.get("encoder_feature_mode", self.encoder_feature_mode))
            use_fpn = bool(profile_config["use_fpn"])
            self.fpn_mode = str(profile_config["fpn_mode"])
            self.fpn_type = str(profile_config["fpn_type"])
            self.task_head_profile = str(profile_config["task_head_profile"])
            self.task_decoder_profile = str(profile_config["task_decoder_profile"])
            self.task_adapter_profile = str(profile_config["task_adapter_profile"])
            self.input_crop_mode = str(profile_config.get("input_crop_mode", self.input_crop_mode))
        self.input_size = int(checkpoint_meta.get("input_size", self.input_size))
        self.heatmap_size = tuple(checkpoint_meta.get("heatmap_size", list(self.heatmap_size)))
        encoder_lora_rank = int(checkpoint_meta.get("encoder_lora_rank", 0))
        encoder_lora_alpha = float(checkpoint_meta.get("encoder_lora_alpha", 16.0))
        encoder_lora_last_blocks = int(checkpoint_meta.get("encoder_lora_last_blocks", 4))
        encoder_lora_dropout = float(checkpoint_meta.get("encoder_lora_dropout", 0.05))
        encoder_lora_task_specific = bool(checkpoint_meta.get("encoder_lora_task_specific", False))
        print(
            "Checkpoint architecture: "
            f"encoder={self.encoder_name}, "
            f"encoder_feature_mode={self.encoder_feature_mode}, "
            f"head={self.head_type}, "
            f"task_head_profile={self.task_head_profile}, "
            f"task_decoder_profile={self.task_decoder_profile}, "
            f"task_adapter_profile={self.task_adapter_profile}, "
            f"input_crop_mode={self.input_crop_mode}, "
            f"fpn_mode={self.fpn_mode}, "
            f"fpn_type={self.fpn_type}, "
            f"heatmap_size={self.heatmap_size}, "
            f"encoder_lora_rank={encoder_lora_rank}, "
            f"encoder_lora_task_specific={encoder_lora_task_specific}, "
            f"domain_adversarial={domain_adversarial}, "
            f"FPN {'ENABLED' if checkpoint_has_fpn else 'DISABLED'}; "
            f"loading model with FPN {'ENABLED' if use_fpn else 'DISABLED'}"
        )
        print(f"Model profile: {self.model_profile if self.model_profile is not None else 'checkpoint/manual'}")
        if self.tta_mode != "none":
            print(
                f"Inference TTA: {self.tta_mode} "
                f"(tasks={sorted(self.tta_task_ids) if self.tta_task_ids is not None else 'all'})"
            )

        self.model = MultiTaskModelFactory(
            encoder_name=self.encoder_name,
            encoder_weights="pretrained",
            encoder_feature_mode=self.encoder_feature_mode,
            task_configs=self.task_configs,
            heatmap_size=self.heatmap_size,
            use_fpn=use_fpn,
            fpn_mode=self.fpn_mode,
            fpn_type=self.fpn_type,
            head_type=self.head_type,
            task_head_profile=self.task_head_profile,
            task_decoder_profile=self.task_decoder_profile,
            task_adapter_profile=self.task_adapter_profile,
            encoder_lora_rank=encoder_lora_rank,
            encoder_lora_alpha=encoder_lora_alpha,
            encoder_lora_last_blocks=encoder_lora_last_blocks,
            encoder_lora_dropout=encoder_lora_dropout,
            encoder_lora_task_specific=encoder_lora_task_specific,
            domain_adversarial=domain_adversarial,
            num_domain_classes=num_domain_classes,
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
        self.input_crop_mode = (
            self.input_crop_mode
            if self.input_crop_mode is not None
            else str(checkpoint_meta.get("input_crop_mode", "none"))
        )
        if self.model_profile is not None:
            profile_config = apply_model_profile(
                self.model_profile,
                "inference",
                {
                    "encoder_name": checkpoint_meta.get("encoder_name", self.encoder_name),
                    "encoder_feature_mode": checkpoint_meta.get("encoder_feature_mode", self.encoder_feature_mode or "final"),
                    "use_fpn": checkpoint_meta.get("use_fpn", True),
                    "fpn_mode": checkpoint_meta.get("fpn_mode", self.fpn_mode or "task_specific"),
                    "fpn_type": checkpoint_meta.get("fpn_type", self.fpn_type or "fpn"),
                    "task_head_profile": checkpoint_meta.get("task_head_profile", self.task_head_profile or "uniform"),
                    "task_decoder_profile": checkpoint_meta.get("task_decoder_profile", self.task_decoder_profile or "uniform"),
                    "task_adapter_profile": checkpoint_meta.get("task_adapter_profile", self.task_adapter_profile or "uniform"),
                    "input_crop_mode": self.input_crop_mode,
                },
            )
            self.input_crop_mode = str(profile_config.get("input_crop_mode", self.input_crop_mode))
        dataset = InferenceDataset(
            data_root=data_root,
            transforms=self.transforms,
            split_csv=split_csv,
            input_size=self.input_size,
            input_crop_mode=self.input_crop_mode,
        )

        # A split CSV may contain only one task, but checkpoints usually contain all
        # task-specific heads/FPNs. Build from the full dataset so strict loading
        # remains compatible, while inference still iterates only the split rows.
        self.task_configs = self._build_task_configs(dataset.full_dataframe)
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
                    outputs_transformed = self._predict_task_transformed_coords(task_images, task_id)
                    outputs = transformed_coords_to_original_normalized(
                        outputs_transformed,
                        [meta[i] for i in task_indices],
                    )
                    outputs = canonicalize_task_coords(outputs, task_id)

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
        "--model-profile",
        type=str,
        choices=MODEL_PROFILE_NAMES,
        default=None,
        help="Named inference architecture preset. When provided, it overrides checkpoint-inferred architecture fields.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="vit_base_patch14_dinov2.lvd142m",
        help="Backbone name used for model construction.",
    )
    parser.add_argument(
        "--encoder-feature-mode",
        type=str,
        choices=("final", "multilayer_fusion_v1", "feature_pyramid_fusion_v1"),
        default=None,
        help="Optional backbone feature extraction mode override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--task-head-profile",
        type=str,
        choices=("uniform", "challenge_legacy_v1", "challenge_v1"),
        default=None,
        help="Optional task-specific head sizing override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--task-decoder-profile",
        type=str,
        choices=tuple(TASK_DECODER_PROFILE_PRESETS.keys()),
        default=None,
        help="Optional task-specific decoder-family override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--task-adapter-profile",
        type=str,
        choices=tuple(TASK_ADAPTER_PROFILE_PRESETS.keys()),
        default=None,
        help="Optional task-specific feature adapter override. If omitted, inferred from checkpoint.",
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
    parser.add_argument(
        "--fpn-type",
        type=str,
        choices=("fpn", "bifpn"),
        default=None,
        help="Optional FPN neck type override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--input-crop-mode",
        type=str,
        choices=(
            "none",
            "content_box",
            "content_box_strict",
            "content_box_wide",
            "content_box_wide_strict",
            "content_box_fugc_wide",
            "content_box_fugc_xwide",
        ),
        default=None,
        help="Optional input crop override. If omitted, inferred from checkpoint metadata.",
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        choices=("none", "hflip", "photometric", "hflip_photometric"),
        default="none",
        help="Inference-time test-time augmentation mode.",
    )
    parser.add_argument(
        "--tta-task-ids",
        type=str,
        default="AOP,FA,FUGC,HC,IVC,PSAX,fetal_femur",
        help="Comma-separated task IDs to apply TTA to. Ignored when --tta-mode none.",
    )
    args = parser.parse_args()

    model = Model(
        checkpoint_path=args.checkpoint_path,
        encoder_name=args.encoder_name,
        encoder_feature_mode=args.encoder_feature_mode,
        model_profile=args.model_profile,
        use_fpn=args.use_fpn,
        fpn_mode=args.fpn_mode,
        fpn_type=args.fpn_type,
        task_head_profile=args.task_head_profile,
        task_decoder_profile=args.task_decoder_profile,
        task_adapter_profile=args.task_adapter_profile,
        input_crop_mode=args.input_crop_mode,
        tta_mode=args.tta_mode,
        tta_task_ids=(DEFAULT_TTA_TASK_IDS if args.tta_mode == "hflip" and args.tta_task_ids == "AOP,FA,FUGC,HC,IVC,PSAX,fetal_femur" else parse_task_id_csv(args.tta_task_ids)),
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
