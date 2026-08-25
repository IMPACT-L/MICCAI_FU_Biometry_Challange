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

from baseline.model_factory import (
    TASK_ADAPTER_PROFILE_PRESETS,
    TASK_DECODER_PROFILE_PRESETS,
    MultiTaskModelFactory,
)
from baseline.model_profiles import MODEL_PROFILE_NAMES, apply_model_profile
from baseline.utils import (
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
OUTPUT_DECIMALS = 6
DEFAULT_TTA_TASK_IDS = {"AOP", "FA", "FUGC", "HC", "IVC", "PSAX", "fetal_femur"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)


def round_float_list(values, decimals: int = OUTPUT_DECIMALS):
    return [round(float(value), decimals) for value in values]


def parse_task_id_csv(value: str | None) -> set[str] | None:
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


class ValidationManifestDataset(Dataset):
    def __init__(self, manifest_path: str, input_size: int = 518, input_crop_mode: str = "none"):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        self.input_size = input_size
        if str(input_crop_mode) not in {"none", *CONTENT_CROP_MODES}:
            raise ValueError(f"Unsupported input_crop_mode: {input_crop_mode}")
        self.input_crop_mode = str(input_crop_mode)
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
        content_crop_box = None
        if content_crop_enabled_for_task(self.input_crop_mode, row["task_id"]):
            image, _, content_crop_box = content_crop_image_and_points(
                image,
                None,
                pad_ratio=content_crop_pad_ratio(self.input_crop_mode),
            )
        dummy_points = np.zeros((int(row["num_points"]), 2), dtype=np.float32)
        image, _, meta = letterbox_image_and_points(image, dummy_points, self.input_size)
        if content_crop_box is not None:
            meta = update_letterbox_meta_for_crop(
                meta,
                content_crop_box,
                original_size=(original_height, original_width),
            )
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


def validate_predictions(predictions, expected_counts: dict[str, int] | None = None):
    counts = Counter(item["task_id"] for item in predictions)
    if expected_counts is not None and len(predictions) != sum(expected_counts.values()):
        raise ValueError(
            f"Submission row count mismatch: got {len(predictions)}, "
            f"expected {sum(expected_counts.values())}"
        )
    if expected_counts is not None and dict(counts) != expected_counts:
        raise ValueError(
            f"Submission task counts mismatch: got {dict(sorted(counts.items()))}, "
            f"expected {expected_counts}"
        )

    seen = set()
    for item in predictions:
        key = (item["task_id"], item["image_path"])
        if key in seen:
            raise ValueError(f"Duplicate prediction entry found: {key}")
        seen.add(key)


def infer_model_config_from_checkpoint(checkpoint: dict, checkpoint_meta: dict) -> tuple[str, str, bool, str, str, str, str, str, str, int, tuple, str]:
    meta_encoder_name = checkpoint_meta.get("encoder_name")
    meta_encoder_feature_mode = checkpoint_meta.get("encoder_feature_mode", "final")
    meta_use_fpn = checkpoint_meta.get("use_fpn")
    meta_fpn_mode = checkpoint_meta.get("fpn_mode", "shared")
    meta_fpn_type = checkpoint_meta.get("fpn_type", "fpn")
    meta_head_type = checkpoint_meta.get("head_type", "basic")
    meta_task_head_profile = checkpoint_meta.get("task_head_profile", "uniform")
    meta_task_decoder_profile = checkpoint_meta.get("task_decoder_profile", "uniform")
    meta_task_adapter_profile = checkpoint_meta.get("task_adapter_profile", "uniform")
    meta_input_size = int(checkpoint_meta.get("input_size", 518))
    meta_heatmap_size = tuple(checkpoint_meta.get("heatmap_size", [64, 64]))
    meta_input_crop_mode = str(checkpoint_meta.get("input_crop_mode", "none"))
    if meta_encoder_name is not None and meta_use_fpn is not None:
        return (
            str(meta_encoder_name),
            str(meta_encoder_feature_mode),
            bool(meta_use_fpn),
            str(meta_fpn_mode),
            str(meta_fpn_type),
            str(meta_head_type),
            str(meta_task_head_profile),
            str(meta_task_decoder_profile),
            str(meta_task_adapter_profile),
            meta_input_size,
            meta_heatmap_size,
            meta_input_crop_mode,
        )

    checkpoint_has_shared_fpn = any(key.startswith("fpn.") for key in checkpoint.keys())
    checkpoint_has_task_fpns = any(key.startswith("task_fpns.") for key in checkpoint.keys())
    checkpoint_has_fpn = checkpoint_has_shared_fpn or checkpoint_has_task_fpns
    inferred_fpn_mode = "task_specific" if checkpoint_has_task_fpns else "shared"
    head_key = next((key for key in checkpoint.keys() if key.endswith("decoder.0.weight")), None)
    if head_key is None:
        raise ValueError("Could not infer model config from checkpoint: missing head weights.")
    in_channels = int(checkpoint[head_key].shape[1])
    if checkpoint_has_fpn:
        if in_channels != 256:
            raise ValueError(f"Unexpected FPN head width in checkpoint: {in_channels}")
        return "vit_base_patch14_dinov2.lvd142m", "final", True, inferred_fpn_mode, "fpn", "basic", "uniform", "uniform", "uniform", meta_input_size, meta_heatmap_size, meta_input_crop_mode
    if in_channels == 384:
        return "vit_small_patch14_dinov2.lvd142m", "final", False, "shared", "fpn", "basic", "uniform", "uniform", "uniform", meta_input_size, meta_heatmap_size, meta_input_crop_mode
    if in_channels == 768:
        return "vit_base_patch14_dinov2.lvd142m", "final", False, "shared", "fpn", "basic", "uniform", "uniform", "uniform", meta_input_size, meta_heatmap_size, meta_input_crop_mode
    raise ValueError(f"Unsupported checkpoint head width: {in_channels}")


def forward_task_coords(model, task_images, task_id: str):
    model_output = model(task_images, task_id=task_id, return_prior=True)
    aux_outputs = {}
    if len(model_output) == 3:
        _, pred_heatmaps, aux_outputs = model_output
    else:
        _, pred_heatmaps = model_output

    refined_coords = aux_outputs.get("refined_coords_transformed")
    return refined_coords if refined_coords is not None else decode_heatmaps_to_normalized_coords(pred_heatmaps)


def predict_task_transformed_coords(model, task_images, task_id: str, tta_mode: str, tta_task_ids: set[str] | None):
    apply_tta = tta_mode != "none" and (tta_task_ids is None or task_id in tta_task_ids)
    use_hflip = tta_mode in {"hflip", "hflip_photometric"}
    use_photometric = tta_mode in {"photometric", "hflip_photometric"}

    def predict_variant(images_variant: torch.Tensor) -> torch.Tensor:
        if not use_photometric:
            return forward_task_coords(model, images_variant, task_id)
        variant_preds = [forward_task_coords(model, variant, task_id) for variant in build_photometric_variants(images_variant)]
        return torch.stack(variant_preds, dim=0).mean(dim=0)

    if not apply_tta:
        return forward_task_coords(model, task_images, task_id)

    base_coords = canonicalize_task_coords(predict_variant(task_images), task_id)
    if not use_hflip:
        return base_coords

    flipped_images = torch.flip(task_images, dims=[-1])
    flip_coords = predict_variant(flipped_images).clone()
    flip_coords[:, 0::2] = 1.0 - flip_coords[:, 0::2]
    flip_coords = canonicalize_task_coords(flip_coords, task_id)
    return 0.5 * (base_coords + flip_coords)


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
        "--encoder-weights",
        default="none",
        choices=("none", "pretrained"),
        help=(
            "Weights used only when constructing the encoder before checkpoint loading. "
            "Use 'none' for offline Docker inference."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="submission_output",
        help="Directory to write regression_predictions.json and submission.zip.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--model-profile",
        default=None,
        choices=MODEL_PROFILE_NAMES,
        help="Named submission architecture preset. When provided, it overrides checkpoint-inferred architecture fields.",
    )
    parser.add_argument(
        "--encoder-name",
        default=None,
        help="Optional backbone override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--encoder-feature-mode",
        default=None,
        choices=("final", "multilayer_fusion_v1", "feature_pyramid_fusion_v1"),
        help="Optional backbone feature extraction mode override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--head-type",
        default=None,
        choices=("basic", "deep"),
        help="Optional decoder head override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--task-head-profile",
        default=None,
        choices=("uniform", "challenge_legacy_v1", "challenge_v1"),
        help="Optional task-specific head sizing override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--task-decoder-profile",
        default=None,
        choices=tuple(TASK_DECODER_PROFILE_PRESETS.keys()),
        help="Optional task-specific decoder-family override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--task-adapter-profile",
        default=None,
        choices=tuple(TASK_ADAPTER_PROFILE_PRESETS.keys()),
        help="Optional task-specific feature adapter override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--fpn-mode",
        default=None,
        choices=("shared", "task_specific"),
        help="Optional FPN mode override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--fpn-type",
        default=None,
        choices=("fpn", "bifpn"),
        help="Optional FPN neck type override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Optional square letterbox input size override. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--input-crop-mode",
        default=None,
        choices=(
            "none",
            "content_box",
            "content_box_strict",
            "content_box_wide",
            "content_box_wide_strict",
            "content_box_fugc_wide",
            "content_box_fugc_xwide",
        ),
        help="Optional model-side crop override. If omitted, inferred from checkpoint metadata.",
    )
    parser.add_argument(
        "--tta-mode",
        default="none",
        choices=("none", "hflip", "photometric", "hflip_photometric"),
        help="Submission-time test-time augmentation mode.",
    )
    parser.add_argument(
        "--tta-task-ids",
        default="AOP,FA,FUGC,HC,IVC,PSAX,fetal_femur",
        help="Comma-separated task IDs to apply TTA to. Ignored when --tta-mode none.",
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
    parser.add_argument(
        "--skip-count-validation",
        action="store_true",
        help=(
            "Do not enforce the public validation manifest row counts. "
            "Use this for hidden/final Docker manifests whose counts are unknown."
        ),
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

    task_configs = build_task_configs(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, checkpoint_meta = load_checkpoint_payload(checkpoint_path, device)
    (
        inferred_encoder_name,
        inferred_encoder_feature_mode,
        inferred_use_fpn,
        inferred_fpn_mode,
        inferred_fpn_type,
        inferred_head_type,
        inferred_task_head_profile,
        inferred_task_decoder_profile,
        inferred_task_adapter_profile,
        inferred_input_size,
        inferred_heatmap_size,
        inferred_input_crop_mode,
    ) = infer_model_config_from_checkpoint(
        checkpoint,
        checkpoint_meta,
    )
    checkpoint_has_domain_classifier = any(key.startswith("domain_classifier.") for key in checkpoint.keys())
    domain_adversarial = bool(checkpoint_meta.get("domain_adversarial", checkpoint_has_domain_classifier))
    num_domain_classes = int(checkpoint_meta.get("num_domain_classes", infer_num_domain_classes(checkpoint)))
    encoder_name = args.encoder_name or inferred_encoder_name
    encoder_feature_mode = args.encoder_feature_mode or inferred_encoder_feature_mode
    fpn_mode = args.fpn_mode or inferred_fpn_mode
    fpn_type = args.fpn_type or inferred_fpn_type
    head_type = args.head_type or inferred_head_type
    task_head_profile = args.task_head_profile or inferred_task_head_profile
    task_decoder_profile = args.task_decoder_profile or inferred_task_decoder_profile
    task_adapter_profile = args.task_adapter_profile or inferred_task_adapter_profile
    use_fpn = inferred_use_fpn if args.use_fpn is None else args.use_fpn
    input_size = int(args.input_size or inferred_input_size)
    input_crop_mode = str(args.input_crop_mode or inferred_input_crop_mode)
    if args.model_profile is not None:
        profile_config = apply_model_profile(
            args.model_profile,
            "inference",
            {
                "encoder_name": encoder_name,
                "encoder_feature_mode": encoder_feature_mode,
                "use_fpn": use_fpn,
                "fpn_mode": fpn_mode,
                "fpn_type": fpn_type,
                "task_head_profile": task_head_profile,
                "task_decoder_profile": task_decoder_profile,
                "task_adapter_profile": task_adapter_profile,
                "input_crop_mode": input_crop_mode,
            },
        )
        encoder_name = str(profile_config["encoder_name"])
        encoder_feature_mode = str(profile_config.get("encoder_feature_mode", encoder_feature_mode))
        use_fpn = bool(profile_config["use_fpn"])
        fpn_mode = str(profile_config["fpn_mode"])
        fpn_type = str(profile_config["fpn_type"])
        task_head_profile = str(profile_config["task_head_profile"])
        task_decoder_profile = str(profile_config["task_decoder_profile"])
        task_adapter_profile = str(profile_config["task_adapter_profile"])
        input_crop_mode = str(profile_config.get("input_crop_mode", input_crop_mode))
    tta_task_ids = (
        DEFAULT_TTA_TASK_IDS
        if args.tta_mode == "hflip" and args.tta_task_ids == "AOP,FA,FUGC,HC,IVC,PSAX,fetal_femur"
        else parse_task_id_csv(args.tta_task_ids)
    )
    encoder_lora_rank = int(checkpoint_meta.get("encoder_lora_rank", 0))
    encoder_lora_alpha = float(checkpoint_meta.get("encoder_lora_alpha", 16.0))
    encoder_lora_last_blocks = int(checkpoint_meta.get("encoder_lora_last_blocks", 4))
    encoder_lora_dropout = float(checkpoint_meta.get("encoder_lora_dropout", 0.05))
    encoder_lora_task_specific = bool(checkpoint_meta.get("encoder_lora_task_specific", False))

    dataset = ValidationManifestDataset(
        manifest_path=manifest_path,
        input_size=input_size,
        input_crop_mode=input_crop_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(
        f"Checkpoint architecture: backbone={inferred_encoder_name}, "
        f"encoder_feature_mode={inferred_encoder_feature_mode}, "
        f"fpn_mode={inferred_fpn_mode}, "
        f"fpn_type={inferred_fpn_type}, "
        f"head={inferred_head_type}, "
        f"task_head_profile={inferred_task_head_profile}, "
        f"task_decoder_profile={inferred_task_decoder_profile}, "
        f"task_adapter_profile={inferred_task_adapter_profile}, "
        f"input_size={inferred_input_size}, "
        f"input_crop_mode={inferred_input_crop_mode}, "
        f"heatmap_size={inferred_heatmap_size}, "
        f"encoder_lora_rank={encoder_lora_rank}, "
        f"encoder_lora_task_specific={encoder_lora_task_specific}, "
        f"domain_adversarial={domain_adversarial}, "
        f"FPN={'ENABLED' if inferred_use_fpn else 'DISABLED'}"
    )
    print(
        f"Submission model config: backbone={encoder_name}, "
        f"encoder_feature_mode={encoder_feature_mode}, "
        f"fpn_mode={fpn_mode}, "
        f"fpn_type={fpn_type}, "
        f"head={head_type}, "
        f"task_head_profile={task_head_profile}, "
        f"task_decoder_profile={task_decoder_profile}, "
        f"task_adapter_profile={task_adapter_profile}, "
        f"input_size={input_size}, "
        f"input_crop_mode={input_crop_mode}, "
        f"heatmap_size={inferred_heatmap_size}, "
        f"FPN={'ENABLED' if use_fpn else 'DISABLED'}"
    )
    print(f"Model profile: {args.model_profile if args.model_profile is not None else 'checkpoint/manual'}")
    if args.tta_mode != "none":
        print(f"Submission TTA: {args.tta_mode} (tasks={sorted(tta_task_ids) if tta_task_ids is not None else 'all'})")

    model = MultiTaskModelFactory(
        encoder_name=encoder_name,
        encoder_weights=None if args.encoder_weights == "none" else "pretrained",
        encoder_feature_mode=encoder_feature_mode,
        task_configs=task_configs,
        heatmap_size=inferred_heatmap_size,
        use_fpn=use_fpn,
        fpn_mode=fpn_mode,
        fpn_type=fpn_type,
        head_type=head_type,
        task_head_profile=task_head_profile,
        task_decoder_profile=task_decoder_profile,
        task_adapter_profile=task_adapter_profile,
        encoder_lora_rank=encoder_lora_rank,
        encoder_lora_alpha=encoder_lora_alpha,
        encoder_lora_last_blocks=encoder_lora_last_blocks,
        encoder_lora_dropout=encoder_lora_dropout,
        encoder_lora_task_specific=encoder_lora_task_specific,
        domain_adversarial=domain_adversarial,
        num_domain_classes=num_domain_classes,
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
                outputs_transformed = predict_task_transformed_coords(
                    model,
                    task_images,
                    task_id,
                    args.tta_mode,
                    tta_task_ids,
                )
                outputs = transformed_coords_to_original_normalized(
                    outputs_transformed,
                    [meta[i] for i in task_indices],
                )
                outputs = canonicalize_task_coords(outputs, task_id)

                for output_idx, batch_idx in enumerate(task_indices):
                    pred = round_float_list(outputs[output_idx].cpu().numpy().tolist())
                    original_height, original_width = batch["original_size"][batch_idx]
                    pixel_coords = []
                    for point_idx in range(0, len(pred), 2):
                        x_norm = float(pred[point_idx])
                        y_norm = float(pred[point_idx + 1])
                        pixel_coords.extend([x_norm * original_width, y_norm * original_height])
                    pixel_coords = round_float_list(pixel_coords)

                    predictions.append(
                        {
                            "image_path": batch["image_path"][batch_idx],
                            "task_id": task_id,
                            "predicted_points_normalized": pred,
                            "predicted_points_pixels": pixel_coords,
                        }
                    )

    expected_counts = None if args.skip_count_validation else EXPECTED_VALIDATION_COUNTS
    validate_predictions(predictions, expected_counts=expected_counts)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "regression_predictions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="regression_predictions.json")

    print(f"Wrote {json_path}")
    print(f"Wrote {zip_path}")
    print(f"Prediction counts: {dict(sorted(Counter(item['task_id'] for item in predictions).items()))}")
    if expected_counts is not None:
        print(f"Validated counts: {expected_counts}")


if __name__ == "__main__":
    main()
