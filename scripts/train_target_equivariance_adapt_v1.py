#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from submit import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ValidationManifestDataset,
    build_task_configs,
    collate_fn,
    infer_model_config_from_checkpoint,
    load_checkpoint_payload,
)
from baseline.model_factory import MultiTaskModelFactory
from baseline.model_profiles import MODEL_PROFILE_NAMES, apply_model_profile
from baseline.utils import decode_heatmaps_to_normalized_coords


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_config(args, checkpoint_state: dict, checkpoint_meta: dict) -> dict:
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
    ) = infer_model_config_from_checkpoint(checkpoint_state, checkpoint_meta)

    config = {
        "encoder_name": args.encoder_name or inferred_encoder_name,
        "encoder_feature_mode": inferred_encoder_feature_mode,
        "use_fpn": inferred_use_fpn if args.use_fpn is None else args.use_fpn,
        "fpn_mode": args.fpn_mode or inferred_fpn_mode,
        "fpn_type": args.fpn_type or inferred_fpn_type,
        "head_type": args.head_type or inferred_head_type,
        "task_head_profile": args.task_head_profile or inferred_task_head_profile,
        "task_decoder_profile": args.task_decoder_profile or inferred_task_decoder_profile,
        "task_adapter_profile": args.task_adapter_profile or inferred_task_adapter_profile,
        "input_size": int(args.input_size or inferred_input_size),
        "heatmap_size": tuple(inferred_heatmap_size),
    }

    if args.model_profile is not None:
        profile_config = apply_model_profile(
            args.model_profile,
            "inference",
            {
                "encoder_name": config["encoder_name"],
                "encoder_feature_mode": config["encoder_feature_mode"],
                "use_fpn": config["use_fpn"],
                "fpn_mode": config["fpn_mode"],
                "fpn_type": config["fpn_type"],
                "task_head_profile": config["task_head_profile"],
                "task_decoder_profile": config["task_decoder_profile"],
                "task_adapter_profile": config["task_adapter_profile"],
            },
        )
        config.update(
            {
                "encoder_name": str(profile_config["encoder_name"]),
                "encoder_feature_mode": str(profile_config.get("encoder_feature_mode", config["encoder_feature_mode"])),
                "use_fpn": bool(profile_config["use_fpn"]),
                "fpn_mode": str(profile_config["fpn_mode"]),
                "fpn_type": str(profile_config["fpn_type"]),
                "task_head_profile": str(profile_config["task_head_profile"]),
                "task_decoder_profile": str(profile_config["task_decoder_profile"]),
                "task_adapter_profile": str(profile_config["task_adapter_profile"]),
            }
        )
    return config


def build_model(config: dict, task_configs: list[dict], device: torch.device) -> MultiTaskModelFactory:
    return MultiTaskModelFactory(
        encoder_name=config["encoder_name"],
        encoder_weights=None,
        encoder_feature_mode=config.get("encoder_feature_mode", "final"),
        task_configs=task_configs,
        heatmap_size=tuple(config["heatmap_size"]),
        use_fpn=bool(config["use_fpn"]),
        fpn_mode=str(config["fpn_mode"]),
        fpn_type=str(config["fpn_type"]),
        head_type=str(config["head_type"]),
        task_head_profile=str(config["task_head_profile"]),
        task_decoder_profile=str(config["task_decoder_profile"]),
        task_adapter_profile=str(config["task_adapter_profile"]),
    ).to(device)


def forward_coords(model: torch.nn.Module, images: torch.Tensor, task_id: str) -> torch.Tensor:
    output = model(images, task_id=task_id, return_prior=True)
    aux_outputs = {}
    if len(output) == 3:
        _, heatmaps, aux_outputs = output
    else:
        _, heatmaps = output
    refined = aux_outputs.get("refined_coords_transformed")
    return refined if refined is not None else decode_heatmaps_to_normalized_coords(heatmaps)


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images.clamp(0.0, 1.0) - mean) / std


def make_forward_affine(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_rotation_deg: float,
    scale_jitter: float,
    anisotropic_jitter: float,
    max_translation: float,
) -> torch.Tensor:
    angles = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0)
    angles = angles * (math.pi * max_rotation_deg / 180.0)
    base_scale = 1.0 + (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * scale_jitter
    sx = base_scale * (1.0 + (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * anisotropic_jitter)
    sy = base_scale * (1.0 + (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * anisotropic_jitter)
    tx = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * max_translation
    ty = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * max_translation

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    forward = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    forward[:, 0, 0] = sx * cos_a
    forward[:, 0, 1] = -sy * sin_a
    forward[:, 0, 2] = tx
    forward[:, 1, 0] = sx * sin_a
    forward[:, 1, 1] = sy * cos_a
    forward[:, 1, 2] = ty
    forward[:, 2, 2] = 1.0
    return forward


def apply_affine_to_points(coords: torch.Tensor, forward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = coords.shape[0]
    points = coords.view(batch_size, -1, 2)
    points_pm = points.mul(2.0).sub(1.0)
    ones = torch.ones(batch_size, points.shape[1], 1, device=coords.device, dtype=coords.dtype)
    hom = torch.cat([points_pm, ones], dim=-1)
    transformed = torch.bmm(hom, forward.transpose(1, 2))[..., :2]
    target = transformed.add(1.0).mul(0.5)
    inside = (target > 0.01).all(dim=-1) & (target < 0.99).all(dim=-1)
    return target.reshape(batch_size, -1), inside.reshape(batch_size, -1)


def augment_images(
    images: torch.Tensor,
    forward: torch.Tensor,
    brightness_jitter: float,
    contrast_jitter: float,
    gamma_jitter: float,
    noise_std: float,
) -> torch.Tensor:
    raw = unnormalize(images)
    theta = torch.linalg.inv(forward)[:, :2, :]
    grid = F.affine_grid(theta, raw.shape, align_corners=False)
    augmented = F.grid_sample(raw, grid, mode="bilinear", padding_mode="border", align_corners=False)

    batch_size = augmented.shape[0]
    dtype = augmented.dtype
    device = augmented.device
    contrast = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0) * contrast_jitter
    brightness = (torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0) * brightness_jitter
    gamma = torch.exp((torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0) * math.log(max(gamma_jitter, 1.0)))
    augmented = (augmented - 0.5) * contrast + 0.5 + brightness
    augmented = augmented.clamp(0.0, 1.0).pow(gamma)
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return normalize(augmented)


def smooth_l1_masked(pred: torch.Tensor, target: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
    coord_mask = point_mask.repeat_interleave(2, dim=1).to(dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
    denom = coord_mask.sum().clamp_min(1.0)
    return (loss * coord_mask).sum() / denom


def set_trainable_scope(model: torch.nn.Module, scope: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prefixes_by_scope = {
        "heads": ("heads.",),
        "adapters": (
            "soft_adapters.",
            "local_refine_adapters.",
            "context_adapters.",
            "context_expert_adapters.",
            "context_local_adapters.",
            "task_film_adapters.",
        ),
        "heads_adapters": (
            "heads.",
            "soft_adapters.",
            "local_refine_adapters.",
            "context_adapters.",
            "context_expert_adapters.",
            "context_local_adapters.",
            "task_film_adapters.",
        ),
        "fpn_heads_adapters": (
            "task_fpns.",
            "fpn.",
            "heads.",
            "soft_adapters.",
            "local_refine_adapters.",
            "context_adapters.",
            "context_expert_adapters.",
            "context_local_adapters.",
            "task_film_adapters.",
        ),
    }
    if scope not in prefixes_by_scope:
        raise ValueError(f"Unsupported trainable scope: {scope}")
    prefixes = prefixes_by_scope[scope]
    trainable = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            trainable += parameter.numel()
    if trainable == 0:
        raise RuntimeError(f"No trainable parameters selected for scope '{scope}'.")
    return trainable


def parse_task_ids(value: str | None) -> set[str] | None:
    if value is None or str(value).lower() in {"", "all"}:
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unlabeled target-domain equivariance adaptation on the official validation manifest."
    )
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-profile", default=None, choices=MODEL_PROFILE_NAMES)
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--head-type", default=None, choices=("basic", "deep"))
    parser.add_argument("--task-head-profile", default=None, choices=("uniform", "challenge_legacy_v1", "challenge_v1"))
    parser.add_argument("--task-decoder-profile", default=None)
    parser.add_argument("--task-adapter-profile", default=None)
    parser.add_argument("--fpn-mode", default=None, choices=("shared", "task_specific"))
    parser.add_argument("--fpn-type", default=None, choices=("fpn", "bifpn"))
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--fpn", dest="use_fpn", action="store_true")
    parser.add_argument("--no-fpn", dest="use_fpn", action="store_false")
    parser.set_defaults(use_fpn=None)

    parser.add_argument("--task-ids", default="A4C,AOP,HC,IVC,PSAX,fetal_femur")
    parser.add_argument("--trainable-scope", default="heads_adapters", choices=("heads", "adapters", "heads_adapters", "fpn_heads_adapters"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap per epoch for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--max-rotation-deg", type=float, default=3.0)
    parser.add_argument("--scale-jitter", type=float, default=0.035)
    parser.add_argument("--anisotropic-jitter", type=float, default=0.015)
    parser.add_argument("--max-translation", type=float, default=0.025)
    parser.add_argument("--brightness-jitter", type=float, default=0.035)
    parser.add_argument("--contrast-jitter", type=float, default=0.08)
    parser.add_argument("--gamma-jitter", type=float, default=1.08)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_task_ids = parse_task_ids(args.task_ids)

    checkpoint_state, checkpoint_meta = load_checkpoint_payload(args.checkpoint_path, device)
    model_config = resolve_model_config(args, checkpoint_state, checkpoint_meta)
    task_configs = build_task_configs(args.manifest)
    if selected_task_ids is not None:
        task_configs = [config for config in task_configs if str(config["task_id"]) in selected_task_ids]
        if not task_configs:
            raise ValueError(f"No task configs remain after task filter: {sorted(selected_task_ids)}")

    teacher = build_model(model_config, build_task_configs(args.manifest), device)
    student = build_model(model_config, build_task_configs(args.manifest), device)
    teacher.load_state_dict(checkpoint_state, strict=True)
    student.load_state_dict(checkpoint_state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    trainable_params = set_trainable_scope(student, args.trainable_scope)
    total_params = sum(parameter.numel() for parameter in student.parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "model_config": model_config,
                "selected_task_ids": sorted(selected_task_ids) if selected_task_ids else "all",
                "trainable_scope": args.trainable_scope,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_percent": 100.0 * trainable_params / max(total_params, 1),
            },
            indent=2,
            sort_keys=True,
        )
    )

    dataset = ValidationManifestDataset(args.manifest, input_size=model_config["input_size"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        running_count = 0
        for step_idx, batch in enumerate(tqdm(loader, desc=f"Target equivariance epoch {epoch + 1}/{args.epochs}")):
            if args.max_steps is not None and step_idx >= int(args.max_steps):
                break
            images = batch["image"].to(device, non_blocking=True)
            task_ids = batch["task_id"]
            loss_terms = []
            for task_id in sorted(set(task_ids)):
                if selected_task_ids is not None and task_id not in selected_task_ids:
                    continue
                task_indices = [idx for idx, value in enumerate(task_ids) if value == task_id]
                task_images = images[task_indices]
                with torch.no_grad():
                    teacher_coords = forward_coords(teacher, task_images, task_id).detach()

                forward = make_forward_affine(
                    batch_size=task_images.shape[0],
                    device=device,
                    dtype=task_images.dtype,
                    max_rotation_deg=args.max_rotation_deg,
                    scale_jitter=args.scale_jitter,
                    anisotropic_jitter=args.anisotropic_jitter,
                    max_translation=args.max_translation,
                )
                augmented_images = augment_images(
                    task_images,
                    forward,
                    brightness_jitter=args.brightness_jitter,
                    contrast_jitter=args.contrast_jitter,
                    gamma_jitter=args.gamma_jitter,
                    noise_std=args.noise_std,
                )
                target_aug_coords, point_mask = apply_affine_to_points(teacher_coords, forward)
                student_aug_coords = forward_coords(student, augmented_images, task_id)
                student_anchor_coords = forward_coords(student, task_images, task_id)

                equiv_loss = smooth_l1_masked(student_aug_coords, target_aug_coords, point_mask)
                anchor_loss = F.smooth_l1_loss(student_anchor_coords, teacher_coords, beta=0.01)
                loss_terms.append(equiv_loss + float(args.anchor_weight) * anchor_loss)

            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in student.parameters() if parameter.requires_grad],
                    max_norm=args.max_grad_norm,
                )
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            running_count += 1

        epoch_loss = running_loss / max(running_count, 1)
        print(f"epoch={epoch + 1} target_equivariance_loss={epoch_loss:.8f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(student.state_dict())

    if best_state is None:
        raise RuntimeError("No training steps were executed.")

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / "best_model.pth"

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        checkpoint = {"state_dict": checkpoint, "meta": {}}
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["state_dict"] = {key: value.detach().cpu() for key, value in best_state.items()}
    checkpoint.setdefault("meta", {})
    checkpoint["meta"].update(
        {
            "encoder_name": model_config["encoder_name"],
            "use_fpn": bool(model_config["use_fpn"]),
            "fpn_mode": model_config["fpn_mode"],
            "fpn_type": model_config["fpn_type"],
            "head_type": model_config["head_type"],
            "task_head_profile": model_config["task_head_profile"],
            "task_decoder_profile": model_config["task_decoder_profile"],
            "task_adapter_profile": model_config["task_adapter_profile"],
            "input_size": int(model_config["input_size"]),
            "heatmap_size": list(model_config["heatmap_size"]),
            "target_equivariance_adapted_from": str(Path(args.checkpoint_path).resolve()),
            "target_equivariance_manifest": str(Path(args.manifest).resolve()),
            "target_equivariance_tasks": sorted(selected_task_ids) if selected_task_ids else "all",
            "target_equivariance_trainable_scope": args.trainable_scope,
            "target_equivariance_best_loss": float(best_loss),
        }
    )
    torch.save(checkpoint, output_path)
    print(f"Saved target-equivariance adapted checkpoint: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
