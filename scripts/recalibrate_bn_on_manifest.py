#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from tqdm import tqdm

from submit import (
    ValidationManifestDataset,
    build_task_configs,
    collate_fn,
    infer_model_config_from_checkpoint,
    load_checkpoint_payload,
)
from baseline.model_factory import MultiTaskModelFactory
from baseline.model_profiles import MODEL_PROFILE_NAMES, apply_model_profile


def _resolve_model_config(args, checkpoint_state, checkpoint_meta):
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

    encoder_name = args.encoder_name or inferred_encoder_name
    encoder_feature_mode = inferred_encoder_feature_mode
    use_fpn = inferred_use_fpn if args.use_fpn is None else args.use_fpn
    fpn_mode = args.fpn_mode or inferred_fpn_mode
    fpn_type = args.fpn_type or inferred_fpn_type
    head_type = args.head_type or inferred_head_type
    task_head_profile = args.task_head_profile or inferred_task_head_profile
    task_decoder_profile = args.task_decoder_profile or inferred_task_decoder_profile
    task_adapter_profile = args.task_adapter_profile or inferred_task_adapter_profile
    input_size = int(args.input_size or inferred_input_size)
    heatmap_size = inferred_heatmap_size

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

    return {
        "encoder_name": encoder_name,
        "encoder_feature_mode": encoder_feature_mode,
        "use_fpn": use_fpn,
        "fpn_mode": fpn_mode,
        "fpn_type": fpn_type,
        "head_type": head_type,
        "task_head_profile": task_head_profile,
        "task_decoder_profile": task_decoder_profile,
        "task_adapter_profile": task_adapter_profile,
        "input_size": input_size,
        "heatmap_size": heatmap_size,
    }


def _reset_bn_stats(module: torch.nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, _BatchNorm):
            child.reset_running_stats()
            child.momentum = None
            count += 1
    return count


def _set_bn_train_only(module: torch.nn.Module) -> None:
    module.eval()
    for child in module.modules():
        if isinstance(child, _BatchNorm):
            child.train()


def main():
    parser = argparse.ArgumentParser(
        description="Re-estimate BatchNorm statistics on the official validation manifest."
    )
    parser.add_argument("--manifest", default="data/manifests/validation_manifest.csv")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-passes", type=int, default=2)
    parser.add_argument("--model-profile", default=None, choices=MODEL_PROFILE_NAMES)
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--head-type", default=None, choices=("basic", "deep"))
    parser.add_argument("--task-head-profile", default=None, choices=("uniform", "challenge_legacy_v1", "challenge_v1"))
    parser.add_argument(
        "--task-decoder-profile",
        default=None,
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
            "hidden_a4c_hc_ivc_fugc_strip_axis_offset_v1",
            "hidden_a4c_hc_ivc_fugc_segment_specialist_v1",
            "hidden_a4c_hc_ivc_plax_refine_v1",
            "hidden_a4c_hc_ivc_femur_refine_v1",
            "geometry_v1",
            "weak_tasks_v1",
            "dedicated_legacy_v1",
            "dedicated_v1",
        ),
    )
    parser.add_argument(
        "--task-adapter-profile",
        default=None,
        choices=("uniform", "softsharing_v1", "localrefine_v1", "coarse_refine_v1", "context_experts_v1", "context_local_v1", "taskfilm_v1"),
    )
    parser.add_argument("--fpn-mode", default=None, choices=("shared", "task_specific"))
    parser.add_argument("--fpn-type", default=None, choices=("fpn", "bifpn"))
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--fpn", dest="use_fpn", action="store_true")
    parser.add_argument("--no-fpn", dest="use_fpn", action="store_false")
    parser.set_defaults(use_fpn=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_state, checkpoint_meta = load_checkpoint_payload(args.checkpoint_path, device)
    model_config = _resolve_model_config(args, checkpoint_state, checkpoint_meta)
    task_configs = build_task_configs(args.manifest)

    model = MultiTaskModelFactory(
        encoder_name=model_config["encoder_name"],
        encoder_weights=None,
        encoder_feature_mode=model_config.get("encoder_feature_mode", "final"),
        task_configs=task_configs,
        use_fpn=model_config["use_fpn"],
        fpn_mode=model_config["fpn_mode"],
        fpn_type=model_config["fpn_type"],
        head_type=model_config["head_type"],
        task_head_profile=model_config["task_head_profile"],
        task_decoder_profile=model_config["task_decoder_profile"],
        task_adapter_profile=model_config["task_adapter_profile"],
        heatmap_size=tuple(model_config["heatmap_size"]),
    ).to(device)
    model.load_state_dict(checkpoint_state, strict=True)

    dataset = ValidationManifestDataset(args.manifest, input_size=model_config["input_size"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    bn_count = _reset_bn_stats(model)
    _set_bn_train_only(model)

    if bn_count == 0:
        raise RuntimeError("No BatchNorm layers were found for recalibration.")

    with torch.no_grad():
        for pass_idx in range(args.num_passes):
            for batch in tqdm(loader, desc=f"BN recalibration pass {pass_idx + 1}/{args.num_passes}"):
                images = batch["image"].to(device, non_blocking=True)
                task_ids = batch["task_id"]
                unique_task_ids = sorted(set(task_ids))
                for task_id in unique_task_ids:
                    task_indices = [i for i, current_task_id in enumerate(task_ids) if current_task_id == task_id]
                    task_images = images[task_indices]
                    model(task_images, task_id=task_id)

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        checkpoint = {"state_dict": checkpoint, "meta": {}}
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["state_dict"] = copy.deepcopy(model.state_dict())
    checkpoint.setdefault("meta", {})
    checkpoint["meta"]["bn_recalibrated_from"] = str(Path(args.checkpoint_path).resolve())
    checkpoint["meta"]["bn_recalibration_manifest"] = str(Path(args.manifest).resolve())
    checkpoint["meta"]["bn_recalibration_passes"] = int(args.num_passes)
    torch.save(checkpoint, output_path)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "bn_layer_count": bn_count,
                "num_manifest_rows": len(dataset),
                "num_passes": int(args.num_passes),
                "model_config": model_config,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
