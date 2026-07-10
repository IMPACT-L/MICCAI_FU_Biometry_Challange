MODEL_PROFILES = {
    "best_stable_v1": {
        "description": (
            "Most stable hidden-validation recipe so far: DINOv3 ViT-B encoder, "
            "task-specific FPN, challenge_v1 head widths, uniform decoder/adapter, "
            "grouped split, baseline augmentation, and server_proxy_v1 checkpointing."
        ),
        "train": {
            "encoder_name": "vit_base_patch16_dinov3",
            "input_size": 512,
            "use_fpn": True,
            "fpn_mode": "task_specific",
            "fpn_type": "fpn",
            "head_type": "deep",
            "task_head_profile": "challenge_v1",
            "task_decoder_profile": "uniform",
            "task_adapter_profile": "uniform",
            "task_loss_family_profile": "uniform",
            "split_mode": "grouped",
            "augmentation_profile": "baseline",
            "checkpoint_score_mode": "server_proxy_v1",
            "cardiac_split_screen_mode": "keep",
            "measurement_loss_weight": 0.0,
            "dataset_loss_weight": 0.0,
            "femur_shaft_loss_weight": 0.15,
            "fugc_segment_loss_weight": 0.08,
            "ivc_band_loss_weight": 0.0,
        },
        "inference": {
            "encoder_name": "vit_base_patch16_dinov3",
            "use_fpn": True,
            "fpn_mode": "task_specific",
            "fpn_type": "fpn",
            "task_head_profile": "challenge_v1",
            "task_decoder_profile": "uniform",
            "task_adapter_profile": "uniform",
        },
    },
    "hidden_localrefine_ft_v1": {
        "description": (
            "Best stable hidden-transfer recipe with task-specific local image-detail "
            "adapters enabled. This reruns the earlier promising localrefine path after "
            "fixing adapter optimizer registration."
        ),
        "train": {
            "encoder_name": "vit_base_patch16_dinov3",
            "input_size": 512,
            "use_fpn": True,
            "fpn_mode": "task_specific",
            "fpn_type": "fpn",
            "head_type": "deep",
            "task_head_profile": "challenge_v1",
            "task_decoder_profile": "uniform",
            "task_adapter_profile": "localrefine_v1",
            "task_loss_family_profile": "uniform",
            "split_mode": "grouped",
            "augmentation_profile": "baseline",
            "checkpoint_score_mode": "server_proxy_v1",
            "cardiac_split_screen_mode": "keep",
            "measurement_loss_weight": 0.0,
            "dataset_loss_weight": 0.0,
            "femur_shaft_loss_weight": 0.15,
            "fugc_segment_loss_weight": 0.08,
            "ivc_band_loss_weight": 0.0,
        },
        "inference": {
            "encoder_name": "vit_base_patch16_dinov3",
            "use_fpn": True,
            "fpn_mode": "task_specific",
            "fpn_type": "fpn",
            "task_head_profile": "challenge_v1",
            "task_decoder_profile": "uniform",
            "task_adapter_profile": "localrefine_v1",
        },
    },
    "hidden_context_ft_v1": {
        "description": (
            "Best stable hidden-transfer recipe with residual context adapters enabled. "
            "This is the post-fix rerun of the earlier coarse_refine adapter line."
        ),
        "train": {
            "encoder_name": "vit_base_patch16_dinov3",
            "input_size": 512,
            "use_fpn": True,
            "fpn_mode": "task_specific",
            "fpn_type": "fpn",
            "head_type": "deep",
            "task_head_profile": "challenge_v1",
            "task_decoder_profile": "uniform",
            "task_adapter_profile": "coarse_refine_v1",
            "task_loss_family_profile": "uniform",
            "split_mode": "grouped",
            "augmentation_profile": "baseline",
            "checkpoint_score_mode": "server_proxy_v1",
            "cardiac_split_screen_mode": "keep",
            "measurement_loss_weight": 0.0,
            "dataset_loss_weight": 0.0,
            "femur_shaft_loss_weight": 0.15,
            "fugc_segment_loss_weight": 0.08,
            "ivc_band_loss_weight": 0.0,
        },
        "inference": {
            "encoder_name": "vit_base_patch16_dinov3",
            "use_fpn": True,
            "fpn_mode": "task_specific",
            "fpn_type": "fpn",
            "task_head_profile": "challenge_v1",
            "task_decoder_profile": "uniform",
            "task_adapter_profile": "coarse_refine_v1",
        },
    },
}

MODEL_PROFILE_NAMES = tuple(sorted(MODEL_PROFILES.keys()))


def get_model_profile(profile_name: str) -> dict:
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"Unsupported model profile: {profile_name}")
    return MODEL_PROFILES[profile_name]


def apply_model_profile(profile_name: str, stage: str, current_config: dict) -> dict:
    profile = get_model_profile(profile_name)
    if stage not in profile:
        raise ValueError(f"Profile '{profile_name}' does not define stage '{stage}'")
    merged = dict(current_config)
    merged.update(profile[stage])
    return merged
