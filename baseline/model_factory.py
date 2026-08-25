from typing import Dict, List

import importlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .anatomical_query_decoder import SharedAnatomicalQueryDecoder
    from .lora import inject_lora_into_vit_attention, set_active_lora_task
except ImportError:
    from anatomical_query_decoder import SharedAnatomicalQueryDecoder
    from lora import inject_lora_into_vit_attention, set_active_lora_task


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
FPN_MODES = {"shared", "task_specific"}
FPN_TYPES = {"fpn", "bifpn"}
ENCODER_FEATURE_MODES = {"final", "multilayer_fusion_v1", "feature_pyramid_fusion_v1"}
TASK_GROUPS = {
    "pair": {"FUGC", "IVC", "fetal_femur"},
    "ellipse": {"FA", "HC"},
    "context": {"A4C", "AOP", "PLAX", "PSAX"},
}
TASK_ADAPTER_PROFILE_PRESETS = {
    "uniform": {},
    "softsharing_v1": {
        "FUGC": "pair",
        "IVC": "pair",
        "fetal_femur": "pair",
        "FA": "ellipse",
        "HC": "ellipse",
        "A4C": "context",
        "AOP": "context",
        "PLAX": "context",
        "PSAX": "context",
    },
    "localrefine_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "coarse_refine_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "taskfilm_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "context_experts_v1": {
        "A4C": "hard",
        "HC": "hard",
        "IVC": "hard",
        "PLAX": "hard",
        "PSAX": "hard",
        "AOP": "medium",
        "FA": "medium",
        "FUGC": "light",
        "fetal_femur": "light",
    },
    "context_local_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "context_local_pair_transfer_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "context_local_queryrefine_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "context_local_prototype_memory_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "context_local_shared_queryrefine_v2": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "context_local_stylemix_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "texture_context_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "texture_residual_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "texture_residual_v2": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "highres_texture_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "pixel_unet_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "hrnet_residual_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "encoder_task_context_local_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "encoder_task_hard_context_local_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
    "boundary_context_v1": {
        "A4C": "A4C",
        "AOP": "AOP",
        "FA": "FA",
        "FUGC": "FUGC",
        "HC": "HC",
        "IVC": "IVC",
        "PLAX": "PLAX",
        "PSAX": "PSAX",
        "fetal_femur": "fetal_femur",
    },
}
TASK_HEAD_PROFILE_PRESETS = {
    "uniform": {},
    "challenge_legacy_v1": {
        "A4C": "heavy",
        "HC": "heavy",
        "PLAX": "heavy",
        "AOP": "light",
        "FUGC": "light",
        "IVC": "light",
        "fetal_femur": "light",
    },
    "challenge_v1": {
        "A4C": "heavy",
        "HC": "heavy",
        "PLAX": "heavy",
        "AOP": "light",
        "FUGC": "light",
        "IVC": "light",
        "fetal_femur": "heavy",
    },
}
TASK_DECODER_PROFILE_PRESETS = {
    "uniform": {},
    "anatomical_queries_v1": {
        "A4C": "anatomical_query",
        "AOP": "anatomical_query",
        "FA": "anatomical_query",
        "FUGC": "anatomical_query",
        "HC": "anatomical_query",
        "IVC": "anatomical_query",
        "PLAX": "anatomical_query",
        "PSAX": "anatomical_query",
        "fetal_femur": "anatomical_query",
    },
    "cardiac_graph_v1": {
        "A4C": "cardiac_graph",
        "PLAX": "cardiac_graph",
        "PSAX": "cardiac_graph",
        "IVC": "ivc_refine_v2",
    },
    "coarse_refine_v1": {
        "A4C": "refine",
        "FUGC": "refine",
        "IVC": "refine",
        "fetal_femur": "refine",
    },
    "ivc_refine_v1": {
        "IVC": "ivc_refine",
    },
    "ivc_refine_v2": {
        "IVC": "ivc_refine_v2",
    },
    "fugc_refine_v1": {
        "FUGC": "fugc",
    },
    "hc_refine_v1": {
        "HC": "hc_refine",
    },
    "hidden_hc_ivc_refine_v1": {
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
    },
    "hidden_a4c_hc_ivc_refine_v1": {
        "A4C": "cardiac_graph",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
    },
    "hidden_a4cv2_hc_ivc_refine_v1": {
        "A4C": "a4c_v2",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
    },
    "hidden_a4cv3_hc_ivc_refine_v1": {
        "A4C": "a4c_v3",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
    },
    "hidden_a4cv4_hc_ivc_refine_v1": {
        "A4C": "a4c_v4",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
    },
    "hidden_a4c_hc_ivc_fugc_refine_v1": {
        "A4C": "cardiac_graph",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "FUGC": "fugc",
    },
    "hidden_a4c_hc_ivc_fugc_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4cv3_hc_ivc_fugc_offset_v1": {
        "A4C": "a4c_v3",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_structure_hc_ivc_fugc_offset_v1": {
        "A4C": "structure_cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_dualscale_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_dualscale",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_anatomy_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_anatomy",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_anatomy_seg_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_anatomy_seg",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_anatomy_local_offset_v2": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_anatomy_local",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_cervical_canal_offset_v3": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_cervical_canal",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_segmentation_offset_v4": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_cervix_segmentation",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_geometry_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_cervix_geometry",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_highres_segmentation_offset_v5": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_cervix_highres_segmentation",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_vitpose_direct_offset_v7": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_vitpose_direct",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_canonical_pairs_offset_v3": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_cardiac_pair_set_offset_v4": {
        "A4C": "a4c_pair_set",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "psax_pair_set",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_pair_set_offset_v1": {
        "A4C": "a4c_pair_set",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_axis_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "axis_offset_deep",
        "FA": "axis_offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "axis_offset_deep",
        "PSAX": "axis_offset_deep",
        "fetal_femur": "axis_offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_aop_vector_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "aop_vector_offset",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_vector_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_vector",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_strip_axis_offset_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_strip_axis",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_segment_specialist_v1": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc_segment_specialist",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4c_hc_ivc_fugc_offset_v2": {
        "A4C": "cardiac_graph",
        "AOP": "offset_deep",
        "FA": "offset_deep",
        "FUGC": "fugc",
        "HC": "hc_refine_offset",
        "IVC": "ivc_refine_v3",
        "PLAX": "offset_deep",
        "PSAX": "offset_deep",
        "fetal_femur": "offset_deep",
    },
    "hidden_a4cv2_hc_ivc_fugc_refine_v1": {
        "A4C": "a4c_v2",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "FUGC": "fugc",
    },
    "hidden_a4cv3_hc_ivc_fugc_refine_v1": {
        "A4C": "a4c_v3",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "FUGC": "fugc",
    },
    "hidden_a4cv4_hc_ivc_fugc_refine_v1": {
        "A4C": "a4c_v4",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "FUGC": "fugc",
    },
    "hidden_a4c_hc_ivc_plax_refine_v1": {
        "A4C": "cardiac_graph",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "plax",
    },
    "hidden_a4c_hc_ivc_femur_refine_v1": {
        "A4C": "cardiac_graph",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "fetal_femur": "femur",
    },
    "geometry_family_v2": {
        "A4C": "cardiac_graph",
        "AOP": "aop",
        "FA": "fa_refine",
        "FUGC": "fugc",
        "HC": "hc_refine",
        "IVC": "ivc_refine_v2",
        "PLAX": "plax",
        "PSAX": "psax",
        "fetal_femur": "femur",
    },
    "structure_v1": {
        "A4C": "structure_cardiac_graph",
        "AOP": "structure_offset_deep",
        "FA": "structure_offset_deep",
        "FUGC": "structure_fugc",
        "HC": "structure_hc_refine_offset",
        "IVC": "structure_ivc_refine_v3",
        "PLAX": "structure_offset_deep",
        "PSAX": "structure_offset_deep",
        "fetal_femur": "structure_femur",
    },
    "geometry_v1": {
        "A4C": "dense",
        "PLAX": "dense",
        "HC": "compact",
        "AOP": "compact",
        "FA": "compact",
        "PSAX": "compact",
        "FUGC": "line",
        "IVC": "line",
        "fetal_femur": "femur",
    },
    "weak_tasks_v1": {
        "FUGC": "fugc",
        "IVC": "ivc",
        "fetal_femur": "femur",
    },
    "dedicated_legacy_v1": {
        "A4C": "a4c",
        "PLAX": "plax",
        "HC": "hc",
        "AOP": "aop",
        "FA": "fa",
        "PSAX": "psax",
        "FUGC": "fugc",
        "IVC": "ivc",
        "fetal_femur": "line",
    },
    "dedicated_v1": {
        "A4C": "a4c",
        "PLAX": "plax",
        "HC": "hc",
        "AOP": "aop",
        "FA": "fa",
        "PSAX": "psax",
        "FUGC": "fugc",
        "IVC": "ivc",
        "fetal_femur": "femur",
    },
}
HEAD_WIDTH_MULTIPLIERS = {
    "light": 0.75,
    "medium": 1.0,
    "heavy": 1.35,
}


class SoftSharingAdapter(nn.Module):
    """Shared adapter plus group-specific residual experts for soft parameter sharing."""

    def __init__(self, channels: int, group_names: list[str]):
        super().__init__()
        hidden = max(channels // 2, 96)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.group_experts = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.Conv2d(channels, hidden, kernel_size=3, padding=1, groups=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for group_name in group_names
            }
        )
        self.shared_scale = nn.Parameter(torch.tensor(1.0))
        self.group_scale = nn.ParameterDict(
            {
                group_name: nn.Parameter(torch.tensor(0.5))
                for group_name in group_names
            }
        )

    def forward(self, x: torch.Tensor, group_name: str) -> torch.Tensor:
        shared_delta = self.shared(x)
        if group_name not in self.group_experts:
            return x + self.shared_scale * shared_delta
        group_delta = self.group_experts[group_name](x)
        return x + self.shared_scale * shared_delta + self.group_scale[group_name] * group_delta


class LocalRefineAdapter(nn.Module):
    """Fuse high-resolution image evidence back into task features."""

    def __init__(self, feature_channels: int, group_names: list[str], image_channels: int = 128):
        super().__init__()
        self.detail_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, image_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(image_channels),
            nn.GELU(),
        )
        hidden = max(feature_channels // 2, 128)
        self.group_fusions = nn.ModuleDict(
            {
                group_name: nn.ModuleDict(
                    {
                        "pre": nn.Sequential(
                            nn.Conv2d(feature_channels + image_channels, hidden, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden),
                            nn.GELU(),
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden),
                            nn.GELU(),
                        ),
                        "gate": nn.Sequential(
                            nn.Conv2d(hidden, feature_channels, kernel_size=1, bias=True),
                            nn.Sigmoid(),
                        ),
                        "delta": nn.Sequential(
                            nn.Conv2d(hidden, feature_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(feature_channels),
                        ),
                    }
                )
                for group_name in group_names
            }
        )
        self.fallback = nn.ModuleDict(
            {
                "pre": nn.Sequential(
                    nn.Conv2d(feature_channels + image_channels, hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                ),
                "gate": nn.Sequential(
                    nn.Conv2d(hidden, feature_channels, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                ),
                "delta": nn.Sequential(
                    nn.Conv2d(hidden, feature_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feature_channels),
                ),
            }
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def _fuse(self, features: torch.Tensor, detail: torch.Tensor, block: nn.ModuleDict) -> torch.Tensor:
        fused = torch.cat([features, detail], dim=1)
        hidden = block["pre"](fused)
        gate = block["gate"](hidden)
        delta = block["delta"](hidden)
        return features + self.residual_scale * gate * delta

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        detail = self.detail_stem(image)
        if detail.shape[-2:] != features.shape[-2:]:
            detail = F.interpolate(detail, size=features.shape[-2:], mode="bilinear", align_corners=False)
        if group_name is None or group_name not in self.group_fusions:
            return self._fuse(features, detail, self.fallback)
        return self._fuse(features, detail, self.group_fusions[group_name])


class ResidualContextAdapter(nn.Module):
    """Post-neck residual context mixing with shared and task-specific branches."""

    def __init__(self, channels: int, group_names: list[str]):
        super().__init__()
        hidden = max(channels, 192)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.group_blocks = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for group_name in group_names
            }
        )
        self.shared_scale = nn.Parameter(torch.tensor(0.75))
        self.group_scale = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.35)) for group_name in group_names}
        )

    def forward(self, x: torch.Tensor, group_name: str | None) -> torch.Tensor:
        shared_delta = self.shared(x)
        if group_name is None or group_name not in self.group_blocks:
            return x + self.shared_scale * shared_delta
        return x + self.shared_scale * shared_delta + self.group_scale[group_name] * self.group_blocks[group_name](x)


class PreFPNTaskBottleneckAdapter(nn.Module):
    """Task-conditioned residual adapter on DINO feature maps before FPN.

    Heads can only decode what the shared ViT feature map exposes. This adapter
    gives each task a small low-rank correction before the task-specific FPN,
    while the sigmoid gate is initialized near zero so warm-started checkpoints
    remain close to the anchor at the beginning of fine-tuning.
    """

    def __init__(self, channels: int, group_names: list[str], bottleneck_channels: int = 192):
        super().__init__()
        hidden = max(min(bottleneck_channels, channels // 2), 96)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.group_blocks = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden, bias=False),
                    nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for group_name in group_names
            }
        )
        self.global_gate = nn.Parameter(torch.tensor(-4.0))
        self.group_gates = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.0)) for group_name in group_names}
        )
        self.max_residual_scale = 0.35

    def _gate(self, group_name: str | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate_logit = self.global_gate
        if group_name is not None and group_name in self.group_gates:
            gate_logit = gate_logit + self.group_gates[group_name]
        gate = self.max_residual_scale * torch.sigmoid(gate_logit)
        return gate.to(dtype=dtype, device=device).view(1, 1, 1, 1)

    def forward(self, x: torch.Tensor, group_name: str | None) -> torch.Tensor:
        if group_name is None:
            return x
        delta = self.shared(x)
        if group_name is not None and group_name in self.group_blocks:
            delta = delta + self.group_blocks[group_name](x)
        return x + self._gate(group_name, x.dtype, x.device) * delta


class TaskFiLMAdapter(nn.Module):
    """Task-conditioned FiLM modulation on post-neck features."""

    def __init__(self, channels: int, group_names: list[str], embedding_dim: int = 64):
        super().__init__()
        self.group_to_idx = {group_name: idx for idx, group_name in enumerate(group_names)}
        self.task_embedding = nn.Embedding(max(len(group_names), 1), embedding_dim)
        hidden = max(channels // 2, 128)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=max(channels // 16, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.modulator = nn.Sequential(
            nn.Linear(channels + embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 3),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, group_name: str | None) -> torch.Tensor:
        if group_name is None or group_name not in self.group_to_idx:
            return x

        pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        task_idx = torch.full(
            (x.shape[0],),
            fill_value=self.group_to_idx[group_name],
            device=x.device,
            dtype=torch.long,
        )
        task_embed = self.task_embedding(task_idx)
        gamma, beta, gate = self.modulator(torch.cat([pooled, task_embed], dim=1)).chunk(3, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gate = torch.sigmoid(gate).unsqueeze(-1).unsqueeze(-1)

        features = self.pre(x)
        modulated = gamma * features + beta
        delta = self.out(modulated)
        return x + self.residual_scale * gate * delta


class ContextExpertsAdapter(nn.Module):
    """Residual context mixer with capacity matched to task difficulty groups."""

    def __init__(self, channels: int, group_names: list[str]):
        super().__init__()
        hidden = max(channels, 192)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.group_blocks = nn.ModuleDict()
        for group_name in group_names:
            if group_name == "hard":
                expert_hidden = max(int(channels * 1.5), 320)
                block = nn.Sequential(
                    nn.Conv2d(channels, expert_hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(expert_hidden),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, expert_hidden, kernel_size=3, padding=2, dilation=2, bias=False),
                    nn.BatchNorm2d(expert_hidden),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, expert_hidden, kernel_size=3, padding=4, dilation=4, bias=False),
                    nn.BatchNorm2d(expert_hidden),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
            elif group_name == "medium":
                expert_hidden = max(channels, 224)
                block = nn.Sequential(
                    nn.Conv2d(channels, expert_hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(expert_hidden),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, expert_hidden, kernel_size=3, padding=2, dilation=2, bias=False),
                    nn.BatchNorm2d(expert_hidden),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
            else:
                expert_hidden = max(channels // 2, 128)
                block = nn.Sequential(
                    nn.Conv2d(channels, expert_hidden, kernel_size=3, padding=1, groups=max(channels // 32, 1), bias=False),
                    nn.BatchNorm2d(expert_hidden),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
            self.group_blocks[group_name] = block

        self.shared_scale = nn.Parameter(torch.tensor(0.60))
        self.group_scale = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.40)) for group_name in group_names}
        )

    def forward(self, x: torch.Tensor, group_name: str | None) -> torch.Tensor:
        shared_delta = self.shared(x)
        if group_name is None or group_name not in self.group_blocks:
            return x + self.shared_scale * shared_delta
        return x + self.shared_scale * shared_delta + self.group_scale[group_name] * self.group_blocks[group_name](x)


class ContextLocalAdapter(nn.Module):
    """Hybrid adapter that mixes global context first, then corrects with image detail."""

    def __init__(self, channels: int, group_names: list[str]):
        super().__init__()
        self.context = ResidualContextAdapter(channels, group_names)
        self.local = LocalRefineAdapter(channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        return mixed + self.local_scale * local_delta


class ContextLocalPairTransferAdapter(nn.Module):
    """Load-compatible context-local adapter plus shared segment transfer.

    The original context/local modules retain their checkpoint key layout. The
    new residual is shared by related measurement tasks and starts at exactly
    zero, so loading an offset128 checkpoint reproduces the anchor model before
    the transfer branch is trained.
    """

    PAIR_TASK_IDS = ("AOP", "FA", "FUGC", "IVC", "fetal_femur")

    def __init__(self, channels: int, group_names: list[str]):
        super().__init__()
        self.context = ResidualContextAdapter(channels, group_names)
        self.local = LocalRefineAdapter(channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        hidden = max(channels // 2, 96)
        groups = max(hidden // 32, 1)
        self.pair_transfer_input = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )
        self.pair_task_modulation = nn.ParameterDict(
            {
                task_id: nn.Parameter(torch.zeros(2, hidden))
                for task_id in self.PAIR_TASK_IDS
            }
        )
        self.pair_transfer_shared = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.pair_transfer_shared[-1].weight)
        nn.init.zeros_(self.pair_transfer_shared[-1].bias)
        self.pair_task_scale = nn.ParameterDict(
            {task_id: nn.Parameter(torch.tensor(0.10)) for task_id in self.PAIR_TASK_IDS}
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        mixed = features + self.context_scale * (context_features - features)
        refined_features = self.local(mixed, image, group_name)
        base = mixed + self.local_scale * (refined_features - mixed)

        if group_name not in self.pair_task_modulation:
            return base
        hidden = self.pair_transfer_input(base)
        modulation = self.pair_task_modulation[group_name]
        gamma = 0.10 * torch.tanh(modulation[0]).view(1, -1, 1, 1)
        beta = 0.10 * modulation[1].view(1, -1, 1, 1)
        hidden = hidden * (1.0 + gamma) + beta
        delta = self.pair_transfer_shared(hidden)
        return base + self.pair_task_scale[group_name] * delta


class FeatureStyleRandomizer(nn.Module):
    """MixStyle-style feature statistics randomization for hidden-domain robustness.

    The module has no trainable parameters and is active only in training mode.
    It mixes channel-wise feature mean/std between samples in the same task batch,
    encouraging the decoder to depend less on scanner-specific contrast/speckle
    style while preserving spatial landmark content.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0 or x.shape[0] < 2:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        normalized = (x - mean) / std

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.shape[0], 1, 1, 1))
        lam = lam.to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(x.shape[0], device=x.device)
        mixed_mean = lam * mean + (1.0 - lam) * mean[perm]
        mixed_std = lam * std + (1.0 - lam) * std[perm]
        return normalized * mixed_std + mixed_mean


class GradientReverseFn(torch.autograd.Function):
    """Identity in the forward pass; reverses feature gradients in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, float(lambd))


class DomainAdversarialHead(nn.Module):
    """Predict scanner/style pseudo-domain from features through a gradient reversal layer."""

    def __init__(self, in_channels: int, num_domain_classes: int, hidden_channels: int = 256):
        super().__init__()
        if int(num_domain_classes) < 2:
            raise ValueError("DomainAdversarialHead requires at least two pseudo-domain classes.")
        hidden_channels = max(int(hidden_channels), 128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_channels, int(num_domain_classes)),
        )

    def forward(self, features: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        return self.classifier(gradient_reverse(features, grl_lambda))


class UltrasoundTextureContextAdapter(nn.Module):
    """Dual-stream adapter that injects raw ultrasound texture and edge evidence.

    DINO features provide semantic context, but ultrasound biometry landmarks often
    sit on thin boundaries. This adapter builds a high-resolution texture stream
    from the input image plus Sobel edges and gates it into task-specific features.
    """

    def __init__(self, feature_channels: int, group_names: list[str], texture_channels: int = 160):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.texture_stem = nn.Sequential(
            nn.Conv2d(7, 40, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 120, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(120),
            nn.GELU(),
            nn.Conv2d(120, texture_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(texture_channels),
            nn.GELU(),
        )
        hidden = max(feature_channels, 256)
        self.shared_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + texture_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.group_fusions = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(hidden // 32, 1), bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                )
                for group_name in group_names
            }
        )
        self.gate = nn.Sequential(nn.Conv2d(hidden, feature_channels, kernel_size=1), nn.Sigmoid())
        self.delta = nn.Sequential(
            nn.Conv2d(hidden, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        self.context_scale = nn.Parameter(torch.tensor(0.80))
        self.texture_scale = nn.Parameter(torch.tensor(0.45))

    def _texture_inputs(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([image, gray, grad_x, grad_y, grad_mag], dim=1)

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        texture = self.texture_stem(self._texture_inputs(image))
        if texture.shape[-2:] != mixed.shape[-2:]:
            texture = F.interpolate(texture, size=mixed.shape[-2:], mode="bilinear", align_corners=False)

        hidden = self.shared_fusion(torch.cat([mixed, texture], dim=1))
        if group_name is not None and group_name in self.group_fusions:
            hidden = hidden + self.group_fusions[group_name](hidden)
        gate = self.gate(hidden)
        delta = self.delta(hidden)
        return mixed + self.texture_scale * gate * delta


class ContextLocalTextureResidualAdapter(nn.Module):
    """Load-compatible context-local adapter plus a small texture residual.

    This preserves the exact checkpoint key layout for the proven
    ContextLocalAdapter (`context`, `local`, `context_scale`, `local_scale`) and
    adds a zero-initialized raw-image texture correction. Warm-started models
    therefore begin from the old solution instead of relearning the adapter.
    """

    def __init__(self, feature_channels: int, group_names: list[str], texture_channels: int = 64):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        self.local = LocalRefineAdapter(feature_channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.texture_stem = nn.Sequential(
            nn.Conv2d(7, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, texture_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(texture_channels),
            nn.GELU(),
        )
        self.texture_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + texture_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        nn.init.zeros_(self.texture_fusion[-1].weight)
        nn.init.zeros_(self.texture_fusion[-1].bias)
        self.texture_scale = nn.Parameter(torch.tensor(0.10))

    def _texture_inputs(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([image, gray, grad_x, grad_y, grad_mag], dim=1)

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        base = mixed + self.local_scale * local_delta

        texture = self.texture_stem(self._texture_inputs(image))
        if texture.shape[-2:] != base.shape[-2:]:
            texture = F.interpolate(texture, size=base.shape[-2:], mode="bilinear", align_corners=False)
        texture_delta = self.texture_fusion(torch.cat([base, texture], dim=1))
        return base + self.texture_scale * texture_delta


class GatedContextLocalTextureResidualAdapter(nn.Module):
    """Context-local adapter with conservative task-gated texture residuals.

    Compared with texture_residual_v1, the residual correction is controlled by
    per-task gates initialized near zero. This keeps the warm-started model
    stable while allowing ultrasound edge/texture cues to help tasks differently.
    """

    def __init__(self, feature_channels: int, group_names: list[str], texture_channels: int = 64):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        self.local = LocalRefineAdapter(feature_channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.texture_stem = nn.Sequential(
            nn.Conv2d(7, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, texture_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(texture_channels),
            nn.GELU(),
        )
        self.texture_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + texture_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        nn.init.zeros_(self.texture_fusion[-1].weight)
        nn.init.zeros_(self.texture_fusion[-1].bias)
        self.global_texture_gate = nn.Parameter(torch.tensor(-2.0))
        self.group_texture_gates = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.0)) for group_name in group_names}
        )
        self.max_texture_scale = 0.18

    def _texture_inputs(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([image, gray, grad_x, grad_y, grad_mag], dim=1)

    def _texture_gate(self, group_name: str | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate_logit = self.global_texture_gate
        if group_name is not None and group_name in self.group_texture_gates:
            gate_logit = gate_logit + self.group_texture_gates[group_name]
        gate = self.max_texture_scale * torch.sigmoid(gate_logit)
        return gate.to(dtype=dtype, device=device).view(1, 1, 1, 1)

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        base = mixed + self.local_scale * local_delta

        texture = self.texture_stem(self._texture_inputs(image))
        if texture.shape[-2:] != base.shape[-2:]:
            texture = F.interpolate(texture, size=base.shape[-2:], mode="bilinear", align_corners=False)
        texture_delta = self.texture_fusion(torch.cat([base, texture], dim=1))
        return base + self._texture_gate(group_name, base.dtype, base.device) * texture_delta


class HighResolutionTextureResidualAdapter(nn.Module):
    """Context-local adapter with a true high-resolution ultrasound texture branch.

    The proven context-local pathway is kept load-compatible with existing
    checkpoints. A separate CNN branch processes image, grayscale, and Sobel
    channels at 128x128 before being projected into the DINO/FPN feature grid.
    Its final fusion is zero-initialized and task-gated, so warm-started models
    initially behave like the base model and only learn useful texture residuals.
    """

    def __init__(self, feature_channels: int, group_names: list[str], texture_channels: int = 80):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        self.local = LocalRefineAdapter(feature_channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.texture_stem = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.highres_refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, groups=96, bias=False),
            nn.Conv2d(96, texture_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(texture_channels),
            nn.GELU(),
        )
        self.texture_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + texture_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, groups=feature_channels, bias=False),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        nn.init.zeros_(self.texture_fusion[-1].weight)
        nn.init.zeros_(self.texture_fusion[-1].bias)

        self.global_texture_gate = nn.Parameter(torch.tensor(-2.5))
        self.group_texture_gates = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.0)) for group_name in group_names}
        )
        self.max_texture_scale = 0.25

    def _texture_inputs(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([image, gray, grad_x, grad_y, grad_mag], dim=1)

    def _texture_gate(self, group_name: str | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate_logit = self.global_texture_gate
        if group_name is not None and group_name in self.group_texture_gates:
            gate_logit = gate_logit + self.group_texture_gates[group_name]
        gate = self.max_texture_scale * torch.sigmoid(gate_logit)
        return gate.to(dtype=dtype, device=device).view(1, 1, 1, 1)

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        base = mixed + self.local_scale * local_delta

        texture = self.texture_stem(self._texture_inputs(image))
        texture = self.highres_refine(texture)
        if texture.shape[-2:] != base.shape[-2:]:
            texture = F.interpolate(texture, size=base.shape[-2:], mode="area")
        texture_delta = self.texture_fusion(torch.cat([base, texture], dim=1))
        return base + self._texture_gate(group_name, base.dtype, base.device) * texture_delta


class HRNetResidualAdapter(nn.Module):
    """Context-local adapter plus a zero-start HRNet detail residual branch.

    Standalone HRNet changed the coordinate system too much for the hidden set.
    This adapter keeps the warm-started DINO/context-local features as the base
    and lets HRNet contribute only a gated residual correction.
    """

    def __init__(
        self,
        feature_channels: int,
        group_names: list[str],
        hrnet_name: str = "hrnet_w32.ms_in1k",
        detail_channels: int = 128,
    ):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        self.local = LocalRefineAdapter(feature_channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        timm = importlib.import_module("timm")
        self.hrnet_encoder = timm.create_model(hrnet_name, pretrained=True, features_only=True)
        feature_channels_by_level = list(map(int, self.hrnet_encoder.feature_info.channels()))
        feature_reductions = list(map(int, self.hrnet_encoder.feature_info.reduction()))
        self.hrnet_target_index = self._resolve_target_index(feature_reductions)
        self.hrnet_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, detail_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(detail_channels),
                nn.GELU(),
            )
            for channels in feature_channels_by_level
        )
        self.hrnet_layer_logits = nn.Parameter(torch.zeros(len(feature_channels_by_level), dtype=torch.float32))
        self.hrnet_layer_logits.data[self.hrnet_target_index] = 2.5
        self.hrnet_refine = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.GELU(),
        )
        self.hrnet_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + detail_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, groups=feature_channels, bias=False),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        nn.init.zeros_(self.hrnet_fusion[-1].weight)
        nn.init.zeros_(self.hrnet_fusion[-1].bias)
        self.global_hrnet_gate = nn.Parameter(torch.tensor(-3.0))
        self.group_hrnet_gates = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.0)) for group_name in group_names}
        )
        self.max_hrnet_scale = 0.20

    @staticmethod
    def _resolve_target_index(reductions: list[int]) -> int:
        candidates = [idx for idx, reduction in enumerate(reductions) if reduction <= 8]
        if candidates:
            return candidates[-1]
        return max(len(reductions) - 2, 0)

    def _hrnet_gate(self, group_name: str | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate_logit = self.global_hrnet_gate
        if group_name is not None and group_name in self.group_hrnet_gates:
            gate_logit = gate_logit + self.group_hrnet_gates[group_name]
        gate = self.max_hrnet_scale * torch.sigmoid(gate_logit)
        return gate.to(dtype=dtype, device=device).view(1, 1, 1, 1)

    def _hrnet_detail(self, image: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        features = self.hrnet_encoder(image)
        weights = torch.softmax(self.hrnet_layer_logits.to(dtype=features[-1].dtype), dim=0)
        target_feature_size = features[self.hrnet_target_index].shape[-2:]
        detail = None
        for weight, feature, projection in zip(weights, features, self.hrnet_projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_feature_size:
                projected = F.interpolate(projected, size=target_feature_size, mode="bilinear", align_corners=False)
            weighted = weight.view(1, 1, 1, 1) * projected
            detail = weighted if detail is None else detail + weighted
        if detail is None:
            raise RuntimeError("HRNet detail branch did not return feature maps.")
        detail = self.hrnet_refine(detail)
        if detail.shape[-2:] != target_size:
            detail = F.interpolate(detail, size=target_size, mode="bilinear", align_corners=False)
        return detail

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        base = mixed + self.local_scale * local_delta

        detail = self._hrnet_detail(image, base.shape[-2:])
        delta = self.hrnet_fusion(torch.cat([base, detail], dim=1))
        return base + self._hrnet_gate(group_name, base.dtype, base.device) * delta


class PixelPyramidUNetFusionAdapter(nn.Module):
    """Load-compatible context-local adapter with a deeper pixel pyramid branch.

    The previous texture adapters add a shallow edge stream. This branch keeps the
    proven context/local pathway intact and learns a multi-scale CNN pyramid from
    raw ultrasound intensity, grayscale, Sobel, and gradient magnitude. The final
    fusion layer is zero-initialized, so a warm-started model initially behaves
    exactly like the offset128 anchor and can only learn bounded residual feature
    corrections.
    """

    def __init__(self, feature_channels: int, group_names: list[str], texture_channels: int = 96):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        self.local = LocalRefineAdapter(feature_channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.texture_stem = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.texture_enc1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.texture_enc2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, groups=96, bias=False),
            nn.Conv2d(96, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.texture_enc3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4, groups=128, bias=False),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.texture_pyramid_fusion = nn.Sequential(
            nn.Conv2d(32 + 64 + 96 + 128, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(192, texture_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(texture_channels),
            nn.GELU(),
        )
        self.group_texture_blocks = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.Conv2d(texture_channels, texture_channels, kernel_size=3, padding=1, groups=texture_channels, bias=False),
                    nn.Conv2d(texture_channels, texture_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(texture_channels),
                    nn.GELU(),
                )
                for group_name in group_names
            }
        )
        self.texture_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + texture_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, groups=feature_channels, bias=False),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        nn.init.zeros_(self.texture_fusion[-1].weight)
        nn.init.zeros_(self.texture_fusion[-1].bias)

        self.global_texture_gate = nn.Parameter(torch.tensor(-2.0))
        self.group_texture_gates = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.0)) for group_name in group_names}
        )
        self.max_texture_scale = 0.35

    def _texture_inputs(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([image, gray, grad_x, grad_y, grad_mag], dim=1)

    def _texture_gate(self, group_name: str | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate_logit = self.global_texture_gate
        if group_name is not None and group_name in self.group_texture_gates:
            gate_logit = gate_logit + self.group_texture_gates[group_name]
        gate = self.max_texture_scale * torch.sigmoid(gate_logit)
        return gate.to(dtype=dtype, device=device).view(1, 1, 1, 1)

    @staticmethod
    def _resize_to(feature: torch.Tensor, size: torch.Size | tuple[int, int]) -> torch.Tensor:
        if feature.shape[-2:] == size:
            return feature
        return F.interpolate(feature, size=size, mode="area")

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        base = mixed + self.local_scale * local_delta

        tex0 = self.texture_stem(self._texture_inputs(image))
        tex1 = self.texture_enc1(tex0)
        tex2 = self.texture_enc2(tex1)
        tex3 = self.texture_enc3(tex2)
        target_size = base.shape[-2:]
        texture = self.texture_pyramid_fusion(
            torch.cat(
                [
                    self._resize_to(tex0, target_size),
                    self._resize_to(tex1, target_size),
                    self._resize_to(tex2, target_size),
                    self._resize_to(tex3, target_size),
                ],
                dim=1,
            )
        )
        if group_name is not None and group_name in self.group_texture_blocks:
            texture = texture + self.group_texture_blocks[group_name](texture)
        texture_delta = self.texture_fusion(torch.cat([base, texture], dim=1))
        return base + self._texture_gate(group_name, base.dtype, base.device) * texture_delta


class BoundaryContextAdapter(nn.Module):
    """Context-local adapter with an explicit ultrasound boundary prior.

    The base context/local path keeps the same parameter names as
    ContextLocalAdapter, so offset128 checkpoints warm-start the proven feature
    pathway. The added branch computes pseudo-boundary channels from grayscale
    gradient magnitude and Laplacian contrast, then injects a zero-started,
    task-gated residual. This lets the model learn edge-conditioned landmarks
    without changing predictions at initialization.
    """

    def __init__(self, feature_channels: int, group_names: list[str], boundary_channels: int = 64):
        super().__init__()
        self.context = ResidualContextAdapter(feature_channels, group_names)
        self.local = LocalRefineAdapter(feature_channels, group_names)
        self.context_scale = nn.Parameter(torch.tensor(0.85))
        self.local_scale = nn.Parameter(torch.tensor(0.60))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace", laplace, persistent=False)

        self.boundary_stem = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, boundary_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(boundary_channels),
            nn.GELU(),
        )
        self.boundary_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + boundary_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=2, dilation=2, groups=feature_channels, bias=False),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        nn.init.zeros_(self.boundary_fusion[-1].weight)
        nn.init.zeros_(self.boundary_fusion[-1].bias)
        self.global_boundary_gate = nn.Parameter(torch.tensor(-2.4))
        self.group_boundary_gates = nn.ParameterDict(
            {group_name: nn.Parameter(torch.tensor(0.0)) for group_name in group_names}
        )
        self.max_boundary_scale = 0.22

    @staticmethod
    def _robust_unit(x: torch.Tensor) -> torch.Tensor:
        dims = (2, 3)
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
        return (x - x_min) / (x_max - x_min).clamp_min(1e-6)

    def _boundary_inputs(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        laplace = F.conv2d(gray, self.laplace.to(dtype=image.dtype), padding=1).abs()
        boundary = 0.65 * self._robust_unit(grad_mag) + 0.35 * self._robust_unit(laplace)
        return torch.cat(
            [
                gray,
                self._robust_unit(grad_x.abs()),
                self._robust_unit(grad_y.abs()),
                self._robust_unit(grad_mag),
                self._robust_unit(laplace),
                self._robust_unit(boundary),
            ],
            dim=1,
        )

    def _boundary_gate(self, group_name: str | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate_logit = self.global_boundary_gate
        if group_name is not None and group_name in self.group_boundary_gates:
            gate_logit = gate_logit + self.group_boundary_gates[group_name]
        gate = self.max_boundary_scale * torch.sigmoid(gate_logit)
        return gate.to(dtype=dtype, device=device).view(1, 1, 1, 1)

    def forward(self, features: torch.Tensor, image: torch.Tensor, group_name: str | None) -> torch.Tensor:
        context_features = self.context(features, group_name)
        context_delta = context_features - features
        mixed = features + self.context_scale * context_delta

        refined_features = self.local(mixed, image, group_name)
        local_delta = refined_features - mixed
        base = mixed + self.local_scale * local_delta

        boundary = self.boundary_stem(self._boundary_inputs(image))
        if boundary.shape[-2:] != base.shape[-2:]:
            boundary = F.interpolate(boundary, size=base.shape[-2:], mode="area")
        boundary_delta = self.boundary_fusion(torch.cat([base, boundary], dim=1))
        return base + self._boundary_gate(group_name, base.dtype, base.device) * boundary_delta


class LandmarkGraphRefineBlock(nn.Module):
    """Lightweight landmark-relation block for feature-conditioned coordinate refinement."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        x = self.norm1(tokens)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / max(float(q.shape[-1]) ** 0.5, 1.0),
            dim=-1,
        )
        x = residual + self.out_proj(torch.matmul(attn, v))
        x = x + self.ffn(self.norm2(x))
        return x


class LandmarkMemoryCrossAttentionBlock(nn.Module):
    """Let landmark queries retrieve spatial evidence from the full feature map."""

    def __init__(self, token_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.memory_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class LandmarkQueryRefiner(nn.Module):
    """Two-stage global-to-local coordinate refiner shared by any heatmap head.

    The first stage uses landmark queries to retrieve spatially indexed evidence
    from the complete task feature map. The second stage resamples features at
    the updated positions and performs a smaller correction. Both output layers
    are zero initialized, making the refiner an exact identity at initialization.
    """

    def __init__(self, in_channels: int, num_points: int, token_dim: int = 128):
        super().__init__()
        self.num_points = int(num_points)
        self.memory_projection = nn.Conv2d(in_channels, token_dim, kernel_size=1, bias=False)
        self.memory_coord_projection = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.coord_projection = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_feature_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_index_embedding = nn.Parameter(torch.zeros(1, self.num_points, token_dim))
        nn.init.trunc_normal_(self.point_index_embedding, std=0.02)

        self.stage1_cross = nn.ModuleList(
            LandmarkMemoryCrossAttentionBlock(token_dim) for _ in range(2)
        )
        self.stage1_graph = LandmarkGraphRefineBlock(token_dim)
        self.stage1_offset = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
            nn.Tanh(),
        )

        self.stage2_point_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.stage2_cross = LandmarkMemoryCrossAttentionBlock(token_dim)
        self.stage2_graph = LandmarkGraphRefineBlock(token_dim)
        self.stage2_offset = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
            nn.Tanh(),
        )
        self.stage1_max_offset = 0.040
        self.stage2_max_offset = 0.015

        for offset_head in (self.stage1_offset, self.stage2_offset):
            final_linear = offset_head[-2]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _coordinate_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        xs = torch.linspace(0.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).view(1, height * width, 2)

    @staticmethod
    def _sample_point_features(features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        sample_grid = points.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def _memory_tokens(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_projection(features)
        memory = projected.flatten(2).transpose(1, 2).contiguous()
        grid = self._coordinate_grid(projected.shape[-2], projected.shape[-1], projected)
        return memory + self.memory_coord_projection(grid)

    def forward(self, features: torch.Tensor, anchor_coords: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size = features.shape[0]
        anchor_points = anchor_coords.view(batch_size, self.num_points, 2)
        memory = self._memory_tokens(features)
        sampled = self._sample_point_features(features, anchor_points)
        global_feature = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        queries = (
            self.coord_projection(anchor_points)
            + self.point_feature_projection(sampled)
            + self.global_projection(global_feature).unsqueeze(1)
            + self.point_index_embedding
        )
        for block in self.stage1_cross:
            queries = block(queries, memory)
        queries = self.stage1_graph(queries)
        stage1_delta = self.stage1_offset(queries) * self.stage1_max_offset
        stage1_points = torch.clamp(anchor_points + stage1_delta, 0.0, 1.0)

        stage2_sampled = self._sample_point_features(features, stage1_points)
        stage2_queries = (
            queries
            + self.coord_projection(stage1_points)
            + self.stage2_point_projection(stage2_sampled)
        )
        stage2_queries = self.stage2_cross(stage2_queries, memory)
        stage2_queries = self.stage2_graph(stage2_queries)
        stage2_delta = self.stage2_offset(stage2_queries) * self.stage2_max_offset
        refined_points = torch.clamp(stage1_points + stage2_delta, 0.0, 1.0)
        return refined_points.reshape(batch_size, -1), {
            "query_stage1_coords_transformed": stage1_points.reshape(batch_size, -1),
            "query_stage1_offsets": stage1_delta,
            "query_stage2_offsets": stage2_delta,
        }


class AnatomicalPrototypeMemoryRefiner(nn.Module):
    """Content-conditioned shape memory applied to trusted heatmap coordinates.

    The module selects a mixture of learned normalized landmark prototypes from
    global and point-sampled image evidence. The selected shape is aligned to
    the heatmap prediction before a bounded local residual is applied. Both
    correction paths start as an exact identity, which protects warm starts.
    """

    def __init__(
        self,
        in_channels: int,
        num_points: int,
        num_prototypes: int = 12,
        token_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.num_prototypes = int(num_prototypes)
        self.token_dim = int(token_dim)

        self.global_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.point_feature_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.shape_projection = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_embedding = nn.Parameter(torch.zeros(1, self.num_points, token_dim))
        self.prototype_keys = nn.Parameter(torch.empty(self.num_prototypes, token_dim))
        self.prototype_shapes = nn.Parameter(
            self._initial_prototype_shapes(self.num_prototypes, self.num_points)
        )
        nn.init.trunc_normal_(self.point_embedding, std=0.02)
        nn.init.trunc_normal_(self.prototype_keys, std=0.02)

        self.query_norm = nn.LayerNorm(token_dim)
        self.local_refine = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
            nn.Tanh(),
        )
        nn.init.zeros_(self.local_refine[-2].weight)
        nn.init.zeros_(self.local_refine[-2].bias)

        # tanh(0) makes the prototype path an exact identity at warm start.
        self.prototype_blend_logit = nn.Parameter(torch.zeros(()))
        self.max_prototype_blend = 0.35
        self.max_prototype_delta = 0.10
        self.max_local_delta = 0.025

    @staticmethod
    def _initial_prototype_shapes(num_prototypes: int, num_points: int) -> torch.Tensor:
        point_angles = torch.linspace(0.0, 2.0 * math.pi, num_points + 1)[:-1]
        base = torch.stack([torch.cos(point_angles), torch.sin(point_angles)], dim=-1)
        shapes = []
        for prototype_index in range(num_prototypes):
            phase = 2.0 * math.pi * prototype_index / max(num_prototypes, 1)
            rotation = torch.tensor(
                [
                    [math.cos(phase), -math.sin(phase)],
                    [math.sin(phase), math.cos(phase)],
                ],
                dtype=torch.float32,
            )
            anisotropy = 0.75 + 0.5 * ((prototype_index % 3) / 2.0)
            shape = base @ rotation.T
            shape[:, 1] *= anisotropy
            shapes.append(shape)
        return torch.stack(shapes, dim=0)

    @staticmethod
    def _sample_point_features(features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        sample_grid = points.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    @staticmethod
    def _normalize_shape(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = points.mean(dim=1, keepdim=True)
        centered = points - center
        scale = centered.square().sum(dim=-1).mean(dim=1, keepdim=True).sqrt().unsqueeze(-1)
        scale = scale.clamp_min(0.025)
        return centered / scale, center, scale

    def forward(self, features: torch.Tensor, anchor_coords: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size = features.shape[0]
        anchor_points = anchor_coords.view(batch_size, self.num_points, 2)
        anchor_shape, anchor_center, anchor_scale = self._normalize_shape(anchor_points)

        sampled = self._sample_point_features(features, anchor_points)
        global_feature = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        point_tokens = (
            self.point_feature_projection(sampled)
            + self.coordinate_projection(anchor_points)
            + self.shape_projection(anchor_shape)
            + self.point_embedding
        )
        query = self.query_norm(self.global_projection(global_feature) + point_tokens.mean(dim=1))
        prototype_logits = torch.matmul(query, self.prototype_keys.T) / math.sqrt(self.token_dim)
        prototype_weights = torch.softmax(prototype_logits, dim=-1)
        prototype_shape = torch.einsum(
            "bk,kpc->bpc",
            prototype_weights,
            self.prototype_shapes,
        )
        prototype_shape, _, _ = self._normalize_shape(prototype_shape)
        prototype_candidate = anchor_center + anchor_scale * prototype_shape
        prototype_delta = (prototype_candidate - anchor_points).clamp(
            -self.max_prototype_delta,
            self.max_prototype_delta,
        )

        local_delta = self.local_refine(point_tokens) * self.max_local_delta
        prototype_blend = self.max_prototype_blend * torch.tanh(self.prototype_blend_logit)
        refined_points = torch.clamp(
            anchor_points + prototype_blend * prototype_delta + local_delta,
            0.0,
            1.0,
        )
        return refined_points.reshape(batch_size, -1), {
            "prototype_candidate_coords_transformed": prototype_candidate.reshape(batch_size, -1),
            "prototype_weights": prototype_weights,
            "prototype_blend": prototype_blend,
            "prototype_local_offsets": local_delta,
        }


class SharedTaskConditionedLandmarkQueryRefiner(LandmarkQueryRefiner):
    """Low-capacity query refiner shared across all landmark tasks.

    Sharing the spatial retrieval and graph blocks lets low-sample tasks learn
    coordinate corrections from the complete multi-task training set. Task and
    point embeddings retain dataset-specific landmark semantics.
    """

    def __init__(
        self,
        in_channels: int,
        task_ids: list[str],
        max_num_points: int,
        token_dim: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            num_points=max_num_points,
            token_dim=token_dim,
        )
        self.task_ids = tuple(str(task_id) for task_id in task_ids)
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.task_embedding = nn.Embedding(len(self.task_ids), token_dim)
        self.query_dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.task_embedding.weight, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        anchor_coords: torch.Tensor,
        task_id: str,
    ) -> tuple[torch.Tensor, dict]:
        if task_id not in self.task_to_index:
            raise ValueError(f"Task ID '{task_id}' is not registered in the shared query refiner.")

        batch_size = features.shape[0]
        num_points = anchor_coords.shape[1] // 2
        if num_points > self.num_points:
            raise ValueError(
                f"Task '{task_id}' has {num_points} points, exceeding the configured "
                f"maximum of {self.num_points}."
            )

        anchor_points = anchor_coords.view(batch_size, num_points, 2)
        memory = self._memory_tokens(features)
        sampled = self._sample_point_features(features, anchor_points)
        global_feature = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        task_index = torch.full(
            (batch_size,),
            self.task_to_index[task_id],
            device=features.device,
            dtype=torch.long,
        )
        task_token = self.task_embedding(task_index).unsqueeze(1)
        queries = (
            self.coord_projection(anchor_points)
            + self.point_feature_projection(sampled)
            + self.global_projection(global_feature).unsqueeze(1)
            + self.point_index_embedding[:, :num_points]
            + task_token
        )
        queries = self.query_dropout(queries)
        for block in self.stage1_cross:
            queries = block(queries, memory)
        queries = self.stage1_graph(queries)
        stage1_delta = self.stage1_offset(queries) * self.stage1_max_offset
        stage1_points = torch.clamp(anchor_points + stage1_delta, 0.0, 1.0)

        stage2_sampled = self._sample_point_features(features, stage1_points)
        stage2_queries = (
            queries
            + self.coord_projection(stage1_points)
            + self.stage2_point_projection(stage2_sampled)
            + task_token
        )
        stage2_queries = self.query_dropout(stage2_queries)
        stage2_queries = self.stage2_cross(stage2_queries, memory)
        stage2_queries = self.stage2_graph(stage2_queries)
        stage2_delta = self.stage2_offset(stage2_queries) * self.stage2_max_offset
        refined_points = torch.clamp(stage1_points + stage2_delta, 0.0, 1.0)
        return refined_points.reshape(batch_size, -1), {
            "query_stage1_coords_transformed": stage1_points.reshape(batch_size, -1),
            "query_stage1_offsets": stage1_delta,
            "query_stage2_offsets": stage2_delta,
        }


CANONICAL_PAIR_RULES = {
    "A4C": (tuple((index, index + 1) for index in range(0, 16, 2)), "y_asc_stable"),
    "PSAX": (((0, 1), (2, 3)), "y_asc_stable"),
}


def canonicalize_landmark_pairs(coords: torch.Tensor, task_id: str) -> torch.Tensor:
    """Apply the organizer-defined upper-to-lower ordering to cardiac pairs."""
    rule = CANONICAL_PAIR_RULES.get(task_id)
    if rule is None:
        return coords

    pairs, ordering = rule
    if ordering != "y_asc_stable":
        raise ValueError(f"Unsupported cardiac pair ordering: {ordering}")
    points = coords.view(coords.shape[0], -1, 2)
    ordered = points.clone()
    for first_index, second_index in pairs:
        first = points[:, first_index]
        second = points[:, second_index]
        swap = first[:, 1] > second[:, 1]
        ordered[:, first_index] = torch.where(swap.unsqueeze(-1), second, first)
        ordered[:, second_index] = torch.where(swap.unsqueeze(-1), first, second)
    return ordered.reshape(coords.shape[0], -1)


class HeatmapHead(nn.Module):
    """Light decoder that maps DINOv2 feature maps to keypoint heatmaps."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        base_hidden = max(in_channels // 2, 128)
        base_hidden2 = max(base_hidden // 2, 64)
        hidden = max(int(base_hidden * width_multiplier), 64)
        hidden2 = max(int(base_hidden2 * width_multiplier), 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DeepHeatmapHead(nn.Module):
    """Stronger decoder head for finer keypoint localization."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        base_hidden1 = max(in_channels // 2, 192)
        base_hidden2 = max(base_hidden1 // 2, 128)
        base_hidden3 = max(base_hidden2 // 2, 96)
        hidden1 = max(int(base_hidden1 * width_multiplier), 96)
        hidden2 = max(int(base_hidden2 * width_multiplier), 64)
        hidden3 = max(int(base_hidden3 * width_multiplier), 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class SubpixelOffsetDeepHeatmapHead(nn.Module):
    """Deep heatmap decoder plus learned per-landmark subpixel offsets.

    The decoder attribute intentionally matches DeepHeatmapHead so existing
    checkpoints can warm-start the heatmap branch while the offset branch learns
    a local correction from FPN features.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        base_hidden1 = max(in_channels // 2, 192)
        base_hidden2 = max(base_hidden1 // 2, 128)
        base_hidden3 = max(base_hidden2 // 2, 96)
        hidden1 = max(int(base_hidden1 * width_multiplier), 96)
        hidden2 = max(int(base_hidden2 * width_multiplier), 64)
        hidden3 = max(int(base_hidden3 * width_multiplier), 64)
        self.num_points = int(num_points)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )
        token_dim = max(in_channels // 2, 128)
        self.coord_embed = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_feat_proj = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_index_embed = nn.Parameter(torch.zeros(1, self.num_points, token_dim))
        nn.init.trunc_normal_(self.point_index_embed, std=0.02)
        self.refine_blocks = nn.ModuleList([LandmarkGraphRefineBlock(token_dim) for _ in range(2)])
        self.offset_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.030

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    @staticmethod
    def _sample_point_features(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        point_coords = coords.view(bsz, -1, 2)
        grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, out_size):
        refine_features = x
        point_logits = self.decoder(x)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(refine_features, coarse_coords)
        global_feature = F.adaptive_avg_pool2d(refine_features, output_size=1).flatten(1)
        tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(global_feature).unsqueeze(1)
            + self.point_index_embed
        )
        for block in self.refine_blocks:
            tokens = block(tokens)
        offsets = self.offset_head(tokens) * self.refine_max_offset
        refined_coords = torch.clamp(coarse_points + offsets, 0.0, 1.0).reshape(coarse_coords.shape[0], -1)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class AxisDistributionOffsetDeepHeatmapHead(SubpixelOffsetDeepHeatmapHead):
    """Offset head plus a learned SimCC-style X/Y coordinate distribution.

    The inherited decoder/refinement names match SubpixelOffsetDeepHeatmapHead
    so offset128 checkpoints warm-start most parameters. The axis branch starts
    with a small fusion gate and only moves predictions when validation supports
    the additional coordinate evidence.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels=in_channels, num_points=num_points, width_multiplier=width_multiplier)
        token_dim = int(self.point_index_embed.shape[-1])
        self.axis_bins = 128
        self.axis_x_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, self.axis_bins),
        )
        self.axis_y_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, self.axis_bins),
        )
        self.axis_gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer(
            "axis_positions",
            torch.linspace(0.0, 1.0, self.axis_bins).view(1, 1, self.axis_bins),
            persistent=False,
        )

    def _axis_coords_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        axis_positions = self.axis_positions.to(device=tokens.device, dtype=tokens.dtype)
        x_probs = torch.softmax(self.axis_x_head(tokens), dim=-1)
        y_probs = torch.softmax(self.axis_y_head(tokens), dim=-1)
        x_coords = (x_probs * axis_positions).sum(dim=-1)
        y_coords = (y_probs * axis_positions).sum(dim=-1)
        return torch.stack([x_coords, y_coords], dim=-1)

    def forward(self, x: torch.Tensor, out_size):
        refine_features = x
        point_logits = self.decoder(x)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(refine_features, coarse_coords)
        global_feature = F.adaptive_avg_pool2d(refine_features, output_size=1).flatten(1)
        tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(global_feature).unsqueeze(1)
            + self.point_index_embed
        )
        for block in self.refine_blocks:
            tokens = block(tokens)

        offsets = self.offset_head(tokens) * self.refine_max_offset
        offset_points = torch.clamp(coarse_points + offsets, 0.0, 1.0)
        axis_points = self._axis_coords_from_tokens(tokens)
        axis_gate = torch.sigmoid(self.axis_gate_logit)
        fused_points = torch.clamp(offset_points + axis_gate * (axis_points - offset_points), 0.0, 1.0)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "offset_coords_transformed": offset_points.reshape(coarse_coords.shape[0], -1),
            "axis_coords_transformed": axis_points.reshape(coarse_coords.shape[0], -1),
            "refined_coords_transformed": fused_points.reshape(coarse_coords.shape[0], -1),
        }


class AOPSegmentVectorOffsetHeatmapHead(SubpixelOffsetDeepHeatmapHead):
    """AOP-specific offset head with two-line geometry refinement.

    AOP is defined by two line segments. This head keeps the warm-started
    heatmap and endpoint-offset path, then applies a zero-started correction to
    each segment midpoint, angle, and length. The initial prediction is exactly
    the inherited offset model; training can learn AOP geometry without moving
    unrelated task heads.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 4:
            raise ValueError("AOPSegmentVectorOffsetHeatmapHead requires exactly 4 points.")
        super().__init__(in_channels=in_channels, num_points=num_points, width_multiplier=width_multiplier)
        token_dim = int(self.point_index_embed.shape[-1])
        self.segment_refine_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 4),
            nn.Tanh(),
        )
        final_linear = self.segment_refine_head[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        self.segment_gate_logit = nn.Parameter(torch.tensor(-1.6))
        self.segment_mid_max_offset = 0.025
        self.segment_angle_max_delta = 0.18
        self.segment_log_length_max_delta = 0.16

    def _apply_segment_refine(self, points: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        bsz = points.shape[0]
        refined = points.clone()
        pair_indices = ((0, 1), (2, 3))
        pair_tokens = torch.stack(
            [tokens[:, [start_idx, end_idx], :].mean(dim=1) for start_idx, end_idx in pair_indices],
            dim=1,
        )
        params = self.segment_refine_head(pair_tokens)
        midpoint_delta = params[..., 0:2] * self.segment_mid_max_offset
        angle_delta = params[..., 2] * self.segment_angle_max_delta
        log_length_delta = params[..., 3] * self.segment_log_length_max_delta

        for pair_idx, (start_idx, end_idx) in enumerate(pair_indices):
            segment = points[:, [start_idx, end_idx], :]
            midpoint = segment.mean(dim=1) + midpoint_delta[:, pair_idx, :]
            vector = segment[:, 1, :] - segment[:, 0, :]
            length = torch.norm(vector, dim=-1, keepdim=True).clamp_min(1e-4)
            unit = vector / length
            cos_delta = torch.cos(angle_delta[:, pair_idx]).unsqueeze(-1)
            sin_delta = torch.sin(angle_delta[:, pair_idx]).unsqueeze(-1)
            rotated_unit = torch.stack(
                [
                    unit[:, 0] * cos_delta[:, 0] - unit[:, 1] * sin_delta[:, 0],
                    unit[:, 0] * sin_delta[:, 0] + unit[:, 1] * cos_delta[:, 0],
                ],
                dim=-1,
            )
            refined_length = length * torch.exp(log_length_delta[:, pair_idx]).unsqueeze(-1)
            refined_vector = rotated_unit * refined_length
            refined[:, start_idx, :] = midpoint - 0.5 * refined_vector
            refined[:, end_idx, :] = midpoint + 0.5 * refined_vector
        return refined.clamp(0.0, 1.0).reshape(bsz, self.num_points * 2)

    def forward(self, x: torch.Tensor, out_size):
        refine_features = x
        point_logits = self.decoder(x)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(refine_features, coarse_coords)
        global_feature = F.adaptive_avg_pool2d(refine_features, output_size=1).flatten(1)
        tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(global_feature).unsqueeze(1)
            + self.point_index_embed
        )
        for block in self.refine_blocks:
            tokens = block(tokens)
        offsets = self.offset_head(tokens) * self.refine_max_offset
        endpoint_points = torch.clamp(coarse_points + offsets, 0.0, 1.0)
        segment_coords = self._apply_segment_refine(endpoint_points, tokens)
        endpoint_coords = endpoint_points.reshape(coarse_coords.shape[0], -1)
        segment_gate = torch.sigmoid(self.segment_gate_logit)
        refined_coords = torch.clamp(
            endpoint_coords + segment_gate * (segment_coords - endpoint_coords),
            0.0,
            1.0,
        )
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "endpoint_refined_coords_transformed": endpoint_coords,
            "segment_refined_coords_transformed": segment_coords,
            "refined_coords_transformed": refined_coords,
        }


class StructureAuxiliaryMixin:
    """Adds a differentiable anatomy-structure map branch to landmark heads."""

    def _init_structure_auxiliary(self, num_points: int, hidden_channels: int = 32) -> None:
        hidden_channels = max(int(hidden_channels), 16)
        self.structure_head = nn.Sequential(
            nn.Conv2d(num_points, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def _append_structure_logits(self, point_logits: torch.Tensor, aux_outputs: dict | None) -> dict:
        if aux_outputs is None:
            aux_outputs = {}
        else:
            aux_outputs = dict(aux_outputs)
        aux_outputs["structure_logits"] = self.structure_head(point_logits)
        return aux_outputs


class StructureOffsetDeepHeatmapHead(StructureAuxiliaryMixin, SubpixelOffsetDeepHeatmapHead):
    """Offset heatmap head regularized by an anatomy structure map."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        self._init_structure_auxiliary(num_points=num_points, hidden_channels=max(32, num_points * 4))

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        return point_logits, self._append_structure_logits(point_logits, aux_outputs)


class LogitPatchOffsetRefiner(nn.Module):
    """Small zero-started local correction from heatmap evidence around each point."""

    def __init__(
        self,
        in_channels: int,
        num_points: int,
        hidden_channels: int = 64,
        patch_size: int = 13,
        span: float = 0.055,
        max_offset: float = 0.012,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.patch_size = int(patch_size)
        self.span = float(span)
        self.max_offset = float(max_offset)
        self.refine_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 2),
            nn.Tanh(),
        )
        final_linear = self.refine_head[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _sample_patches(self, context: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        point_coords = coords.view(bsz, self.num_points, 2)
        grid_lin = torch.linspace(-1.0, 1.0, self.patch_size, device=context.device, dtype=context.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, self.patch_size, self.patch_size, 2)
        patches = []
        for point_idx in range(self.num_points):
            center = point_coords[:, point_idx, :].view(bsz, 1, 1, 2)
            sample_grid = center + base_grid * self.span
            sample_grid = sample_grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            patches.append(
                F.grid_sample(
                    context,
                    sample_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
            )
        return torch.cat(patches, dim=0)

    def forward(self, context: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        patches = self._sample_patches(context, coords)
        offsets = self.refine_head(patches).view(self.num_points, bsz, 2).transpose(0, 1)
        points = coords.view(bsz, self.num_points, 2)
        refined = torch.clamp(points + offsets * self.max_offset, 0.0, 1.0)
        return refined.reshape(bsz, self.num_points * 2)


class LineHeatmapHead(nn.Module):
    """Decoder biased toward elongated 2-point structures."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 160) * width_multiplier), 96)
        hidden2 = max(int(max(hidden1 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.axial_h = nn.Conv2d(hidden1, hidden1, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.axial_v = nn.Conv2d(hidden1, hidden1, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.stem(x)
        x = self.bn(self.axial_h(x) + self.axial_v(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class CompactHeatmapHead(nn.Module):
    """Balanced decoder for compact anatomical targets."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 160) * width_multiplier), 96)
        hidden2 = max(int(max(hidden1 // 2, 96) * width_multiplier), 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DenseRelationalHeatmapHead(nn.Module):
    """Heavier decoder for dense multi-landmark relational structures."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 224) * width_multiplier), 128)
        hidden2 = max(int(max(hidden1 // 2, 160) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 128) * width_multiplier), 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
        )
        self.proj = nn.Conv2d(in_channels, hidden1, kernel_size=1, bias=False)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.block1(x) + self.proj(x)
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class FemurHeatmapHead(nn.Module):
    """Dedicated decoder for fetal femur endpoint localization plus shaft evidence."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 224) * width_multiplier), 128)
        hidden2 = max(int(max(hidden1 // 2, 160) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 128) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.axial_h = nn.Conv2d(hidden1, hidden1, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.axial_v = nn.Conv2d(hidden1, hidden1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.axial_bn = nn.BatchNorm2d(hidden1)
        self.shared_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.shaft_head = nn.Sequential(
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.axial_bn(self.axial_h(x) + self.axial_v(x))
        x = self.shared_decoder(x)
        point_logits = self.point_head(x)
        shaft_logits = self.shaft_head(x)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if shaft_logits.shape[-2:] != out_size:
            shaft_logits = F.interpolate(shaft_logits, size=out_size, mode="bilinear", align_corners=False)
        return point_logits, {"shaft_logits": shaft_logits}


class FUGCHeatmapHead(nn.Module):
    """Dedicated local-refinement decoder for short 2-point FUGC structures."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 192) * width_multiplier), 112)
        hidden2 = max(int(max(hidden1 // 2, 128) * width_multiplier), 80)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.segment_head = nn.Sequential(
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, 1, kernel_size=1),
        )
        self.refine_patch_size = 14
        refine_in_channels = hidden3 + num_points + 1
        refine_hidden = max(hidden3, 64)
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, num_points * 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.075

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def _sample_refine_roi(self, feat: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.refine_patch_size
        point_pairs = coarse_coords.reshape(bsz, -1, 2)
        midpoint = point_pairs.mean(dim=1)
        segment_length = torch.norm(point_pairs[:, 1] - point_pairs[:, 0], dim=-1)
        half_span = (segment_length * 1.75).clamp(0.10, 0.28)

        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)

        center = midpoint.view(bsz, 1, 1, 2)
        span = half_span.view(bsz, 1, 1, 1)
        sample_grid = center + base_grid * span
        sample_grid = sample_grid.clamp(0.0, 1.0)
        sample_grid = sample_grid * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.refine(x)
        point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(segment_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat(
            [
                feat_for_refine,
                torch.sigmoid(point_logits),
                torch.sigmoid(segment_logits),
            ],
            dim=1,
        )
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        offsets = self.refine_head(roi_patch) * self.refine_max_offset
        refined_coords = (coarse_coords + offsets).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class FUGCDirectViTPoseHead(nn.Module):
    """Direct ViTPose-style decoder over the shared encoder patch map."""

    uses_encoder_features = True

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        if int(num_points) != 2:
            raise ValueError("FUGCDirectViTPoseHead requires exactly 2 points.")
        hidden = 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden, num_points, kernel_size=1)
        self.segment_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, height, width = logits.shape
        probs = torch.softmax(logits.flatten(2), dim=-1).view(bsz, num_points, height, width)
        xs = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        expected_x = (probs * xs.view(1, 1, 1, width)).sum(dim=(-2, -1))
        expected_y = (probs * ys.view(1, 1, height, 1)).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def forward(self, x: torch.Tensor, out_size):
        features = self.decoder(x)
        point_logits = self.point_head(features)
        segment_logits = self.segment_head(features)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(segment_logits, size=out_size, mode="bilinear", align_corners=False)
        coords = self._softargmax_coords(point_logits)
        return point_logits, {
            "segment_logits": segment_logits,
            "refined_coords_transformed": coords,
        }


class FUGCDualScaleHeatmapHead(FUGCHeatmapHead):
    """Warm-start-compatible FUGC head with an image-detail residual path.

    The inherited DINO/FPN decoder provides semantic context. A separate
    grayscale-and-gradient CNN retains boundary detail at heatmap resolution.
    Its output layers are zero initialized, making the initial prediction
    exactly equal to the trusted FUGC head.
    """

    requires_input_image = True

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 2:
            raise ValueError("FUGCDualScaleHeatmapHead requires exactly 2 points.")
        super().__init__(in_channels, num_points, width_multiplier)
        detail_channels = 64
        self.register_buffer(
            "detail_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "detail_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("detail_sobel_x", sobel_x, persistent=False)
        self.register_buffer("detail_sobel_y", sobel_y, persistent=False)

        self.detail_stem = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, detail_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
            nn.Conv2d(
                detail_channels,
                detail_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=detail_channels,
                bias=False,
            ),
            nn.Conv2d(detail_channels, detail_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
        )
        self.detail_point_delta = nn.Conv2d(detail_channels, num_points, kernel_size=1)
        self.detail_segment_delta = nn.Conv2d(detail_channels, 1, kernel_size=1)
        nn.init.zeros_(self.detail_point_delta.weight)
        nn.init.zeros_(self.detail_point_delta.bias)
        nn.init.zeros_(self.detail_segment_delta.weight)
        nn.init.zeros_(self.detail_segment_delta.bias)

        detail_refine_channels = detail_channels + num_points + 1
        self.detail_refine_head = nn.Sequential(
            nn.Conv2d(detail_refine_channels, detail_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(detail_channels, 4),
            nn.Tanh(),
        )
        detail_final = self.detail_refine_head[-2]
        nn.init.zeros_(detail_final.weight)
        nn.init.zeros_(detail_final.bias)
        self.detail_refine_max_offset = 0.035

    def _detail_inputs(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.detail_image_mean.to(dtype=image.dtype, device=image.device)
        std = self.detail_image_std.to(dtype=image.dtype, device=image.device)
        restored = (image * std + mean).clamp(0.0, 1.0)
        gray = (
            0.2989 * restored[:, 0:1]
            + 0.5870 * restored[:, 1:2]
            + 0.1140 * restored[:, 2:3]
        )
        grad_x = F.conv2d(gray, self.detail_sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.detail_sobel_y.to(dtype=image.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([restored, grad_mag], dim=1)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        if input_image is None:
            raise ValueError("FUGCDualScaleHeatmapHead requires the normalized input image.")

        x = self.stem(x)
        x = self.refine(x)
        base_point_logits = self.point_head(x)
        base_segment_logits = self.segment_head(x)
        feat_for_refine = x
        if base_point_logits.shape[-2:] != out_size:
            base_point_logits = F.interpolate(
                base_point_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if base_segment_logits.shape[-2:] != out_size:
            base_segment_logits = F.interpolate(
                base_segment_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(
                feat_for_refine, size=out_size, mode="bilinear", align_corners=False
            )

        detail_features = self.detail_stem(self._detail_inputs(input_image))
        if detail_features.shape[-2:] != out_size:
            detail_features = F.interpolate(
                detail_features, size=out_size, mode="bilinear", align_corners=False
            )
        detail_point_logits = self.detail_point_delta(detail_features)
        detail_segment_logits = self.detail_segment_delta(detail_features)
        point_logits = base_point_logits + detail_point_logits
        segment_logits = base_segment_logits + detail_segment_logits

        coarse_coords = self._softargmax_coords(point_logits)
        semantic_refine_input = torch.cat(
            [feat_for_refine, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)],
            dim=1,
        )
        semantic_patch = self._sample_refine_roi(semantic_refine_input, coarse_coords)
        semantic_offsets = self.refine_head(semantic_patch) * self.refine_max_offset
        semantic_coords = (coarse_coords + semantic_offsets).clamp(0.0, 1.0)

        detail_refine_input = torch.cat(
            [detail_features, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)],
            dim=1,
        )
        detail_patch = self._sample_refine_roi(detail_refine_input, semantic_coords)
        detail_offsets = self.detail_refine_head(detail_patch) * self.detail_refine_max_offset
        refined_coords = (semantic_coords + detail_offsets).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "detail_point_logits": detail_point_logits,
            "detail_segment_logits": detail_segment_logits,
            "coarse_coords_transformed": coarse_coords,
            "pre_detail_coords_transformed": semantic_coords,
            "refined_coords_transformed": refined_coords,
        }


class FUGCAnatomyEndpointHeatmapHead(FUGCHeatmapHead):
    """Asymmetric FUGC refinement for the two distinct endpoint anatomies.

    Point 1 is localized on the fetal-head boundary, whereas point 2 is
    localized on the bright pubic-symphysis ridge. The inherited decoder is
    retained as a frozen semantic anchor. Zero-started endpoint-specific image
    branches add heatmap and local coordinate residuals without changing the
    warm-start prediction at initialization.
    """

    requires_input_image = True

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 2:
            raise ValueError("FUGCAnatomyEndpointHeatmapHead requires exactly 2 points.")
        super().__init__(in_channels, num_points, width_multiplier)
        semantic_channels = 24
        anatomy_channels = 48
        self.register_buffer(
            "anatomy_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_laplacian",
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_dxx",
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_dyy",
            torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "anatomy_dxy",
            torch.tensor(
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 4.0,
            persistent=False,
        )

        def image_stem():
            return nn.Sequential(
                nn.Conv2d(13, 32, kernel_size=5, padding=2, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, anatomy_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, anatomy_channels),
                nn.GELU(),
                nn.Conv2d(
                    anatomy_channels,
                    anatomy_channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    groups=anatomy_channels,
                    bias=False,
                ),
                nn.Conv2d(anatomy_channels, anatomy_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, anatomy_channels),
                nn.GELU(),
            )

        base_feature_channels = self.point_head.in_channels
        self.anatomy_semantic_proj = nn.Sequential(
            nn.Conv2d(base_feature_channels, semantic_channels, kernel_size=1, bias=False),
            nn.GroupNorm(6, semantic_channels),
            nn.GELU(),
        )
        self.anatomy_boundary_stem = image_stem()
        self.anatomy_ridge_stem = image_stem()
        fused_channels = anatomy_channels + semantic_channels
        self.anatomy_boundary_fuse = nn.Sequential(
            nn.Conv2d(fused_channels, anatomy_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, anatomy_channels),
            nn.GELU(),
        )
        self.anatomy_ridge_fuse = nn.Sequential(
            nn.Conv2d(fused_channels, anatomy_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, anatomy_channels),
            nn.GELU(),
        )
        self.anatomy_boundary_logit_delta = nn.Conv2d(anatomy_channels, 1, kernel_size=1)
        self.anatomy_ridge_logit_delta = nn.Conv2d(anatomy_channels, 1, kernel_size=1)
        nn.init.zeros_(self.anatomy_boundary_logit_delta.weight)
        nn.init.zeros_(self.anatomy_boundary_logit_delta.bias)
        nn.init.zeros_(self.anatomy_ridge_logit_delta.weight)
        nn.init.zeros_(self.anatomy_ridge_logit_delta.bias)

        local_channels = anatomy_channels + semantic_channels + num_points + 2

        def endpoint_refiner():
            return nn.Sequential(
                nn.Conv2d(local_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 2),
                nn.Tanh(),
            )

        self.anatomy_boundary_refiner = endpoint_refiner()
        self.anatomy_ridge_refiner = endpoint_refiner()
        for refiner in (self.anatomy_boundary_refiner, self.anatomy_ridge_refiner):
            final = refiner[-2]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.anatomy_boundary_max_offset = 0.055
        self.anatomy_ridge_max_offset = 0.090
        self.anatomy_patch_size = 24

    def _anatomy_inputs(self, image: torch.Tensor, out_size) -> torch.Tensor:
        mean = self.anatomy_image_mean.to(dtype=image.dtype, device=image.device)
        std = self.anatomy_image_std.to(dtype=image.dtype, device=image.device)
        restored = (image * std + mean).clamp(0.0, 1.0)
        gray = (
            0.2989 * restored[:, 0:1]
            + 0.5870 * restored[:, 1:2]
            + 0.1140 * restored[:, 2:3]
        )
        gray = F.interpolate(gray, size=out_size, mode="bilinear", align_corners=False)
        smooth3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        smooth7 = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        sobel_x = self.anatomy_sobel_x.to(dtype=image.dtype, device=image.device)
        sobel_y = self.anatomy_sobel_y.to(dtype=image.dtype, device=image.device)
        laplacian = self.anatomy_laplacian.to(dtype=image.dtype, device=image.device)
        grad_x = F.conv2d(smooth3, sobel_x, padding=1) / 4.0
        grad_y = F.conv2d(smooth3, sobel_y, padding=1) / 4.0
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        lap3 = F.conv2d(smooth3, laplacian, padding=1) / 4.0
        lap7 = F.conv2d(smooth7, laplacian, padding=1) / 4.0
        dxx = F.conv2d(smooth3, self.anatomy_dxx.to(dtype=image.dtype), padding=1)
        dyy = F.conv2d(smooth3, self.anatomy_dyy.to(dtype=image.dtype), padding=1)
        dxy = F.conv2d(smooth3, self.anatomy_dxy.to(dtype=image.dtype), padding=1)
        anisotropy = torch.sqrt((dxx - dyy).square() + 4.0 * dxy.square() + 1e-6)
        min_eigenvalue = 0.5 * (dxx + dyy - anisotropy)
        bright_ridge = F.relu(-min_eigenvalue)
        variance = (F.avg_pool2d(gray.square(), 7, 1, 3) - smooth7.square()).clamp_min(0.0)
        height, width = gray.shape[-2:]
        coord_x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        coord_y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        coord_x = coord_x.view(1, 1, 1, width).expand(gray.shape[0], -1, height, -1)
        coord_y = coord_y.view(1, 1, height, 1).expand(gray.shape[0], -1, -1, width)
        return torch.cat(
            [
                gray,
                smooth3,
                smooth7,
                grad_x,
                grad_y,
                grad_mag,
                lap3,
                lap7,
                bright_ridge,
                anisotropy,
                variance,
                coord_x,
                coord_y,
            ],
            dim=1,
        )

    def _sample_endpoint_patch(
        self,
        feature_map: torch.Tensor,
        coords: torch.Tensor,
        point_index: int,
        half_span: float,
    ) -> torch.Tensor:
        batch_size = feature_map.shape[0]
        center = coords.reshape(batch_size, 2, 2)[:, point_index]
        axis = torch.linspace(
            -1.0,
            1.0,
            self.anatomy_patch_size,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        relative_grid = torch.stack([grid_x, grid_y], dim=-1)
        sample_grid = center[:, None, None, :] + float(half_span) * relative_grid[None]
        patch = F.grid_sample(
            feature_map,
            sample_grid.clamp(0.0, 1.0) * 2.0 - 1.0,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        relative_channels = relative_grid.permute(2, 0, 1).unsqueeze(0)
        relative_channels = relative_channels.expand(batch_size, -1, -1, -1)
        return torch.cat([patch, relative_channels], dim=1)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        if input_image is None:
            raise ValueError("FUGCAnatomyEndpointHeatmapHead requires the normalized input image.")

        x = self.stem(x)
        x = self.refine(x)
        base_point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        if base_point_logits.shape[-2:] != out_size:
            base_point_logits = F.interpolate(
                base_point_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(
                segment_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        anatomy_inputs = self._anatomy_inputs(input_image, out_size)
        semantic_features = self.anatomy_semantic_proj(x)
        boundary_features = self.anatomy_boundary_fuse(
            torch.cat([self.anatomy_boundary_stem(anatomy_inputs), semantic_features], dim=1)
        )
        ridge_features = self.anatomy_ridge_fuse(
            torch.cat([self.anatomy_ridge_stem(anatomy_inputs), semantic_features], dim=1)
        )
        anatomy_point_delta = torch.cat(
            [
                self.anatomy_boundary_logit_delta(boundary_features),
                self.anatomy_ridge_logit_delta(ridge_features),
            ],
            dim=1,
        )
        point_logits = base_point_logits + anatomy_point_delta
        coarse_coords = self._softargmax_coords(point_logits)

        semantic_refine_input = torch.cat(
            [x, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)], dim=1
        )
        semantic_patch = self._sample_refine_roi(semantic_refine_input, coarse_coords)
        semantic_offsets = self.refine_head(semantic_patch) * self.refine_max_offset
        semantic_coords = (coarse_coords + semantic_offsets).clamp(0.0, 1.0)

        point_probabilities = torch.sigmoid(point_logits)
        boundary_local_map = torch.cat(
            [boundary_features, semantic_features, point_probabilities], dim=1
        )
        ridge_local_map = torch.cat(
            [ridge_features, semantic_features, point_probabilities], dim=1
        )
        boundary_patch = self._sample_endpoint_patch(
            boundary_local_map, semantic_coords, point_index=0, half_span=0.16
        )
        ridge_patch = self._sample_endpoint_patch(
            ridge_local_map, semantic_coords, point_index=1, half_span=0.22
        )
        boundary_offset = (
            self.anatomy_boundary_refiner(boundary_patch) * self.anatomy_boundary_max_offset
        )
        ridge_offset = self.anatomy_ridge_refiner(ridge_patch) * self.anatomy_ridge_max_offset
        endpoint_offsets = torch.stack([boundary_offset, ridge_offset], dim=1).reshape(
            semantic_coords.shape[0], 4
        )
        refined_coords = (semantic_coords + endpoint_offsets).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "anatomy_point_delta": anatomy_point_delta,
            "anatomy_boundary_features": boundary_features,
            "anatomy_ridge_features": ridge_features,
            "anatomy_boundary_patch": boundary_patch,
            "anatomy_ridge_patch": ridge_patch,
            "coarse_coords_transformed": coarse_coords,
            "pre_anatomy_coords_transformed": semantic_coords,
            "refined_coords_transformed": refined_coords,
        }


class FUGCAnatomySegmentationHeatmapHead(FUGCAnatomyEndpointHeatmapHead):
    """Anatomy endpoint head jointly pretrained with public tissue masks.

    The public masks provide a useful image-level supervision signal, while the
    challenge landmark loss remains the final coordinate objective. The added
    endpoint residual starts at zero so loading an anatomy checkpoint preserves
    its predictions before the new branch is trained.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        self.cervix_segmentation_head = nn.Sequential(
            nn.Conv2d(98, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1),
        )
        self.cervix_endpoint_residual_head = nn.Sequential(
            nn.Conv2d(99, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_points * 2),
            nn.Tanh(),
        )
        nn.init.zeros_(self.cervix_segmentation_head[-1].weight)
        nn.init.zeros_(self.cervix_segmentation_head[-1].bias)
        nn.init.zeros_(self.cervix_endpoint_residual_head[-2].weight)
        nn.init.zeros_(self.cervix_endpoint_residual_head[-2].bias)
        self.cervix_endpoint_max_offset = 0.045

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        point_logits, aux_outputs = super().forward(x, out_size, input_image)
        boundary_features = aux_outputs["anatomy_boundary_features"]
        ridge_features = aux_outputs["anatomy_ridge_features"]
        # The two 48-channel experts plus the inherited two-channel semantic
        # point probabilities form a compact tissue-aware endpoint context.
        point_probabilities = torch.sigmoid(point_logits)
        if boundary_features.shape[-2:] != point_probabilities.shape[-2:]:
            point_probabilities = F.interpolate(
                point_probabilities,
                size=boundary_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        context = torch.cat([boundary_features, ridge_features, point_probabilities], dim=1)
        segmentation_logits = self.cervix_segmentation_head(context)
        segmentation_probabilities = torch.softmax(segmentation_logits, dim=1)
        endpoint_context = torch.cat(
            [boundary_features, ridge_features, segmentation_probabilities], dim=1
        )
        residual = (
            self.cervix_endpoint_residual_head(endpoint_context)
            * self.cervix_endpoint_max_offset
        )
        refined_coords = (
            aux_outputs["refined_coords_transformed"] + residual
        ).clamp(0.0, 1.0)
        aux_outputs["cervix_segmentation_logits"] = segmentation_logits
        aux_outputs["cervix_endpoint_residual"] = residual
        aux_outputs["refined_coords_transformed"] = refined_coords
        return point_logits, aux_outputs


class FUGCAnatomyLocalHeatmapHead(FUGCAnatomyEndpointHeatmapHead):
    """FUGC anatomy head with directly supervised endpoint-local heatmaps."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        local_channels = self.anatomy_boundary_refiner[0].in_channels

        def local_heatmap_head():
            head = nn.Sequential(
                nn.Conv2d(local_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 48),
                nn.GELU(),
                nn.Conv2d(48, 1, kernel_size=1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        self.anatomy_local_boundary_head = local_heatmap_head()
        self.anatomy_local_ridge_head = local_heatmap_head()
        self.anatomy_local_blend = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    @staticmethod
    def _local_softargmax(local_logits: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = local_logits.shape
        probabilities = torch.softmax(local_logits.reshape(batch_size, -1), dim=-1).reshape(
            batch_size, height, width
        )
        axis_x = torch.linspace(-1.0, 1.0, width, device=local_logits.device, dtype=local_logits.dtype)
        axis_y = torch.linspace(-1.0, 1.0, height, device=local_logits.device, dtype=local_logits.dtype)
        relative_x = (probabilities * axis_x.view(1, 1, width)).sum(dim=(-2, -1))
        relative_y = (probabilities * axis_y.view(1, height, 1)).sum(dim=(-2, -1))
        return torch.stack([relative_x, relative_y], dim=-1)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        point_logits, aux_outputs = super().forward(x, out_size, input_image=input_image)
        boundary_patch = aux_outputs["anatomy_boundary_patch"]
        ridge_patch = aux_outputs["anatomy_ridge_patch"]
        boundary_local_logits = self.anatomy_local_boundary_head(boundary_patch)
        ridge_local_logits = self.anatomy_local_ridge_head(ridge_patch)

        centers = aux_outputs["pre_anatomy_coords_transformed"].reshape(-1, 2, 2)
        boundary_relative = self._local_softargmax(boundary_local_logits)
        ridge_relative = self._local_softargmax(ridge_local_logits)
        boundary_local_coords = centers[:, 0] + 0.16 * boundary_relative
        ridge_local_coords = centers[:, 1] + 0.22 * ridge_relative
        local_coords = torch.stack([boundary_local_coords, ridge_local_coords], dim=1).clamp(0.0, 1.0)

        base_refined = aux_outputs["refined_coords_transformed"].reshape(-1, 2, 2)
        blend = 0.75 * torch.tanh(self.anatomy_local_blend).view(1, 2, 1)
        refined_coords = (base_refined + blend * (local_coords - base_refined)).clamp(0.0, 1.0)
        aux_outputs.update(
            {
                "anatomy_boundary_local_logits": boundary_local_logits,
                "anatomy_ridge_local_logits": ridge_local_logits,
                "anatomy_local_coords_transformed": local_coords.reshape(-1, 4),
                "pre_local_heatmap_coords_transformed": base_refined.reshape(-1, 4),
                "refined_coords_transformed": refined_coords.reshape(-1, 4),
            }
        )
        return point_logits, aux_outputs


class FUGCCervicalCanalHeatmapHead(FUGCHeatmapHead):
    """Refine the internal/external cervical os in a canal-aligned strip.

    The inherited FUGC decoder remains a warm-started semantic anchor. A
    compact image/feature branch samples an elongated canonical strip along
    the predicted cervical canal, predicts a dense canal response, and
    localizes both endpoints in tangent/normal coordinates. The final gate is
    initialized near zero so the new branch cannot destroy the anchor before
    it has learned useful cervical anatomy.
    """

    requires_input_image = True

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 2:
            raise ValueError("FUGCCervicalCanalHeatmapHead requires exactly 2 points.")
        super().__init__(in_channels, num_points, width_multiplier)
        image_channels = 40
        semantic_channels = 24
        strip_channels = 64

        self.register_buffer(
            "canal_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "canal_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "canal_sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "canal_sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "canal_laplacian",
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )

        self.canal_image_stem = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, image_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, image_channels),
            nn.GELU(),
            nn.Conv2d(
                image_channels,
                image_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=image_channels,
                bias=False,
            ),
            nn.Conv2d(image_channels, image_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, image_channels),
            nn.GELU(),
        )
        self.canal_semantic_proj = nn.Sequential(
            nn.Conv2d(self.point_head.in_channels, semantic_channels, kernel_size=1, bias=False),
            nn.GroupNorm(6, semantic_channels),
            nn.GELU(),
        )
        strip_in_channels = image_channels + semantic_channels + num_points + 1 + 2
        self.canal_strip_encoder = nn.Sequential(
            nn.Conv2d(strip_in_channels, strip_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, strip_channels),
            nn.GELU(),
            nn.Conv2d(strip_channels, strip_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, strip_channels),
            nn.GELU(),
            nn.Conv2d(
                strip_channels,
                strip_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=strip_channels,
                bias=False,
            ),
            nn.Conv2d(strip_channels, strip_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, strip_channels),
            nn.GELU(),
        )
        self.canal_response_head = nn.Conv2d(strip_channels, 1, kernel_size=1)
        self.canal_endpoint_delta = nn.Conv2d(strip_channels, num_points, kernel_size=1)
        nn.init.zeros_(self.canal_endpoint_delta.weight)
        nn.init.zeros_(self.canal_endpoint_delta.bias)
        self.canal_blend_logits = nn.Parameter(torch.full((num_points,), -5.0))
        self.canal_strip_height = 32
        self.canal_strip_width = 64

    def _canal_image_inputs(self, image: torch.Tensor, out_size) -> torch.Tensor:
        mean = self.canal_image_mean.to(dtype=image.dtype, device=image.device)
        std = self.canal_image_std.to(dtype=image.dtype, device=image.device)
        restored = (image * std + mean).clamp(0.0, 1.0)
        gray = (
            0.2989 * restored[:, 0:1]
            + 0.5870 * restored[:, 1:2]
            + 0.1140 * restored[:, 2:3]
        )
        gray = F.interpolate(gray, size=out_size, mode="bilinear", align_corners=False)
        smooth3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        smooth9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        sobel_x = self.canal_sobel_x.to(dtype=image.dtype, device=image.device)
        sobel_y = self.canal_sobel_y.to(dtype=image.dtype, device=image.device)
        grad_x = F.conv2d(smooth3, sobel_x, padding=1) / 4.0
        grad_y = F.conv2d(smooth3, sobel_y, padding=1) / 4.0
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        laplacian = F.conv2d(
            smooth3,
            self.canal_laplacian.to(dtype=image.dtype, device=image.device),
            padding=1,
        ) / 4.0
        variance = (F.avg_pool2d(gray.square(), 9, 1, 4) - smooth9.square()).clamp_min(0.0)
        return torch.cat(
            [gray, smooth3, smooth9, grad_x, grad_y, grad_mag, laplacian, variance],
            dim=1,
        )

    def _sample_canal_strip(
        self,
        feature_map: torch.Tensor,
        anchor_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = anchor_coords.shape[0]
        points = anchor_coords.reshape(batch_size, 2, 2)
        midpoint = points.mean(dim=1)
        segment = points[:, 1] - points[:, 0]
        length = torch.norm(segment, dim=-1).clamp_min(1e-4)
        tangent = segment / length[:, None]
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        half_length = (0.78 * length).clamp(0.14, 0.48)
        half_width = (0.22 * length).clamp(0.055, 0.14)

        axis_u = torch.linspace(
            -1.0,
            1.0,
            self.canal_strip_width,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        axis_v = torch.linspace(
            -1.0,
            1.0,
            self.canal_strip_height,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        grid_v, grid_u = torch.meshgrid(axis_v, axis_u, indexing="ij")
        world_grid = (
            midpoint[:, None, None, :]
            + grid_u[None, :, :, None]
            * half_length[:, None, None, None]
            * tangent[:, None, None, :]
            + grid_v[None, :, :, None]
            * half_width[:, None, None, None]
            * normal[:, None, None, :]
        )
        strip = F.grid_sample(
            feature_map,
            world_grid.clamp(0.0, 1.0) * 2.0 - 1.0,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        canonical_coords = torch.stack([grid_u, grid_v], dim=0).unsqueeze(0)
        canonical_coords = canonical_coords.expand(batch_size, -1, -1, -1)
        strip = torch.cat([strip, canonical_coords], dim=1)

        relative_points = points - midpoint[:, None, :]
        anchor_u = (relative_points * tangent[:, None, :]).sum(dim=-1) / half_length[:, None]
        anchor_v = (relative_points * normal[:, None, :]).sum(dim=-1) / half_width[:, None]
        anchor_canonical = torch.stack([anchor_u, anchor_v], dim=-1)
        return strip, world_grid, anchor_canonical, torch.stack([half_length, half_width], dim=-1)

    @staticmethod
    def _canonical_softargmax(logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = logits.shape
        probabilities = torch.softmax(logits.reshape(batch_size, num_points, -1), dim=-1)
        probabilities = probabilities.reshape(batch_size, num_points, height, width)
        axis_u = torch.linspace(-1.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        axis_v = torch.linspace(-1.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        expected_u = (probabilities * axis_u.view(1, 1, 1, width)).sum(dim=(-2, -1))
        expected_v = (probabilities * axis_v.view(1, 1, height, 1)).sum(dim=(-2, -1))
        return torch.stack([expected_u, expected_v], dim=-1)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        if input_image is None:
            raise ValueError("FUGCCervicalCanalHeatmapHead requires the normalized input image.")

        x = self.stem(x)
        x = self.refine(x)
        point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(
                segment_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        semantic_refine_input = torch.cat(
            [x, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)], dim=1
        )
        semantic_patch = self._sample_refine_roi(semantic_refine_input, coarse_coords)
        semantic_offsets = self.refine_head(semantic_patch) * self.refine_max_offset
        anchor_coords = (coarse_coords + semantic_offsets).clamp(0.0, 1.0)

        image_features = self.canal_image_stem(self._canal_image_inputs(input_image, out_size))
        semantic_features = self.canal_semantic_proj(x)
        canal_feature_map = torch.cat(
            [image_features, semantic_features, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)],
            dim=1,
        )
        strip, world_grid, anchor_canonical, frame_scale = self._sample_canal_strip(
            canal_feature_map,
            anchor_coords,
        )
        strip_features = self.canal_strip_encoder(strip)
        canal_logits = self.canal_response_head(strip_features)

        height, width = strip_features.shape[-2:]
        axis_u = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        axis_v = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        grid_v, grid_u = torch.meshgrid(axis_v, axis_u, indexing="ij")
        prior_logits = -0.5 * (
            ((grid_u[None, None] - anchor_canonical[:, :, 0, None, None]) / 0.075).square()
            + ((grid_v[None, None] - anchor_canonical[:, :, 1, None, None]) / 0.14).square()
        )
        endpoint_logits = prior_logits + self.canal_endpoint_delta(strip_features)
        endpoint_canonical = self._canonical_softargmax(endpoint_logits)

        anchor_points = anchor_coords.reshape(-1, 2, 2)
        midpoint = anchor_points.mean(dim=1)
        segment = anchor_points[:, 1] - anchor_points[:, 0]
        tangent = F.normalize(segment, dim=-1, eps=1e-6)
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        local_coords = (
            midpoint[:, None, :]
            + endpoint_canonical[:, :, 0, None]
            * frame_scale[:, None, 0, None]
            * tangent[:, None, :]
            + endpoint_canonical[:, :, 1, None]
            * frame_scale[:, None, 1, None]
            * normal[:, None, :]
        ).clamp(0.0, 1.0)
        blend = (0.65 * torch.sigmoid(self.canal_blend_logits)).view(1, 2, 1)
        refined_coords = (anchor_points + blend * (local_coords - anchor_points)).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "coarse_coords_transformed": coarse_coords,
            "pre_canal_coords_transformed": anchor_coords,
            "canal_local_coords_transformed": local_coords.reshape(-1, 4),
            "canal_endpoint_logits": endpoint_logits,
            "canal_logits": canal_logits,
            "canal_sample_grid_transformed": world_grid,
            "canal_blend": blend,
            "refined_coords_transformed": refined_coords.reshape(-1, 4),
        }


class FUGCCervixSegmentationHeatmapHead(FUGCHeatmapHead):
    """Localize both cervical os endpoints through explicit tissue parsing.

    The public FUGC masks label the two cervical regions whose common
    interface terminates at the annotated internal and external os. The new
    branch therefore predicts those regions and endpoint heatmaps jointly,
    while retaining the inherited offset128 FUGC decoder as a warm-started
    semantic anchor.
    """

    requires_input_image = True

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 2:
            raise ValueError("FUGCCervixSegmentationHeatmapHead requires exactly 2 points.")
        super().__init__(in_channels, num_points, width_multiplier)
        image_channels = 48
        semantic_channels = 32
        fusion_channels = 80

        self.register_buffer(
            "cervix_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "cervix_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.cervix_image_stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, image_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, image_channels),
            nn.GELU(),
            nn.Conv2d(image_channels, image_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, image_channels),
            nn.GELU(),
            nn.Conv2d(
                image_channels,
                image_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=image_channels,
                bias=False,
            ),
            nn.Conv2d(image_channels, image_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, image_channels),
            nn.GELU(),
        )
        self.cervix_semantic_proj = nn.Sequential(
            nn.Conv2d(self.point_head.in_channels, semantic_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, semantic_channels),
            nn.GELU(),
        )
        self.cervix_fusion = nn.Sequential(
            nn.Conv2d(image_channels + semantic_channels, fusion_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, fusion_channels),
            nn.GELU(),
            nn.Conv2d(
                fusion_channels,
                fusion_channels,
                3,
                padding=2,
                dilation=2,
                groups=fusion_channels,
                bias=False,
            ),
            nn.Conv2d(fusion_channels, fusion_channels, 1, bias=False),
            nn.GroupNorm(8, fusion_channels),
            nn.GELU(),
        )
        self.cervix_segmentation_head = nn.Conv2d(fusion_channels, 3, kernel_size=1)
        self.cervix_endpoint_head = nn.Sequential(
            nn.Conv2d(fusion_channels + 3, fusion_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, num_points, kernel_size=1),
        )
        endpoint_output = self.cervix_endpoint_head[-1]
        nn.init.zeros_(endpoint_output.weight)
        nn.init.zeros_(endpoint_output.bias)

        local_channels = 64
        self.cervix_local_refine_head = nn.Sequential(
            nn.Conv2d(fusion_channels + 3 + num_points, local_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, local_channels),
            nn.GELU(),
            nn.Conv2d(local_channels, local_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, local_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(local_channels, local_channels),
            nn.GELU(),
            nn.Linear(local_channels, num_points * 2),
            nn.Tanh(),
        )
        local_output = self.cervix_local_refine_head[-2]
        nn.init.zeros_(local_output.weight)
        nn.init.zeros_(local_output.bias)
        self.cervix_local_max_offset = 0.06

    def _restored_input(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.cervix_image_mean.to(dtype=image.dtype, device=image.device)
        std = self.cervix_image_std.to(dtype=image.dtype, device=image.device)
        return (image * std + mean).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        if input_image is None:
            raise ValueError("FUGCCervixSegmentationHeatmapHead requires the normalized input image.")

        x = self.stem(x)
        x = self.refine(x)
        base_point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        if base_point_logits.shape[-2:] != out_size:
            base_point_logits = F.interpolate(
                base_point_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(
                segment_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        image_features = self.cervix_image_stem(self._restored_input(input_image))
        if image_features.shape[-2:] != out_size:
            image_features = F.interpolate(
                image_features, size=out_size, mode="bilinear", align_corners=False
            )
        semantic_features = self.cervix_semantic_proj(x)
        fused_features = self.cervix_fusion(
            torch.cat([image_features, semantic_features], dim=1)
        )
        segmentation_logits = self.cervix_segmentation_head(fused_features)
        segmentation_probabilities = torch.softmax(segmentation_logits, dim=1)
        endpoint_delta_logits = self.cervix_endpoint_head(
            torch.cat([fused_features, segmentation_probabilities], dim=1)
        )
        point_logits = base_point_logits + endpoint_delta_logits

        coarse_coords = self._softargmax_coords(point_logits)
        semantic_refine_input = torch.cat(
            [x, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)], dim=1
        )
        semantic_patch = self._sample_refine_roi(semantic_refine_input, coarse_coords)
        semantic_offsets = self.refine_head(semantic_patch) * self.refine_max_offset
        semantic_coords = (coarse_coords + semantic_offsets).clamp(0.0, 1.0)

        local_refine_input = torch.cat(
            [fused_features, segmentation_probabilities, torch.sigmoid(point_logits)], dim=1
        )
        local_patch = self._sample_refine_roi(local_refine_input, semantic_coords)
        local_offsets = self.cervix_local_refine_head(local_patch) * self.cervix_local_max_offset
        refined_coords = (semantic_coords + local_offsets).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "cervix_segmentation_logits": segmentation_logits,
            "cervix_endpoint_delta_logits": endpoint_delta_logits,
            "coarse_coords_transformed": coarse_coords,
            "pre_cervix_coords_transformed": semantic_coords,
            "refined_coords_transformed": refined_coords,
        }


class FUGCCervixGeometryHeatmapHead(FUGCCervixSegmentationHeatmapHead):
    """Use soft tissue-boundary extrema as an anatomical coordinate decoder.

    FUGC landmarks follow the lower boundary of class 1 and the upper boundary
    of class 2 in the public cervical-region masks. The extrema are computed
    differentiably from the predicted semantic probabilities, avoiding a
    brittle argmax or external post-processing step.
    """

    # A high temperature approximates the annotated boundary extrema while
    # remaining differentiable for end-to-end training.
    geometry_temperature = 20.0
    geometry_roi_sigma = 0.08
    # About one pixel at the 336 px FUGC image height.
    geometry_max_delta = 0.003

    @classmethod
    def _soft_boundary_coords(
        cls,
        segmentation_logits: torch.Tensor,
        prior_coords: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(segmentation_logits, dim=1)
        height, width = segmentation_logits.shape[-2:]
        y = torch.linspace(
            0.0,
            1.0,
            height,
            device=segmentation_logits.device,
            dtype=segmentation_logits.dtype,
        ).view(1, 1, height, 1)
        x = torch.linspace(
            0.0,
            1.0,
            width,
            device=segmentation_logits.device,
            dtype=segmentation_logits.dtype,
        ).view(1, 1, 1, width)
        prior = prior_coords.reshape(-1, 2, 2).to(dtype=segmentation_logits.dtype)
        prior_x = prior[:, :, 0].view(-1, 2, 1, 1)
        prior_y = prior[:, :, 1].view(-1, 2, 1, 1)
        roi = torch.exp(
            -(
                (x - prior_x).square()
                + (y - prior_y).square()
            )
            / (2.0 * cls.geometry_roi_sigma**2)
        )

        class_one = probabilities[:, 1:2].clamp_min(1e-6)
        class_two = probabilities[:, 2:3].clamp_min(1e-6)
        lower_boundary = torch.softmax(
            (
                class_one.log() + roi[:, 0:1].clamp_min(1e-6).log()
                + cls.geometry_temperature * y
            ).flatten(1),
            dim=1,
        ).view(-1, 1, height, width)
        upper_boundary = torch.softmax(
            (
                class_two.log() + roi[:, 1:2].clamp_min(1e-6).log()
                - cls.geometry_temperature * y
            ).flatten(1),
            dim=1,
        ).view(-1, 1, height, width)

        lower = torch.cat(
            [
                (lower_boundary * x).sum(dim=(-2, -1)),
                (lower_boundary * y).sum(dim=(-2, -1)),
            ],
            dim=1,
        )
        upper = torch.cat(
            [
                (upper_boundary * x).sum(dim=(-2, -1)),
                (upper_boundary * y).sum(dim=(-2, -1)),
            ],
            dim=1,
        )
        return torch.cat([lower, upper], dim=1)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        point_logits, aux_outputs = super().forward(x, out_size, input_image=input_image)
        geometry_coords = self._soft_boundary_coords(
            aux_outputs["cervix_segmentation_logits"],
            aux_outputs["coarse_coords_transformed"],
        )

        # Keep the learned endpoint decoder as the stable anchor. The geometry
        # branch contributes a bounded residual while it learns to calibrate
        # the public-mask anatomy to challenge landmark coordinates.
        anchor_coords = aux_outputs["refined_coords_transformed"]
        geometry_delta = geometry_coords - anchor_coords
        bounded_delta = self.geometry_max_delta * torch.tanh(
            geometry_delta / self.geometry_max_delta
        )
        refined_coords = (anchor_coords + bounded_delta).clamp(0.0, 1.0)
        aux_outputs = dict(aux_outputs)
        aux_outputs["geometry_coords_transformed"] = geometry_coords
        aux_outputs["refined_coords_transformed"] = refined_coords
        return point_logits, aux_outputs


class FUGCCervixHighResolutionSegmentationHead(FUGCHeatmapHead):
    """Parse cervical regions at 256 resolution before localizing both os endpoints."""

    requires_input_image = True

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 2:
            raise ValueError("FUGCCervixHighResolutionSegmentationHead requires 2 points.")
        super().__init__(in_channels, num_points, width_multiplier)
        self.register_buffer(
            "cervix_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "cervix_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.cervix_encoder_256 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.cervix_encoder_128 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.cervix_semantic_proj = nn.Sequential(
            nn.Conv2d(self.point_head.in_channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.cervix_bottleneck = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.cervix_decoder_256 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.cervix_segmentation_head = nn.Conv2d(64, 3, 1)
        self.cervix_endpoint_head = nn.Sequential(
            nn.Conv2d(67, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, num_points, 1),
        )
        endpoint_output = self.cervix_endpoint_head[-1]
        nn.init.zeros_(endpoint_output.weight)
        nn.init.zeros_(endpoint_output.bias)
        self.cervix_local_refine_head = nn.Sequential(
            nn.Conv2d(69, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, num_points * 2),
            nn.Tanh(),
        )
        local_output = self.cervix_local_refine_head[-2]
        nn.init.zeros_(local_output.weight)
        nn.init.zeros_(local_output.bias)
        self.cervix_local_max_offset = 0.07

    def _restore_input(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.cervix_image_mean.to(dtype=image.dtype, device=image.device)
        std = self.cervix_image_std.to(dtype=image.dtype, device=image.device)
        return (image * std + mean).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, out_size, input_image: torch.Tensor | None = None):
        if input_image is None:
            raise ValueError(
                "FUGCCervixHighResolutionSegmentationHead requires the normalized input image."
            )
        x = self.stem(x)
        x = self.refine(x)
        base_point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        if base_point_logits.shape[-2:] != out_size:
            base_point_logits = F.interpolate(
                base_point_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(
                segment_logits, size=out_size, mode="bilinear", align_corners=False
            )
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        feature_256 = self.cervix_encoder_256(self._restore_input(input_image))
        feature_128 = self.cervix_encoder_128(feature_256)
        semantic = self.cervix_semantic_proj(x)
        if semantic.shape[-2:] != feature_128.shape[-2:]:
            semantic = F.interpolate(
                semantic, size=feature_128.shape[-2:], mode="bilinear", align_corners=False
            )
        bottleneck = self.cervix_bottleneck(torch.cat([feature_128, semantic], dim=1))
        decoder_input = torch.cat(
            [F.interpolate(bottleneck, size=feature_256.shape[-2:], mode="bilinear", align_corners=False), feature_256],
            dim=1,
        )
        highres_features = self.cervix_decoder_256(decoder_input)
        segmentation_logits = self.cervix_segmentation_head(highres_features)
        segmentation_probabilities = torch.softmax(segmentation_logits, dim=1)
        highres_endpoint_delta = self.cervix_endpoint_head(
            torch.cat([highres_features, segmentation_probabilities], dim=1)
        )
        endpoint_delta = F.interpolate(
            highres_endpoint_delta, size=out_size, mode="bilinear", align_corners=False
        )
        point_logits = base_point_logits + endpoint_delta

        coarse_coords = self._softargmax_coords(point_logits)
        semantic_refine_input = torch.cat(
            [x, torch.sigmoid(point_logits), torch.sigmoid(segment_logits)], dim=1
        )
        semantic_patch = self._sample_refine_roi(semantic_refine_input, coarse_coords)
        semantic_offsets = self.refine_head(semantic_patch) * self.refine_max_offset
        semantic_coords = (coarse_coords + semantic_offsets).clamp(0.0, 1.0)
        highres_base = F.interpolate(
            base_point_logits,
            size=highres_endpoint_delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        highres_endpoint_logits = highres_base + highres_endpoint_delta
        local_input = torch.cat(
            [highres_features, segmentation_probabilities, torch.sigmoid(highres_endpoint_logits)],
            dim=1,
        )
        local_patch = self._sample_refine_roi(local_input, semantic_coords)
        local_offsets = self.cervix_local_refine_head(local_patch) * self.cervix_local_max_offset
        refined_coords = (semantic_coords + local_offsets).clamp(0.0, 1.0)
        return point_logits, {
            "segment_logits": segment_logits,
            "cervix_segmentation_logits": segmentation_logits,
            "cervix_highres_endpoint_logits": highres_endpoint_logits,
            "cervix_endpoint_delta_logits": endpoint_delta,
            "coarse_coords_transformed": coarse_coords,
            "pre_cervix_coords_transformed": semantic_coords,
            "refined_coords_transformed": refined_coords,
        }


class FUGCVectorRefineHeatmapHead(FUGCHeatmapHead):
    """FUGC head with segment-parameter refinement.

    The base FUGC head predicts endpoint heatmaps and independent endpoint
    offsets. This head keeps that path load-compatible, then applies a
    zero-started correction in midpoint/angle/length space so FUGC remains a
    coherent short segment rather than two unrelated points.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        if int(num_points) != 2:
            raise ValueError("FUGCVectorRefineHeatmapHead requires exactly 2 points.")
        super().__init__(in_channels, num_points, width_multiplier)
        refine_hidden = max(self.refine_head[0].out_channels, 64)
        refine_in_channels = self.refine_head[0].in_channels
        self.vector_refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, 4),
            nn.Tanh(),
        )
        final_linear = self.vector_refine_head[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        self.vector_mid_max_offset = 0.030
        self.vector_angle_max_delta = 0.22
        self.vector_log_length_max_delta = 0.22

    def _apply_vector_refine(self, coords: torch.Tensor, roi_patch: torch.Tensor) -> torch.Tensor:
        bsz = coords.shape[0]
        points = coords.view(bsz, 2, 2)
        params = self.vector_refine_head(roi_patch)

        midpoint_delta = params[:, 0:2] * self.vector_mid_max_offset
        angle_delta = params[:, 2] * self.vector_angle_max_delta
        log_length_delta = params[:, 3] * self.vector_log_length_max_delta

        midpoint = points.mean(dim=1) + midpoint_delta
        vector = points[:, 1, :] - points[:, 0, :]
        length = torch.norm(vector, dim=-1, keepdim=True).clamp_min(1e-4)
        unit = vector / length

        cos_delta = torch.cos(angle_delta).unsqueeze(-1)
        sin_delta = torch.sin(angle_delta).unsqueeze(-1)
        rotated_unit = torch.stack(
            [
                unit[:, 0] * cos_delta[:, 0] - unit[:, 1] * sin_delta[:, 0],
                unit[:, 0] * sin_delta[:, 0] + unit[:, 1] * cos_delta[:, 0],
            ],
            dim=-1,
        )
        refined_length = length * torch.exp(log_length_delta).unsqueeze(-1)
        refined_vector = rotated_unit * refined_length
        refined_points = torch.stack(
            [
                midpoint - 0.5 * refined_vector,
                midpoint + 0.5 * refined_vector,
            ],
            dim=1,
        )
        return refined_points.clamp(0.0, 1.0).reshape(bsz, 4)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.refine(x)
        point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(segment_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat(
            [
                feat_for_refine,
                torch.sigmoid(point_logits),
                torch.sigmoid(segment_logits),
            ],
            dim=1,
        )
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        endpoint_offsets = self.refine_head(roi_patch) * self.refine_max_offset
        endpoint_refined = (coarse_coords + endpoint_offsets).clamp(0.0, 1.0)
        vector_refined = self._apply_vector_refine(endpoint_refined, roi_patch)

        return point_logits, {
            "segment_logits": segment_logits,
            "coarse_coords_transformed": coarse_coords,
            "endpoint_refined_coords_transformed": endpoint_refined,
            "refined_coords_transformed": vector_refined,
        }


class FUGCStripAxisHeatmapHead(FUGCVectorRefineHeatmapHead):
    """FUGC head that refines the pair from an oriented strip around the canal.

    FUGC is a short two-point structure. A square ROI can miss the actual
    boundary evidence when the segment is oblique, so this branch samples an
    elongated local strip aligned with the current segment estimate and predicts
    a final midpoint/angle/length correction.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        refine_hidden = max(self.refine_head[0].out_channels, 64)
        refine_in_channels = self.refine_head[0].in_channels
        self.strip_patch_h = 12
        self.strip_patch_w = 32
        self.strip_refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, 4),
            nn.Tanh(),
        )
        final_linear = self.strip_refine_head[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        self.strip_gate_logit = nn.Parameter(torch.tensor(-1.5))
        self.strip_mid_max_offset = 0.035
        self.strip_angle_max_delta = 0.30
        self.strip_log_length_max_delta = 0.28

    def _sample_axis_strip(self, feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        points = coords.view(bsz, 2, 2)
        midpoint = points.mean(dim=1)
        vector = points[:, 1, :] - points[:, 0, :]
        length = torch.norm(vector, dim=-1, keepdim=True).clamp_min(1e-4)
        unit = vector / length
        perp = torch.stack([-unit[:, 1], unit[:, 0]], dim=-1)

        long_axis = torch.linspace(-1.0, 1.0, self.strip_patch_w, device=feat.device, dtype=feat.dtype)
        short_axis = torch.linspace(-1.0, 1.0, self.strip_patch_h, device=feat.device, dtype=feat.dtype)
        grid_short, grid_long = torch.meshgrid(short_axis, long_axis, indexing="ij")
        grid_long = grid_long.view(1, self.strip_patch_h, self.strip_patch_w, 1)
        grid_short = grid_short.view(1, self.strip_patch_h, self.strip_patch_w, 1)

        half_long = (0.70 * length + 0.025).clamp(0.075, 0.25).view(bsz, 1, 1, 1)
        half_short = (0.22 * length + 0.020).clamp(0.030, 0.095).view(bsz, 1, 1, 1)
        center = midpoint.view(bsz, 1, 1, 2)
        unit = unit.view(bsz, 1, 1, 2)
        perp = perp.view(bsz, 1, 1, 2)

        sample_grid = center + grid_long * half_long * unit + grid_short * half_short * perp
        sample_grid = sample_grid.clamp(0.0, 1.0) * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _apply_strip_refine(self, coords: torch.Tensor, strip_patch: torch.Tensor) -> torch.Tensor:
        bsz = coords.shape[0]
        points = coords.view(bsz, 2, 2)
        params = self.strip_refine_head(strip_patch)

        midpoint_delta = params[:, 0:2] * self.strip_mid_max_offset
        angle_delta = params[:, 2] * self.strip_angle_max_delta
        log_length_delta = params[:, 3] * self.strip_log_length_max_delta

        midpoint = points.mean(dim=1) + midpoint_delta
        vector = points[:, 1, :] - points[:, 0, :]
        length = torch.norm(vector, dim=-1, keepdim=True).clamp_min(1e-4)
        unit = vector / length
        cos_delta = torch.cos(angle_delta).unsqueeze(-1)
        sin_delta = torch.sin(angle_delta).unsqueeze(-1)
        rotated_unit = torch.stack(
            [
                unit[:, 0] * cos_delta[:, 0] - unit[:, 1] * sin_delta[:, 0],
                unit[:, 0] * sin_delta[:, 0] + unit[:, 1] * cos_delta[:, 0],
            ],
            dim=-1,
        )
        refined_length = length * torch.exp(log_length_delta).unsqueeze(-1)
        refined_vector = rotated_unit * refined_length
        refined_points = torch.stack(
            [
                midpoint - 0.5 * refined_vector,
                midpoint + 0.5 * refined_vector,
            ],
            dim=1,
        )
        return refined_points.clamp(0.0, 1.0).reshape(bsz, 4)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.refine(x)
        point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(segment_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat(
            [
                feat_for_refine,
                torch.sigmoid(point_logits),
                torch.sigmoid(segment_logits),
            ],
            dim=1,
        )
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        endpoint_offsets = self.refine_head(roi_patch) * self.refine_max_offset
        endpoint_refined = (coarse_coords + endpoint_offsets).clamp(0.0, 1.0)
        vector_refined = self._apply_vector_refine(endpoint_refined, roi_patch)

        strip_patch = self._sample_axis_strip(refine_input, vector_refined)
        strip_refined = self._apply_strip_refine(vector_refined, strip_patch)
        gate = torch.sigmoid(self.strip_gate_logit)
        refined_coords = (vector_refined + gate * (strip_refined - vector_refined)).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "coarse_coords_transformed": coarse_coords,
            "endpoint_refined_coords_transformed": endpoint_refined,
            "vector_refined_coords_transformed": vector_refined,
            "strip_refined_coords_transformed": strip_refined,
            "refined_coords_transformed": refined_coords,
        }


class FUGCSegmentSpecialistHeatmapHead(FUGCStripAxisHeatmapHead):
    """FUGC-specific endpoint head with explicit 1D segment evidence.

    The parent FUGC head already warm-starts from the selected offset128 model
    and refines endpoints from a rotated strip. This specialist adds a small
    axis-profile branch over that strip: it predicts two endpoint positions
    along the segment axis and a bounded confidence gate. The branch is
    initialized conservatively so a warm-started checkpoint remains close to the
    anchor, while FUGC-only fine-tuning can learn a coherent short-segment
    detector instead of moving two endpoints independently.
    """

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        refine_hidden = max(self.refine_head[0].out_channels, 64)
        refine_in_channels = self.refine_head[0].in_channels
        self.axis_profile_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, 2, kernel_size=(self.strip_patch_h, 1), padding=0),
        )
        self.axis_gate_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, 1),
        )
        nn.init.zeros_(self.axis_profile_head[-1].weight)
        nn.init.zeros_(self.axis_profile_head[-1].bias)
        nn.init.zeros_(self.axis_gate_head[-1].weight)
        nn.init.constant_(self.axis_gate_head[-1].bias, -2.0)
        self.axis_gate_logit = nn.Parameter(torch.tensor(-1.3))
        self.axis_length_margin = 0.12

    def _axis_profile_coords(self, coords: torch.Tensor, strip_patch: torch.Tensor) -> torch.Tensor:
        bsz = coords.shape[0]
        points = coords.view(bsz, 2, 2)
        midpoint = points.mean(dim=1)
        vector = points[:, 1, :] - points[:, 0, :]
        length = torch.norm(vector, dim=-1, keepdim=True).clamp_min(1e-4)
        unit = vector / length

        logits = self.axis_profile_head(strip_patch).squeeze(2)
        positions = torch.linspace(
            -1.0,
            1.0,
            logits.shape[-1],
            device=logits.device,
            dtype=logits.dtype,
        ).view(1, 1, -1)
        axis_pos = (torch.softmax(logits, dim=-1) * positions).sum(dim=-1)
        axis_pos = torch.sort(axis_pos, dim=-1).values

        half_span = (0.70 * length + 0.025).clamp(0.075, 0.25)
        min_pos = -1.0 + self.axis_length_margin
        max_pos = 1.0 - self.axis_length_margin
        axis_pos = axis_pos.clamp(min_pos, max_pos)
        refined_points = midpoint.unsqueeze(1) + axis_pos.unsqueeze(-1) * half_span.unsqueeze(1) * unit.unsqueeze(1)
        return refined_points.clamp(0.0, 1.0).reshape(bsz, 4)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.refine(x)
        point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(segment_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat(
            [
                feat_for_refine,
                torch.sigmoid(point_logits),
                torch.sigmoid(segment_logits),
            ],
            dim=1,
        )
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        endpoint_offsets = self.refine_head(roi_patch) * self.refine_max_offset
        endpoint_refined = (coarse_coords + endpoint_offsets).clamp(0.0, 1.0)
        vector_refined = self._apply_vector_refine(endpoint_refined, roi_patch)

        strip_patch = self._sample_axis_strip(refine_input, vector_refined)
        strip_refined = self._apply_strip_refine(vector_refined, strip_patch)
        strip_gate = torch.sigmoid(self.strip_gate_logit)
        strip_fused = (vector_refined + strip_gate * (strip_refined - vector_refined)).clamp(0.0, 1.0)

        axis_refined = self._axis_profile_coords(strip_fused, strip_patch)
        sample_gate = torch.sigmoid(self.axis_gate_head(strip_patch)).view(-1, 1)
        axis_gate = torch.sigmoid(self.axis_gate_logit).view(1, 1) * sample_gate
        refined_coords = (strip_fused + axis_gate * (axis_refined - strip_fused)).clamp(0.0, 1.0)

        return point_logits, {
            "segment_logits": segment_logits,
            "coarse_coords_transformed": coarse_coords,
            "endpoint_refined_coords_transformed": endpoint_refined,
            "vector_refined_coords_transformed": vector_refined,
            "strip_refined_coords_transformed": strip_refined,
            "axis_refined_coords_transformed": axis_refined,
            "refined_coords_transformed": refined_coords,
        }


class ROIRefineHeatmapHead(nn.Module):
    """Generic coarse-to-fine head for tasks that benefit from local refinement."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 192) * width_multiplier), 112)
        hidden2 = max(int(max(hidden1 // 2, 144) * width_multiplier), 80)
        hidden3 = max(int(max(hidden2 // 2, 112) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.refine_patch_size = 18
        refine_in_channels = hidden3 + num_points
        refine_hidden = max(hidden3, 80)
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, num_points * 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.060

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def _sample_refine_roi(self, feat: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.refine_patch_size
        point_pairs = coarse_coords.reshape(bsz, -1, 2)
        min_xy = point_pairs.min(dim=1).values
        max_xy = point_pairs.max(dim=1).values
        midpoint = 0.5 * (min_xy + max_xy)
        span = (max_xy - min_xy).amax(dim=-1).clamp(0.10, 0.45) * 0.90
        span = span.clamp(0.12, 0.32)

        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)

        center = midpoint.view(bsz, 1, 1, 2)
        sample_grid = center + base_grid * span.view(bsz, 1, 1, 1)
        sample_grid = sample_grid.clamp(0.0, 1.0)
        sample_grid = sample_grid * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat([feat_for_refine, torch.sigmoid(point_logits)], dim=1)
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        offsets = self.refine_head(roi_patch) * self.refine_max_offset
        refined_coords = (coarse_coords + offsets).clamp(0.0, 1.0)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class CardiacGraphRefineHeatmapHead(nn.Module):
    """Cardiac landmark head with structural coordinate refinement."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 224) * width_multiplier), 128)
        hidden2 = max(int(max(hidden1 // 2, 160) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 128) * width_multiplier), 80)
        token_dim = max(hidden3, 96)
        self.num_points = num_points

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.coord_embed = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_feat_proj = nn.Sequential(
            nn.Linear(hidden3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_index_embed = nn.Parameter(torch.zeros(1, num_points, token_dim))
        nn.init.trunc_normal_(self.point_index_embed, std=0.02)
        self.graph_blocks = nn.ModuleList(
            [LandmarkGraphRefineBlock(token_dim) for _ in range(3)]
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.045 if num_points >= 12 else 0.035

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    @staticmethod
    def _sample_point_features(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz, channels = feat.shape[:2]
        point_coords = coords.view(bsz, -1, 2)
        grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(feat_for_refine, coarse_coords)
        pooled_feature = F.adaptive_avg_pool2d(feat_for_refine, output_size=1).flatten(1)
        tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(pooled_feature).unsqueeze(1)
            + self.point_index_embed
        )
        for block in self.graph_blocks:
            tokens = block(tokens)
        offsets = self.offset_head(tokens) * self.refine_max_offset
        refined_coords = torch.clamp(coarse_points + offsets, 0.0, 1.0).reshape(coarse_coords.shape[0], -1)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class CardiacPairSetRefiner(nn.Module):
    """Order-invariant geometric refinement for ambiguous cardiac endpoint pairs."""

    def __init__(self, in_channels: int, task_id: str, token_dim: int = 96):
        super().__init__()
        if task_id not in CANONICAL_PAIR_RULES:
            raise ValueError(f"Cardiac pair-set refinement is not defined for task '{task_id}'.")
        self.task_id = task_id
        self.pair_indices, _ = CANONICAL_PAIR_RULES[task_id]
        self.num_pairs = len(self.pair_indices)
        self.point_feature_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(5, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.pair_embedding = nn.Parameter(torch.zeros(1, self.num_pairs, token_dim))
        nn.init.trunc_normal_(self.pair_embedding, std=0.02)
        self.pair_set_block = LandmarkGraphRefineBlock(token_dim)
        self.pair_mixer = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 4),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.max_midpoint_offset = 0.025
        self.max_angle_offset = math.radians(12.0)
        self.max_log_length_offset = 0.20

    @staticmethod
    def _sample_pair_features(features: torch.Tensor, pair_points: torch.Tensor) -> torch.Tensor:
        batch_size, num_pairs = pair_points.shape[:2]
        sample_grid = pair_points.reshape(batch_size, num_pairs * 2, 2)
        sample_grid = sample_grid.mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sampled.reshape(batch_size, num_pairs, 2, features.shape[1]).mean(dim=2)

    def forward(self, features: torch.Tensor, coords: torch.Tensor):
        canonical_coords = canonicalize_landmark_pairs(coords, self.task_id)
        points = canonical_coords.view(canonical_coords.shape[0], -1, 2)
        pair_index = torch.as_tensor(self.pair_indices, device=points.device, dtype=torch.long)
        pair_points = points[:, pair_index]
        first = pair_points[:, :, 0]
        second = pair_points[:, :, 1]
        midpoint = 0.5 * (first + second)
        vector = second - first
        length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-4)
        unit_vector = vector / length

        pair_features = self._sample_pair_features(features, pair_points)
        global_feature = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        geometry = torch.cat([midpoint, unit_vector, length], dim=-1)
        pair_tokens = (
            self.point_feature_projection(pair_features)
            + self.global_projection(global_feature).unsqueeze(1)
            + self.geometry_projection(geometry)
            + self.pair_embedding
        )
        pair_tokens = self.pair_set_block(pair_tokens)
        corrections = torch.tanh(self.correction_head(self.pair_mixer(pair_tokens)))
        midpoint_delta = corrections[..., :2] * self.max_midpoint_offset
        angle_delta = corrections[..., 2:3] * self.max_angle_offset
        length_scale = torch.exp(corrections[..., 3:4] * self.max_log_length_offset)

        cos_delta = torch.cos(angle_delta)
        sin_delta = torch.sin(angle_delta)
        rotated_unit = torch.cat(
            [
                cos_delta * unit_vector[..., :1] - sin_delta * unit_vector[..., 1:2],
                sin_delta * unit_vector[..., :1] + cos_delta * unit_vector[..., 1:2],
            ],
            dim=-1,
        )
        refined_midpoint = midpoint + midpoint_delta
        half_vector = 0.5 * length * length_scale * rotated_unit
        refined_points = points.clone()
        refined_points[:, pair_index[:, 0]] = refined_midpoint - half_vector
        refined_points[:, pair_index[:, 1]] = refined_midpoint + half_vector
        refined_coords = canonicalize_landmark_pairs(
            refined_points.clamp(0.0, 1.0).reshape(canonical_coords.shape[0], -1),
            self.task_id,
        )
        return refined_coords, {
            "pair_set_anchor_coords_transformed": canonical_coords,
            "pair_set_midpoint_offsets": midpoint_delta,
            "pair_set_angle_offsets": angle_delta,
            "pair_set_length_scales": length_scale,
        }


class A4CPairSetHeatmapHead(CardiacGraphRefineHeatmapHead):
    """Checkpoint-compatible A4C head with an order-invariant pair-set decoder."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels=in_channels, num_points=num_points, width_multiplier=width_multiplier)
        self.pair_set_refiner = CardiacPairSetRefiner(in_channels=in_channels, task_id="A4C")

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        anchor_coords = aux_outputs["refined_coords_transformed"]
        refined_coords, pair_outputs = self.pair_set_refiner(x, anchor_coords)
        aux_outputs = dict(aux_outputs)
        aux_outputs["pre_canonical_coords_transformed"] = anchor_coords
        aux_outputs["refined_coords_transformed"] = refined_coords
        aux_outputs.update(pair_outputs)
        return point_logits, aux_outputs


class PSAXPairSetHeatmapHead(SubpixelOffsetDeepHeatmapHead):
    """Checkpoint-compatible PSAX head with an order-invariant pair-set decoder."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels=in_channels, num_points=num_points, width_multiplier=width_multiplier)
        self.pair_set_refiner = CardiacPairSetRefiner(in_channels=in_channels, task_id="PSAX")

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        anchor_coords = aux_outputs["refined_coords_transformed"]
        refined_coords, pair_outputs = self.pair_set_refiner(x, anchor_coords)
        aux_outputs = dict(aux_outputs)
        aux_outputs["pre_canonical_coords_transformed"] = anchor_coords
        aux_outputs["refined_coords_transformed"] = refined_coords
        aux_outputs.update(pair_outputs)
        return point_logits, aux_outputs


class A4CHeatmapHead(nn.Module):
    """Dedicated chamber-aware decoder for four-chamber cardiac structure landmarks."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 224) * width_multiplier), 128)
        hidden2 = max(int(max(hidden1 // 2, 160) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 128) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.context_d1 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.context_d2 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.context_d3 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=3, dilation=3, bias=False)
        self.context_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        context = self.context_d1(x) + self.context_d2(x) + self.context_d3(x)
        x = self.context_bn(context)
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class A4CStructuredV2HeatmapHead(nn.Module):
    """Group-aware A4C decoder with graph reasoning and local per-point refinement."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 224) * width_multiplier), 128)
        hidden2 = max(int(max(hidden1 // 2, 176) * width_multiplier), 112)
        hidden3 = max(int(max(hidden2 // 2, 144) * width_multiplier), 96)
        token_dim = max(hidden3, 128)
        self.num_points = num_points
        self.local_patch_size = 12
        self.local_span = 0.08
        self.num_groups = max(1, min(4, num_points // 4 if num_points >= 8 else num_points))

        group_ids = torch.arange(num_points, dtype=torch.long) * self.num_groups // max(num_points, 1)
        self.register_buffer("group_ids", group_ids.clamp(0, self.num_groups - 1), persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.coord_embed = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_feat_proj = nn.Sequential(
            nn.Linear(hidden3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.group_feat_proj = nn.Sequential(
            nn.Linear(hidden3 + 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_index_embed = nn.Parameter(torch.zeros(1, num_points, token_dim))
        self.group_index_embed = nn.Parameter(torch.zeros(1, self.num_groups, token_dim))
        nn.init.trunc_normal_(self.point_index_embed, std=0.02)
        nn.init.trunc_normal_(self.group_index_embed, std=0.02)
        self.graph_blocks = nn.ModuleList(
            [LandmarkGraphRefineBlock(token_dim) for _ in range(4)]
        )
        self.graph_offset_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
            nn.Tanh(),
        )
        local_hidden = max(token_dim, 128)
        self.local_refine_head = nn.Sequential(
            nn.Conv2d(hidden3 + 1, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden3, local_hidden),
            nn.GELU(),
            nn.Linear(local_hidden, 2),
            nn.Tanh(),
        )
        self.graph_refine_max_offset = 0.040
        self.local_refine_max_offset = 0.020

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    @staticmethod
    def _sample_point_features(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz, channels = feat.shape[:2]
        point_coords = coords.view(bsz, -1, 2)
        grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def _build_group_tokens(self, coarse_points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        bsz = coarse_points.shape[0]
        group_tokens = []
        for group_idx in range(self.num_groups):
            mask = self.group_ids == group_idx
            coords_group = coarse_points[:, mask, :]
            feats_group = point_features[:, mask, :]
            pooled = torch.cat([feats_group.mean(dim=1), coords_group.mean(dim=1)], dim=-1)
            token = self.group_feat_proj(pooled)
            group_tokens.append(token)
        return torch.stack(group_tokens, dim=1) + self.group_index_embed

    def _sample_local_patches(self, feat: torch.Tensor, point_logits: torch.Tensor, coarse_points: torch.Tensor) -> torch.Tensor:
        bsz, channels, _, _ = feat.shape
        patches = []
        patch_size = self.local_patch_size
        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, patch_size, patch_size, 2)
        for point_idx in range(self.num_points):
            refine_feat = torch.cat(
                [feat, torch.sigmoid(point_logits[:, point_idx : point_idx + 1])],
                dim=1,
            )
            center = coarse_points[:, point_idx, :].view(bsz, 1, 1, 2)
            sample_grid = center + base_grid * self.local_span
            sample_grid = sample_grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            patch = F.grid_sample(
                refine_feat,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            patches.append(patch)
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(feat_for_refine, coarse_coords)
        pooled_feature = F.adaptive_avg_pool2d(feat_for_refine, output_size=1).flatten(1)
        group_tokens = self._build_group_tokens(coarse_points, point_features)
        group_context = group_tokens[:, self.group_ids, :]
        point_tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(pooled_feature).unsqueeze(1)
            + group_context
            + self.point_index_embed
        )
        for block in self.graph_blocks:
            point_tokens = block(point_tokens)
        graph_offsets = self.graph_offset_head(point_tokens) * self.graph_refine_max_offset

        local_patches = self._sample_local_patches(feat_for_refine, point_logits, coarse_points)
        patch_input = local_patches.view(
            coarse_points.shape[0] * self.num_points,
            local_patches.shape[2],
            local_patches.shape[3],
            local_patches.shape[4],
        )
        local_offsets = self.local_refine_head(patch_input).view(coarse_points.shape[0], self.num_points, 2)
        local_offsets = local_offsets * self.local_refine_max_offset

        refined_coords = torch.clamp(coarse_points + graph_offsets + local_offsets, 0.0, 1.0).reshape(coarse_coords.shape[0], -1)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class A4CResidualLocalRefineHeatmapHead(CardiacGraphRefineHeatmapHead):
    """Checkpoint-compatible A4C upgrade with small zero-init local residual refinement."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier=width_multiplier)
        hidden3 = self.point_head.in_channels
        refine_hidden = max(hidden3, 96)
        self.local_patch_size = 12
        self.local_span = 0.06
        self.local_refine_head = nn.Sequential(
            nn.Conv2d(hidden3 + 1, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden3, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, 2),
            nn.Tanh(),
        )
        self.local_refine_max_offset = 0.010
        final_linear = self.local_refine_head[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _sample_local_patches(self, feat: torch.Tensor, point_logits: torch.Tensor, coarse_points: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.local_patch_size
        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, patch_size, patch_size, 2)
        patches = []
        for point_idx in range(self.num_points):
            refine_feat = torch.cat(
                [feat, torch.sigmoid(point_logits[:, point_idx : point_idx + 1])],
                dim=1,
            )
            center = coarse_points[:, point_idx, :].view(bsz, 1, 1, 2)
            sample_grid = center + base_grid * self.local_span
            sample_grid = sample_grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            patch = F.grid_sample(
                refine_feat,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            patches.append(patch)
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(feat_for_refine, coarse_coords)
        pooled_feature = F.adaptive_avg_pool2d(feat_for_refine, output_size=1).flatten(1)
        tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(pooled_feature).unsqueeze(1)
            + self.point_index_embed
        )
        for block in self.graph_blocks:
            tokens = block(tokens)
        graph_offsets = self.offset_head(tokens) * self.refine_max_offset

        local_patches = self._sample_local_patches(feat_for_refine, point_logits, coarse_points)
        patch_input = local_patches.view(
            coarse_points.shape[0] * self.num_points,
            local_patches.shape[2],
            local_patches.shape[3],
            local_patches.shape[4],
        )
        local_offsets = self.local_refine_head(patch_input).view(coarse_points.shape[0], self.num_points, 2)
        local_offsets = local_offsets * self.local_refine_max_offset

        refined_coords = torch.clamp(coarse_points + graph_offsets + local_offsets, 0.0, 1.0).reshape(coarse_coords.shape[0], -1)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class A4CStrongResidualHeatmapHead(CardiacGraphRefineHeatmapHead):
    """Stronger checkpoint-compatible A4C head with multiscale residual heatmaps and local refinement."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier=width_multiplier)
        hidden3 = self.point_head.in_channels
        refine_hidden = max(hidden3 + 32, 128)
        self.local_patch_size = 14
        self.local_span = 0.08
        self.ms_d1 = nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, dilation=1, bias=False)
        self.ms_d2 = nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=2, dilation=2, bias=False)
        self.ms_d4 = nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=4, dilation=4, bias=False)
        self.ms_bn = nn.BatchNorm2d(hidden3)
        self.ms_act = nn.GELU()
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(hidden3 * 2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.heatmap_residual_head = nn.Sequential(
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )
        self.local_refine_head = nn.Sequential(
            nn.Conv2d(hidden3 + 1, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden3, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, 2),
            nn.Tanh(),
        )
        self.heatmap_residual_scale = 0.35
        self.local_refine_max_offset = 0.014
        heatmap_last = self.heatmap_residual_head[-1]
        nn.init.zeros_(heatmap_last.weight)
        nn.init.zeros_(heatmap_last.bias)
        local_last = self.local_refine_head[-2]
        nn.init.zeros_(local_last.weight)
        nn.init.zeros_(local_last.bias)

    def _sample_local_patches(self, feat: torch.Tensor, point_logits: torch.Tensor, coarse_points: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.local_patch_size
        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, patch_size, patch_size, 2)
        patches = []
        for point_idx in range(self.num_points):
            refine_feat = torch.cat(
                [feat, torch.sigmoid(point_logits[:, point_idx : point_idx + 1])],
                dim=1,
            )
            center = coarse_points[:, point_idx, :].view(bsz, 1, 1, 2)
            sample_grid = center + base_grid * self.local_span
            sample_grid = sample_grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            patch = F.grid_sample(
                refine_feat,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            patches.append(patch)
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        ms_context = self.ms_act(self.ms_bn(self.ms_d1(x) + self.ms_d2(x) + self.ms_d4(x)))
        fused_feat = self.ms_fuse(torch.cat([x, ms_context], dim=1))

        base_logits = self.point_head(x)
        residual_logits = self.heatmap_residual_head(fused_feat) * self.heatmap_residual_scale
        point_logits = base_logits + residual_logits
        feat_for_refine = fused_feat
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        coarse_points = coarse_coords.view(coarse_coords.shape[0], self.num_points, 2)
        point_features = self._sample_point_features(feat_for_refine, coarse_coords)
        pooled_feature = F.adaptive_avg_pool2d(feat_for_refine, output_size=1).flatten(1)
        tokens = (
            self.coord_embed(coarse_points)
            + self.point_feat_proj(point_features)
            + self.global_proj(pooled_feature).unsqueeze(1)
            + self.point_index_embed
        )
        for block in self.graph_blocks:
            tokens = block(tokens)
        graph_offsets = self.offset_head(tokens) * self.refine_max_offset

        local_patches = self._sample_local_patches(feat_for_refine, point_logits, coarse_points)
        patch_input = local_patches.view(
            coarse_points.shape[0] * self.num_points,
            local_patches.shape[2],
            local_patches.shape[3],
            local_patches.shape[4],
        )
        local_offsets = self.local_refine_head(patch_input).view(coarse_points.shape[0], self.num_points, 2)
        local_offsets = local_offsets * self.local_refine_max_offset

        refined_coords = torch.clamp(coarse_points + graph_offsets + local_offsets, 0.0, 1.0).reshape(coarse_coords.shape[0], -1)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class AOPHeatmapHead(nn.Module):
    """Dedicated arc-aware decoder for compact curved AOP anatomy."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 176) * width_multiplier), 96)
        hidden2 = max(int(max(hidden1 // 2, 120) * width_multiplier), 72)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.arc_h = nn.Conv2d(hidden1, hidden1, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.arc_v = nn.Conv2d(hidden1, hidden1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.arc_d = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.arc_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.arc_bn(self.arc_h(x) + self.arc_v(x) + self.arc_d(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class FAHeatmapHead(nn.Module):
    """Dedicated axis-aware decoder for fetal abdomen biometry landmarks."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 192) * width_multiplier), 112)
        hidden2 = max(int(max(hidden1 // 2, 128) * width_multiplier), 80)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.axis_h = nn.Conv2d(hidden1, hidden1, kernel_size=(1, 9), padding=(0, 4), bias=False)
        self.axis_v = nn.Conv2d(hidden1, hidden1, kernel_size=(9, 1), padding=(4, 0), bias=False)
        self.axis_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.axis_bn(self.axis_h(x) + self.axis_v(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class FARefineHeatmapHead(nn.Module):
    """Axis-aware coarse-to-fine decoder for fetal abdomen landmarks."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 208) * width_multiplier), 120)
        hidden2 = max(int(max(hidden1 // 2, 144) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 112) * width_multiplier), 72)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.axis_context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=(9, 1), padding=(4, 0), bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.refine_patch_size = 18
        refine_hidden = max(hidden3, 96)
        refine_in_channels = hidden3 + num_points
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, num_points * 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.045

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def _sample_refine_roi(self, feat: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.refine_patch_size
        points = coarse_coords.reshape(bsz, -1, 2)
        min_xy = points.min(dim=1).values
        max_xy = points.max(dim=1).values
        center = 0.5 * (min_xy + max_xy)
        span_xy = (max_xy - min_xy).clamp(0.08, 0.60)
        span = span_xy.amax(dim=-1).clamp(0.16, 0.34)

        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)

        sample_grid = center.view(bsz, 1, 1, 2) + base_grid * span.view(bsz, 1, 1, 1)
        sample_grid = sample_grid.clamp(0.0, 1.0)
        sample_grid = sample_grid * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.axis_context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat([feat_for_refine, torch.sigmoid(point_logits)], dim=1)
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        offsets = self.refine_head(roi_patch) * self.refine_max_offset
        refined_coords = (coarse_coords + offsets).clamp(0.0, 1.0)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class HCHeatmapHead(nn.Module):
    """Dedicated ring-aware decoder for head circumference landmarks."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 192) * width_multiplier), 112)
        hidden2 = max(int(max(hidden1 // 2, 128) * width_multiplier), 80)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.ring_d1 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.ring_d2 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.ring_d4 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=4, dilation=4, bias=False)
        self.ring_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.ring_bn(self.ring_d1(x) + self.ring_d2(x) + self.ring_d4(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class HCRefineHeatmapHead(nn.Module):
    """Ring-aware coarse-to-fine decoder for head circumference landmarks."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 208) * width_multiplier), 128)
        hidden2 = max(int(max(hidden1 // 2, 144) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 112) * width_multiplier), 72)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.ring_context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.refine_patch_size = 20
        refine_hidden = max(hidden3, 96)
        refine_in_channels = hidden3 + num_points
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, num_points * 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.055

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def _sample_refine_roi(self, feat: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.refine_patch_size
        points = coarse_coords.reshape(bsz, -1, 2)
        min_xy = points.min(dim=1).values
        max_xy = points.max(dim=1).values
        center = 0.5 * (min_xy + max_xy)
        span_xy = (max_xy - min_xy).clamp(0.08, 0.60)
        span = span_xy.amax(dim=-1).clamp(0.14, 0.34) * 0.95

        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)

        sample_grid = center.view(bsz, 1, 1, 2) + base_grid * span.view(bsz, 1, 1, 1)
        sample_grid = sample_grid.clamp(0.0, 1.0)
        sample_grid = sample_grid * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.ring_context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat([feat_for_refine, torch.sigmoid(point_logits)], dim=1)
        roi_patch = self._sample_refine_roi(refine_input, coarse_coords)
        offsets = self.refine_head(roi_patch) * self.refine_max_offset
        refined_coords = (coarse_coords + offsets).clamp(0.0, 1.0)
        return point_logits, {
            "coarse_coords_transformed": coarse_coords,
            "refined_coords_transformed": refined_coords,
        }


class HCRefineOffsetHeatmapHead(HCRefineHeatmapHead):
    """HC refine head with an additional zero-started local heatmap offset stage."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        self.post_offset_refiner = LogitPatchOffsetRefiner(
            in_channels=num_points,
            num_points=num_points,
            hidden_channels=64,
            patch_size=13,
            span=0.050,
            max_offset=0.010,
        )

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        refined_coords = aux_outputs["refined_coords_transformed"]
        post_refined_coords = self.post_offset_refiner(torch.sigmoid(point_logits), refined_coords)
        aux_outputs = dict(aux_outputs)
        aux_outputs["pre_post_offset_coords_transformed"] = refined_coords
        aux_outputs["refined_coords_transformed"] = post_refined_coords
        return point_logits, aux_outputs


class IVCHeatmapHead(nn.Module):
    """Dedicated diameter-aware decoder for noisy 2-point IVC measurements."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 176) * width_multiplier), 96)
        hidden2 = max(int(max(hidden1 // 2, 120) * width_multiplier), 72)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.dir_h = nn.Conv2d(hidden1, hidden1, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.dir_v = nn.Conv2d(hidden1, hidden1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.dir_d1 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.dir_d2 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=3, dilation=3, bias=False)
        self.context_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.context_bn(self.dir_h(x) + self.dir_v(x) + self.dir_d1(x) + self.dir_d2(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class IVCRefineHeatmapHead(nn.Module):
    """Coarse-to-fine local refinement head for short IVC diameter landmarks."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 192) * width_multiplier), 112)
        hidden2 = max(int(max(hidden1 // 2, 128) * width_multiplier), 80)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.band_head = nn.Sequential(
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, 1, kernel_size=1),
        )
        self.refine_patch_size = 16
        refine_hidden = max(hidden3, 64)
        refine_in_channels = hidden3 + num_points + 1
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, num_points * 2),
            nn.Tanh(),
        )
        self.refine_max_offset = 0.10

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def _sample_refine_roi(self, feat: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_size = self.refine_patch_size
        point_pairs = coarse_coords.reshape(bsz, -1, 2)
        midpoint = point_pairs.mean(dim=1)
        segment_length = torch.norm(point_pairs[:, 1] - point_pairs[:, 0], dim=-1)
        half_span = (segment_length * 2.25).clamp(0.10, 0.30)

        grid_lin = torch.linspace(-1.0, 1.0, patch_size, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)

        center = midpoint.view(bsz, 1, 1, 2)
        span = half_span.view(bsz, 1, 1, 1)
        sample_grid = center + base_grid * span
        sample_grid = sample_grid.clamp(0.0, 1.0)
        sample_grid = sample_grid * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        band_logits = self.band_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if band_logits.shape[-2:] != out_size:
            band_logits = F.interpolate(band_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat(
            [
                feat_for_refine,
                torch.sigmoid(point_logits),
                torch.sigmoid(band_logits),
            ],
            dim=1,
        )
        refine_roi = self._sample_refine_roi(refine_input, coarse_coords)
        refine_offsets = self.refine_head(refine_roi) * self.refine_max_offset
        refined_coords = torch.clamp(coarse_coords + refine_offsets, 0.0, 1.0)
        return point_logits, {
            "band_logits": band_logits,
            "refined_coords_transformed": refined_coords,
        }


class IVCRefineV2HeatmapHead(nn.Module):
    """Rotated strip refinement head for short vessel diameter localization."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 224) * width_multiplier), 144)
        hidden2 = max(int(max(hidden1 // 2, 160) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 112) * width_multiplier), 80)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden1, hidden1, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=(9, 1), padding=(4, 0), bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
        )
        self.point_head = nn.Conv2d(hidden3, num_points, kernel_size=1)
        self.band_head = nn.Sequential(
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, 1, kernel_size=1),
        )
        self.refine_hw = (18, 26)
        refine_hidden = max(hidden3, 96)
        refine_in_channels = hidden3 + num_points + 1
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.Conv2d(refine_hidden, refine_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, 4),
            nn.Tanh(),
        )
        self.center_offset_scale = 0.08
        self.angle_offset_scale = 0.40
        self.length_scale = 0.35

    @staticmethod
    def _softargmax_coords(logits: torch.Tensor) -> torch.Tensor:
        bsz, num_points, h, w = logits.shape
        probs = torch.softmax(logits.view(bsz, num_points, -1), dim=-1).view(bsz, num_points, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
        grid_x = xs.view(1, 1, 1, w)
        grid_y = ys.view(1, 1, h, 1)
        expected_x = (probs * grid_x).sum(dim=(-2, -1))
        expected_y = (probs * grid_y).sum(dim=(-2, -1))
        return torch.stack([expected_x, expected_y], dim=-1).reshape(bsz, num_points * 2)

    def _sample_rotated_strip(self, feat: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        bsz = feat.shape[0]
        patch_h, patch_w = self.refine_hw
        points = coarse_coords.reshape(bsz, 2, 2)
        p0 = points[:, 0]
        p1 = points[:, 1]
        center = 0.5 * (p0 + p1)
        vec = p1 - p0
        seg_len = torch.norm(vec, dim=-1).clamp_min(1e-4)
        unit_long = vec / seg_len.unsqueeze(-1)
        unit_short = torch.stack([-unit_long[:, 1], unit_long[:, 0]], dim=-1)
        long_span = (0.5 * seg_len * 2.8).clamp(0.12, 0.34)
        short_span = (0.22 * seg_len + 0.05).clamp(0.05, 0.14)

        long_lin = torch.linspace(-1.0, 1.0, patch_w, device=feat.device, dtype=feat.dtype)
        short_lin = torch.linspace(-1.0, 1.0, patch_h, device=feat.device, dtype=feat.dtype)
        grid_short, grid_long = torch.meshgrid(short_lin, long_lin, indexing="ij")
        grid_long = grid_long.view(1, patch_h, patch_w, 1)
        grid_short = grid_short.view(1, patch_h, patch_w, 1)

        center = center.view(bsz, 1, 1, 2)
        unit_long = unit_long.view(bsz, 1, 1, 2)
        unit_short = unit_short.view(bsz, 1, 1, 2)
        long_span = long_span.view(bsz, 1, 1, 1)
        short_span = short_span.view(bsz, 1, 1, 1)

        sample_grid = center + grid_long * long_span * unit_long + grid_short * short_span * unit_short
        sample_grid = sample_grid.clamp(0.0, 1.0)
        sample_grid = sample_grid * 2.0 - 1.0
        return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _decode_parametric_refine(self, coarse_coords: torch.Tensor, refine_params: torch.Tensor) -> torch.Tensor:
        points = coarse_coords.reshape(coarse_coords.shape[0], 2, 2)
        p0 = points[:, 0]
        p1 = points[:, 1]
        center = 0.5 * (p0 + p1)
        vec = p1 - p0
        coarse_half_len = (0.5 * torch.norm(vec, dim=-1)).clamp_min(1e-4)
        base_angle = torch.atan2(vec[:, 1], vec[:, 0])

        center_delta = refine_params[:, 0:2] * self.center_offset_scale
        angle_delta = refine_params[:, 2] * self.angle_offset_scale
        length_delta = 1.0 + refine_params[:, 3] * self.length_scale

        refined_center = torch.clamp(center + center_delta, 0.0, 1.0)
        refined_angle = base_angle + angle_delta
        refined_half_len = (coarse_half_len * length_delta).clamp(0.01, 0.25)

        direction = torch.stack([torch.cos(refined_angle), torch.sin(refined_angle)], dim=-1)
        offset = direction * refined_half_len.unsqueeze(-1)
        refined_p0 = torch.clamp(refined_center - offset, 0.0, 1.0)
        refined_p1 = torch.clamp(refined_center + offset, 0.0, 1.0)
        return torch.cat([refined_p0, refined_p1], dim=-1)

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = x + self.context(x)
        x = self.decoder(x)
        point_logits = self.point_head(x)
        band_logits = self.band_head(x)
        feat_for_refine = x
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if band_logits.shape[-2:] != out_size:
            band_logits = F.interpolate(band_logits, size=out_size, mode="bilinear", align_corners=False)
        if feat_for_refine.shape[-2:] != out_size:
            feat_for_refine = F.interpolate(feat_for_refine, size=out_size, mode="bilinear", align_corners=False)

        coarse_coords = self._softargmax_coords(point_logits)
        refine_input = torch.cat(
            [
                feat_for_refine,
                torch.sigmoid(point_logits),
                torch.sigmoid(band_logits),
            ],
            dim=1,
        )
        refine_roi = self._sample_rotated_strip(refine_input, coarse_coords)
        refine_params = self.refine_head(refine_roi)
        refined_coords = self._decode_parametric_refine(coarse_coords, refine_params)
        return point_logits, {
            "band_logits": band_logits,
            "refined_coords_transformed": refined_coords,
        }


class IVCRefineV3HeatmapHead(IVCRefineV2HeatmapHead):
    """IVC rotated-strip head with a zero-started local endpoint correction stage."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        self.post_offset_refiner = LogitPatchOffsetRefiner(
            in_channels=num_points + 1,
            num_points=num_points,
            hidden_channels=64,
            patch_size=15,
            span=0.060,
            max_offset=0.014,
        )

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        refined_coords = aux_outputs["refined_coords_transformed"]
        band_logits = aux_outputs["band_logits"]
        post_context = torch.cat([torch.sigmoid(point_logits), torch.sigmoid(band_logits)], dim=1)
        post_refined_coords = self.post_offset_refiner(post_context, refined_coords)
        aux_outputs = dict(aux_outputs)
        aux_outputs["pre_post_offset_coords_transformed"] = refined_coords
        aux_outputs["refined_coords_transformed"] = post_refined_coords
        return point_logits, aux_outputs


class PLAXHeatmapHead(nn.Module):
    """Dedicated long-axis cardiac decoder for dense PLAX landmark layouts."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 240) * width_multiplier), 144)
        hidden2 = max(int(max(hidden1 // 2, 176) * width_multiplier), 96)
        hidden3 = max(int(max(hidden2 // 2, 128) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.axis_h = nn.Conv2d(hidden1, hidden1, kernel_size=(1, 9), padding=(0, 4), bias=False)
        self.axis_v = nn.Conv2d(hidden1, hidden1, kernel_size=(9, 1), padding=(4, 0), bias=False)
        self.axis_d2 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.axis_d4 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=4, dilation=4, bias=False)
        self.axis_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.axis_bn(self.axis_h(x) + self.axis_v(x) + self.axis_d2(x) + self.axis_d4(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class PSAXHeatmapHead(nn.Module):
    """Dedicated short-axis ring decoder for localized PSAX paired diameters."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__()
        hidden1 = max(int(max(in_channels // 2, 184) * width_multiplier), 104)
        hidden2 = max(int(max(hidden1 // 2, 128) * width_multiplier), 80)
        hidden3 = max(int(max(hidden2 // 2, 96) * width_multiplier), 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.GELU(),
        )
        self.ring_d1 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.ring_d2 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.ring_d3 = nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=3, dilation=3, bias=False)
        self.diag_mix = nn.Conv2d(hidden1, hidden1, kernel_size=5, padding=2, bias=False)
        self.context_bn = nn.BatchNorm2d(hidden1)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden2, hidden3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.GELU(),
            nn.Conv2d(hidden3, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.context_bn(self.ring_d1(x) + self.ring_d2(x) + self.ring_d3(x) + self.diag_mix(x))
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class StructureCardiacGraphRefineHeatmapHead(StructureAuxiliaryMixin, CardiacGraphRefineHeatmapHead):
    """Cardiac graph head with an auxiliary paired-structure map."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        self._init_structure_auxiliary(num_points=num_points, hidden_channels=max(48, num_points * 3))

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        return point_logits, self._append_structure_logits(point_logits, aux_outputs)


class StructureHCRefineOffsetHeatmapHead(StructureAuxiliaryMixin, HCRefineOffsetHeatmapHead):
    """HC refine-offset head with an auxiliary ellipse-diameter structure map."""

    def __init__(self, in_channels: int, num_points: int, width_multiplier: float = 1.0):
        super().__init__(in_channels, num_points, width_multiplier)
        self._init_structure_auxiliary(num_points=num_points, hidden_channels=32)

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        return point_logits, self._append_structure_logits(point_logits, aux_outputs)


class StructureFUGCHeatmapHead(FUGCHeatmapHead):
    """FUGC head exposing the short-segment map as the generic structure map."""

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        aux_outputs = dict(aux_outputs)
        aux_outputs["structure_logits"] = aux_outputs["segment_logits"]
        return point_logits, aux_outputs


class StructureIVCRefineV3HeatmapHead(IVCRefineV3HeatmapHead):
    """IVC head exposing the vessel band map as the generic structure map."""

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        aux_outputs = dict(aux_outputs)
        aux_outputs["structure_logits"] = aux_outputs["band_logits"]
        return point_logits, aux_outputs


class StructureFemurHeatmapHead(FemurHeatmapHead):
    """Femur head exposing the shaft map as the generic structure map."""

    def forward(self, x: torch.Tensor, out_size):
        point_logits, aux_outputs = super().forward(x, out_size)
        aux_outputs = dict(aux_outputs)
        aux_outputs["structure_logits"] = aux_outputs["shaft_logits"]
        return point_logits, aux_outputs


class FPN(nn.Module):
    """Feature Pyramid Network (FPN) neck for multi-scale feature enrichment.

    Takes the single-scale feature map from DINOv2 backbone and builds a
    multi-scale feature pyramid via bottom-up downsampling and top-down
    fusion, producing enriched features for keypoint heatmap prediction.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # 1×1 conv to reduce backbone channels
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Bottom-up pyramid levels
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Lateral connections for top-down pathway
        self.lateral2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        # Final output smoothing
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] from backbone
        x = self.reduce(x)  # [B, out_channels, H, W]

        # Bottom-up: build 3-level pyramid
        c1 = F.relu(self.bn1(self.conv1(x)))                              # stride 1  (finest)
        c2 = F.relu(self.bn2(self.conv2(F.max_pool2d(c1, 2))))            # stride 2
        c3 = F.relu(self.bn3(self.conv3(F.max_pool2d(c2, 2))))            # stride 4  (coarsest)

        # Top-down pathway with lateral connections
        m3 = c3
        m2 = self.lateral2(c2) + F.interpolate(m3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        m1 = self.lateral1(c1) + F.interpolate(m2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        # Fuse all pyramid levels at finest resolution
        out = m1 + F.interpolate(m2, size=m1.shape[-2:], mode="bilinear", align_corners=False) \
                  + F.interpolate(m3, size=m1.shape[-2:], mode="bilinear", align_corners=False)

        return self.smooth(out)


class BiFPNBlock(nn.Module):
    """Single bidirectional fusion block over a 3-level feature pyramid."""

    def __init__(self, channels: int):
        super().__init__()
        self.w_td_2 = nn.Parameter(torch.ones(2))
        self.w_td_1 = nn.Parameter(torch.ones(2))
        self.w_out_2 = nn.Parameter(torch.ones(3))
        self.w_out_3 = nn.Parameter(torch.ones(2))

        self.refine_td2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.refine_td1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.refine_out2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.refine_out3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    @staticmethod
    def _normalize(weights: torch.Tensor) -> torch.Tensor:
        weights = F.relu(weights)
        return weights / (weights.sum() + 1e-4)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor):
        w_td_2 = self._normalize(self.w_td_2)
        td2 = self.refine_td2(
            w_td_2[0] * p2
            + w_td_2[1] * F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        )

        w_td_1 = self._normalize(self.w_td_1)
        td1 = self.refine_td1(
            w_td_1[0] * p1
            + w_td_1[1] * F.interpolate(td2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        )

        w_out_2 = self._normalize(self.w_out_2)
        out2 = self.refine_out2(
            w_out_2[0] * p2
            + w_out_2[1] * td2
            + w_out_2[2] * F.max_pool2d(td1, kernel_size=2)
        )

        w_out_3 = self._normalize(self.w_out_3)
        out3 = self.refine_out3(
            w_out_3[0] * p3
            + w_out_3[1] * F.max_pool2d(out2, kernel_size=2)
        )
        return td1, out2, out3


class BiFPN(nn.Module):
    """Bidirectional feature pyramid neck for stronger multi-scale fusion."""

    def __init__(self, in_channels: int, out_channels: int = 256, num_blocks: int = 2):
        super().__init__()
        self.out_channels = out_channels
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.p1_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.p2_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.p3_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(BiFPNBlock(out_channels) for _ in range(num_blocks))
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        p1 = self.p1_conv(x)
        p2 = self.p2_conv(F.max_pool2d(p1, kernel_size=2))
        p3 = self.p3_conv(F.max_pool2d(p2, kernel_size=2))

        for block in self.blocks:
            p1, p2, p3 = block(p1, p2, p3)

        out = (
            p1
            + F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        )
        return self.smooth(out)


def build_neck(fpn_type: str, in_channels: int, out_channels: int = 256) -> nn.Module:
    if fpn_type == "fpn":
        return FPN(in_channels=in_channels, out_channels=out_channels)
    if fpn_type == "bifpn":
        return BiFPN(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unsupported fpn_type: {fpn_type}")


class MLPRegressionHead(nn.Module):
    """Simple MLP regression head that directly predicts keypoint coordinates.
    
    Takes DINOv2 feature map [B, C, H, W], applies adaptive average pooling,
    flattens, and predicts normalized coordinates [0, 1] via MLP.
    """

    def __init__(self, in_channels: int, num_points: int, hidden_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_points * 2),  # Output: [x1, y1, x2, y2, ...]
            nn.Sigmoid(),  # Normalize to [0, 1] for normalized coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            Normalized coordinates [B, num_points*2] in range [0, 1]
        """
        x = self.pool(x)  # [B, C, 1, 1]
        return self.mlp(x)


class DINOv2Backbone(nn.Module):
    """Returns a 2D feature map [B, C, H, W] from ViT or timm CNN backbones."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        feature_mode: str = "final",
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_last_blocks: int = 4,
        lora_dropout: float = 0.05,
        lora_task_ids: List[str] | None = None,
    ):
        super().__init__()
        if feature_mode not in ENCODER_FEATURE_MODES:
            raise ValueError(f"Unsupported encoder feature mode: {feature_mode}")
        timm = importlib.import_module("timm")
        self.model_name = model_name
        self.feature_mode = feature_mode
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_last_blocks = int(lora_last_blocks)
        self.lora_dropout = float(lora_dropout)
        self.lora_task_ids = tuple(map(str, lora_task_ids or ()))
        self.lora_injected_modules: tuple[str, ...] = ()

        probe_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.is_vit_style = hasattr(probe_backbone, "patch_embed")
        if self.is_vit_style:
            self.backbone = probe_backbone
            self.out_channels = int(self.backbone.num_features)
            self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
            self.intermediate_indices = self._resolve_intermediate_indices()
        else:
            # Non-ViT timm models expose hierarchical feature maps. Fuse them to a
            # single high-resolution tensor so the existing task heads can be reused.
            del probe_backbone
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
            )
            self.out_channels = 256
            self.num_prefix_tokens = 0
            self.intermediate_indices = ()
            feature_channels = list(map(int, self.backbone.feature_info.channels()))
            feature_reductions = list(map(int, self.backbone.feature_info.reduction()))
            self.feature_fusion_target_index = self._resolve_feature_fusion_target_index(feature_reductions)
            self.feature_fusion_projections = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(channels, self.out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.GELU(),
                )
                for channels in feature_channels
            )
            self.feature_fusion_refine = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
            )
            init_logits = torch.linspace(-1.0, 1.0, steps=len(feature_channels))
            init_logits[self.feature_fusion_target_index] = 2.5
            self.feature_fusion_logits = nn.Parameter(init_logits)
            print(
                "Feature backbone fusion: "
                f"model={model_name}, channels={feature_channels}, "
                f"reductions={feature_reductions}, target_index={self.feature_fusion_target_index}"
            )
            if self.feature_mode == "multilayer_fusion_v1":
                print("Feature backbone does not use ViT layer tokens; using feature_pyramid_fusion_v1 behavior.")
        if self.is_vit_style and self.feature_mode == "multilayer_fusion_v1":
            if not hasattr(self.backbone, "forward_intermediates"):
                raise ValueError(f"Model '{model_name}' does not expose forward_intermediates().")
            init_logits = torch.linspace(-4.0, -1.5, steps=len(self.intermediate_indices))
            init_logits[-1] = 4.0
            self.feature_layer_logits = nn.Parameter(init_logits)
        else:
            self.register_parameter("feature_layer_logits", None)

        if self.lora_rank > 0:
            if not self.is_vit_style:
                raise ValueError("Encoder LoRA currently requires a ViT-style backbone")
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            injected = inject_lora_into_vit_attention(
                self.backbone,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                last_blocks=self.lora_last_blocks,
                dropout=self.lora_dropout,
                task_ids=self.lora_task_ids,
            )
            self.lora_injected_modules = tuple(injected)
            print(
                "Encoder LoRA: ENABLED "
                f"(rank={self.lora_rank}, alpha={self.lora_alpha:g}, "
                f"last_blocks={self.lora_last_blocks}, projections={len(injected)})"
            )

    def set_lora_task(self, task_id: str) -> None:
        if self.lora_task_ids:
            set_active_lora_task(self.backbone, task_id)

    @staticmethod
    def _resolve_feature_fusion_target_index(reductions: list[int]) -> int:
        # Use stride-8 when available. It preserves enough spatial detail for
        # ultrasound landmarks while keeping memory practical at 512 input size.
        candidates = [idx for idx, reduction in enumerate(reductions) if reduction <= 8]
        if candidates:
            return candidates[-1]
        return max(len(reductions) - 2, 0)

    def _resolve_intermediate_indices(self) -> tuple[int, ...]:
        depth = len(getattr(self.backbone, "blocks", []))
        if depth <= 0:
            return ()
        raw_indices = [
            max(depth // 4 - 1, 0),
            max(depth // 2 - 1, 0),
            max((3 * depth) // 4 - 1, 0),
            depth - 1,
        ]
        return tuple(dict.fromkeys(raw_indices))

    def _final_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                all_tokens = feats["x_prenorm"]
                patch_tokens = all_tokens[:, self.num_prefix_tokens :, :]
            else:
                raise RuntimeError("Unsupported forward_features output from DINOv2 backbone.")
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                patch_tokens = feats[:, self.num_prefix_tokens :, :]
            else:
                raise RuntimeError("Unexpected tensor shape from forward_features.")
        else:
            raise RuntimeError("Unexpected feature type from DINOv2 backbone.")

        bsz, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError("Patch token count is not square; input size may be incompatible.")

        return patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side)

    def _multilayer_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_intermediates(
            x,
            indices=list(self.intermediate_indices),
            return_prefix_tokens=False,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        if not isinstance(features, (list, tuple)) or not features:
            raise RuntimeError("forward_intermediates did not return feature maps.")

        target_size = features[-1].shape[-2:]
        aligned = []
        for feature in features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(feature)

        weights = torch.softmax(self.feature_layer_logits.to(dtype=aligned[-1].dtype), dim=0)
        fused = torch.zeros_like(aligned[-1])
        for weight, feature in zip(weights, aligned):
            fused = fused + weight.view(1, 1, 1, 1) * feature
        return fused

    def _feature_pyramid_map(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)) or not features:
            raise RuntimeError("features_only backbone did not return feature maps.")
        target_size = features[self.feature_fusion_target_index].shape[-2:]
        weights = torch.softmax(self.feature_fusion_logits.to(dtype=features[-1].dtype), dim=0)
        fused = None
        for weight, feature, projection in zip(weights, features, self.feature_fusion_projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode="bilinear", align_corners=False)
            weighted = weight.view(1, 1, 1, 1) * projected
            fused = weighted if fused is None else fused + weighted
        if fused is None:
            raise RuntimeError("Failed to fuse feature pyramid maps.")
        return self.feature_fusion_refine(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_vit_style:
            return self._feature_pyramid_map(x)
        if self.feature_mode == "multilayer_fusion_v1":
            return self._multilayer_feature_map(x)
        return self._final_feature_map(x)


class MultiTaskModelFactory(nn.Module):
    """Keypoint heatmap model with FPN neck using a shared DINOv2 encoder and task-specific heads."""

    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str,
        task_configs: List[Dict],
        heatmap_size=(64, 64),
        use_fpn: bool = False,
        fpn_mode: str = "shared",
        fpn_type: str = "fpn",
        encoder_feature_mode: str = "final",
        head_type: str = "basic",
        task_head_profile: str = "uniform",
        task_decoder_profile: str = "uniform",
        task_adapter_profile: str = "uniform",
        encoder_lora_rank: int = 0,
        encoder_lora_alpha: float = 16.0,
        encoder_lora_last_blocks: int = 4,
        encoder_lora_dropout: float = 0.05,
        encoder_lora_task_specific: bool = False,
        domain_adversarial: bool = False,
        num_domain_classes: int = 0,
    ):
        super().__init__()

        self.heatmap_size = heatmap_size
        self.use_fpn = use_fpn
        self.fpn_mode = fpn_mode
        self.fpn_type = fpn_type
        self.encoder_feature_mode = encoder_feature_mode
        self.head_type = head_type
        self.task_head_profile = task_head_profile
        self.task_decoder_profile = task_decoder_profile
        self.task_adapter_profile = task_adapter_profile
        self.encoder_lora_rank = int(encoder_lora_rank)
        self.encoder_lora_alpha = float(encoder_lora_alpha)
        self.encoder_lora_last_blocks = int(encoder_lora_last_blocks)
        self.encoder_lora_dropout = float(encoder_lora_dropout)
        self.encoder_lora_task_specific = bool(encoder_lora_task_specific)
        self.domain_adversarial = bool(domain_adversarial)
        self.num_domain_classes = int(num_domain_classes)

        print(f"Initializing encoder: {encoder_name}")
        print(f"Encoder feature mode: {encoder_feature_mode}")
        self.encoder = DINOv2Backbone(
            model_name=encoder_name,
            pretrained=(encoder_weights is not None),
            feature_mode=encoder_feature_mode,
            lora_rank=self.encoder_lora_rank,
            lora_alpha=self.encoder_lora_alpha,
            lora_last_blocks=self.encoder_lora_last_blocks,
            lora_dropout=self.encoder_lora_dropout,
            lora_task_ids=(
                sorted(str(config["task_id"]) for config in task_configs)
                if self.encoder_lora_task_specific
                else None
            ),
        )

        # FPN neck (optional)
        self.fpn = None
        self.task_fpns = None
        self.soft_adapters = None
        self.local_refine_adapters = None
        self.context_adapters = None
        self.task_film_adapters = None
        self.context_expert_adapters = None
        self.context_local_adapters = None
        self.encoder_task_adapters = None
        self.encoder_task_adapter_task_ids = None
        self.feature_style_randomizer = None
        self.domain_classifier = None
        self.landmark_query_refiners = nn.ModuleDict()
        self.shared_landmark_query_refiner = None
        self.shared_anatomical_query_decoder = None
        head_channels = self.encoder.out_channels
        if use_fpn:
            if fpn_mode not in FPN_MODES:
                raise ValueError(f"Unsupported fpn_mode: {fpn_mode}")
            if fpn_type not in FPN_TYPES:
                raise ValueError(f"Unsupported fpn_type: {fpn_type}")
            if fpn_mode == "shared":
                self.fpn = build_neck(fpn_type=fpn_type, in_channels=self.encoder.out_channels, out_channels=256)
                head_channels = self.fpn.out_channels
                print(f"FPN neck: ENABLED (shared, type={fpn_type})")
            else:
                self.task_fpns = nn.ModuleDict()
                head_channels = 256
                print(f"FPN neck: ENABLED (task-specific, type={fpn_type})")
        else:
            print("FPN neck: DISABLED")

        self.heads = nn.ModuleDict()
        print(f"Creating keypoint heads for {len(task_configs)} tasks...")
        if task_head_profile not in TASK_HEAD_PROFILE_PRESETS:
            raise ValueError(f"Unsupported task_head_profile: {task_head_profile}")
        if task_decoder_profile not in TASK_DECODER_PROFILE_PRESETS:
            raise ValueError(f"Unsupported task_decoder_profile: {task_decoder_profile}")
        if task_adapter_profile not in TASK_ADAPTER_PROFILE_PRESETS:
            raise ValueError(f"Unsupported task_adapter_profile: {task_adapter_profile}")
        print(f"Head type: {head_type}")
        print(f"Task head profile: {task_head_profile}")
        print(f"Task decoder profile: {task_decoder_profile}")
        print(f"Task adapter profile: {task_adapter_profile}")
        task_variant_map = TASK_HEAD_PROFILE_PRESETS[task_head_profile]
        task_decoder_map = TASK_DECODER_PROFILE_PRESETS[task_decoder_profile]
        task_adapter_map = TASK_ADAPTER_PROFILE_PRESETS[task_adapter_profile]
        self.task_to_adapter_group = {task_id: task_adapter_map.get(task_id) for task_id in task_adapter_map}
        active_groups = sorted({group for group in task_adapter_map.values() if group is not None})
        if task_adapter_profile == "softsharing_v1" and active_groups:
            self.soft_adapters = SoftSharingAdapter(head_channels, active_groups)
        elif task_adapter_profile == "localrefine_v1" and active_groups:
            self.local_refine_adapters = LocalRefineAdapter(head_channels, active_groups)
        elif task_adapter_profile == "coarse_refine_v1" and active_groups:
            self.context_adapters = ResidualContextAdapter(head_channels, active_groups)
        elif task_adapter_profile == "context_experts_v1" and active_groups:
            self.context_expert_adapters = ContextExpertsAdapter(head_channels, active_groups)
        elif task_adapter_profile in {
            "context_local_v1",
            "context_local_queryrefine_v1",
            "context_local_prototype_memory_v1",
            "context_local_shared_queryrefine_v2",
        } and active_groups:
            self.context_local_adapters = ContextLocalAdapter(head_channels, active_groups)
        elif task_adapter_profile == "context_local_pair_transfer_v1" and active_groups:
            self.context_local_adapters = ContextLocalPairTransferAdapter(head_channels, active_groups)
        elif task_adapter_profile == "context_local_stylemix_v1" and active_groups:
            self.context_local_adapters = ContextLocalAdapter(head_channels, active_groups)
            self.feature_style_randomizer = FeatureStyleRandomizer(p=0.5, alpha=0.2)
        elif task_adapter_profile == "texture_context_v1" and active_groups:
            self.context_local_adapters = UltrasoundTextureContextAdapter(head_channels, active_groups)
        elif task_adapter_profile == "texture_residual_v1" and active_groups:
            self.context_local_adapters = ContextLocalTextureResidualAdapter(head_channels, active_groups)
        elif task_adapter_profile == "texture_residual_v2" and active_groups:
            self.context_local_adapters = GatedContextLocalTextureResidualAdapter(head_channels, active_groups)
        elif task_adapter_profile == "highres_texture_v1" and active_groups:
            self.context_local_adapters = HighResolutionTextureResidualAdapter(head_channels, active_groups)
        elif task_adapter_profile == "pixel_unet_v1" and active_groups:
            self.context_local_adapters = PixelPyramidUNetFusionAdapter(head_channels, active_groups)
        elif task_adapter_profile == "hrnet_residual_v1" and active_groups:
            self.context_local_adapters = HRNetResidualAdapter(head_channels, active_groups)
        elif task_adapter_profile == "encoder_task_context_local_v1" and active_groups:
            self.encoder_task_adapters = PreFPNTaskBottleneckAdapter(self.encoder.out_channels, active_groups)
            self.encoder_task_adapter_task_ids = None
            self.context_local_adapters = ContextLocalAdapter(head_channels, active_groups)
        elif task_adapter_profile == "encoder_task_hard_context_local_v1" and active_groups:
            self.encoder_task_adapters = PreFPNTaskBottleneckAdapter(self.encoder.out_channels, active_groups)
            self.encoder_task_adapter_task_ids = {"FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}
            self.context_local_adapters = ContextLocalAdapter(head_channels, active_groups)
        elif task_adapter_profile == "boundary_context_v1" and active_groups:
            self.context_local_adapters = BoundaryContextAdapter(head_channels, active_groups)
        elif task_adapter_profile == "taskfilm_v1" and active_groups:
            self.task_film_adapters = TaskFiLMAdapter(head_channels, active_groups)
        if self.domain_adversarial:
            self.domain_classifier = DomainAdversarialHead(
                in_channels=head_channels,
                num_domain_classes=self.num_domain_classes,
            )
            print(f"Domain adversarial head: ENABLED ({self.num_domain_classes} pseudo-domain classes)")
        decoder_family_to_head = {
            "anatomical_query": None,
            "basic": HeatmapHead,
            "deep": DeepHeatmapHead,
            "offset_deep": SubpixelOffsetDeepHeatmapHead,
            "axis_offset_deep": AxisDistributionOffsetDeepHeatmapHead,
            "aop_vector_offset": AOPSegmentVectorOffsetHeatmapHead,
            "structure_offset_deep": StructureOffsetDeepHeatmapHead,
            "cardiac_graph": CardiacGraphRefineHeatmapHead,
            "a4c_pair_set": A4CPairSetHeatmapHead,
            "psax_pair_set": PSAXPairSetHeatmapHead,
            "structure_cardiac_graph": StructureCardiacGraphRefineHeatmapHead,
            "a4c_v2": A4CStructuredV2HeatmapHead,
            "a4c_v3": A4CResidualLocalRefineHeatmapHead,
            "a4c_v4": A4CStrongResidualHeatmapHead,
            "refine": ROIRefineHeatmapHead,
            "ivc_refine": IVCRefineHeatmapHead,
            "ivc_refine_v2": IVCRefineV2HeatmapHead,
            "ivc_refine_v3": IVCRefineV3HeatmapHead,
            "structure_ivc_refine_v3": StructureIVCRefineV3HeatmapHead,
            "line": LineHeatmapHead,
            "compact": CompactHeatmapHead,
            "dense": DenseRelationalHeatmapHead,
            "femur": FemurHeatmapHead,
            "structure_femur": StructureFemurHeatmapHead,
            "fugc": FUGCHeatmapHead,
            "fugc_dualscale": FUGCDualScaleHeatmapHead,
            "fugc_anatomy": FUGCAnatomyEndpointHeatmapHead,
            "fugc_anatomy_seg": FUGCAnatomySegmentationHeatmapHead,
            "fugc_anatomy_local": FUGCAnatomyLocalHeatmapHead,
            "fugc_cervical_canal": FUGCCervicalCanalHeatmapHead,
            "fugc_cervix_segmentation": FUGCCervixSegmentationHeatmapHead,
            "fugc_cervix_geometry": FUGCCervixGeometryHeatmapHead,
            "fugc_cervix_highres_segmentation": FUGCCervixHighResolutionSegmentationHead,
            "fugc_vitpose_direct": FUGCDirectViTPoseHead,
            "fugc_vector": FUGCVectorRefineHeatmapHead,
            "fugc_strip_axis": FUGCStripAxisHeatmapHead,
            "fugc_segment_specialist": FUGCSegmentSpecialistHeatmapHead,
            "structure_fugc": StructureFUGCHeatmapHead,
            "hc_refine": HCRefineHeatmapHead,
            "hc_refine_offset": HCRefineOffsetHeatmapHead,
            "structure_hc_refine_offset": StructureHCRefineOffsetHeatmapHead,
            "a4c": A4CHeatmapHead,
            "aop": AOPHeatmapHead,
            "fa": FAHeatmapHead,
            "fa_refine": FARefineHeatmapHead,
            "hc": HCHeatmapHead,
            "ivc": IVCHeatmapHead,
            "plax": PLAXHeatmapHead,
            "psax": PSAXHeatmapHead,
        }

        for config in task_configs:
            task_id = config["task_id"]
            task_name = config["task_name"]
            if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
                continue

            num_points = int(config["num_classes"])
            head_variant = task_variant_map.get(task_id, "medium")
            width_multiplier = HEAD_WIDTH_MULTIPLIERS[head_variant]
            decoder_family = task_decoder_map.get(task_id, head_type)
            if decoder_family not in decoder_family_to_head:
                raise ValueError(f"Unsupported decoder family '{decoder_family}' for task '{task_id}'")
            head_cls = decoder_family_to_head[decoder_family]
            print(
                f"  - {task_id}: {head_variant} width, decoder={decoder_family} "
                f"(width_multiplier={width_multiplier:.2f}, num_points={num_points})"
            )
            if self.task_fpns is not None:
                self.task_fpns[task_id] = build_neck(
                    fpn_type=fpn_type,
                    in_channels=self.encoder.out_channels,
                    out_channels=256,
                )
            if decoder_family == "anatomical_query":
                self.heads[task_id] = nn.Identity()
            else:
                decoder_in_channels = (
                    self.encoder.out_channels
                    if getattr(head_cls, "uses_encoder_features", False)
                    else head_channels
                )
                self.heads[task_id] = head_cls(
                    in_channels=decoder_in_channels,
                    num_points=num_points,
                    width_multiplier=width_multiplier,
                )
            if task_adapter_profile == "context_local_queryrefine_v1":
                self.landmark_query_refiners[task_id] = LandmarkQueryRefiner(
                    in_channels=head_channels,
                    num_points=num_points,
                )
            elif task_adapter_profile == "context_local_prototype_memory_v1":
                self.landmark_query_refiners[task_id] = AnatomicalPrototypeMemoryRefiner(
                    in_channels=head_channels,
                    num_points=num_points,
                )

        if task_adapter_profile == "context_local_shared_queryrefine_v2":
            task_num_points = {
                str(config["task_id"]): int(config["num_classes"])
                for config in task_configs
                if str(config["task_id"]) in self.heads
            }
            self.shared_landmark_query_refiner = SharedTaskConditionedLandmarkQueryRefiner(
                in_channels=head_channels,
                task_ids=list(self.heads.keys()),
                max_num_points=max(task_num_points.values()),
            )

        if task_decoder_profile == "anatomical_queries_v1":
            task_num_points = {
                str(config["task_id"]): int(config["num_classes"])
                for config in task_configs
                if str(config["task_id"]) in self.heads
            }
            self.shared_anatomical_query_decoder = SharedAnatomicalQueryDecoder(
                in_channels=head_channels,
                task_num_points=task_num_points,
            )
            print(
                "Shared anatomical query decoder: ENABLED "
                f"({len(task_num_points)} tasks, max_points={max(task_num_points.values())})"
            )

        if not self.heads:
            raise ValueError("No keypoint heads were created. Check task_configs with task_name == 'Regression'.")

    def _extract_encoder_features(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        self.encoder.set_lora_task(task_id)
        features = self.encoder(x)
        if self.encoder_task_adapters is not None:
            if self.encoder_task_adapter_task_ids is not None and task_id not in self.encoder_task_adapter_task_ids:
                adapter_group = None
            else:
                adapter_group = self.task_to_adapter_group.get(task_id)
            features = self.encoder_task_adapters(features, adapter_group)
        return features

    def _extract_task_features(
        self,
        x: torch.Tensor,
        task_id: str,
        apply_style_randomizer: bool = True,
        apply_adapters: bool = True,
    ) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")

        input_image = x
        features = self._extract_encoder_features(x, task_id)

        # Apply FPN neck if enabled
        if self.fpn is not None:
            features = self.fpn(features)
        elif self.task_fpns is not None:
            features = self.task_fpns[task_id](features)
        if apply_style_randomizer and self.feature_style_randomizer is not None:
            features = self.feature_style_randomizer(features)
        if not apply_adapters:
            return features
        if self.soft_adapters is not None:
            adapter_group = self.task_to_adapter_group.get(task_id)
            if adapter_group is not None:
                features = self.soft_adapters(features, adapter_group)
        if self.local_refine_adapters is not None:
            adapter_group = self.task_to_adapter_group.get(task_id)
            features = self.local_refine_adapters(features, input_image, adapter_group)
        if self.context_adapters is not None:
            adapter_group = self.task_to_adapter_group.get(task_id)
            features = self.context_adapters(features, adapter_group)
        if self.context_expert_adapters is not None:
            adapter_group = self.task_to_adapter_group.get(task_id)
            features = self.context_expert_adapters(features, adapter_group)
        if self.context_local_adapters is not None:
            adapter_group = self.task_to_adapter_group.get(task_id)
            features = self.context_local_adapters(features, input_image, adapter_group)
        if self.task_film_adapters is not None:
            adapter_group = self.task_to_adapter_group.get(task_id)
            features = self.task_film_adapters(features, adapter_group)
        return features

    def domain_logits(self, x: torch.Tensor, task_id: str, grl_lambda: float = 1.0) -> torch.Tensor:
        if self.domain_classifier is None:
            raise RuntimeError("domain_logits() called but domain_adversarial is disabled.")
        features = self._extract_task_features(
            x,
            task_id=task_id,
            apply_style_randomizer=False,
            apply_adapters=False,
        )
        return self.domain_classifier(features, grl_lambda=grl_lambda)

    def forward(self, x: torch.Tensor, task_id: str, return_prior: bool = False) -> torch.Tensor:
        head = self.heads[task_id]
        if getattr(head, "uses_encoder_features", False):
            features = self._extract_encoder_features(x, task_id)
        else:
            features = self._extract_task_features(
                x,
                task_id=task_id,
                apply_style_randomizer=True,
                apply_adapters=True,
            )
        if self.shared_anatomical_query_decoder is not None:
            head_output = self.shared_anatomical_query_decoder(
                features,
                task_id=task_id,
                out_size=self.heatmap_size,
            )
        else:
            if getattr(head, "requires_input_image", False):
                head_output = head(features, out_size=self.heatmap_size, input_image=x)
            else:
                head_output = head(features, out_size=self.heatmap_size)
        aux_outputs = None
        if isinstance(head_output, tuple):
            pred_logits, aux_outputs = head_output
        else:
            pred_logits = head_output

        if self.task_decoder_profile == "hidden_canonical_pairs_offset_v3" and task_id in CANONICAL_PAIR_RULES:
            if aux_outputs is None:
                aux_outputs = {}
            else:
                aux_outputs = dict(aux_outputs)
            pre_canonical_coords = aux_outputs.get("refined_coords_transformed")
            if pre_canonical_coords is None:
                pre_canonical_coords = SubpixelOffsetDeepHeatmapHead._softargmax_coords(pred_logits)
            aux_outputs["pre_canonical_coords_transformed"] = pre_canonical_coords
            aux_outputs["refined_coords_transformed"] = canonicalize_landmark_pairs(
                pre_canonical_coords,
                task_id,
            )

        query_refiner = None
        query_refiner_is_shared = False
        if self.shared_landmark_query_refiner is not None:
            query_refiner = self.shared_landmark_query_refiner
            query_refiner_is_shared = True
        elif task_id in self.landmark_query_refiners:
            query_refiner = self.landmark_query_refiners[task_id]

        if query_refiner is not None:
            if aux_outputs is None:
                aux_outputs = {}
            else:
                aux_outputs = dict(aux_outputs)
            anchor_coords = aux_outputs.get("refined_coords_transformed")
            if anchor_coords is None:
                anchor_coords = SubpixelOffsetDeepHeatmapHead._softargmax_coords(pred_logits)
            if query_refiner_is_shared:
                query_coords, query_outputs = query_refiner(features, anchor_coords, task_id)
            else:
                query_coords, query_outputs = query_refiner(features, anchor_coords)
            aux_outputs["pre_query_coords_transformed"] = anchor_coords
            aux_outputs["refined_coords_transformed"] = query_coords
            aux_outputs.update(query_outputs)

        if return_prior:
            pred_heatmaps = torch.sigmoid(pred_logits)
            if aux_outputs is not None:
                return pred_logits, pred_heatmaps, aux_outputs
            return pred_logits, pred_heatmaps
        return pred_logits
