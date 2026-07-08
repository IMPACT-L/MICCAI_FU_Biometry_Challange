from typing import Dict, List

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
FPN_MODES = {"shared", "task_specific"}
FPN_TYPES = {"fpn", "bifpn"}
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
    "coarse_refine_v1": {
        "A4C": "refine",
        "FUGC": "refine",
        "IVC": "refine",
        "fetal_femur": "refine",
    },
    "fugc_refine_v1": {
        "FUGC": "fugc",
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
    """Returns the last patch feature map as a 2D tensor [B, C, H, W]."""

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", pretrained: bool = True):
        super().__init__()
        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if not hasattr(self.backbone, "patch_embed"):
            raise ValueError(f"Model '{model_name}' is not a ViT-style backbone with patch_embed.")
        self.out_channels = int(self.backbone.num_features)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        feat_map = patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side)
        return feat_map


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
        head_type: str = "basic",
        task_head_profile: str = "uniform",
        task_decoder_profile: str = "uniform",
        task_adapter_profile: str = "uniform",
    ):
        super().__init__()

        self.heatmap_size = heatmap_size
        self.use_fpn = use_fpn
        self.fpn_mode = fpn_mode
        self.fpn_type = fpn_type
        self.head_type = head_type
        self.task_head_profile = task_head_profile
        self.task_decoder_profile = task_decoder_profile
        self.task_adapter_profile = task_adapter_profile

        print(f"Initializing encoder: {encoder_name}")
        self.encoder = DINOv2Backbone(model_name=encoder_name, pretrained=(encoder_weights is not None))

        # FPN neck (optional)
        self.fpn = None
        self.task_fpns = None
        self.soft_adapters = None
        self.local_refine_adapters = None
        self.context_adapters = None
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
        decoder_family_to_head = {
            "basic": HeatmapHead,
            "deep": DeepHeatmapHead,
            "refine": ROIRefineHeatmapHead,
            "line": LineHeatmapHead,
            "compact": CompactHeatmapHead,
            "dense": DenseRelationalHeatmapHead,
            "femur": FemurHeatmapHead,
            "fugc": FUGCHeatmapHead,
            "a4c": A4CHeatmapHead,
            "aop": AOPHeatmapHead,
            "fa": FAHeatmapHead,
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
            self.heads[task_id] = head_cls(
                in_channels=head_channels,
                num_points=num_points,
                width_multiplier=width_multiplier,
            )

        if not self.heads:
            raise ValueError("No keypoint heads were created. Check task_configs with task_name == 'Regression'.")

    def forward(self, x: torch.Tensor, task_id: str, return_prior: bool = False) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")

        input_image = x
        features = self.encoder(x)

        # Apply FPN neck if enabled
        if self.fpn is not None:
            features = self.fpn(features)
        elif self.task_fpns is not None:
            features = self.task_fpns[task_id](features)
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

        head_output = self.heads[task_id](features, out_size=self.heatmap_size)
        aux_outputs = None
        if isinstance(head_output, tuple):
            pred_logits, aux_outputs = head_output
        else:
            pred_logits = head_output

        if return_prior:
            pred_heatmaps = torch.sigmoid(pred_logits)
            if aux_outputs is not None:
                return pred_logits, pred_heatmaps, aux_outputs
            return pred_logits, pred_heatmaps
        return pred_logits
