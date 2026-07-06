from typing import Dict, List

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
FPN_MODES = {"shared", "task_specific"}
TASK_HEAD_PROFILE_PRESETS = {
    "uniform": {},
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

    def forward(self, x: torch.Tensor, out_size):
        x = self.stem(x)
        x = self.refine(x)
        point_logits = self.point_head(x)
        segment_logits = self.segment_head(x)
        if point_logits.shape[-2:] != out_size:
            point_logits = F.interpolate(point_logits, size=out_size, mode="bilinear", align_corners=False)
        if segment_logits.shape[-2:] != out_size:
            segment_logits = F.interpolate(segment_logits, size=out_size, mode="bilinear", align_corners=False)
        return point_logits, {"segment_logits": segment_logits}


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
        head_type: str = "basic",
        task_head_profile: str = "uniform",
        task_decoder_profile: str = "uniform",
    ):
        super().__init__()

        self.heatmap_size = heatmap_size
        self.use_fpn = use_fpn
        self.fpn_mode = fpn_mode
        self.head_type = head_type
        self.task_head_profile = task_head_profile
        self.task_decoder_profile = task_decoder_profile

        print(f"Initializing encoder: {encoder_name}")
        self.encoder = DINOv2Backbone(model_name=encoder_name, pretrained=(encoder_weights is not None))

        # FPN neck (optional)
        self.fpn = None
        self.task_fpns = None
        head_channels = self.encoder.out_channels
        if use_fpn:
            if fpn_mode not in FPN_MODES:
                raise ValueError(f"Unsupported fpn_mode: {fpn_mode}")
            if fpn_mode == "shared":
                self.fpn = FPN(in_channels=self.encoder.out_channels, out_channels=256)
                head_channels = self.fpn.out_channels
                print("FPN neck: ENABLED (shared)")
            else:
                self.task_fpns = nn.ModuleDict()
                head_channels = 256
                print("FPN neck: ENABLED (task-specific)")
        else:
            print("FPN neck: DISABLED")

        self.heads = nn.ModuleDict()
        print(f"Creating keypoint heads for {len(task_configs)} tasks...")
        if task_head_profile not in TASK_HEAD_PROFILE_PRESETS:
            raise ValueError(f"Unsupported task_head_profile: {task_head_profile}")
        if task_decoder_profile not in TASK_DECODER_PROFILE_PRESETS:
            raise ValueError(f"Unsupported task_decoder_profile: {task_decoder_profile}")
        print(f"Head type: {head_type}")
        print(f"Task head profile: {task_head_profile}")
        print(f"Task decoder profile: {task_decoder_profile}")
        task_variant_map = TASK_HEAD_PROFILE_PRESETS[task_head_profile]
        task_decoder_map = TASK_DECODER_PROFILE_PRESETS[task_decoder_profile]
        decoder_family_to_head = {
            "basic": HeatmapHead,
            "deep": DeepHeatmapHead,
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
                self.task_fpns[task_id] = FPN(in_channels=self.encoder.out_channels, out_channels=256)
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

        features = self.encoder(x)

        # Apply FPN neck if enabled
        if self.fpn is not None:
            features = self.fpn(features)
        elif self.task_fpns is not None:
            features = self.task_fpns[task_id](features)

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
