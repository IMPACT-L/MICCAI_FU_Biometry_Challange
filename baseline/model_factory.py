from typing import Dict, List

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}


class HeatmapHead(nn.Module):
    """Light decoder that maps DINOv2 feature maps to keypoint heatmaps."""

    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        hidden = max(in_channels // 2, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DeepHeatmapHead(nn.Module):
    """Stronger decoder head for finer keypoint localization."""

    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        hidden1 = max(in_channels // 2, 192)
        hidden2 = max(hidden1 // 2, 128)
        hidden3 = max(hidden2 // 2, 96)
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
        head_type: str = "basic",
    ):
        super().__init__()

        self.heatmap_size = heatmap_size
        self.use_fpn = use_fpn
        self.head_type = head_type

        print(f"Initializing encoder: {encoder_name}")
        self.encoder = DINOv2Backbone(model_name=encoder_name, pretrained=(encoder_weights is not None))

        # FPN neck (optional)
        self.fpn = None
        head_channels = self.encoder.out_channels
        if use_fpn:
            self.fpn = FPN(in_channels=self.encoder.out_channels, out_channels=256)
            head_channels = self.fpn.out_channels
            print("FPN neck: ENABLED")
        else:
            print("FPN neck: DISABLED")

        self.heads = nn.ModuleDict()
        print(f"Creating keypoint heads for {len(task_configs)} tasks...")
        if head_type == "basic":
            head_cls = HeatmapHead
        elif head_type == "deep":
            head_cls = DeepHeatmapHead
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        print(f"Head type: {head_type}")

        for config in task_configs:
            task_id = config["task_id"]
            task_name = config["task_name"]
            if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
                continue

            num_points = int(config["num_classes"])
            self.heads[task_id] = head_cls(in_channels=head_channels, num_points=num_points)

        if not self.heads:
            raise ValueError("No keypoint heads were created. Check task_configs with task_name == 'Regression'.")

    def forward(self, x: torch.Tensor, task_id: str, return_prior: bool = False) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")

        features = self.encoder(x)

        # Apply FPN neck if enabled
        if self.fpn is not None:
            features = self.fpn(features)

        pred_logits = self.heads[task_id](features, out_size=self.heatmap_size)

        if return_prior:
            pred_heatmaps = torch.sigmoid(pred_logits)
            return pred_logits, pred_heatmaps
        return pred_logits
