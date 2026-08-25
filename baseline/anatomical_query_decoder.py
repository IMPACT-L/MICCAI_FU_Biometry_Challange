"""Shared task-conditioned landmark query decoder.

This module is intentionally isolated from the legacy decoder collection. It
uses one attention stack for all tasks so low-sample ultrasound views can learn
spatial retrieval and landmark-geometry reasoning from the full training set.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class AnatomicalQueryBlock(nn.Module):
    """Pre-normalized landmark self-attention and image cross-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(queries)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        attended, _ = self.cross_attention(
            self.cross_query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class SharedAnatomicalQueryDecoder(nn.Module):
    """Decode every task with shared landmark queries and dense image evidence.

    The task and point embeddings define landmark identity. Shared self- and
    cross-attention blocks learn anatomical configuration and image retrieval
    across all datasets. Query-conditioned dense similarities produce heatmaps;
    a bounded local residual then removes heatmap discretization error.
    """

    def __init__(
        self,
        in_channels: int,
        task_num_points: Mapping[str, int],
        query_dim: int = 192,
        dense_dim: int = 96,
        num_layers: int = 3,
        num_heads: int = 6,
        dropout: float = 0.10,
        max_offset: float = 0.025,
    ):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        if not task_num_points:
            raise ValueError("task_num_points cannot be empty")

        self.task_ids: Sequence[str] = tuple(sorted(map(str, task_num_points)))
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.task_num_points = {str(key): int(value) for key, value in task_num_points.items()}
        self.max_num_points = max(self.task_num_points.values())
        self.query_dim = int(query_dim)
        self.dense_dim = int(dense_dim)
        self.max_offset = float(max_offset)

        self.memory_stem = nn.Sequential(
            nn.Conv2d(in_channels, query_dim, kernel_size=1, bias=False),
            _group_norm(query_dim),
            nn.GELU(),
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1, bias=False),
            _group_norm(query_dim),
            nn.GELU(),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(2, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.task_embedding = nn.Embedding(len(self.task_ids), query_dim)
        self.landmark_embedding = nn.Parameter(
            torch.empty(len(self.task_ids), self.max_num_points, query_dim)
        )
        self.global_projection = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.blocks = nn.ModuleList(
            AnatomicalQueryBlock(query_dim, num_heads, dropout) for _ in range(num_layers)
        )
        self.query_output_norm = nn.LayerNorm(query_dim)

        # Decode at four times the patch-grid resolution before matching the
        # configured heatmap size. This preserves boundaries without BatchNorm.
        mid_channels = max(query_dim // 2, dense_dim)
        self.dense_decoder = nn.Sequential(
            nn.Conv2d(query_dim, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(mid_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(mid_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, dense_dim, kernel_size=3, padding=1, bias=False),
            _group_norm(dense_dim),
            nn.GELU(),
        )
        self.heatmap_key = nn.Conv2d(dense_dim, dense_dim, kernel_size=1, bias=False)
        self.heatmap_query = nn.Linear(query_dim, dense_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        self.local_projection = nn.Sequential(
            nn.Linear(dense_dim + query_dim + 2, query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
        )
        self.local_geometry = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 2),
            nn.Tanh(),
        )

        nn.init.trunc_normal_(self.task_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.landmark_embedding, std=0.02)
        nn.init.zeros_(self.offset_head[-2].weight)
        nn.init.zeros_(self.offset_head[-2].bias)

    @staticmethod
    def _coordinate_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        xs = torch.linspace(0.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).view(1, height * width, 2)

    @staticmethod
    def _softargmax(logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, height, width = logits.shape
        probabilities = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
        ys = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        xs = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        expected_x = (probabilities * xs.view(1, 1, 1, width)).sum(dim=(-2, -1))
        expected_y = (probabilities * ys.view(1, 1, height, 1)).sum(dim=(-2, -1))
        return torch.stack((expected_x, expected_y), dim=-1).view(batch_size, num_points, 2)

    @staticmethod
    def _sample_dense_features(features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        grid = points.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def forward(self, features: torch.Tensor, task_id: str, out_size) -> tuple[torch.Tensor, dict]:
        if task_id not in self.task_to_index:
            raise ValueError(f"Task ID '{task_id}' is not registered in the anatomical query decoder")
        task_index = self.task_to_index[task_id]
        num_points = self.task_num_points[task_id]
        batch_size = features.shape[0]

        memory_map = self.memory_stem(features)
        memory = memory_map.flatten(2).transpose(1, 2).contiguous()
        memory_grid = self._coordinate_grid(memory_map.shape[-2], memory_map.shape[-1], memory_map)
        memory = memory + self.position_mlp(memory_grid)

        task_tokens = self.task_embedding.weight[task_index].view(1, 1, -1)
        point_tokens = self.landmark_embedding[task_index, :num_points].unsqueeze(0)
        global_token = self.global_projection(memory_map.mean(dim=(-2, -1))).unsqueeze(1)
        queries = point_tokens.expand(batch_size, -1, -1) + task_tokens + global_token
        for block in self.blocks:
            queries = block(queries, memory)
        queries = self.query_output_norm(queries)

        dense_features = self.dense_decoder(memory_map)
        dense_keys = F.normalize(self.heatmap_key(dense_features), dim=1, eps=1e-6)
        heatmap_queries = F.normalize(self.heatmap_query(queries), dim=-1, eps=1e-6)
        logit_scale = self.logit_scale.exp().clamp(max=30.0)
        logits = logit_scale * torch.einsum("bnd,bdhw->bnhw", heatmap_queries, dense_keys)
        if logits.shape[-2:] != tuple(out_size):
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

        coarse_points = self._softargmax(logits)
        sampled_features = self._sample_dense_features(dense_features, coarse_points)
        local_tokens = self.local_projection(
            torch.cat((queries, sampled_features, coarse_points), dim=-1)
        )
        normalized = F.layer_norm(local_tokens, (self.query_dim,))
        geometric_context, _ = self.local_geometry(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        local_tokens = local_tokens + geometric_context
        offsets = self.offset_head(local_tokens) * self.max_offset
        refined_points = torch.clamp(coarse_points + offsets, 0.0, 1.0)
        return logits, {
            "coarse_coords_transformed": coarse_points.reshape(batch_size, -1),
            "refined_coords_transformed": refined_points.reshape(batch_size, -1),
            "anatomical_query_offsets": offsets,
        }
