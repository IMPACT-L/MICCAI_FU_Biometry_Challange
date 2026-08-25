"""Low-rank adaptation utilities for timm vision-transformer backbones."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen linear projection with a trainable zero-initialized low-rank path."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        low_rank = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base_output + low_rank * self.scaling


class TaskSpecificLoRALinear(nn.Module):
    """Frozen projection with an independently routed LoRA path per task."""

    def __init__(
        self,
        base: nn.Linear,
        task_ids: Sequence[str],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not task_ids:
            raise ValueError("Task-specific LoRA requires at least one task")
        self.base = base
        self.task_ids = tuple(map(str, task_ids))
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.active_task_index = 0
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(len(self.task_ids), self.rank, base.in_features)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(len(self.task_ids), base.out_features, self.rank)
        )
        for task_index in range(len(self.task_ids)):
            nn.init.kaiming_uniform_(self.lora_a[task_index], a=math.sqrt(5.0))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def set_active_task(self, task_id: str) -> None:
        if task_id not in self.task_to_index:
            raise ValueError(f"Task ID '{task_id}' is not registered in task-specific LoRA")
        self.active_task_index = self.task_to_index[task_id]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        task_index = self.active_task_index
        low_rank = F.linear(
            F.linear(self.dropout(inputs), self.lora_a[task_index]),
            self.lora_b[task_index],
        )
        return base_output + low_rank * self.scaling


def inject_lora_into_vit_attention(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    last_blocks: int,
    dropout: float,
    task_ids: Sequence[str] | None = None,
) -> list[str]:
    """Inject LoRA into qkv and output projections of the final ViT blocks."""

    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("The selected backbone does not expose transformer blocks")
    block_count = min(max(int(last_blocks), 1), len(blocks))
    start_index = len(blocks) - block_count
    injected = []
    for block_index in range(start_index, len(blocks)):
        attention = getattr(blocks[block_index], "attn", None)
        if attention is None:
            raise ValueError(f"Transformer block {block_index} has no attention module")
        for projection_name in ("qkv", "proj"):
            projection = getattr(attention, projection_name, None)
            if not isinstance(projection, nn.Linear):
                raise ValueError(
                    f"Expected block {block_index} attention {projection_name} to be nn.Linear"
                )
            if task_ids:
                wrapped = TaskSpecificLoRALinear(
                    projection,
                    task_ids=task_ids,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
            else:
                wrapped = LoRALinear(
                    projection,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
            setattr(attention, projection_name, wrapped)
            injected.append(f"blocks.{block_index}.attn.{projection_name}")
    return injected


def set_active_lora_task(backbone: nn.Module, task_id: str) -> None:
    """Route all task-specific LoRA projections to the requested task."""

    for module in backbone.modules():
        if isinstance(module, TaskSpecificLoRALinear):
            module.set_active_task(task_id)


def remap_checkpoint_for_lora(
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ordinary Linear checkpoint keys into LoRALinear base projections."""

    remapped = dict(checkpoint_state)
    for key, value in checkpoint_state.items():
        if ".attn.qkv." not in key and ".attn.proj." not in key:
            continue
        if key.endswith(".weight"):
            candidate = key[: -len(".weight")] + ".base.weight"
        elif key.endswith(".bias"):
            candidate = key[: -len(".bias")] + ".base.bias"
        else:
            continue
        if candidate in model_state and model_state[candidate].shape == value.shape:
            remapped[candidate] = value
    return remapped
