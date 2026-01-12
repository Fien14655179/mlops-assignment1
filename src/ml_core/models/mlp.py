from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_units: Sequence[int],
        num_classes: int,
    ) -> None:
        super().__init__()

        # input_shape is like [3, 96, 96]
        if len(input_shape) != 3:
            raise ValueError(f"Expected input_shape of length 3 (C,H,W), got {input_shape}")

        c, h, w = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
        input_dim = c * h * w  # 3*96*96 = 27648

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hu in hidden_units:
            hu = int(hu)
            layers.append(nn.Linear(prev_dim, hu))
            layers.append(nn.ReLU())
            prev_dim = hu

        layers.append(nn.Linear(prev_dim, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x comes in as (N, C, H, W); flatten to (N, C*H*W)
        x = torch.flatten(x, start_dim=1)
        return self.net(x)
