"""Shared 1-D building blocks used by the proposed model and the ablations."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation over the channel dimension of a 1-D feature map.

    ``(B, C, L) -> (B, C, L)`` with the channel responses rescaled by a
    learned bottleneck gate. The excitation weights are view-expanded rather
    than materialised, so the gate costs negligible memory.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        gate = self.avg_pool(x).view(batch, channels)
        gate = self.fc(gate).view(batch, channels, 1)
        return x * gate.expand_as(x)


class ResBlock1D(nn.Module):
    """Pre-activation residual block: conv-BN-ReLU-conv-BN(-SE)-add-ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_se = use_se
        self.se = SEBlock1D(out_channels, se_reduction) if use_se else None

        needs_projection = stride != 1 or in_channels != out_channels
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if needs_projection
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se and self.se is not None:
            out = self.se(out)
        return self.relu(out + identity)


class FeatureExtractor1D(nn.Module):
    """Stack of residual blocks mapping ``(B, D, L) -> (B, C_out, L)``."""

    def __init__(
        self,
        in_channels: int,
        channel_plan: tuple[int, ...] = (64, 128),
        use_se: bool = True,
        se_reduction: int = 16,
        kernel_size: int = 3,
        num_blocks: int | None = None,
    ):
        super().__init__()
        plan = list(channel_plan)
        if num_blocks is not None:
            plan = plan + [plan[-1]] * max(0, num_blocks - len(plan))
        blocks = []
        current = in_channels
        for out_channels in plan:
            blocks.append(
                ResBlock1D(
                    current,
                    out_channels,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    kernel_size=kernel_size,
                )
            )
            current = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def masked_dropout_only_train(module: nn.Module) -> None:
    """Put every Dropout into train mode and every other module into eval mode.

    This is the correct way to run MC Dropout: BatchNorm must stay in eval mode
    or it will use (and mutate) running statistics from the test batch, which
    silently corrupts both the mean prediction and the uncertainty estimate.
    """
    for child in module.modules():
        if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            child.train()
