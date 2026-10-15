"""Ablation variants of the proposed architecture.

Rather than four near-duplicate classes, there is one configurable
``SequenceRegressor`` plus named factories. That way an ablation can never
accidentally differ from the full model in any respect other than the factor
being ablated -- which is exactly the criticism the thesis version invited,
because the full model and the ablations were trained with different losses.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .blocks import FeatureExtractor1D

CellName = Literal["gru", "lstm"]
MMDFeature = Literal["flatten", "pooled"]


class SequenceRegressor(nn.Module):
    """Optional 1-D residual extractor (optionally with SE) + recurrent head."""

    def __init__(
        self,
        input_dim: int = 140,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_extractor: bool = True,
        use_se: bool = True,
        se_reduction: int = 16,
        channel_plan: tuple[int, ...] = (64, 128),
        cell: CellName = "gru",
        bidirectional: bool = True,
        mmd_feature: MMDFeature = "flatten",
    ):
        super().__init__()
        self.use_extractor = use_extractor
        self.mmd_feature = mmd_feature

        if use_extractor:
            self.extractor = FeatureExtractor1D(
                in_channels=input_dim,
                channel_plan=channel_plan,
                use_se=use_se,
                se_reduction=se_reduction,
            )
            rnn_input = self.extractor.out_channels
            self.conv_out = rnn_input
        else:
            self.extractor = None
            rnn_input = input_dim
            self.conv_out = input_dim

        cell_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.cell_name = cell
        self.rnn = cell_cls(
            rnn_input,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    # ------------------------------------------------------------- forward
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.extractor is not None:
            x = self.extractor(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encode(x)
        return self.fc(self.dropout(out[:, -1, :]))

    def mmd_representation(self, x: torch.Tensor) -> torch.Tensor:
        if self.mmd_feature == "pooled" or self.extractor is None:
            return self.encode(x).mean(dim=1)
        conv = self.extractor(x.transpose(1, 2))
        return conv.reshape(conv.size(0), -1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.mmd_representation(x)


# ----------------------------------------------------------------- factories
def build_resnet_bigru(**kwargs) -> SequenceRegressor:
    """Ablation 1: residual extractor without SE gating."""
    return SequenceRegressor(use_extractor=True, use_se=False, cell="gru", bidirectional=True, **kwargs)


def build_seresnet_gru(**kwargs) -> SequenceRegressor:
    """Ablation 2: SE-ResNet with a unidirectional GRU."""
    return SequenceRegressor(use_extractor=True, use_se=True, cell="gru", bidirectional=False, **kwargs)


def build_seresnet_lstm(**kwargs) -> SequenceRegressor:
    """Ablation 3: SE-ResNet with a unidirectional LSTM."""
    return SequenceRegressor(use_extractor=True, use_se=True, cell="lstm", bidirectional=False, **kwargs)


def build_pure_bigru(**kwargs) -> SequenceRegressor:
    """Ablation 4: BiGRU straight on the 140-D features, no extractor at all."""
    return SequenceRegressor(use_extractor=False, use_se=False, cell="gru", bidirectional=True, **kwargs)


def build_resnet_lstm(**kwargs) -> SequenceRegressor:
    """Extra control: residual extractor without SE and a bidirectional LSTM."""
    return SequenceRegressor(use_extractor=True, use_se=False, cell="lstm", bidirectional=True, **kwargs)
