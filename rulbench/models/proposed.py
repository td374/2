"""The proposed model: 1-D SE-ResNet feature extractor followed by a BiGRU head.

Two behaviours are deliberately parameterised because they are exactly the
points a reviewer will probe:

``mmd_feature``
    ``"flatten"`` reproduces the thesis (128 ch x 32 steps = 4096-D). ``"pooled"``
    uses a 128-D time-pooled embedding, which is the usual choice in the domain
    adaptation literature and is far cheaper inside the RBF kernel.

``mc_dropout``
    The wrapper keeps BatchNorm in eval mode and only switches Dropout to train
    mode, so Monte-Carlo sampling actually measures model uncertainty instead of
    re-estimating batch statistics from the test batch.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn

from .blocks import FeatureExtractor1D, masked_dropout_only_train

MMDFeature = Literal["flatten", "pooled"]


class SEResNetBiGRU(nn.Module):
    """Proposed architecture.

    Parameters
    ----------
    input_dim
        Feature width per time step (140 in the thesis).
    hidden_dim, num_layers
        BiGRU configuration.
    channel_plan
        Output channels of the residual blocks.
    mmd_feature
        Representation handed to the MK-MMD term (see module docstring).
    """

    def __init__(
        self,
        input_dim: int = 140,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        se_reduction: int = 16,
        channel_plan: tuple[int, ...] = (64, 128),
        use_se: bool = True,
        mmd_feature: MMDFeature = "flatten",
    ):
        super().__init__()
        self.extractor = FeatureExtractor1D(
            in_channels=input_dim,
            channel_plan=channel_plan,
            use_se=use_se,
            se_reduction=se_reduction,
        )
        conv_out = self.extractor.out_channels
        self.bigru = nn.GRU(
            conv_out,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_out = conv_out
        self.mmd_feature = mmd_feature

    # ------------------------------------------------------------- forward
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T, 2 * hidden_dim)``."""
        x = x.transpose(1, 2)                      # (B, D, T) for Conv1d
        x = self.extractor(x)                      # (B, C, T)
        x = x.transpose(1, 2)                      # (B, T, C)
        out, _ = self.bigru(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encode(x)
        last = out[:, -1, :]
        return self.fc(self.head_dropout(last))    # (B, 1)

    # ---------------------------------------------------- MMD representation
    def mmd_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Return the vector fed into MK-MMD.

        ``flatten`` mirrors the thesis (conv output flattened, 4096-D).
        ``pooled`` averages the BiGRU sequence, giving a 128-D embedding.
        """
        if self.mmd_feature == "pooled":
            return self.encode(x).mean(dim=1)
        conv = self.extractor(x.transpose(1, 2))
        return conv.reshape(conv.size(0), -1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backwards-compatible alias used by the legacy training scripts."""
        return self.mmd_representation(x)

    # --------------------------------------------------------------- helpers
    @property
    def mmd_dim(self) -> int:
        if self.mmd_feature == "pooled":
            return self.hidden_dim * 2
        return self.conv_out * 32  # assumes the thesis window length of 32

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MCDropoutModel(nn.Module):
    """Monte-Carlo Dropout wrapper with correct BatchNorm handling."""

    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.2):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.set_dropout_rate(dropout_rate)

    def set_dropout_rate(self, rate: float) -> None:
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def _eval_all(self) -> None:
        self.base_model.eval()
        # keep the head dropout deterministic inside the deterministic pass
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def _stochastic(self) -> None:
        self.base_model.eval()          # BatchNorm et al. stay in eval mode
        masked_dropout_only_train(self.base_model)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        if mc_dropout:
            self._stochastic()
        else:
            self._eval_all()
        return self.base_model(x)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_iter: int = 50,
        z: float = 1.96,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mean, std, lower, upper)`` for the 1-D prediction array."""
        self._stochastic()
        samples = [self.base_model(x).detach().cpu().numpy() for _ in range(n_iter)]
        stacked = np.stack(samples, axis=0)            # (n_iter, B, 1)
        mean = stacked.mean(axis=0).reshape(-1)
        std = stacked.std(axis=0).reshape(-1)
        self._eval_all()
        return mean, std, mean - z * std, mean + z * std

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        self._eval_all()
        return self.base_model(x).detach().cpu().numpy().reshape(-1)

    def mmd_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model.mmd_representation(x)
