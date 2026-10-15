"""Comparison baselines: CNN / LSTM / GRU / Informer-SEnet.

All baselines consume the same ``(B, T, D)`` tensor as the proposed model so
that the only difference between runs is architecture plus loss configuration.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------------------ CNN
class CNN1D(nn.Module):
    """Two conv layers -> global average pooling -> linear."""

    def __init__(
        self,
        input_dim: int = 140,
        hidden_dims: tuple[int, ...] = (64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        current = input_dim
        for out_channels in hidden_dims:
            layers += [
                nn.Conv1d(current, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ]
            current = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(current, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2))
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


# ------------------------------------------------------------------ recurrent
class _RecurrentRegressor(nn.Module):
    """Shared implementation for the LSTM / GRU / BiGRU baselines."""

    def __init__(
        self,
        cell: type[nn.RNNBase],
        input_dim: int = 140,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = cell(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


class LSTMRegressor(_RecurrentRegressor):
    def __init__(self, input_dim: int = 140, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__(nn.LSTM, input_dim, hidden_dim, num_layers, dropout, bidirectional)


class GRURegressor(_RecurrentRegressor):
    def __init__(self, input_dim: int = 140, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__(nn.GRU, input_dim, hidden_dim, num_layers, dropout, bidirectional)


# ----------------------------------------------------------- Informer encoder
class ProbSparseSelfAttention(nn.Module):
    """Informer-style probabilistic sparse attention.

    Faithful to the original criterion (select the top-``u`` queries by the
    max-minus-mean sparsity measurement against ``u`` sampled keys) but
    *deterministic*: the key sample is drawn from a seeded generator instead of
    the global RNG, so a run is reproducible from its seed alone.
    """

    def __init__(self, d_model: int, n_heads: int = 8, factor: int = 5, dropout: float = 0.0,
                 deterministic: bool = True, seed: int = 0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.deterministic = deterministic
        self.seed = seed
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        u = min(length, self.factor * max(1, int(math.ceil(math.log(max(length, 2))))))
        if u >= length:                      # 32-step windows: sparsity is a no-op
            attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
            out = attn @ v
        else:
            if self.deterministic:
                generator = torch.Generator(device=x.device).manual_seed(self.seed)
                idx = torch.randperm(length, generator=generator, device=x.device)[:u]
            else:
                idx = torch.randperm(length, device=x.device)[:u]
            k_sample = k[:, :, idx, :]
            scores = q @ k_sample.transpose(-2, -1) / math.sqrt(self.head_dim)   # (B,H,L,u)
            sparsity = scores.max(dim=-1).values - scores.mean(dim=-1)           # (B,H,L)
            top_q = sparsity.topk(u, dim=-1).indices                             # (B,H,u)

            gathered = scores.gather(2, top_q.unsqueeze(-1).expand(-1, -1, -1, u))
            weights = torch.softmax(gathered, dim=-1)                            # (B,H,u,u)
            out_sparse = weights @ v[:, :, idx, :]                               # (B,H,u,hd)

            # inactive queries keep the mean of the sampled values, as in Informer
            context = (
                v[:, :, idx, :].mean(dim=2, keepdim=True)
                .expand(-1, -1, length, -1)
                .contiguous()
            )
            context.scatter_(
                2,
                top_q.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
                out_sparse,
            )
            out = context
        out = out.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.dropout(self.out_proj(out))


class InformerEncoderLayer(nn.Module):
    """Pre-norm attention block + position-wise feed-forward network."""

    def __init__(self, d_model: int, n_heads: int = 8, factor: int = 5, dropout: float = 0.2,
                 seed: int = 0):
        super().__init__()
        self.attention = ProbSparseSelfAttention(d_model, n_heads, factor, dropout=dropout, seed=seed)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class SEBlock1D(nn.Module):
    """Channel gate used by the Informer-SEnet baseline."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        gate = self.fc(self.pool(x).view(batch, channels)).view(batch, channels, 1)
        return x * gate


class InformerSEnet(nn.Module):
    """Transformer encoder with SE channel gating, as used in the comparison."""

    def __init__(self, input_dim: int = 140, d_model: int = 128, n_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.2, factor: int = 5, seed: int = 0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.layers = nn.ModuleList(
            InformerEncoderLayer(d_model, n_heads, factor, dropout, seed=seed)
            for _ in range(num_layers)
        )
        self.se = SEBlock1D(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        x = self.input_proj(x) + self.positional[:, :length, :]
        for layer in self.layers:
            x = layer(x)
        x = self.se(x.transpose(1, 2)).mean(dim=-1)
        return self.fc(self.dropout(x))
