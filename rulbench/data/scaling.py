"""Leakage-free feature standardisation.

The single most important correctness fix relative to the original pipeline:
the scaler is fitted on **training bearings only** and merely applied to the
validation and test bearings. The thesis version pooled all 15 bearings,
including the test bearings, which lets test statistics into the model and
invalidates the reported metrics.

For cross-condition runs the scaler is fitted on the source-domain training
split and reused unchanged on the target domain -- that is the only setting in
which the MK-MMD alignment measures a genuine domain shift.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

import numpy as np


class ChannelScaler:
    """Column-wise z-score scaler with an explicit ``fit``/``transform`` split."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean_: Optional[np.ndarray] = np.asarray(mean) if mean is not None else None
        self.std_: Optional[np.ndarray] = np.asarray(std) if std is not None else None

    # ------------------------------------------------------------------ api
    @property
    def n_features(self) -> int:
        return 0 if self.mean_ is None else int(self.mean_.shape[0])

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.std_ is not None

    def fit(self, X: np.ndarray) -> "ChannelScaler":
        """Estimate mean / std. ``X`` is ``(N, D)`` or ``(N, T, D)``."""
        flat = self._flatten(X)
        self.mean_ = flat.mean(axis=0)
        std = flat.std(axis=0)
        # guard against constant features which would blow up the division
        std[std < 1e-8] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ChannelScaler must be fitted before transform()")
        arr = np.asarray(X, dtype=np.float64)
        out = (arr - self.mean_) / self.std_
        return out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # -------------------------------------------------------------- plumbing
    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            return arr[:, None]
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, arr.shape[-1])

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ChannelScaler":
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "mean": self.mean_.tolist() if self.is_fitted else None,
            "std": self.std_.tolist() if self.is_fitted else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "fitted" if self.is_fitted else "unfitted"
        return f"ChannelScaler({state}, n_features={self.n_features})"


def save_json(obj: Dict[str, Any], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
