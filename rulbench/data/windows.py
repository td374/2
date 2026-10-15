"""Sliding-window sample generation.

A fixed-length window with stride 1 turns a bearing of any lifetime into the
same model input shape ``(window_len, n_features)``, which is what aligns
bearings with very different lifetimes (52 min vs 2538 min) onto a common
representation. The window label is the RUL ratio at the **last** time step,
matching the supervised target used in the thesis.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def make_windows(
    features: np.ndarray,
    ratio: np.ndarray,
    window_len: int = 32,
    stride: int = 1,
) -> Dict[str, np.ndarray]:
    """Cut one bearing's feature sequence into overlapping windows.

    Parameters
    ----------
    features
        ``(T, D)`` scaled feature sequence.
    ratio
        ``(T,)`` normalised RUL ratio aligned with ``features``.
    window_len, stride
        Sliding-window geometry. The thesis uses ``32`` and ``1``.

    Returns
    -------
    dict with ``X`` ``(N, window_len, D)``, ``ratio`` ``(N,)`` and
    ``window_end`` ``(N,)`` (index of the label's time step, useful to map
    predictions back onto the timeline for plots).
    """
    features = np.asarray(features)
    ratio = np.asarray(ratio).ravel()
    if features.ndim != 2:
        raise ValueError(f"features must be (T, D), got {features.shape}")
    if features.shape[0] != ratio.shape[0]:
        raise ValueError("features and ratio must share the time dimension")
    if window_len < 1 or stride < 1:
        raise ValueError("window_len and stride must be >= 1")

    n_time = features.shape[0]
    n_win = n_time - window_len + 1
    if n_win <= 0:
        raise ValueError(
            f"bearing too short: {n_time} snapshots for a window of {window_len}"
        )

    idx = np.arange(n_win)[:, None] + np.arange(window_len)[None, :]  # (N, L)
    X = features[idx].astype(np.float32)
    ends = idx[:, -1]
    return {
        "X": X,
        "ratio": ratio[ends].astype(np.float32),
        "window_end": ends.astype(np.int32),
    }


def concatenate_windows(parts: list[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack the windows of several bearings into one split."""
    if not parts:
        raise ValueError("no window parts to concatenate")
    keys = ("X", "ratio")
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    out["window_end"] = np.concatenate([p["window_end"] for p in parts], axis=0)
    # lifetime is per-window: every window of a bearing shares that bearing's life
    if "lifetime" in parts[0]:
        out["lifetime"] = np.concatenate([p["lifetime"] for p in parts], axis=0)
    return out


def attach_lifetime(split: Dict[str, np.ndarray], lifetime: float) -> Dict[str, np.ndarray]:
    """Broadcast a scalar lifetime over every window in a split."""
    n = split["ratio"].shape[0]
    split["lifetime"] = np.full(n, float(lifetime), dtype=np.float32)
    return split


def combined_window_counts(window_len: int, stride: int, lifetimes: Tuple[int, ...]) -> int:
    """Total number of windows produced by a set of bearing lifetimes."""
    return sum(max(0, int(t) - window_len + 1 + (stride - 1)) // stride for t in lifetimes)
