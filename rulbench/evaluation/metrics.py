"""Evaluation metrics.

Two PHM scores are provided on purpose.

``phm_score_legacy``
    Reproduces the thesis implementation exactly: relative error expressed as a
    *fraction* inside ``exp(-ln2 * Er / 20)`` for under-estimation and
    ``exp(-ln2 * |Er| / 5)`` for over-estimation.

``phm_score_official``
    The IEEE PHM 2012 challenge formulation. It is structurally identical but
    takes the relative error as a **percentage**, so it is roughly 100x more
    punishing at the same absolute deviation. Quoting only the former makes the
    reported scores look far better than the literature's; reporting both is
    both honest and cheap.

Also included is ``constant_baseline_rmse``: the closed-form RMSE of the
"always answer 1.0" predictor under a piecewise-constant label. It is the lower
bound every model must beat, and it is the fastest way to spot a broken
baseline. It supports two regimes -- the textbook *continuous limit* and the
*exact windowed* value that matches how the model is actually scored; see the
function docstring, and do not compare a measured RMSE against the wrong one.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_EPS = 1e-12


# ------------------------------------------------------------------- errors
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, mode: str = "range") -> float:
    """Range-normalised RMSE in percent.

    ``range`` divides by ``max - min`` of the true RUL (the thesis definition);
    ``mean`` divides by the mean, which is the more common convention elsewhere.
    """
    y_true = np.asarray(y_true, float)
    if mode == "range":
        denom = y_true.max() - y_true.min()
    elif mode == "mean":
        denom = np.mean(np.abs(y_true))
    else:
        raise ValueError(f"unknown nrmse mode: {mode!r}")
    return float(rmse(y_true, y_pred) / (denom + _EPS) * 100.0)


# ----------------------------------------------------------------- PHM score
def _phm_from_relative(err: np.ndarray) -> np.ndarray:
    """``err < 0`` is over-estimation (late alarm) and is penalised harder."""
    return np.where(
        err > 0,
        np.exp(np.log(0.5) * err / 20.0),
        np.exp(np.log(0.5) * (-err) / 5.0),
    )


def phm_score_legacy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Thesis implementation: relative error used as a fraction."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return 1.0
    err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.mean(_phm_from_relative(err)))


def phm_score_official(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """IEEE PHM 2012 challenge score: relative error used as a percentage."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return 1.0
    err = 100.0 * (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.mean(_phm_from_relative(err)))


# --------------------------------------------------------------- uncertainty
def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction-interval coverage probability (nominal 0.95)."""
    y_true = np.asarray(y_true, float)
    mask = (y_true >= np.asarray(lower, float)) & (y_true <= np.asarray(upper, float))
    return float(np.mean(mask))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction-interval width, in the same unit as the inputs."""
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def crps_gaussian(mean: np.ndarray, std: np.ndarray, y_true: np.ndarray) -> float:
    """CRPS for a Gaussian predictive distribution (proper scoring rule)."""
    from scipy.stats import norm

    mean = np.asarray(mean, float)
    std = np.maximum(np.asarray(std, float), _EPS)
    y_true = np.asarray(y_true, float)
    z = (y_true - mean) / std
    return float(
        np.mean(
            std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        )
    )


# ------------------------------------------------------------ trivial baseline
def constant_baseline_rmse(
    healthy_ratio: float = 0.8,
    lifetime: float = 1.0,
    n_samples: int | None = None,
    window_len: int | None = None,
) -> float:
    """Closed-form RMSE of the constant-1.0 predictor.

    Two regimes, and picking the wrong one is a real trap:

    * **Continuous limit** (``n_samples`` / ``window_len`` omitted). For a label
      that is 1 on ``[0, h)`` and linear to 0 on ``[h, 1]`` the mean squared
      error is ``(1 - h) / 3``, hence ``RMSE = lifetime * sqrt((1-h)/3)``;
      with ``h = 0.8`` that is ``0.258 * lifetime``. This is the textbook
      anchor, but it assumes an infinite and uniformly sampled life.

    * **Windowed, exact** (both ``n_samples`` and ``window_len`` given). The
      model is actually scored on *sliding windows* whose label is the ratio at
      the window's **last** step, so the first ``window_len - 1`` snapshots
      never appear and the integer cut ``int(h * n_samples)`` bites. Both
      effects inflate the baseline, badly for short lives. On XJTU-SY bearing
      1_5 (52 snapshots, window 32) the continuous formula says 13.4 min while
      the scored value is 20.2 min -- a 51 % gap; on 2_5 (339 snapshots) the
      gap is only 4 %.

    Use the windowed form for anything compared against a measured RMSE
    (acceptance gates, sanity anchors). Keep the continuous form only as the
    analytic lower bound it is.
    """
    healthy_ratio = float(healthy_ratio)
    if not 0.0 < healthy_ratio < 1.0:
        raise ValueError("healthy_ratio must lie in (0, 1)")

    if n_samples is None or window_len is None:
        return float(lifetime * np.sqrt((1.0 - healthy_ratio) / 3.0))

    n_samples = int(n_samples)
    window_len = int(window_len)
    if window_len < 1 or window_len > n_samples:
        raise ValueError("need 1 <= window_len <= n_samples")

    # Same label construction as rulbench.data.labels.piecewise_linear_rul:
    # the tail ends at 1/span, not at exactly 0. The window label is the ratio
    # at the window's last index, i.e. ends = window_len - 1 .. n_samples - 1.
    threshold = int(n_samples * healthy_ratio)
    threshold = max(1, min(threshold, n_samples - 1))
    span = n_samples - threshold
    ratio = np.ones(n_samples, dtype=np.float64)
    ratio[threshold:] = (n_samples - np.arange(threshold, n_samples)) / span

    ends = np.arange(window_len - 1, n_samples)
    err = 1.0 - ratio[ends]
    return float(lifetime * np.sqrt(np.mean(err**2)))


def constant_baseline_table(
    lifetimes: Dict[str, float],
    healthy_ratio: float = 0.8,
    window_len: int | None = None,
    n_samples: Dict[str, int] | None = None,
) -> Dict[str, float]:
    """Per-bearing baseline.

    ``lifetimes`` are the physical lives used for the minute conversion;
    ``n_samples`` (optional) maps a bearing name to its snapshot count. When
    both ``window_len`` and the matching ``n_samples`` entry are present the
    exact windowed form is used, otherwise the continuous limit.
    """
    out: Dict[str, float] = {}
    for name, life in lifetimes.items():
        ns = None if n_samples is None else n_samples.get(name)
        out[name] = constant_baseline_rmse(
            healthy_ratio,
            life,
            n_samples=ns,
            window_len=window_len if ns is not None else None,
        )
    return out


# --------------------------------------------------------- statistics helpers
def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a scalar statistic."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float).ravel()
    if values.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test; returns ``(statistic, p_value)``."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 6:
        return (float("nan"), float("nan"))
    try:
        res = wilcoxon(a, b)
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return (float("nan"), float("nan"))


# ------------------------------------------------------------------ umbrella
def evaluate_predictions(
    y_true_minutes: np.ndarray,
    y_pred_minutes: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    nrmse_mode: str = "range",
) -> Dict[str, float]:
    """Compute the full metric block for one prediction vector (minutes)."""
    out: Dict[str, float] = {
        "rmse": rmse(y_true_minutes, y_pred_minutes),
        "mae": mae(y_true_minutes, y_pred_minutes),
        "nrmse": nrmse(y_true_minutes, y_pred_minutes, mode=nrmse_mode),
        "phm_legacy": phm_score_legacy(y_true_minutes, y_pred_minutes),
        "phm_official": phm_score_official(y_true_minutes, y_pred_minutes),
    }
    if lower is not None and upper is not None:
        out["picp"] = picp(y_true_minutes, lower, upper)
        out["mpiw"] = mpiw(lower, upper)
    if std is not None:
        out["crps"] = crps_gaussian(y_pred_minutes, std, y_true_minutes)
    return out


METRIC_COLUMNS = ("rmse", "mae", "nrmse", "phm_legacy", "phm_official", "picp", "mpiw", "crps")
