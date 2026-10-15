"""RUL label construction.

The thesis' central data-side design choice is kept and made explicit: the
label is **not** a straight line. Instead the first ``healthy_ratio`` of the
life is held constant at 1.0 and only the tail decays linearly to zero, which
matches the physical "stable then accelerating" degradation rhythm and, more
importantly, makes the label distribution identical across operating
conditions -- the precondition for the MK-MMD alignment to be meaningful.

Note on semantics
-----------------
The ratio is multiplied by the bearing lifetime to obtain a *nominal* RUL in
minutes; because the constant phase is held at 1.0 for the whole healthy
period, the nominal RUL overstates the physical remaining life early on. This
is intentional (it is what "piecewise constant + linear tail" means) but it
must be stated in the paper, otherwise the reported RMSE in minutes invites the
wrong comparison.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def piecewise_linear_rul(n_samples: int, healthy_ratio: float = 0.8) -> np.ndarray:
    """Normalised RUL ratio in ``[0, 1]`` for one bearing life.

    ``ratio[i] = 1`` for ``i < healthy_ratio * n_samples`` and decays linearly
    afterwards. Mirrors the original implementation, including the fact that
    the final value is ``1 / (n_samples - threshold)`` rather than exactly 0.
    """
    if not 0.0 < healthy_ratio < 1.0:
        raise ValueError("healthy_ratio must lie in (0, 1)")
    n_samples = int(n_samples)
    threshold = int(n_samples * healthy_ratio)
    threshold = max(1, min(threshold, n_samples - 1))
    span = n_samples - threshold
    ratio = np.ones(n_samples, dtype=np.float32)
    ratio[threshold:] = (n_samples - np.arange(threshold, n_samples)) / span
    return ratio


def linear_rul(n_samples: int) -> np.ndarray:
    """Classic full-life linear decay, kept as an ablation baseline label."""
    n_samples = int(n_samples)
    tail = np.arange(1, n_samples + 1, dtype=np.float32) / n_samples
    ratio = np.ones(n_samples, dtype=np.float32)
    ratio[:] = tail[::-1]
    return ratio


def exp_concave_rul(n_samples: int, alpha: float = 2.0) -> np.ndarray:
    """Exp-Concave label (Kim & Lee 2024) for accelerated degradation.
    
    Models "slow-then-fast" degradation pattern where early life degrades slowly
    and late life accelerates (e.g., Paris crack growth law with m>1).
    
    Parameters
    ----------
    n_samples : int
        Number of time steps (bearing lifetime).
    alpha : float, optional
        Curvature parameter controlling concavity. Larger alpha = steeper decline.
        Default 2.0 follows Kim & Lee (IET Electronics Letters 2024).
    
    Returns
    -------
    ratio : np.ndarray
        Normalised RUL in [0, 1], shape (n_samples,).
    
    Notes
    -----
    Formula: RUL(t) = 1 - (1 - exp(-alpha)) * (1 - exp(-alpha * t/N)) / (1 - exp(-alpha))
    Simplified: RUL(t) = 1 - (1 - exp(-alpha * t/N)) / (1 - exp(-alpha))
    
    This produces a concave curve (accelerating decay), in contrast to:
    - piecewise: plateau then linear
    - linear: constant decay rate
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    n_samples = int(n_samples)
    t = np.arange(n_samples, dtype=np.float32)
    ratio_t = t / n_samples
    
    # Exp-concave formula
    exp_alpha = np.exp(-alpha)
    exp_alpha_t = np.exp(-alpha * ratio_t)
    numerator = 1.0 - exp_alpha_t
    denominator = 1.0 - exp_alpha
    
    ratio = 1.0 - (numerator / denominator)
    return np.clip(ratio, 0.0, 1.0)


def rul_minutes(ratio: np.ndarray, lifetime_minutes: float) -> np.ndarray:
    """Convert a normalised ratio into nominal minutes for a given lifetime."""
    return np.asarray(ratio, dtype=np.float64) * float(lifetime_minutes)


LABEL_FACTORIES = {
    "piecewise": piecewise_linear_rul,
    "linear": linear_rul,
    "exp_concave": exp_concave_rul,
}


def build_labels(
    lifetime_minutes: int,
    strategy: str = "piecewise",
    healthy_ratio: float = 0.8,
    alpha: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Return ``{"ratio": ..., "minutes": ...}`` for one bearing.
    
    Parameters
    ----------
    lifetime_minutes : int
        Bearing lifetime in minutes.
    strategy : str, optional
        Label strategy: "piecewise", "linear", or "exp_concave".
    healthy_ratio : float, optional
        For piecewise: plateau length (default 0.8).
    alpha : float, optional
        For exp_concave: curvature parameter (default 2.0).
    """
    if strategy == "piecewise":
        ratio = piecewise_linear_rul(lifetime_minutes, healthy_ratio=healthy_ratio)
    elif strategy == "linear":
        ratio = linear_rul(lifetime_minutes)
    elif strategy == "exp_concave":
        ratio = exp_concave_rul(lifetime_minutes, alpha=alpha)
    else:
        raise ValueError(f"unknown label strategy: {strategy!r}")
    return {"ratio": ratio, "minutes": rul_minutes(ratio, lifetime_minutes)}
