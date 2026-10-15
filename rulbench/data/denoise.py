"""Wavelet denoising of raw vibration snapshots.

The paper preprocesses every snapshot with a 3-level db4 soft-threshold wavelet
filter. We keep that exact choice as the default but expose level / wavelet /
threshold rule so the sensitivity analysis planned in ``docs/experiment_plan.md``
can vary them without touching the rest of the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PYWT_HINT = (
    "PyWavelets is required for wavelet denoising: pip install PyWavelets. "
    "Set data.denoise.enable=false to skip denoising instead."
)


def _pywt() -> Any:
    """Import PyWavelets on demand so the rest of the package stays usable."""
    try:
        import pywt  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_PYWT_HINT) from exc
    return pywt


def _universal_threshold(detail: np.ndarray, rule: str = "soft") -> float:
    """Estimate the noise level from the finest detail band (MAD / 0.6745)."""
    sigma = np.median(np.abs(detail)) / 0.6745
    return float(sigma * np.sqrt(2.0 * np.log(max(detail.size, 2))))


def denoise_snapshot(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    mode: str = "soft",
) -> np.ndarray:
    """Denoise a single 1-D snapshot.

    Only the detail coefficients are thresholded; the approximation band is kept
    untouched so the low-frequency degradation trend survives.
    """
    pywt = _pywt()
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx, details = coeffs[0], coeffs[1:]
    thresh = _universal_threshold(details[-1])
    denoised_details = [pywt.threshold(d, thresh, mode=mode) for d in details]
    rec = pywt.waverec([approx, *denoised_details], wavelet)
    return rec[: signal.shape[-1]]


def denoise_bearing(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    mode: str = "soft",
) -> np.ndarray:
    """Denoise a whole bearing array of shape ``(T, C, L)``.

    Parameters mirror :func:`denoise_snapshot`; the filter is applied
    independently to every (snapshot, channel) pair.
    """
    if signal.ndim != 3:
        raise ValueError(f"expected (T, C, L), got shape {signal.shape}")
    out = np.empty_like(signal)
    for t in range(signal.shape[0]):
        for c in range(signal.shape[1]):
            out[t, c] = denoise_snapshot(signal[t, c], wavelet=wavelet, level=level, mode=mode)
    return out
