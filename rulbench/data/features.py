"""Feature construction: 128-D FFT amplitudes + 42-D multi-domain descriptors.

Final input vector is 140-D per monitoring minute::

    128 = 64 spectral-amplitude bins x 2 channels   (horizontal + vertical)
     12 = mRMR-selected subset of the 42 multi-domain descriptors
    ---
    140

All feature extraction is *stateless* except :func:`select_mrmr`, which must be
fitted on the **training bearings only** and then applied to the validation and
test bearings. Doing otherwise leaks test information into the feature space.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import kurtosis, skew

try:  # optional
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

#: number of leading spectral bins kept per channel
FFT_BINS_PER_CHANNEL: int = 64

#: total size of the multi-domain descriptor vector
MULTI_DOMAIN_DIM: int = 42

#: named blocks inside the 42-D descriptor
MULTI_DOMAIN_BLOCKS: Dict[str, int] = {
    "time": 16,
    "frequency": 14,
    "wavelet_packet": 8,
    "entropy": 4,
}

_EPS = 1e-12

TIME_FEATURE_NAMES: Tuple[str, ...] = (
    "mean", "std", "rms", "peak", "peak_to_peak", "abs_mean", "skewness",
    "kurtosis", "crest_factor", "shape_factor", "impulse_factor",
    "clearance_factor", "variance", "median", "iqr", "zero_crossing_rate",
)

FREQ_FEATURE_NAMES: Tuple[str, ...] = (
    "freq_centroid", "freq_rms", "freq_std", "freq_skewness", "freq_kurtosis",
    "spectral_spread", "spectral_entropy", "dominant_freq", "dominant_amplitude",
    "band_power_low", "band_power_mid", "band_power_high", "total_power",
    "spectral_rolloff",
)

WAVELET_BANDS: int = 8
ENTROPY_FEATURE_NAMES: Tuple[str, ...] = (
    "time_entropy", "singular_entropy", "permutation_entropy", "wavelet_energy_entropy",
)


# --------------------------------------------------------------------- FFT band
def extract_fft_amplitude(
    snapshots: np.ndarray,
    bins: int = FFT_BINS_PER_CHANNEL,
    bin_start: int = 0,
    log_scale: bool = False,
    chunk: int = 256,
) -> np.ndarray:
    """Leading spectral amplitudes for a batch of snapshots.

    Parameters
    ----------
    snapshots
        ``(T, C, L)`` denoised vibration data.
    bins, bin_start
        Which slice of the one-sided amplitude spectrum to keep. Defaults
        reproduce the paper's "first 64 components per channel". Because the
        bin width is ``fs / L = 25.6 kHz / 32768 = 0.78 Hz``, the default window
        only covers 0-50 Hz; ``bin_start`` is exposed so that the planned
        sensitivity analysis can widen the band.
    log_scale
        Apply ``log1p`` to compress the dynamic range.

    Returns
    -------
    ``(T, C * bins)`` float32
    """
    if snapshots.ndim != 3:
        raise ValueError(f"expected (T, C, L), got {snapshots.shape}")
    t_total, n_ch, n_pts = snapshots.shape
    out = np.empty((t_total, n_ch * bins), dtype=np.float32)
    stop = bin_start + bins
    for start in range(0, t_total, chunk):
        block = snapshots[start : start + chunk]
        spec = np.abs(np.fft.rfft(block, axis=-1))[:, :, bin_start:stop]
        if spec.shape[-1] < bins:  # pad defensively for tiny synthetic inputs
            spec = np.pad(spec, ((0, 0), (0, 0), (0, bins - spec.shape[-1])))
        spec = spec.astype(np.float32)
        if log_scale:
            spec = np.log1p(spec)
        out[start : start + chunk] = spec.reshape(block.shape[0], -1)
    return out


# ------------------------------------------------------------- multi-domain 42
def _time_domain(x: np.ndarray) -> np.ndarray:
    absx = np.abs(x)
    mean_abs = absx.mean() + _EPS
    root_mean_sq = np.sqrt(np.mean(x ** 2)) + _EPS
    peak = absx.max() + _EPS
    sqrt_abs_mean = (np.mean(np.sqrt(absx)) + _EPS) ** 2
    q75, q25 = np.percentile(x, [75, 25])
    zcr = float(np.mean(np.diff(np.signbit(x)) != 0))
    return np.array(
        [
            x.mean(),
            x.std(),
            np.sqrt(np.mean(x ** 2)),
            absx.max(),
            x.max() - x.min(),
            absx.mean(),
            float(skew(x)),
            float(kurtosis(x)),
            peak / root_mean_sq,
            root_mean_sq / mean_abs,
            peak / mean_abs,
            peak / sqrt_abs_mean,
            x.var(),
            float(np.median(x)),
            float(q75 - q25),
            zcr,
        ],
        dtype=np.float64,
    )


def _frequency_domain(amplitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    amp = amplitude + _EPS
    total = amp.sum()
    p = amp / total
    centroid = float((freqs * p).sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * p).sum()))
    spec_entropy = float(-(p * np.log(p + _EPS)).sum() / np.log(len(p) + _EPS))
    dom_idx = int(np.argmax(amp))
    third = len(amp) // 3
    return np.array(
        [
            centroid,
            float(np.sqrt((freqs ** 2 * p).sum())),
            spread,
            float((((freqs - centroid) / (spread + _EPS)) ** 3 * p).sum()),
            float((((freqs - centroid) / (spread + _EPS)) ** 4 * p).sum()),
            spread,
            spec_entropy,
            float(freqs[dom_idx]),
            float(amp[dom_idx]),
            float((amp[:third] ** 2).sum()),
            float((amp[third : 2 * third] ** 2).sum()),
            float((amp[2 * third :] ** 2).sum()),
            float((amp ** 2).sum()),
            float(freqs[min(int(np.searchsorted(np.cumsum(p), 0.85)), len(freqs) - 1)]),
        ],
        dtype=np.float64,
    )


def _wavelet_packet_energy(x: np.ndarray, bands: int = WAVELET_BANDS) -> np.ndarray:
    if pywt is None:
        return np.zeros(bands, dtype=np.float64)
    level = int(np.log2(bands))
    wp = pywt.WaveletPacket(data=x, wavelet="db4", mode="symmetric", maxlevel=level)
    nodes = [n.path for n in wp.get_level(level, order="natural")]
    energies = np.array([float(np.sum(np.asarray(wp[p].data) ** 2)) for p in nodes])
    total = energies.sum() + _EPS
    return energies / total


def _time_entropy(x: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(x, bins=bins)
    p = hist / (hist.sum() + _EPS)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _singular_entropy(x: np.ndarray, order: int = 16) -> float:
    n = len(x) - order + 1
    if n <= order:
        return 0.0
    traj = np.lib.stride_tricks.sliding_window_view(x[: n + order - 1], order)
    sv = np.linalg.svd(traj[:n], compute_uv=False)
    sv = sv / (sv.sum() + _EPS)
    sv = sv[sv > 1e-12]
    return float(-(sv * np.log(sv)).sum())


def _permutation_entropy(x: np.ndarray, order: int = 4, delay: int = 1) -> float:
    n = len(x) - (order - 1) * delay
    if n <= 0:
        return 0.0
    idx = np.arange(order)[None, :] * delay + np.arange(n)[:, None]
    patterns = np.argsort(x[idx], axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / (counts.sum() + _EPS)
    return float(-(p * np.log(p)).sum())


def _entropy_block(x: np.ndarray, wp_energy: np.ndarray) -> np.ndarray:
    p = wp_energy[wp_energy > 0]
    return np.array(
        [
            _time_entropy(x),
            _singular_entropy(x),
            _permutation_entropy(x),
            float(-(p * np.log(p)).sum()),
        ],
        dtype=np.float64,
    )


def extract_multi_domain_single(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """42-D descriptor for one snapshot of one channel."""
    amp = np.abs(np.fft.rfft(x))
    wp_energy = _wavelet_packet_energy(x)
    return np.concatenate(
        [
            _time_domain(x),
            _frequency_domain(amp, freqs),
            wp_energy,
            _entropy_block(x, wp_energy),
        ]
    )


def extract_multi_domain(
    snapshots: np.ndarray,
    sample_rate: float = 25600.0,
    source: str = "horizontal",
) -> np.ndarray:
    """42-D multi-domain descriptors, shape ``(T, 42)``.

    Parameters
    ----------
    source
        Which channel(s) feed the descriptor block:

        ``"horizontal"`` / ``"vertical"``
            descriptor computed on that channel only -- the only option that
            yields exactly 42 dimensions.
        ``"mean"``
            descriptor computed on the channel-mean waveform (still 42 dims,
            uses both sensors).
        ``"concat"``
            descriptors for both channels (84 dims); pair with a larger mRMR
            quota if you want to keep two-sensor information.

    Notes
    -----
    The original pipeline that produced the thesis numbers is not available, so
    this module is a documented reconstruction of the described blocks
    (time / frequency / wavelet-packet / entropy = 42). Verify the exact block
    sizes against your own notes before quoting numbers derived from it.
    """
    if snapshots.ndim != 3:
        raise ValueError(f"expected (T, C, L), got {snapshots.shape}")
    n_pts = snapshots.shape[-1]
    freqs = np.fft.rfftfreq(n_pts, d=1.0 / sample_rate)

    if source == "mean" or source == "concat":
        channels: List[np.ndarray] = (
            [snapshots.mean(axis=1)] if source == "mean"
            else [snapshots[:, 0], snapshots[:, 1]] if snapshots.shape[1] > 1
            else [snapshots[:, 0]]
        )
    elif source == "horizontal":
        channels = [snapshots[:, 0]]
    elif source == "vertical":
        channels = [snapshots[:, min(1, snapshots.shape[1] - 1)]]
    else:
        raise ValueError(f"unknown multi-domain source: {source!r}")

    out = np.empty((snapshots.shape[0], MULTI_DOMAIN_DIM * len(channels)), dtype=np.float32)
    for t in range(snapshots.shape[0]):
        feats = [extract_multi_domain_single(ch[t], freqs) for ch in channels]
        out[t] = np.concatenate(feats)
    return out


# ------------------------------------------------------------------- mRMR
def quantile_discretize(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Rank-based discretisation used for discrete mutual information."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    out = np.empty_like(values, dtype=np.int64)
    for j in range(values.shape[1]):
        col = values[:, j]
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
        out[:, j] = np.digitize(col, edges) if edges.size else 0
    return out


def select_mrmr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_bins: int = 10,
) -> List[int]:
    """Greedy max-relevance min-redundancy feature selection.

    Relevance is ``I(feature; target)``, redundancy is the mean mutual
    information against already-selected features. **Fit on training bearings
    only**, then reuse the returned indices everywhere else.
    """
    from sklearn.metrics import mutual_info_score

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must share the first dimension")
    k = min(k, X.shape[1])

    Xd = quantile_discretize(X, n_bins)
    yd = quantile_discretize(y, n_bins).ravel()

    relevance = np.array([mutual_info_score(Xd[:, i], yd) for i in range(X.shape[1])])
    n_feat = X.shape[1]
    redundancy = np.zeros((n_feat, n_feat), dtype=np.float64)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            value = mutual_info_score(Xd[:, i], Xd[:, j])
            redundancy[i, j] = redundancy[j, i] = value

    selected: List[int] = []
    remaining = list(range(n_feat))
    while len(selected) < k and remaining:
        if not selected:
            scores = relevance[remaining]
        else:
            scores = np.array(
                [
                    relevance[i] - redundancy[i, selected].mean()
                    for i in remaining
                ]
            )
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected
