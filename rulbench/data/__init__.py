"""Data layer: raw signal -> features -> labels -> scaling -> windowed samples.

Two datasets are wired in (XJTU-SY and IEEE PHM 2012 / PRONOSTIA) behind one
interface; :mod:`rulbench.data.registry` is the entry point that resolves which
raw loader to use.
"""

from .build import build_all, build_cross_experiment, build_single_condition, fft_dim
from .features import (
    FFT_BINS_PER_CHANNEL,
    MULTI_DOMAIN_DIM,
    extract_fft_amplitude,
    extract_multi_domain,
    select_mrmr,
)
from .labels import piecewise_linear_rul
from .raw import CONDITIONS, N_SAMPLES_PER_FILE, SAMPLE_INTERVAL_MIN, list_bearings, load_bearing_signal
from .registry import DATASETS, RAW_MODULES, dataset_info, get_raw, list_datasets, time_unit
from .scaling import ChannelScaler
from .windows import make_windows

__all__ = [
    "CONDITIONS",
    "N_SAMPLES_PER_FILE",
    "SAMPLE_INTERVAL_MIN",
    "list_bearings",
    "load_bearing_signal",
    "DATASETS",
    "RAW_MODULES",
    "dataset_info",
    "get_raw",
    "list_datasets",
    "time_unit",
    "FFT_BINS_PER_CHANNEL",
    "MULTI_DOMAIN_DIM",
    "extract_fft_amplitude",
    "extract_multi_domain",
    "select_mrmr",
    "piecewise_linear_rul",
    "ChannelScaler",
    "make_windows",
    "build_single_condition",
    "build_cross_experiment",
    "build_all",
    "fft_dim",
]
