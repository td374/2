"""Raw XJTU-SY acquisition loading.

Official layout::

    <root>/<condition_name>/Bearing<cond>_<id>/<1..N>.csv

Each CSV holds ``N_SAMPLES_PER_FILE`` rows sampled at 25.6 kHz (1.28 s per
snapshot) and two columns: horizontal acceleration, vertical acceleration, in g.
One CSV therefore equals one monitoring minute, so the number of CSV files is
numerically equal to the bearing lifetime in minutes.
"""

from __future__ import annotations

import pathlib
from typing import List, Sequence

import numpy as np

#: Column order inside every XJTU-SY csv.
CHANNEL_NAMES: tuple[str, str] = ("horizontal", "vertical")

#: Condition id -> folder name (speed / radial load).
CONDITIONS: dict[int, str] = {
    1: "35Hz12kN",
    2: "37.5Hz11kN",
    3: "40Hz10kN",
}

#: Sampling frequency of the vibration channels, in Hz.
SAMPLE_RATE: float = 25600.0

#: 25.6 kHz * 1.28 s
N_SAMPLES_PER_FILE: int = 32768

#: One acquisition per minute.
SAMPLE_INTERVAL_MIN: float = 1.0

#: Physical unit of the reported lifetime, and the interval between snapshots
#: expressed in that unit. Consumed by :mod:`rulbench.data.build` so that the
#: label axis (snapshot index) and the reported lifetime never get conflated.
TIME_UNIT: str = "min"
SNAPSHOT_INTERVAL: float = 1.0

#: Bearing ids available per condition.
BEARING_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Human-readable description of each operating condition.
CONDITION_META: dict[int, dict[str, object]] = {
    1: {"name": "35Hz12kN", "speed_hz": 35.0, "radial_load_kn": 12.0},
    2: {"name": "37.5Hz11kN", "speed_hz": 37.5, "radial_load_kn": 11.0},
    3: {"name": "40Hz10kN", "speed_hz": 40.0, "radial_load_kn": 10.0},
}


def bearing_dir(root: str | pathlib.Path, condition: int, bearing_id: int) -> pathlib.Path:
    root = pathlib.Path(root)
    return root / CONDITIONS[condition] / f"Bearing{condition}_{bearing_id}"


def available_bearings(root: str | pathlib.Path, condition: int) -> List[int]:
    """Bearing ids that actually exist on disk, without raising when empty.

    The non-raising variant exists so that a split configured for one dataset
    can be safely reused on another with fewer bearings per condition: the
    generic builder intersects the requested ids with what is present.
    """
    return [bid for bid in BEARING_IDS if bearing_dir(root, condition, bid).is_dir()]


def list_bearings(root: str | pathlib.Path, condition: int) -> List[int]:
    """Return the bearing ids that actually exist on disk for a condition."""
    found = available_bearings(root, condition)
    if not found:
        raise FileNotFoundError(
            f"no bearing folders found for condition {condition} under {root}. "
            f"Expected e.g. {bearing_dir(root, condition, 1)}"
        )
    return found


def _sorted_csv_files(folder: pathlib.Path) -> List[pathlib.Path]:
    files = [f for f in folder.iterdir() if f.suffix.lower() == ".csv"]
    # numeric sort on the file stem; some releases use zero padding, some do not
    files.sort(key=lambda p: int(p.stem))
    return files


def load_bearing_signal(
    root: str | pathlib.Path,
    condition: int,
    bearing_id: int,
    max_snapshots: int | None = None,
    channels: Sequence[int] = (0, 1),
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load one bearing lifetime as an array of shape ``(T, len(channels), 32768)``.

    Loading is streamed per csv so that a full-lifetime bearing (up to 2538
    snapshots) never needs more than a few tens of MiB of transient memory.

    Parameters
    ----------
    max_snapshots
        Debug aid: only read the first ``N`` monitoring minutes.
    channels
        Column indices to keep. Default reads both the horizontal and the
        vertical accelerometer, matching the paper's 2-channel feature design.
    """
    folder = bearing_dir(root, condition, bearing_id)
    files = _sorted_csv_files(folder)
    if max_snapshots is not None:
        files = files[:max_snapshots]
    if not files:
        raise FileNotFoundError(f"no csv files in {folder}")

    chunks: List[np.ndarray] = []
    for csv_path in files:
        # first row is the sensor header, e.g. "Horizontal acceleration g"
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=tuple(channels), dtype=dtype)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != N_SAMPLES_PER_FILE:
            raise ValueError(
                f"{csv_path} has {arr.shape[0]} rows, expected {N_SAMPLES_PER_FILE}"
            )
        chunks.append(arr)
    # A csv is (length, n_channels); stacking gives (T, length, n_channels).
    # The rest of the pipeline expects the contract (T, n_channels, length) --
    # denoise_bearing and extract_fft_amplitude transform along the last axis --
    # so the channel axis is moved forward explicitly.
    stacked = np.stack(chunks, axis=0)
    return np.ascontiguousarray(stacked.transpose(0, 2, 1))


def snapshot_count(root: str | pathlib.Path, condition: int, bearing_id: int) -> int:
    """Number of monitoring snapshots captured for a bearing.

    This is the length of the label axis, i.e. ``T`` in ``(T, D)``. It is *not*
    the physical lifetime -- see :func:`lifetime`.
    """
    return len(_sorted_csv_files(bearing_dir(root, condition, bearing_id)))


def lifetime(root: str | pathlib.Path, condition: int, bearing_id: int) -> float:
    """Physical lifetime of a bearing, expressed in :data:`TIME_UNIT`.

    For XJTU-SY one snapshot equals one minute, so this numerically equals
    :func:`snapshot_count`. Datasets whose snapshot interval is not the unit
    (e.g. PHM 2012, one snapshot per 10 s) differ, which is exactly why the two
    quantities are kept apart.
    """
    return snapshot_count(root, condition, bearing_id) * SNAPSHOT_INTERVAL


def lifetime_minutes(root: str | pathlib.Path, condition: int, bearing_id: int) -> int:
    """Deprecated alias of :func:`lifetime`, kept for backward compatibility."""
    return int(lifetime(root, condition, bearing_id))
