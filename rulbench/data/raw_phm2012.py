"""Raw IEEE PHM 2012 / PRONOSTIA acquisition loading.

Official layout (FEMTO-ST, redistributed on GitHub)::

    <root>/Learning_set/Bearing<cond>_<id>/acc_00001.csv
    <root>/Learning_set/Bearing<cond>_<id>/temp_00001.csv
    <root>/Test_set/...
    <root>/Full_Test_Set/...

Differences from XJTU-SY that the rest of the pipeline has to absorb:

======================  ======================  ======================
property                XJTU-SY                 PHM 2012
======================  ======================  ======================
samples per snapshot    32768 (1.28 s)          2560 (0.1 s)
interval per snapshot   1 min                   10 s
units per file          2 (h / v acceleration)  6 (4 timestamp + h / v)
header row              yes                     no
condition encoding      folder name             bearing id prefix
======================  ======================  ======================

Both datasets sample at 25.6 kHz, but because the record length differs by
12.8x the same FFT bin index maps to a 12.8x wider frequency band. The default
``data.fft.bins: 64`` therefore spans 0-50 Hz on XJTU-SY and 0-640 Hz here.
Pick ``data.fft.bins`` / ``data.fft.bin_start`` per dataset -- see
``docs/dataset_selection.md``.

One mirror-specific quirk is absorbed by the loader rather than the caller:
``Full_Test_Set/Bearing1_4`` is **semicolon**-separated while the rest of the
tree is comma-separated, so ``_read_numeric_csv`` sniffs the delimiter from the
first line of every file instead of hard-coding ``,``.

Failure criterion in the original challenge is "vibration amplitude exceeds
20 g". ``lifetime`` here is the recorded duration in **seconds**.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Sequence

import numpy as np

#: Column layout inside every ``acc_*.csv``: hour, minute, second, microsecond,
#: horizontal acceleration, vertical acceleration. The files carry no header.
ACC_COLUMNS: tuple[int, int] = (4, 5)

#: Column layout inside every ``temp_*.csv``: hour, minute, second, microsecond,
#: bearing temperature in degrees Celsius.
TEMP_COLUMNS: tuple[int, int] = (4,)

#: Directories that may hold bearing folders, in preference order. Which ones
#: exist depends on the mirror; the loader resolves each bearing independently.
SUBSETS: tuple[str, ...] = ("Learning_set", "Full_Test_Set", "Test_set")

#: Condition id -> human-readable label. Unlike XJTU-SY there is no per-condition
#: folder; the condition is encoded in the bearing name ``Bearing<cond>_<id>``.
CONDITIONS: Dict[int, str] = {
    1: "1800rpm4000N",
    2: "1650rpm4200N",
    3: "1500rpm5000N",
}

#: Speed and radial load of each condition, as published in the challenge notes.
CONDITION_META: Dict[int, Dict[str, object]] = {
    1: {"speed_rpm": 1800.0, "radial_load_n": 4000.0},
    2: {"speed_rpm": 1650.0, "radial_load_n": 4200.0},
    3: {"speed_rpm": 1500.0, "radial_load_n": 5000.0},
}

#: Sampling frequency of the accelerometers, in Hz.
SAMPLE_RATE: float = 25600.0

#: 25.6 kHz * 0.1 s
N_SAMPLES_PER_FILE: int = 2560

#: Physical unit of the reported lifetime, and the interval between snapshots
#: expressed in that unit. One snapshot is recorded every 10 s, so a bearing
#: that yields N csv files lived for 10*N seconds.
TIME_UNIT: str = "s"
SNAPSHOT_INTERVAL: float = 10.0

#: Bearing ids that may exist under a condition (condition 3 stops at 3).
BEARING_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)


# --------------------------------------------------------------------- layout
def _subset_dirs(root: str | pathlib.Path) -> List[pathlib.Path]:
    root = pathlib.Path(root)
    return [root / name for name in SUBSETS]


def bearing_dir(
    root: str | pathlib.Path,
    condition: int,
    bearing_id: int,
    subset: str | None = None,
) -> pathlib.Path:
    """Resolve the folder of one bearing.

    When ``subset`` is given the path is returned unconditionally (the caller
    wants to create or inspect it). Otherwise the first existing subset wins, so
    a partial download still resolves whatever it does contain.
    """
    root = pathlib.Path(root)
    folder_name = f"Bearing{condition}_{bearing_id}"
    if subset is not None:
        return root / subset / folder_name
    for candidate in _subset_dirs(root):
        path = candidate / folder_name
        if path.is_dir():
            return path
    # nothing on disk yet -- report the canonical location to make the error useful
    return root / SUBSETS[0] / folder_name


def which_subset(root: str | pathlib.Path, condition: int, bearing_id: int) -> str | None:
    """Name of the subset that holds a bearing, or ``None`` if absent."""
    folder_name = f"Bearing{condition}_{bearing_id}"
    for name in SUBSETS:
        if (pathlib.Path(root) / name / folder_name).is_dir():
            return name
    return None


def available_bearings(root: str | pathlib.Path, condition: int) -> List[int]:
    """Bearing ids present on disk for a condition, without raising when empty."""
    return [
        bid
        for bid in BEARING_IDS
        if which_subset(root, condition, bid) is not None
    ]


def list_bearings(root: str | pathlib.Path, condition: int) -> List[int]:
    """Bearing ids present on disk; raises when the condition is empty."""
    found = available_bearings(root, condition)
    if not found:
        raise FileNotFoundError(
            f"no Bearing{condition}_* folder found under any of "
            f"{[str(p) for p in _subset_dirs(root)]}"
        )
    return found


# --------------------------------------------------------------------- reading
def _sorted_acc_files(folder: pathlib.Path) -> List[pathlib.Path]:
    files = [f for f in folder.iterdir() if f.name.startswith("acc_") and f.suffix.lower() == ".csv"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return files


def _sorted_temp_files(folder: pathlib.Path) -> List[pathlib.Path]:
    files = [f for f in folder.iterdir() if f.name.startswith("temp_") and f.suffix.lower() == ".csv"]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return files


def _all_numeric(fields: Sequence[str]) -> bool:
    for tok in fields:
        try:
            float(tok.strip())
        except ValueError:
            return False
    return True


def _read_numeric_csv(
    csv_path: pathlib.Path,
    columns: Sequence[int],
    dtype: np.dtype,
) -> np.ndarray:
    """Read selected columns of one numeric csv, tolerating delimiter and header.

    Two quirks exist in the wild and both are present in the community mirror:

    * Most files are **comma** separated, but
      ``Full_Test_Set/Bearing1_4`` ships **semicolon** separated (a European
      locale export) even though it sits next to comma-separated siblings. So
      the delimiter is sniffed from the first line rather than hard-coded.
    * The original files carry no header, but some redistributions prepend one.
      That is detected by whether the first line parses as pure numbers.

    Sniffing up front also avoids the old try/except dance that read the whole
    file twice on every headerless file.
    """
    with csv_path.open("r", errors="replace") as fh:
        first_line = fh.readline()

    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    has_header = not _all_numeric(first_line.rstrip("\r\n").split(delimiter))

    kwargs = {"delimiter": delimiter, "usecols": tuple(columns), "dtype": dtype}
    skip = 1 if has_header else 0
    try:
        arr = np.loadtxt(csv_path, skiprows=skip, **kwargs)
    except ValueError:
        # last resort: the header-guess was wrong, flip it
        arr = np.loadtxt(csv_path, skiprows=1 - skip, **kwargs)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def load_bearing_signal(
    root: str | pathlib.Path,
    condition: int,
    bearing_id: int,
    max_snapshots: int | None = None,
    channels: Sequence[int] = ACC_COLUMNS,
    dtype: np.dtype = np.float32,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Load one bearing lifetime as an array of shape ``(T, len(channels), L)``.

    Parameters
    ----------
    max_snapshots
        Debug aid: read only the first ``N`` recordings.
    channels
        **Column indices** inside ``acc_*.csv``. Default reads the horizontal
        and vertical accelerometer.
    expected_rows
        Row count every recording must share. ``None`` adopts the row count of
        the first file, so a mirror that stores a different record length than
        the documented 2560 still loads consistently.
    """
    folder = bearing_dir(root, condition, bearing_id)
    files = _sorted_acc_files(folder)
    if max_snapshots is not None:
        files = files[:max_snapshots]
    if not files:
        raise FileNotFoundError(f"no acc_*.csv files in {folder}")

    n_rows = expected_rows
    chunks: List[np.ndarray] = []
    for csv_path in files:
        arr = _read_numeric_csv(csv_path, channels, dtype)
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(
                f"{csv_path.name} has {arr.shape[0]} rows but the first file had "
                f"{n_rows}; ragged recordings cannot be stacked"
            )
        chunks.append(arr)
    # A recording is (length, n_channels); stacking gives (T, length, n_channels).
    # The pipeline contract is (T, n_channels, length), matching XJTU-SY and what
    # denoise_bearing / extract_fft_amplitude assume (they act on the last axis).
    stacked = np.stack(chunks, axis=0)
    return np.ascontiguousarray(stacked.transpose(0, 2, 1))


def load_bearing_temperature(
    root: str | pathlib.Path,
    condition: int,
    bearing_id: int,
    max_snapshots: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load the PT100 channel as ``(T, 1, L_temp)``.

    The temperature probe runs at 10 Hz and is *not* time-aligned with the
    accelerometer recordings, so this is returned separately and is not part of
    the default feature pipeline. Kept available because it is a legitimate
    second modality if a reviewer asks for one.
    """
    folder = bearing_dir(root, condition, bearing_id)
    files = _sorted_temp_files(folder)
    if max_snapshots is not None:
        files = files[:max_snapshots]
    if not files:
        raise FileNotFoundError(f"no temp_*.csv files in {folder}")
    stacked = np.stack(
        [_read_numeric_csv(path, TEMP_COLUMNS, dtype) for path in files], axis=0
    )  # (T, L_temp, 1)
    return np.ascontiguousarray(stacked.transpose(0, 2, 1))


# ------------------------------------------------------------------ accounting
def snapshot_count(root: str | pathlib.Path, condition: int, bearing_id: int) -> int:
    """Number of accelerometer recordings = length of the label axis."""
    return len(_sorted_acc_files(bearing_dir(root, condition, bearing_id)))


def lifetime(root: str | pathlib.Path, condition: int, bearing_id: int) -> float:
    """Recorded lifetime in seconds (``snapshot_count`` * 10)."""
    return snapshot_count(root, condition, bearing_id) * SNAPSHOT_INTERVAL


# -------------------------------------------------------------------- auditing
def verify(root: str | pathlib.Path, conditions: Sequence[int] = (1, 2, 3)) -> List[Dict[str, object]]:
    """Walk the download and report what is there, without loading any signal.

    Used by ``scripts/download_phm2012.py`` and by the S1 acceptance gate in
    ``docs/experiment_plan.md``: the output is the evidence that the download
    is complete and that the file layout matches the documented one.
    """
    report: List[Dict[str, object]] = []
    for condition in conditions:
        for bearing_id in available_bearings(root, condition):
            folder = bearing_dir(root, condition, bearing_id)
            n_acc = snapshot_count(root, condition, bearing_id)
            n_temp = len(_sorted_temp_files(folder))
            report.append(
                {
                    "condition": int(condition),
                    "bearing": f"{condition}_{bearing_id}",
                    "subset": which_subset(root, condition, bearing_id),
                    "n_acc_files": n_acc,
                    "n_temp_files": n_temp,
                    "lifetime_seconds": n_acc * SNAPSHOT_INTERVAL,
                    "lifetime_minutes": n_acc * SNAPSHOT_INTERVAL / 60.0,
                    "path": str(folder),
                }
            )
    return report
