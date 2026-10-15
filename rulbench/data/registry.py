"""Dataset registry: the single place that knows which raw loaders exist.

Adding a third dataset means adding one module under ``rulbench/data/`` that
exposes the same seven names, plus one entry in each table below. Nothing else
in the pipeline needs to change: :mod:`rulbench.data.build` resolves the loader
through :func:`get_raw` and drives it through the common interface.

The common interface a raw module must expose
------------------------------------------------
``CONDITIONS``        ``{condition_id: label}``
``CONDITION_META``    ``{condition_id: {physical parameters}}``
``BEARING_IDS``       candidate ids inside one condition
``SAMPLE_RATE``       accelerometer sampling frequency in Hz
``N_SAMPLES_PER_FILE`` rows per recording
``TIME_UNIT``         ``"min"`` or ``"s"`` -- unit of the reported lifetime
``SNAPSHOT_INTERVAL`` physical time between recordings, in ``TIME_UNIT``
``bearing_dir(root, condition, bearing_id)``
``available_bearings(root, condition)``            (never raises)
``list_bearings(root, condition)``                 (raises when empty)
``load_bearing_signal(root, condition, bearing_id, ...)``   ``(T, C, L)``
``snapshot_count(root, condition, bearing_id)``    label-axis length
``lifetime(root, condition, bearing_id)``          physical lifetime
"""

from __future__ import annotations

import types
from typing import Dict, List

from . import raw as xjtu_sy
from . import raw_phm2012 as phm2012

#: Short dataset id -> raw loader module.
RAW_MODULES: Dict[str, types.ModuleType] = {
    "xjtu_sy": xjtu_sy,
    "phm2012": phm2012,
}

#: Provenance and shape of every supported dataset, for configs and write-ups.
DATASETS: Dict[str, Dict[str, object]] = {
    "xjtu_sy": {
        "display_name": "XJTU-SY",
        "institution": "Xi'an Jiaotong University & Changxing Sumyoung Technology",
        "rig": "hydraulic loading rig, run-to-failure",
        "n_conditions": 3,
        "bearings_per_condition": 5,
        "sample_rate": 25600.0,
        "samples_per_snapshot": 32768,
        "snapshot_interval": 1.0,
        "time_unit": "min",
        "channels": ["horizontal", "vertical"],
        "reference": (
            "Wang B, Lei Y, Li N, Li N. A hybrid prognostics approach for "
            "estimating remaining useful life of rolling element bearings. "
            "IEEE Trans. Reliability, 2020, 69(1): 401-412."
        ),
        "homepage": "https://biaowang.tech/xjtu-sy-bearing-datasets/",
    },
    "phm2012": {
        "display_name": "IEEE PHM 2012 / PRONOSTIA",
        "institution": "FEMTO-ST Institute (Besancon, France)",
        "rig": "PRONOSTIA accelerated life test platform",
        "n_conditions": 3,
        "bearings_per_condition": 7,  # 7 / 7 / 3 -> 17 in total
        "sample_rate": 25600.0,
        "samples_per_snapshot": 2560,
        "snapshot_interval": 10.0,
        "time_unit": "s",
        "channels": ["horizontal", "vertical", "temperature"],
        "reference": (
            "Nectoux P, Gouriveau R, Medjaher K, Ramasso E, Morello B, "
            "Zerhouni N, Varnier C. PRONOSTIA: An experimental platform for "
            "bearings accelerated life test. IEEE Int. Conf. on Prognostics "
            "and Health Management, Denver, CO, 2012."
        ),
        "homepage": "https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset",
    },
}


def list_datasets() -> List[str]:
    return sorted(DATASETS)


def get_raw(dataset: str) -> types.ModuleType:
    """Return the raw loader module for a dataset id."""
    try:
        return RAW_MODULES[dataset]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(RAW_MODULES)}"
        ) from None


def dataset_info(dataset: str) -> Dict[str, object]:
    try:
        return DATASETS[dataset]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}"
        ) from None


def time_unit(dataset: str) -> str:
    return str(dataset_info(dataset)["time_unit"])
