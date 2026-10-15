"""Turn trained checkpoints into the result tables the paper needs."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..utils import get_logger, jsonable
from .inference import (
    clip_to_lifetime,
    median_scaling_factor,
    predict_ensemble,
    ratio_to_minutes,
)
from .metrics import METRIC_COLUMNS, evaluate_predictions

log = get_logger()


def _split_truth(artifact: Dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    payload = artifact["splits"][split]
    lifetime = np.asarray(payload["lifetime"], float)
    truth = np.asarray(payload["ratio"], float) * lifetime
    return truth, lifetime


def evaluate_dataset(
    artifact: Dict,
    model_name: str,
    cfg: Config,
    run_dir: str | pathlib.Path,
    seeds: Sequence[int],
    device=None,
) -> Dict[str, Any]:
    """Ensemble-evaluate one (dataset, model) pair.

    Returns corrected and uncorrected metrics, the fitted scaling factor, the
    per-bearing breakdown and the raw prediction arrays for plotting.
    """
    is_cross = artifact["meta"].get("protocol") == "cross"

    val_pred = predict_ensemble(artifact, model_name, cfg, run_dir, seeds, split="val", device=device)
    test_pred = predict_ensemble(artifact, model_name, cfg, run_dir, seeds, split="test", device=device)

    val_true, val_life = _split_truth(artifact, "val")
    test_true, test_life = _split_truth(artifact, "test")

    val_pred_min = ratio_to_minutes(val_pred["mean"], val_life)
    correction_enabled = bool(cfg.get_path("evaluation.correction.enabled", True)) and is_cross
    factor = (
        median_scaling_factor(
            val_true,
            val_pred_min,
            bounds=tuple(cfg.get_path("evaluation.correction.bounds", [0.1, 10.0])),
        )
        if correction_enabled
        else 1.0
    )

    test_mean_min = ratio_to_minutes(test_pred["mean"], test_life)
    lower_raw = ratio_to_minutes(test_pred["lower"], test_life)
    upper_raw = ratio_to_minutes(test_pred["upper"], test_life)
    std_raw = ratio_to_minutes(test_pred["std"], test_life)

    # Corrected (post-processed) arm: rescale the mean AND the spread by the same
    # factor, so CRPS (from ``std_min``) and the interval (lower/upper) stay on the
    # same scale. Previously ``std_min`` was left unscaled -> asymmetric arms (audit D2).
    corrected_min = clip_to_lifetime(test_mean_min * factor, test_life)
    lower = clip_to_lifetime(lower_raw * factor, test_life)
    upper = clip_to_lifetime(upper_raw * factor, test_life)
    std_min = std_raw * factor

    # Uncorrected arm: factor removed from the mean AND from the interval/spread.
    # Previously this block reused the *corrected* lower/upper -> PICP of the
    # "uncorrected" row was computed against corrected intervals (audit D3).
    uncorrected_min = clip_to_lifetime(test_mean_min, test_life)
    lower_unc = clip_to_lifetime(lower_raw, test_life)
    upper_unc = clip_to_lifetime(upper_raw, test_life)

    metrics = evaluate_predictions(test_true, corrected_min, lower, upper, std_min)
    metrics_uncorrected = evaluate_predictions(
        test_true, uncorrected_min, lower_unc, upper_unc, std_raw
    )

    result = {
        "dataset": artifact["name"],
        "model": model_name,
        "protocol": artifact["meta"].get("protocol"),
        "seeds": list(int(s) for s in seeds),
        "scaling_factor": float(factor),
        "metrics": metrics,
        "metrics_uncorrected": metrics_uncorrected,
        "correction_delta_rmse_pct": float(
            100.0 * (metrics_uncorrected["rmse"] - metrics["rmse"]) / (metrics_uncorrected["rmse"] + 1e-12)
        ),
        "per_bearing": per_bearing_metrics(artifact, test_true, corrected_min, lower, upper),
        "predictions": {
            "true": test_true,
            "pred": corrected_min,
            "lower": lower,
            "upper": upper,
            "bearing_id": np.asarray(artifact["splits"]["test"]["bearing_id"]),
            "window_end": np.asarray(artifact["splits"]["test"]["window_end"]),
        },
    }
    log.info(
        "%s | %s -> RMSE %.2f MAE %.2f NRMSE %.2f%% PHM(official) %.4f PICP %.3f",
        artifact["name"], model_name,
        metrics["rmse"], metrics["mae"], metrics["nrmse"], metrics["phm_official"], metrics["picp"],
    )
    return result


def per_bearing_metrics(
    artifact: Dict,
    truth: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> List[Dict[str, float]]:
    """Metric block computed separately for every bearing in the test split."""
    bearing_ids = np.asarray(artifact["splits"]["test"]["bearing_id"])
    out: List[Dict[str, float]] = []
    for bid in sorted(set(bearing_ids.tolist())):
        mask = bearing_ids == bid
        row = {"bearing_id": int(bid), "n_windows": int(mask.sum())}
        row.update(
            evaluate_predictions(truth[mask], pred[mask], lower[mask], upper[mask])
        )
        out.append(row)
    return out


# ------------------------------------------------------------------- tables
def results_to_records(results: Sequence[Dict[str, Any]], corrected: bool = True) -> List[Dict[str, Any]]:
    """Flatten evaluation results into rows suitable for CSV / markdown."""
    key = "metrics" if corrected else "metrics_uncorrected"
    rows = []
    for res in results:
        row = {
            "dataset": res["dataset"],
            "model": res["model"],
            "protocol": res["protocol"],
            "scaling_factor": round(res["scaling_factor"], 4),
        }
        for metric in METRIC_COLUMNS:
            if metric in res[key]:
                row[metric] = round(float(res[key][metric]), 4)
        rows.append(row)
    return rows


def summary_by_model(results: Sequence[Dict[str, Any]], metric: str = "rmse") -> List[Dict[str, Any]]:
    """Aggregate a metric across datasets, per model (the headline table)."""
    buckets: Dict[str, List[float]] = {}
    for res in results:
        if metric in res["metrics"]:
            buckets.setdefault(res["model"], []).append(float(res["metrics"][metric]))
    rows = []
    for model, values in buckets.items():
        arr = np.asarray(values, float)
        rows.append({"model": model, "n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std())})
    return sorted(rows, key=lambda r: r["mean"])


def _first_seen_fields(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Column order = dict insertion order of the first record that has the key.

    pandas emitted columns in insertion order (``dataset, model, protocol, ...``)
    and the released CSVs are read by eye as well as by script, so keep them
    aligned rather than sorting.
    """
    fieldnames: List[str] = []
    for r in records:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    return fieldnames


def _read_csv_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read an existing result CSV back into ordered dicts (empty if absent)."""
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def save_table(
    records: Sequence[Dict[str, Any]],
    path: str | pathlib.Path,
    merge_keys: Optional[Sequence[str]] = None,
) -> pathlib.Path:
    """Write records to CSV with the standard library.

    This used to build a ``pandas.DataFrame`` and call ``to_csv``. That is a
    hard crash on Windows once a CUDA context exists in the process: importing
    pandas is fine, but *constructing a DataFrame* after ``torch`` has touched
    the GPU segfaults (reproduced on pandas 3.0.3 / torch 2.6.0+cu124). Because
    ``save_table`` runs at the end of every ``run()`` -- after training and MC
    Dropout evaluation -- the first model of a multi-model driver wrote its CSV
    and then killed the process, so multi-model blocks never completed.

    The stdlib writer is byte-identical for our purposes (same columns, same
    ``utf-8-sig`` BOM, same value formatting through ``jsonable``) and has no
    native dependency, so the crash cannot recur. Do not reintroduce pandas
    here; if a DataFrame is ever genuinely needed, import pandas at module
    import time, before any CUDA context is created.

    ``merge_keys`` (B11) lets several *processes* accumulate one shared table.
    When the supervised Stage 4 driver runs one model per subprocess into a
    directory shared by the whole arm, a plain overwrite makes every model
    truncate the previous model's rows, so the released CSV ends up holding only
    the last writer's 3 rows. Passing ``merge_keys=("dataset", "model")`` makes
    the writer read the existing table, drop rows whose key matches an incoming
    row, and rewrite the union in a deterministic order. ``None`` (the default)
    keeps the original overwrite semantics for every existing caller.
    """
    import csv

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [jsonable(dict(r)) for r in records]

    if merge_keys:
        keys = tuple(merge_keys)
        incoming = {tuple(str(r.get(k, "")) for k in keys): r for r in records}
        kept = [
            r for r in _read_csv_rows(path)
            if tuple(str(r.get(k, "")) for k in keys) not in incoming
        ]
        merged = kept + records
        merged.sort(key=lambda r: tuple(str(r.get(k, "")) for k in keys))
        records = merged

    fieldnames = _first_seen_fields(records)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path
