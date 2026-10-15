"""End-to-end runners.

One generic ``run`` drives every scenario; the named helpers only decide which
datasets, which models and which splits are involved. Keeping training and
evaluation in a single place is what makes the comparison table trustworthy.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config, as_list
from ..data.build import load_dataset
from ..evaluation.inference import clip_to_lifetime, median_scaling_factor, ratio_to_minutes
from ..evaluation.metrics import evaluate_predictions
from ..evaluation.report import evaluate_dataset, per_bearing_metrics, results_to_records, save_table, summary_by_model
from ..models.registry import is_trivial, resolve
from ..models.trivial import build_trivial
from ..train.protocols import checkpoint_path, train_one
from ..utils import get_logger, jsonable, resolve_device

log = get_logger()


# --------------------------------------------------------------- trivial path
def evaluate_trivial(artifact: Dict, model_name: str, cfg: Config) -> Dict[str, Any]:
    """Fit and score a non-neural baseline (constant / ridge / hold)."""
    model = build_trivial(model_name)
    train = artifact["splits"]["train"]
    model.fit(train["X"], train["ratio"])

    val = artifact["splits"]["val"]
    test = artifact["splits"]["test"]
    val_pred = np.asarray(model.predict(val["X"]), float)
    test_pred_ratio = np.asarray(model.predict(test["X"]), float)

    val_true = np.asarray(val["ratio"], float) * np.asarray(val["lifetime"], float)
    test_true = np.asarray(test["ratio"], float) * np.asarray(test["lifetime"], float)
    test_life = np.asarray(test["lifetime"], float)

    is_cross = artifact["meta"].get("protocol") == "cross"
    factor = 1.0
    if is_cross and bool(cfg.get_path("evaluation.correction.enabled", True)):
        factor = median_scaling_factor(val_true, ratio_to_minutes(val_pred, val["lifetime"]))

    corrected = clip_to_lifetime(ratio_to_minutes(test_pred_ratio, test_life) * factor, test_life)
    metrics = evaluate_predictions(test_true, corrected)
    metrics_uncorrected = evaluate_predictions(test_true, clip_to_lifetime(ratio_to_minutes(test_pred_ratio, test_life), test_life))

    return {
        "dataset": artifact["name"],
        "model": model_name,
        "protocol": artifact["meta"].get("protocol"),
        "seeds": [],
        "scaling_factor": float(factor),
        "metrics": metrics,
        "metrics_uncorrected": metrics_uncorrected,
        "correction_delta_rmse_pct": float(
            100.0 * (metrics_uncorrected["rmse"] - metrics["rmse"]) / (metrics_uncorrected["rmse"] + 1e-12)
        ),
        "per_bearing": per_bearing_metrics(artifact, test_true, corrected, corrected, corrected),
        "predictions": {},
    }


# --------------------------------------------------------------------- runner
def run(
    cfg: Config,
    dataset_paths: Sequence[str | pathlib.Path],
    models: Sequence[str],
    out_dir: str | pathlib.Path,
    seeds: Optional[Sequence[int]] = None,
    device=None,
    train_missing: bool = True,
    merge_tables: bool = False,
) -> Dict[str, Any]:
    """Train (if needed) and evaluate every (dataset, model) pair.

    ``merge_tables`` (B11) is for the supervised Stage 4 driver, which runs one
    model per subprocess into a directory shared by a whole arm. There a plain
    overwrite would make each model truncate its predecessors, leaving only the
    last writer's rows. When true the CSV writers merge on ``(dataset, model)``
    instead of overwriting; the default ``False`` preserves the original
    behaviour for every existing caller.
    """
    out_dir = pathlib.Path(out_dir)
    run_dir = out_dir / "checkpoints"
    # ``runtime.seeds`` is the documented key (see configs/base.yaml); the
    # ``train.seeds`` fallback is kept so older configs keep working. Reading
    # only ``train.seeds`` -- as this did -- silently ignored the config and
    # fell back to the hard-coded default, which happened to be identical.
    # ``as_list`` lets ``--set runtime.seeds=42`` (a scalar) work too.
    configured_seeds = cfg.get_path("runtime.seeds") or cfg.get_path("train.seeds")
    source = seeds if seeds is not None else (configured_seeds or [42, 123, 456, 789, 1011])
    seeds = [int(s) for s in as_list(source)]
    device = device or resolve_device(cfg.get_path("runtime.device", "cuda"))

    results: List[Dict[str, Any]] = []
    # Fail fast on a typo before any GPU time is spent. ``resolve`` raises on
    # unknown neural names; the trivial controls are validated by ``is_trivial``.
    for model_name in models:
        if not is_trivial(model_name):
            resolve(model_name)

    for ds_path in dataset_paths:
        artifact = load_dataset(ds_path)
        for model_name in models:
            if is_trivial(model_name):
                results.append(evaluate_trivial(artifact, model_name, cfg))
                continue

            for seed in seeds:
                ckpt = checkpoint_path(run_dir, artifact["name"], model_name, seed)
                if train_missing and not ckpt.exists():
                    train_one(artifact, model_name, cfg, seed=seed, out_path=ckpt, device=device)
            results.append(evaluate_dataset(artifact, model_name, cfg, run_dir, seeds, device=device))

    records = results_to_records(results, corrected=True)
    records_uncorrected = results_to_records(results, corrected=False)
    merge_keys = ("dataset", "model") if merge_tables else None
    save_table(records, out_dir / "results_corrected.csv", merge_keys=merge_keys)
    save_table(records_uncorrected, out_dir / "results_uncorrected.csv", merge_keys=merge_keys)

    headline = summary_by_model(results, metric=cfg.get_path("report.headline_metric", "rmse"))
    save_table(headline, out_dir / "summary_by_model.csv",
               merge_keys=("model",) if merge_tables else None)

    with (out_dir / "results_full.json").open("w", encoding="utf-8") as fh:
        json.dump(
            [jsonable({k: v for k, v in r.items() if k != "predictions"}) for r in results],
            fh, indent=2, ensure_ascii=False, default=str,
        )
    return {"results": results, "records": records, "headline": headline, "out_dir": str(out_dir)}


# ------------------------------------------------------------- scenario sugar
def _datasets_for(cfg: Config, data_dir: str | pathlib.Path, pattern: str,
                  include: Optional[Sequence[str]] = None) -> List[pathlib.Path]:
    """Datasets matching ``pattern``; ``include`` restricts to named stems.

    ``include`` matches the file stem's task suffix (e.g. ``"1+2_to_3"``), so a
    caller can run a *subset* of the cross tasks without touching the data dir
    or inventing a new pattern. ``None`` reproduces the original glob result, so
    every existing call site is unchanged.
    """
    data_dir = pathlib.Path(data_dir)
    paths = sorted(data_dir.glob(pattern))
    if include is None:
        return paths
    wanted = {s.strip() for s in include if s and s.strip()}
    out = []
    for p in paths:
        stem = p.stem
        # accept either the full stem (``cross_1+2_to_3``) or the task part
        if stem in wanted or stem.replace("cross_", "") in wanted:
            out.append(p)
    return out


def run_single(cfg: Config, data_dir: str | pathlib.Path, out_dir: str | pathlib.Path, **kwargs) -> Dict[str, Any]:
    """Single-condition experiments (train 1-3, val 4, test 5 per condition)."""
    paths = _datasets_for(cfg, data_dir, "single_cond*.npz", kwargs.pop("datasets", None))
    models = as_list(kwargs.pop("models", cfg.get_path("experiments.single.models", ["se_resnet_bigru", "cnn", "lstm", "gru", "informer_senet", "constant_one", "ridge"])))
    return run(cfg, paths, models, out_dir, **kwargs)


def run_cross(cfg: Config, data_dir: str | pathlib.Path, out_dir: str | pathlib.Path, **kwargs) -> Dict[str, Any]:
    """Cross-condition transfer experiments, with or without MK-MMD."""
    paths = _datasets_for(cfg, data_dir, "cross_*.npz", kwargs.pop("datasets", None))
    models = as_list(kwargs.pop("models", cfg.get_path("experiments.cross.models", ["se_resnet_bigru", "cnn", "lstm", "gru", "informer_senet", "constant_one", "ridge"])))
    return run(cfg, paths, models, out_dir, **kwargs)


def run_ablation(cfg: Config, data_dir: str | pathlib.Path, out_dir: str | pathlib.Path, **kwargs) -> Dict[str, Any]:
    """Module ablations, trained under exactly the same protocol as the full model."""
    paths = _datasets_for(cfg, data_dir, "single_cond*.npz")
    models = as_list(kwargs.pop("models", ["se_resnet_bigru", "resnet_bigru", "se_resnet_gru", "se_resnet_lstm", "pure_bigru"]))
    return run(cfg, paths, models, out_dir, **kwargs)


def run_ablation_cross(cfg: Config, data_dir: str | pathlib.Path, out_dir: str | pathlib.Path, **kwargs) -> Dict[str, Any]:
    """Cross-condition ablation: proposed vs pure BiGRU vs no-SE."""
    paths = _datasets_for(cfg, data_dir, "cross_*.npz")
    models = as_list(kwargs.pop("models", ["se_resnet_bigru", "pure_bigru", "resnet_bigru"]))
    return run(cfg, paths, models, out_dir, **kwargs)
