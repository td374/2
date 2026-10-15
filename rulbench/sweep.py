"""Parameter sweeps.

A sweep entry names one dotted config path and a list of values. The runner
detects whether the path lives under ``data.``: if it does, the prepared
datasets (and therefore the feature cache) must be rebuilt for each value,
because changing the window length, the FFT band or the label rule changes the
tensors themselves. Overrides under ``model.`` or ``train.`` only require
retraining.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

import yaml

from .config import Config
from .pipelines.prepare import prepare
from .pipelines.run import run_ablation, run_ablation_cross, run_cross, run_single
from .utils import get_logger, resolve_device, safe_tag

log = get_logger()

SWEEP_FILE = "configs/sweep.yaml"

FAMILIES = {
    "single": "run_single",
    "cross": "run_cross",
    "ablation_single": "run_ablation",
    "ablation_cross": "run_ablation_cross",
}


def load_sweeps(path: str | pathlib.Path = SWEEP_FILE) -> Dict[str, Any]:
    with pathlib.Path(path).open("r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("sweeps", {})


def run_sweep(
    cfg: Config,
    sweep_name: str,
    out_root: str | pathlib.Path | None = None,
    sweep_file: str | pathlib.Path = SWEEP_FILE,
) -> Dict[str, Any]:
    sweeps = load_sweeps(sweep_file)
    if sweep_name not in sweeps:
        raise KeyError(f"unknown sweep {sweep_name!r}; available: {sorted(sweeps)}")

    spec = sweeps[sweep_name]
    path = spec["path"]
    values = spec["values"]
    family = spec.get("family", "cross")
    models = spec.get("models", ["se_resnet_bigru"])
    out_root = pathlib.Path(out_root or spec.get("report_dir", f"artifacts/reports/sweep_{sweep_name}"))
    touches_data = path.startswith("data.")
    device = resolve_device(cfg.get_path("runtime.device", "cuda"))

    rows: List[Dict[str, Any]] = []
    for value in values:
        tag = safe_tag(str(value))
        variant = cfg.override(**{path.replace(".", "__"): value})
        data_dir = out_root / f"value_{tag}" / "data"
        report_dir = out_root / f"value_{tag}" / "reports"

        if touches_data or not pathlib.Path(cfg.get_path("paths.data_dir")).exists():
            prepare(variant, data_dir)
        else:
            data_dir = pathlib.Path(cfg.get_path("paths.data_dir"))

        run_fn = {
            "single": lambda: run_single(variant, data_dir, report_dir, models=models, device=device),
            "cross": lambda: run_cross(variant, data_dir, report_dir, models=models, device=device),
            "ablation_single": lambda: run_ablation(variant, data_dir, report_dir, models=models, device=device),
            "ablation_cross": lambda: run_ablation_cross(variant, data_dir, report_dir, models=models, device=device),
        }[family]
        result = run_fn()

        row = {
            "sweep": sweep_name,
            "path": path,
            "value": value,
            "family": family,
            "headline": result["headline"],
        }
        rows.append(row)
        log.info("sweep %s: %s=%s done", sweep_name, path, value)

    summary_path = out_root / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False, default=str)

    headline_rows = []
    for row in rows:
        entry = {"value": row["value"]}
        for item in row["headline"]:
            entry[f"{item['model']}_mean"] = round(item["mean"], 4)
            entry[f"{item['model']}_std"] = round(item["std"], 4)
        headline_rows.append(entry)
    return {sweep_name: headline_rows}
