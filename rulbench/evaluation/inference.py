"""Inference: MC Dropout, seed ensembling and post-hoc scaling correction.

Seed ensembling uses the law of total variance to fuse within-seed MC Dropout
spread with between-seed spread::

    var_total = mean_s(var_s) + var_s(mean_s)

The thesis averaged the per-seed interval bounds independently, which silently
discards the between-seed term and therefore understates uncertainty -- one of
the reasons the reported PICP sat at 0.745 against a nominal 0.95.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import Config
from ..models.proposed import MCDropoutModel
from ..train.loop import load_checkpoint
from ..train.protocols import build_for_artifact, checkpoint_path
from ..utils import get_logger, resolve_device, seed_everything

log = get_logger()


def _mc_wrap(model: torch.nn.Module, cfg: Config) -> MCDropoutModel:
    """Wrap a plain model so MC Dropout sampling can be applied to it too."""
    return MCDropoutModel(model, dropout_rate=float(cfg.get_path("model.dropout", 0.2)))


@torch.no_grad()
def predict_seed(
    artifact: Dict,
    model_name: str,
    cfg: Config,
    ckpt: str | pathlib.Path,
    split: str = "test",
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Per-seed predictive mean / std / interval for one split."""
    device = device or resolve_device(cfg.get_path("runtime.device", "cuda"))
    # Seed immediately before the MC-Dropout loop so that PREDICTION IS A PURE
    # FUNCTION of (checkpoint, data, seed). ``predict_with_uncertainty`` draws
    # ``evaluation.mc_samples`` stochastic dropout masks; without seeding them the
    # result depends on whatever RNG state the process happens to be in, so
    # re-evaluating EXISTING checkpoints is not reproducible (~0.05-2% per cell).
    # Training happened to seed the RNG, which is why ``scripts/17`` measured
    # 0.0% for the train->eval pipeline but pure re-evaluation still drifted.
    if seed is not None:
        seed_everything(int(seed))
    model = build_for_artifact(model_name, artifact, cfg).to(device)
    load_checkpoint(model, str(ckpt), device)
    model.eval()

    wrapper = _mc_wrap(model, cfg)
    X = torch.as_tensor(
        np.ascontiguousarray(artifact["splits"][split]["X"]),
        dtype=torch.float32,
        device=device,
    )
    n_iter = int(cfg.get_path("evaluation.mc_samples", 50))
    z = float(cfg.get_path("evaluation.z", 1.96))
    mean, std, lower, upper = wrapper.predict_with_uncertainty(X, n_iter=n_iter, z=z)
    return {"mean": mean, "std": std, "lower": lower, "upper": upper}


def predict_ensemble(
    artifact: Dict,
    model_name: str,
    cfg: Config,
    run_dir: str | pathlib.Path,
    seeds: Sequence[int],
    split: str = "test",
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Ensemble several seeds and return merged mean / uncertainty."""
    per_seed_mean: List[np.ndarray] = []
    per_seed_var: List[np.ndarray] = []

    for seed in seeds:
        ckpt = checkpoint_path(run_dir, artifact["name"], model_name, seed)
        if not pathlib.Path(ckpt).exists():
            raise FileNotFoundError(f"missing checkpoint for seed {seed}: {ckpt}")
        pred = predict_seed(artifact, model_name, cfg, ckpt, split=split, device=device, seed=seed)
        per_seed_mean.append(pred["mean"])
        per_seed_var.append(pred["std"] ** 2)

    means = np.stack(per_seed_mean, axis=0)      # (S, N)
    variances = np.stack(per_seed_var, axis=0)   # (S, N)

    mean = means.mean(axis=0)
    total_var = variances.mean(axis=0) + means.var(axis=0)
    std = np.sqrt(np.maximum(total_var, 0.0))

    z = float(cfg.get_path("evaluation.z", 1.96))
    return {
        "mean": mean,
        "std": std,
        "lower": mean - z * std,
        "upper": mean + z * std,
        "per_seed_mean": means,
    }


# -------------------------------------------------------- scaling correction
def median_scaling_factor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bounds: tuple[float, float] = (0.1, 10.0),
) -> float:
    """``median(true / pred)`` on a labelled split, guarded against outliers."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    valid = np.abs(y_pred) > 1e-6
    if not np.any(valid):
        return 1.0
    factor = float(np.median(y_true[valid] / y_pred[valid]))
    if not np.isfinite(factor) or not (bounds[0] <= factor <= bounds[1]):
        return 1.0
    return factor


def ratio_to_minutes(ratio: np.ndarray, lifetime: np.ndarray) -> np.ndarray:
    """Convert a predicted ratio into minutes using the per-window lifetime."""
    return np.asarray(ratio, float) * np.asarray(lifetime, float)


def clip_to_lifetime(values: np.ndarray, lifetime: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, float), 0.0, np.asarray(lifetime, float))
