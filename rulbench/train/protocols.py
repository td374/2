"""Training protocols: single-condition, cross-condition UDA, source-only.

The cross-condition protocol makes one experimental setting explicit that the
thesis left unstated: under unsupervised domain adaptation the *unlabelled*
target-domain windows take part in training through the MK-MMD term. The
``target_split`` argument is how that is declared, and setting ``uda.enabled``
to ``false`` produces the source-only control run that isolates how much of the
gain actually comes from alignment rather than from having seen the target data.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..config import Config
from ..models import build_model, resolve, supports_mmd
from ..utils import get_logger, resolve_device, safe_tag, seed_everything
from .loop import Trainer

log = get_logger()


# ---------------------------------------------------------------- model build
def model_kwargs_from_config(model_name: str, artifact: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Assemble constructor kwargs from the config ``models:`` block."""
    base = {
        "input_dim": int(artifact["feature_dim"]),
        "hidden_dim": int(cfg.get_path("model.hidden_dim", 64)),
        "num_layers": int(cfg.get_path("model.num_layers", 2)),
        "dropout": float(cfg.get_path("model.dropout", 0.2)),
    }
    per_model = (cfg.get_path("models", {}) or {}).get(model_name, {}) or {}
    if resolve(model_name) == "se_resnet_bigru":
        base.update(
            {
                "se_reduction": int(cfg.get_path("model.se_reduction", 16)),
                "channel_plan": tuple(cfg.get_path("model.channel_plan", [64, 128])),
                "use_se": bool(cfg.get_path("model.use_se", True)),
                "mmd_feature": cfg.get_path("model.mmd_feature", "flatten"),
            }
        )
    base.update(per_model)
    return base


def build_for_artifact(model_name: str, artifact: Dict[str, Any], cfg: Config) -> torch.nn.Module:
    return build_model(model_name, **model_kwargs_from_config(model_name, artifact, cfg))


# ------------------------------------------------------------------- protocols
def target_split_for_mmd(artifact: Dict[str, Any], cfg: Config) -> Optional[Dict[str, np.ndarray]]:
    """Return the unlabelled target windows used by MK-MMD, or ``None``.

    ``None`` means a *source-only* run: no target data of any kind reaches the
    model, which is the control condition for the domain-adaptation claim.
    """
    if artifact["meta"].get("protocol") != "cross":
        return None
    if not bool(cfg.get_path("uda.enabled", True)):
        return None
    return {"X": artifact["splits"]["test"]["X"]}


def train_one(
    artifact: Dict[str, Any],
    model_name: str,
    cfg: Config,
    seed: int,
    out_path: str | pathlib.Path | None = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train one (model, dataset, seed) triple and optionally checkpoint it."""
    device = device or resolve_device(cfg.get_path("runtime.device", "cuda"))

    # Seed BEFORE the model exists. ``Trainer.fit`` seeds too, but by then the
    # initial weights have already been drawn from whatever global RNG state the
    # process happened to be in, so the weights depended on process history
    # (which task ran first, how many epochs it ran) instead of on ``seed``.
    # That is the run-to-run spread measured by scripts/17_reproducibility_probe.py:
    # the first model of every process drifted, and the leftover RNG state could
    # carry the drift into the next one. Seeding here makes the initial weights a
    # pure function of ``seed``.
    seed_everything(seed)

    model = build_for_artifact(model_name, artifact, cfg)
    trainer = Trainer(model, cfg, device)

    train_split = artifact["splits"]["train"]
    val_split = artifact["splits"]["val"]
    target = target_split_for_mmd(artifact, cfg)

    tag = f"{artifact['name']}|{model_name}|s{seed}"
    result = trainer.fit(train_split, val_split, target=target, seed=seed, tag=tag)

    # ``target is not None`` only means the unlabelled target windows were
    # *offered*. The loop cycles them only when the model actually exposes an
    # ``mmd_representation`` head (see Trainer.fit), so for a CNN/LSTM/GRU the
    # "UDA" arm is by construction identical to the source-only arm. Report what
    # happened, not what was requested -- otherwise a reader cannot tell an
    # alignment result from a model that simply has no alignment module.
    used_target = target is not None and hasattr(model, "mmd_representation")

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(
            result,
            str(out_path),
            extra={
                "model_name": model_name,
                "dataset": artifact["name"],
                "seed": seed,
                "feature_dim": artifact["feature_dim"],
                "used_target_for_mmd": used_target,
                "n_parameters": model.n_parameters() if hasattr(model, "n_parameters") else None,
            },
        )

    summary = {
        "dataset": artifact["name"],
        "model": model_name,
        "seed": int(seed),
        "best_val_loss": float(result.best_val_loss),
        "best_epoch": int(result.best_epoch),
        "epochs_run": int(result.epochs_run),
        "used_target_for_mmd": used_target,
        "history": {k: [float(x) for x in v] for k, v in result.history.items()},
        "checkpoint": str(out_path) if out_path is not None else None,
    }
    log.info("%s -> val %.6f @ epoch %d", tag, result.best_val_loss, result.best_epoch)
    return summary


def checkpoint_path(
    run_dir: str | pathlib.Path,
    dataset: str,
    model_name: str,
    seed: int,
    suffix: str = "pt",
) -> pathlib.Path:
    """Stable checkpoint naming: ``<run_dir>/<dataset>__<model>__seed<seed>.pt``."""
    return pathlib.Path(run_dir) / f"{safe_tag(dataset)}__{model_name}__seed{seed}.{suffix}"


def should_use_mmd(model_name: str) -> bool:
    return supports_mmd(model_name)
