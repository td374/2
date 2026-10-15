"""Cross-cutting helpers: seeding, logging, paths, device selection."""

from __future__ import annotations

import logging
import os
import pathlib
import random
import sys
from typing import Any

import numpy as np

_LOGGER_NAME = "rulbench"
_CONFIGURED = False


# --------------------------------------------------------------------- device
def resolve_device(prefer: str = "cuda") -> "Any":
    """Return a torch device, falling back to CPU when CUDA is unavailable."""
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------- seeding
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch in one call.

    Called once per training run so that any run is reproducible from its seed.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # pragma: no cover - DataLoader hook
    worker_seed = (torch_initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_initial_seed() -> int:
    import torch

    return int(torch.initial_seed())


# --------------------------------------------------------------------- logging
def get_logger(name: str = _LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger(_LOGGER_NAME)
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logger


# --------------------------------------------------------------------- paths
def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(root: str | pathlib.Path, run_id: str, create: bool = True) -> pathlib.Path:
    """Standard per-run artefact directory: ``<root>/<run_id>/``."""
    path = pathlib.Path(root) / run_id
    return ensure_dir(path) if create else path


def safe_tag(name: str) -> str:
    """Filesystem-safe tag for task names such as ``1->2`` or ``1+2->3``."""
    return name.replace("->", "_").replace("\u2192", "_").replace("+", "_").replace(" ", "")


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars into plain python types."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
