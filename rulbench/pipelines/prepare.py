"""Dataset preparation pipeline: raw XJTU-SY -> cached, split, scaled artefacts."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

from ..config import Config
from ..data.build import build_all, dataset_fingerprint, load_dataset
from ..utils import get_logger, jsonable

log = get_logger()


def prepare(cfg: Config, out_dir: str | pathlib.Path | None = None) -> Dict[str, Any]:
    """Build every dataset the plan needs and write an index of what was made."""
    root = cfg.get_path("data.root")
    if not root:
        raise ValueError("data.root is not set in the config")

    dataset = cfg.get_path("data.dataset", "xjtu_sy")
    out_dir = pathlib.Path(out_dir or cfg.get_path("paths.data_dir", "artifacts/data"))
    written = build_all(root, cfg, out_dir, dataset=dataset)

    index = []
    for path in written:
        artifact = load_dataset(path)
        index.append(
            {
                "file": str(path),
                "name": artifact["name"],
                "dataset": artifact["meta"].get("dataset", dataset),
                "protocol": artifact["meta"].get("protocol"),
                "feature_dim": artifact["feature_dim"],
                "time_unit": artifact["meta"].get("time_unit"),
                "n_windows": artifact["meta"].get("n_windows"),
                "groups": artifact["meta"].get("groups"),
                "lifetimes": artifact["meta"].get("lifetimes"),
                "fingerprint": dataset_fingerprint(artifact),
            }
        )
    index_path = out_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(jsonable(index), fh, indent=2, ensure_ascii=False)
    log.info("prepared %d %s datasets -> %s", len(index), dataset, index_path)
    return {"datasets": index, "out_dir": str(out_dir), "index": str(index_path), "dataset": dataset}
