"""Lightweight YAML-backed configuration object.

Design goal: keep every experiment knob in files under ``configs/`` so that a
run can be reproduced from (config, seed) alone, and so that the agent or a
human can hand a *small* config file to a run instead of re-deriving settings
from conversation context.
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict, Iterable

import yaml


class Config(dict):
    """A dict with attribute access, deep-merge and dotted-path lookup."""

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, *paths: str | pathlib.Path) -> "Config":
        """Load and deep-merge YAML files left to right (later wins)."""
        merged: Dict[str, Any] = {}
        for p in paths:
            p = pathlib.Path(p)
            if not p.exists():
                raise FileNotFoundError(f"config not found: {p}")
            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            merged = _deep_merge(merged, data)
        return cls(merged)

    # ------------------------------------------------------------- accessors
    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return ``cfg.get_path('train.optim.lr')`` style lookups."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def override(self, **kwargs: Any) -> "Config":
        """Return a copy with dotted keys overridden, e.g. ``seed=1``."""
        out = Config(copy.deepcopy(dict(self)))
        for key, value in kwargs.items():
            if value is None:
                continue
            _assign_dotted(out, key.replace("__", "."), value)
        return out

    # ------------------------------------------------------------ utilities
    def to_json(self) -> str:
        return json.dumps(self, indent=2, ensure_ascii=False, default=str)

    def freeze(self) -> "Config":
        """Recursively convert nested dicts to Config (for attribute access)."""
        return Config(_freeze(dict(self)))


def _freeze(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _freeze(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_freeze(v) for v in node]
    return node


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _assign_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def dump_config(cfg: Config, path: str | pathlib.Path) -> None:
    """Persist the resolved config next to a run's artefacts."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, allow_unicode=True, sort_keys=False)


def as_list(value: Any, cast: type | None = None) -> list:
    """Normalise a scalar / iterable config value into a list."""
    if value is None:
        return []
    items: Iterable
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return [cast(v) if cast else v for v in items]
