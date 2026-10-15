"""Model registry: one name -> one factory, shared by every pipeline.

Having a single registry is what guarantees the proposed model, the baselines
and the ablations are all built through the same code path and then trained by
the same loop with the same loss.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .ablations import (
    build_pure_bigru,
    build_resnet_bigru,
    build_resnet_lstm,
    build_seresnet_gru,
    build_seresnet_lstm,
)
from .baselines import CNN1D, GRURegressor, InformerSEnet, LSTMRegressor
from .proposed import SEResNetBiGRU
from .trivial import TRIVIAL_FACTORIES

__all__ = [
    "MODELS", "FAMILIES", "ALIASES", "DEFAULT_KWARGS",
    "build_model", "filter_kwargs", "list_models", "resolve", "supports_mmd",
    "is_trivial", "family_of",
]

#: name -> (factory, family)
FAMILIES: Dict[str, str] = {}


def _register(name: str, family: str, factory: Callable[..., nn.Module]) -> None:
    MODELS[name] = factory
    FAMILIES[name] = family


MODELS: Dict[str, Callable[..., nn.Module]] = {}

# proposed
_register("se_resnet_bigru", "proposed", SEResNetBiGRU)

# deep baselines
_register("cnn", "baseline", CNN1D)
_register("lstm", "baseline", LSTMRegressor)
_register("gru", "baseline", GRURegressor)
_register("informer_senet", "baseline", InformerSEnet)

# ablations
_register("resnet_bigru", "ablation", build_resnet_bigru)
_register("se_resnet_gru", "ablation", build_seresnet_gru)
_register("se_resnet_lstm", "ablation", build_seresnet_lstm)
_register("pure_bigru", "ablation", build_pure_bigru)
_register("resnet_lstm", "ablation", build_resnet_lstm)

#: aliases kept so the legacy script names still resolve
ALIASES = {
    "SEResNetBiGRU": "se_resnet_bigru",
    "CNN": "cnn",
    "LSTM": "lstm",
    "GRU": "gru",
    "InformerSEnet": "informer_senet",
    "ResNetBiGRU": "resnet_bigru",
    "SEResNetGRU": "se_resnet_gru",
    "SEResNetLSTM": "se_resnet_lstm",
    "PureBiGRU": "pure_bigru",
}


def resolve(name: str) -> str:
    """Map a user-facing name (old or new) onto a registry key."""
    if name in MODELS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise KeyError(f"unknown model {name!r}. Available: {sorted(MODELS)}")


def is_trivial(name: str) -> bool:
    """True for the non-neural controls (``constant_*`` / ``ridge`` / ``last_value``).

    These live in :mod:`rulbench.models.trivial` and are *not* in :data:`MODELS`,
    so :func:`resolve` raises on them by design. Any dispatcher that wants to
    route both kinds must ask this first -- calling ``resolve`` inside a
    membership test is the bug that silently kept the trivial baselines out of
    every comparison table.
    """
    return name in TRIVIAL_FACTORIES


def family_of(name: str) -> str:
    """Family of any model name, including the trivial baselines.

    Never raises for a *known* trivial name; an unknown neural name still
    raises via :func:`resolve`, which is the desired loud failure.
    """
    if name in TRIVIAL_FACTORIES:
        return "trivial"
    return FAMILIES.get(resolve(name), "baseline")


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name, forwarding only the arguments it accepts."""
    key = resolve(name)
    return MODELS[key](**filter_kwargs(name, kwargs))


#: constructor defaults for the models whose signature differs from the shared
#: ``input_dim / hidden_dim / num_layers / dropout`` quartet
DEFAULT_KWARGS: Dict[str, Dict[str, object]] = {
    "cnn": {"hidden_dims": (64, 128)},
    "informer_senet": {"d_model": 128, "n_heads": 8},
}


def filter_kwargs(name: str, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Merge per-model defaults and drop anything the constructor cannot take."""
    import inspect

    signature = inspect.signature(MODELS[resolve(name)])
    params = signature.parameters
    merged = {**DEFAULT_KWARGS.get(resolve(name), {}), **kwargs}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return merged
    return {k: v for k, v in merged.items() if k in params}


def list_models(family: str | None = None) -> list[str]:
    if family is None:
        return sorted(MODELS)
    return sorted(k for k, v in FAMILIES.items() if v == family)


#: models that expose a ``mmd_representation`` and can therefore take the
#: domain-adaptation loss during cross-condition training
MMD_CAPABLE = ("se_resnet_bigru", "resnet_bigru", "se_resnet_gru", "se_resnet_lstm", "pure_bigru")


def supports_mmd(name: str) -> bool:
    return resolve(name) in MMD_CAPABLE
