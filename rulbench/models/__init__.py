"""Model zoo: proposed architecture, deep baselines, ablations, trivial controls."""

from .ablations import (
    SequenceRegressor,
    build_pure_bigru,
    build_resnet_bigru,
    build_resnet_lstm,
    build_seresnet_gru,
    build_seresnet_lstm,
)
from .baselines import CNN1D, GRURegressor, InformerSEnet, LSTMRegressor
from .blocks import FeatureExtractor1D, ResBlock1D, SEBlock1D
from .proposed import MCDropoutModel, SEResNetBiGRU
from .registry import (
    DEFAULT_KWARGS,
    FAMILIES,
    build_model,
    filter_kwargs,
    list_models,
    resolve,
    supports_mmd,
)
from .trivial import ConstantPredictor, LastValueHold, RidgeRUL, build_trivial

__all__ = [
    "SEBlock1D",
    "ResBlock1D",
    "FeatureExtractor1D",
    "SEResNetBiGRU",
    "MCDropoutModel",
    "CNN1D",
    "LSTMRegressor",
    "GRURegressor",
    "InformerSEnet",
    "SequenceRegressor",
    "build_resnet_bigru",
    "build_seresnet_gru",
    "build_seresnet_lstm",
    "build_pure_bigru",
    "build_resnet_lstm",
    "ConstantPredictor",
    "LastValueHold",
    "RidgeRUL",
    "build_trivial",
    "MODELS",
    "FAMILIES",
    "DEFAULT_KWARGS",
    "build_model",
    "filter_kwargs",
    "list_models",
    "resolve",
    "supports_mmd",
]
