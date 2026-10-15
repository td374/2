"""Trivial and classical baselines.

The constant predictor is not padding: with the piecewise-constant label, 80 %
of the training targets equal 1.0, so a predictor that always answers 1.0 is a
genuinely competitive *lower bound*. Reporting it is what stops a reviewer from
noticing that some deep baselines fall below it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ConstantPredictor:
    """Always predict the same normalised RUL ratio.

    Parameters
    ----------
    value
        ``"one"`` -> 1.0 (upper bound of the label range, the strong trivial
        baseline), ``"mean"`` -> mean training label, or an explicit float.
    """

    def __init__(self, value: str | float = "one"):
        self.value_spec = value
        self.constant_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantPredictor":
        if self.value_spec == "one":
            self.constant_ = 1.0
        elif self.value_spec == "mean":
            self.constant_ = float(np.mean(y))
        elif self.value_spec == "zero":
            self.constant_ = 0.0
        else:
            self.constant_ = float(self.value_spec)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.constant_, dtype=np.float64)


class RidgeRUL:
    """Ridge regression on the flattened window -- a classical ML reference.

    Cheap, deterministic and surprisingly strong on this label, so it is a
    useful sanity anchor between the trivial constant and the deep models.
    """

    def __init__(self, alpha: float = 1.0, standardize: bool = True):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = Ridge(alpha=alpha)
        self.model = make_pipeline(StandardScaler(), base) if standardize else base

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return arr.reshape(arr.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRUL":
        self.model.fit(self._flatten(X), np.asarray(y, dtype=np.float64).ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X))


class LastValueHold:
    """Hold the first observed ratio for the whole life.

    Kept for completeness of the "no learning" family: it answers the question
    "what if the model never reacts to degradation at all?".
    """

    def __init__(self, hold_value: float = 1.0):
        self.hold_value = hold_value

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastValueHold":
        if np.asarray(y).size:
            self.hold_value = float(np.asarray(y).ravel()[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.hold_value, dtype=np.float64)


TRIVIAL_FACTORIES = {
    "constant_one": lambda: ConstantPredictor("one"),
    "constant_mean": lambda: ConstantPredictor("mean"),
    "constant_zero": lambda: ConstantPredictor("zero"),
    "ridge": lambda: RidgeRUL(alpha=1.0),
    "last_value": lambda: LastValueHold(),
}


def build_trivial(name: str):
    if name not in TRIVIAL_FACTORIES:
        raise KeyError(f"unknown trivial baseline {name!r}; options: {sorted(TRIVIAL_FACTORIES)}")
    return TRIVIAL_FACTORIES[name]()
