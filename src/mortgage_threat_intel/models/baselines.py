"""Logistic regression and HistGradientBoosting baselines (J-04)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class LogisticBaseline:
    """Reference utility baseline."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=2000, random_state=0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        config: dict[str, Any] | None = None,
    ) -> LogisticBaseline:
        params: dict[str, Any] = {"random_state": seed, "max_iter": 2000}
        if config:
            for key in ("C", "max_iter", "penalty", "solver"):
                if key in config:
                    params[key] = config[key]
        self._model.set_params(**params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class HistGradientBoostingBaseline:
    """Preselected audit model (fairness / TreeSHAP / robustness subject)."""

    def __init__(self) -> None:
        self._model = HistGradientBoostingClassifier(random_state=0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        config: dict[str, Any] | None = None,
    ) -> HistGradientBoostingBaseline:
        params: dict[str, Any] = {
            "random_state": seed,
            "max_depth": 6,
            "learning_rate": 0.1,
            "max_iter": 100,
            "min_samples_leaf": 20,
        }
        if config:
            for key in (
                "max_depth",
                "learning_rate",
                "max_iter",
                "min_samples_leaf",
                "l2_regularization",
            ):
                if key in config:
                    params[key] = config[key]
        self._model.set_params(**params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
