"""Shared fit / predict_proba contract for model adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ProbabilisticModel(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        config: dict[str, Any] | None = None,
    ) -> ProbabilisticModel: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
