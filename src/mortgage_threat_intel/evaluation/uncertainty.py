"""Paired bootstrap helpers (shared; Krati may extend)."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return mean and (1-alpha) percentile CI."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        means.append(float(np.mean(sample)))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), lo, hi
