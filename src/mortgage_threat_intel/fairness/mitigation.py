"""Fairness mitigations for the boosted-tree audit model — Krati K-03.

Reweighting is fit on 2023 training members only. Equalized-odds thresholds are
selected on 2024 validation only. Protected attributes are never model inputs;
they are used only for audit weighting / post-processing thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mortgage_threat_intel.fairness.metrics import (
    _as_positive_proba,
    _binary_rates,
    equalized_odds_difference,
)


def reweight_by_group_label(
    y: np.ndarray,
    group: pd.Series | np.ndarray,
) -> np.ndarray:
    """Inverse joint weighting so (A, Y) factors approximately: w ∝ P(A)P(Y)/P(A,Y)."""
    y = np.asarray(y).astype(int)
    group = pd.Series(group).astype(str).reset_index(drop=True)
    if len(y) != len(group):
        raise ValueError("y and group length mismatch")
    n = len(y)
    if n == 0:
        return np.asarray([], dtype=float)

    p_y = pd.Series(y).value_counts(normalize=True)
    p_a = group.value_counts(normalize=True)
    joint = pd.DataFrame({"a": group, "y": y}).value_counts(normalize=True)

    weights = np.empty(n, dtype=float)
    for i in range(n):
        a = group.iloc[i]
        yi = int(y[i])
        py = float(p_y.get(yi, 0.0))
        pa = float(p_a.get(a, 0.0))
        paj = float(joint.get((a, yi), 0.0))
        weights[i] = (pa * py / paj) if paj > 0 else 1.0

    # Normalize to mean 1 for stable HGB fitting
    mean_w = float(np.mean(weights))
    if mean_w > 0:
        weights = weights / mean_w
    return weights


def fit_equalized_odds_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    group: pd.Series | np.ndarray,
    *,
    grid: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Select per-group probability thresholds on validation to reduce EO gap.

    Greedy coordinate search over a shared grid; thresholds are chosen only from
    the provided validation labels/scores (must be 2024 in the experiment runner).
    """
    del seed  # deterministic grid search; kept for API symmetry
    y_true = np.asarray(y_true).astype(int)
    p1 = _as_positive_proba(proba)
    group = pd.Series(group).astype(str).reset_index(drop=True)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    groups = sorted(group.unique().tolist())
    thresholds = {g: 0.5 for g in groups}

    def _eo_at(th: dict[str, float]) -> float:
        tprs: list[float] = []
        fprs: list[float] = []
        for g in groups:
            mask = (group == g).to_numpy()
            if not np.any(mask):
                continue
            pred = (p1[mask] >= th[g]).astype(int)
            rates = _binary_rates(y_true[mask], pred)
            tprs.append(float(rates["tpr"]))
            fprs.append(float(rates["fpr"]))
        return equalized_odds_difference(tprs, fprs)

    best = dict(thresholds)
    best_eo = _eo_at(best)
    improved = True
    while improved:
        improved = False
        for g in groups:
            local_best = best[g]
            local_eo = best_eo
            for t in grid:
                cand = dict(best)
                cand[g] = float(t)
                eo = _eo_at(cand)
                if np.isfinite(eo) and (not np.isfinite(local_eo) or eo < local_eo - 1e-12):
                    local_eo = eo
                    local_best = float(t)
            if local_best != best[g]:
                best[g] = local_best
                best_eo = local_eo
                improved = True
    return best


def apply_group_thresholds(
    proba: np.ndarray,
    group: pd.Series | np.ndarray,
    thresholds: dict[str, float],
    *,
    default_threshold: float = 0.5,
) -> np.ndarray:
    """Return hard labels using per-group thresholds."""
    p1 = _as_positive_proba(proba)
    group = pd.Series(group).astype(str).reset_index(drop=True)
    pred = np.empty(len(p1), dtype=int)
    for i in range(len(p1)):
        thr = thresholds.get(group.iloc[i], default_threshold)
        pred[i] = int(p1[i] >= thr)
    return pred


def mitigation_summary(
    *,
    name: str,
    utility: dict[str, float],
    fairness: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate before/after row for reports (utility + calibration + disparity)."""
    payload: dict[str, Any] = {
        "variant": name,
        "utility": utility,
        "fairness": fairness,
    }
    if extra:
        payload["extra"] = extra
    return payload
