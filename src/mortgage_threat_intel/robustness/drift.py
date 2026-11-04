"""Temporal drift evaluation — Krati K-05."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mortgage_threat_intel.evaluation.metrics import utility_metrics
from mortgage_threat_intel.fairness.metrics import audit_attributes_fairness


def feature_drift_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, float]:
    """Mean absolute standardized mean difference across columns (simple drift proxy)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        raise ValueError("reference and current must be 2D with the same n_features")
    if len(ref) == 0 or len(cur) == 0:
        return {
            "mean_abs_smd": float("nan"),
            "max_abs_smd": float("nan"),
            "n_features": int(ref.shape[1]) if ref.ndim == 2 else 0,
        }
    smds: list[float] = []
    for j in range(ref.shape[1]):
        mu0 = float(np.nanmean(ref[:, j]))
        mu1 = float(np.nanmean(cur[:, j]))
        s0 = float(np.nanstd(ref[:, j]))
        s1 = float(np.nanstd(cur[:, j]))
        pooled = np.sqrt(0.5 * (s0**2 + s1**2))
        if pooled <= 1e-12:
            smd = 0.0 if abs(mu1 - mu0) < 1e-12 else float("inf")
        else:
            smd = abs(mu1 - mu0) / pooled
        if np.isfinite(smd):
            smds.append(float(smd))
    if not smds:
        return {
            "mean_abs_smd": float("nan"),
            "max_abs_smd": float("nan"),
            "n_features": int(ref.shape[1]),
        }
    return {
        "mean_abs_smd": float(np.mean(smds)),
        "max_abs_smd": float(np.max(smds)),
        "n_features": int(ref.shape[1]),
    }


def temporal_evaluation(
    *,
    y_ref: np.ndarray,
    p_ref: np.ndarray,
    audit_ref: pd.DataFrame,
    y_new: np.ndarray,
    p_new: np.ndarray,
    audit_new: pd.DataFrame,
    X_ref: np.ndarray,
    X_new: np.ndarray,
    primary_attrs: list[str],
    min_cell_count: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare utility/fairness on a later year vs reference validation; add drift stat."""
    util_ref = utility_metrics(y_ref, p_ref)
    util_new = utility_metrics(y_new, p_new)
    fair_ref = audit_attributes_fairness(
        y_ref,
        p_ref,
        audit_ref,
        primary_attrs=primary_attrs,
        intersectional=False,
        min_cell_count=min_cell_count,
        n_bootstrap=50,
        seed=seed,
    )
    fair_new = audit_attributes_fairness(
        y_new,
        p_new,
        audit_new,
        primary_attrs=primary_attrs,
        intersectional=False,
        min_cell_count=min_cell_count,
        n_bootstrap=50,
        seed=seed + 1,
    )
    drift = feature_drift_statistic(X_ref, X_new)
    deltas = {
        key: (
            float(util_new[key] - util_ref[key])
            if np.isfinite(util_new.get(key, np.nan)) and np.isfinite(util_ref.get(key, np.nan))
            else float("nan")
        )
        for key in ("auroc", "auprc", "balanced_accuracy", "brier_score")
    }
    return {
        "label": "temporal_reliability_shift",
        "interpretation": (
            "Treat as reliability/drift unless evidence supports malicious causation."
        ),
        "utility_reference": util_ref,
        "utility_new": util_new,
        "utility_delta_new_minus_ref": deltas,
        "fairness_reference": fair_ref,
        "fairness_new": fair_new,
        "feature_drift": drift,
    }
