"""Utility and calibration metrics (scaffold helpers)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def utility_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Aggregate utility metrics from positive-class probabilities."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    if proba.ndim == 2:
        p1 = proba[:, 1]
    else:
        p1 = proba
    pred = (p1 >= 0.5).astype(int)
    out: dict[str, float] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier_score": float(brier_score_loss(y_true, p1)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, p1))
        out["auprc"] = float(average_precision_score(y_true, p1))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out
