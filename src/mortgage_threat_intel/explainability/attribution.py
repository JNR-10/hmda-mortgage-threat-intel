"""TreeSHAP attribution for the boosted-tree audit model — Krati K-04."""

from __future__ import annotations

from typing import Any

import numpy as np

from mortgage_threat_intel.models.baselines import HistGradientBoostingBaseline


def _require_shap():
    try:
        import shap  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'shap' package is required for TreeSHAP (K-04). "
            "Install with: pip install 'shap>=0.44'"
        ) from exc
    import shap

    return shap


def _underlying_sklearn_model(model: HistGradientBoostingBaseline):
    return model._model  # noqa: SLF001 — adapter exposes sklearn estimator for TreeExplainer


def tree_shap_values(
    model: HistGradientBoostingBaseline,
    X: np.ndarray,
    *,
    background: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute TreeSHAP values; return aggregate-safe arrays (no row IDs)."""
    shap = _require_shap()
    est = _underlying_sklearn_model(model)
    explainer = shap.TreeExplainer(est, data=background)
    values = explainer.shap_values(X)
    # Binary classifiers may return list [class0, class1] or 2D array for positive class.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        # (n, features, classes) → positive class
        values = values[:, :, -1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected_val = float(np.asarray(expected).reshape(-1)[-1])
    else:
        expected_val = float(expected)

    mean_abs = np.mean(np.abs(values), axis=0)
    n_features = int(mean_abs.shape[0])
    names = list(feature_names) if feature_names else [f"f{i}" for i in range(n_features)]
    if len(names) < n_features:
        names = names + [f"f{i}" for i in range(len(names), n_features)]
    elif len(names) > n_features:
        names = names[:n_features]
    order = np.argsort(-mean_abs)
    global_ranking = [
        {
            "feature": names[i],
            "mean_abs_shap": float(mean_abs[i]),
            "rank": int(rank),
        }
        for rank, i in enumerate(order)
    ]
    return {
        "shap_values": values,
        "expected_value": expected_val,
        "global_mean_abs": mean_abs,
        "global_ranking": global_ranking,
        "n_explained": int(values.shape[0]),
        "n_features": n_features,
        "feature_names": names,
    }


def select_background(
    X: np.ndarray,
    *,
    max_rows: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Fixed-size background sample for TreeExplainer."""
    X = np.asarray(X)
    n = len(X)
    if n <= max_rows:
        return X.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx]


def select_explain_rows(
    X: np.ndarray,
    *,
    max_rows: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample rows to explain; return (X_sub, indices)."""
    X = np.asarray(X)
    n = len(X)
    if n <= max_rows:
        return X.copy(), np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx], idx


def synthetic_local_illustration(
    shap_row: np.ndarray,
    feature_names: list[str],
    *,
    expected_value: float,
    label: str = "SYNTHETIC_HMDA_LIKE_RECORD",
) -> dict[str, Any]:
    """Aggregate-safe local waterfall payload for a clearly labeled synthetic row."""
    order = np.argsort(-np.abs(shap_row))
    contributions = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "shap": float(shap_row[i]),
        }
        for i in order
    ]
    return {
        "record_label": label,
        "note": (
            "Synthetic HMDA-like illustration only — not a real applicant; "
            "not advice; not a legal adverse-action notice."
        ),
        "expected_value": float(expected_value),
        "contributions": contributions,
    }
