"""Input missingness stress test — Krati K-05."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mortgage_threat_intel.data.prepare import NUMERIC_CANDIDATES
from mortgage_threat_intel.evaluation.metrics import utility_metrics
from mortgage_threat_intel.fairness.metrics import audit_attributes_fairness


def select_top_non_sensitive_features(
    importance: np.ndarray | list[float],
    feature_names: list[str],
    *,
    allowlist: list[str],
    protected: list[str],
    k: int = 5,
) -> list[str]:
    """Pick top-k allowlisted, non-protected raw feature names by importance.

    ``importance`` is aligned to transformed ``feature_names`` (post one-hot).
    Importance is aggregated onto original allowlist columns by name matching.
    """
    protected_set = set(protected)
    scores: dict[str, float] = {c: 0.0 for c in allowlist if c not in protected_set}
    for name, imp in zip(feature_names, importance, strict=False):
        matched = None
        for raw in scores:
            if (
                name == raw
                or name.endswith(f"__{raw}")
                or f"__{raw}_" in f"{name}_"
                or name.startswith(f"{raw}_")
            ):
                matched = raw
                break
        if matched is not None:
            scores[matched] += float(abs(imp))
    ordered = sorted(scores.keys(), key=lambda c: (-scores[c], c))
    return ordered[:k]


def mask_cells_as_missing(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Set a fraction of cells among ``columns`` to missing on a copy."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Missingness column not in frame: {col}")
        n = len(out)
        n_mask = int(round(fraction * n))
        if n_mask <= 0:
            continue
        idx = rng.choice(n, size=min(n_mask, n), replace=False)
        # Numeric and categorical both become missing; preprocessor imputes.
        _ = NUMERIC_CANDIDATES  # documented path: numeric vs cat both set to NA
        out.loc[out.index[idx], col] = np.nan
    return out


def missingness_stress_eval(
    *,
    model: Any,
    preprocessor: Any,
    eval_frame: pd.DataFrame,
    y_true: np.ndarray,
    audit_frame: pd.DataFrame,
    stress_columns: list[str],
    primary_attrs: list[str],
    fraction: float = 0.10,
    seed: int = 42,
    min_cell_count: int = 50,
) -> dict[str, Any]:
    """Apply missingness, reuse fitted preprocessor, compare utility/fairness."""
    X_clean = preprocessor.transform(eval_frame)
    p_clean = model.predict_proba(X_clean)[:, 1]
    util_clean = utility_metrics(y_true, p_clean)
    fair_clean = audit_attributes_fairness(
        y_true,
        p_clean,
        audit_frame,
        primary_attrs=primary_attrs,
        intersectional=False,
        min_cell_count=min_cell_count,
        n_bootstrap=40,
        seed=seed,
    )

    stressed = mask_cells_as_missing(eval_frame, stress_columns, fraction=fraction, seed=seed)
    X_stress = preprocessor.transform(stressed)
    p_stress = model.predict_proba(X_stress)[:, 1]
    util_stress = utility_metrics(y_true, p_stress)
    fair_stress = audit_attributes_fairness(
        y_true,
        p_stress,
        audit_frame,
        primary_attrs=primary_attrs,
        intersectional=False,
        min_cell_count=min_cell_count,
        n_bootstrap=40,
        seed=seed + 2,
    )
    deltas = {
        key: (
            float(util_stress[key] - util_clean[key])
            if np.isfinite(util_stress.get(key, np.nan))
            and np.isfinite(util_clean.get(key, np.nan))
            else float("nan")
        )
        for key in ("auroc", "auprc", "balanced_accuracy", "brier_score")
    }
    return {
        "label": "controlled_input_missingness_failure",
        "interpretation": (
            "Controlled input-failure test on an experimental copy — not evidence "
            "of a demonstrated live evasion attack."
        ),
        "fraction": fraction,
        "stress_columns": list(stress_columns),
        "utility_clean": util_clean,
        "utility_stressed": util_stress,
        "utility_delta_stressed_minus_clean": deltas,
        "fairness_clean": fair_clean,
        "fairness_stressed": fair_stress,
    }
