"""Group fairness metrics on the boosted-tree audit model — Krati K-02."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_positive_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba


def _binary_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "n_positive": 0,
            "n_negative": 0,
            "selection_rate": float("nan"),
            "tpr": float("nan"),
            "fpr": float("nan"),
            "tnr": float("nan"),
            "fnr": float("nan"),
        }
    selection = float(np.mean(y_pred == 1))
    pos = y_true == 1
    neg = ~pos
    tpr = float(np.mean(y_pred[pos] == 1)) if int(pos.sum()) else float("nan")
    fpr = float(np.mean(y_pred[neg] == 1)) if int(neg.sum()) else float("nan")
    tnr = float(np.mean(y_pred[neg] == 0)) if int(neg.sum()) else float("nan")
    fnr = float(np.mean(y_pred[pos] == 0)) if int(pos.sum()) else float("nan")
    return {
        "n": n,
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "selection_rate": selection,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
    }


def demographic_parity_difference(selection_rates: list[float]) -> float:
    """Max−min selection rate across groups with finite rates."""
    vals = [float(v) for v in selection_rates if np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    return float(max(vals) - min(vals))


def equalized_odds_difference(tprs: list[float], fprs: list[float]) -> float:
    """Maximum of (max−min TPR, max−min FPR) across groups with finite rates."""
    tpr_vals = [float(v) for v in tprs if np.isfinite(v)]
    fpr_vals = [float(v) for v in fprs if np.isfinite(v)]
    gaps: list[float] = []
    if len(tpr_vals) >= 2:
        gaps.append(max(tpr_vals) - min(tpr_vals))
    if len(fpr_vals) >= 2:
        gaps.append(max(fpr_vals) - min(fpr_vals))
    if not gaps:
        return float("nan")
    return float(max(gaps))


def _group_labels(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        raise ValueError("group columns required")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing audit columns for grouping: {missing}")
    if len(columns) == 1:
        return frame[columns[0]].astype(str)
    return frame[columns].astype(str).agg("|".join, axis=1)


def _bootstrap_disparity_ci(
    y_true: np.ndarray,
    p1: np.ndarray,
    group: np.ndarray,
    *,
    metric: str,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    max_bootstrap_rows: int = 20_000,
) -> tuple[float, float, float]:
    """Bootstrap CIs for disparity metrics.

    For large validation sets, each replicate draws ``min(n, max_bootstrap_rows)``
    rows with replacement so real-HMDA runs stay CPU-feasible while still
    reporting uncertainty.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    draw_n = min(n, max_bootstrap_rows)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=draw_n)
        y_b = y_true[idx]
        p_b = p1[idx]
        g_b = group[idx]
        pred = (p_b >= threshold).astype(int)
        sels: list[float] = []
        tprs: list[float] = []
        fprs: list[float] = []
        for gval in np.unique(g_b):
            mask = g_b == gval
            rates = _binary_rates(y_b[mask], pred[mask])
            if int(rates["n"]) == 0:
                continue
            sels.append(float(rates["selection_rate"]))
            tprs.append(float(rates["tpr"]))
            fprs.append(float(rates["fpr"]))
        if metric == "demographic_parity":
            stats.append(demographic_parity_difference(sels))
        else:
            stats.append(equalized_odds_difference(tprs, fprs))
    arr = np.asarray([s for s in stats if np.isfinite(s)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def group_fairness_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    group: pd.Series | np.ndarray,
    *,
    threshold: float = 0.5,
    min_cell_count: int = 50,
    n_bootstrap: int = 200,
    seed: int = 42,
    attribute_name: str = "group",
) -> dict[str, Any]:
    """Per-group rates, disparity metrics, bootstrap CIs, underpowered flags."""
    y_true = np.asarray(y_true).astype(int)
    p1 = _as_positive_proba(proba)
    if len(y_true) != len(p1):
        raise ValueError("y_true and proba length mismatch")
    group = pd.Series(group).astype(str).reset_index(drop=True)
    if len(group) != len(y_true):
        raise ValueError("group length mismatch")

    y_pred = (p1 >= threshold).astype(int)
    per_group: dict[str, Any] = {}
    for gval, idx in group.groupby(group).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        rates = _binary_rates(y_true[idx_arr], y_pred[idx_arr])
        n = int(rates["n"])
        underpowered = n < min_cell_count
        per_group[str(gval)] = {
            **rates,
            "underpowered": underpowered,
            "min_cell_count": min_cell_count,
            "power_note": (
                f"n={n} < min_cell_count={min_cell_count}; treat as descriptive only"
                if underpowered
                else None
            ),
        }

    all_sel = [float(g["selection_rate"]) for g in per_group.values() if int(g["n"]) > 0]
    all_tpr = [float(g["tpr"]) for g in per_group.values() if int(g["n"]) > 0]
    all_fpr = [float(g["fpr"]) for g in per_group.values() if int(g["n"]) > 0]
    powered = [g for g in per_group.values() if not g["underpowered"]]
    powered_sel = [float(g["selection_rate"]) for g in powered]
    powered_tpr = [float(g["tpr"]) for g in powered]
    powered_fpr = [float(g["fpr"]) for g in powered]

    dp_ci = _bootstrap_disparity_ci(
        y_true,
        p1,
        group.to_numpy(),
        metric="demographic_parity",
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    eo_ci = _bootstrap_disparity_ci(
        y_true,
        p1,
        group.to_numpy(),
        metric="equalized_odds",
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )

    return {
        "attribute": attribute_name,
        "threshold": threshold,
        "n_total": int(len(y_true)),
        "n_groups": len(per_group),
        "n_underpowered_groups": int(sum(1 for g in per_group.values() if g["underpowered"])),
        "groups": per_group,
        "demographic_parity_difference": demographic_parity_difference(all_sel),
        "demographic_parity_difference_powered_groups": demographic_parity_difference(powered_sel),
        "demographic_parity_difference_ci": {
            "mean": dp_ci[0],
            "low": dp_ci[1],
            "high": dp_ci[2],
        },
        "equalized_odds_difference": equalized_odds_difference(all_tpr, all_fpr),
        "equalized_odds_difference_powered_groups": equalized_odds_difference(
            powered_tpr, powered_fpr
        ),
        "equalized_odds_difference_ci": {
            "mean": eo_ci[0],
            "low": eo_ci[1],
            "high": eo_ci[2],
        },
    }


def audit_attributes_fairness(
    y_true: np.ndarray,
    proba: np.ndarray,
    audit_frame: pd.DataFrame,
    *,
    primary_attrs: list[str],
    intersectional: bool = True,
    min_cell_count: int = 50,
    threshold: float = 0.5,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run primary (and optional intersectional) fairness reports."""
    reports: dict[str, Any] = {"primary": {}, "intersectional": None}
    for attr in primary_attrs:
        if attr not in audit_frame.columns:
            reports["primary"][attr] = {
                "error": f"missing audit column {attr}",
                "underpowered": True,
            }
            continue
        labels = _group_labels(audit_frame, [attr])
        reports["primary"][attr] = group_fairness_report(
            y_true,
            proba,
            labels,
            threshold=threshold,
            min_cell_count=min_cell_count,
            n_bootstrap=n_bootstrap,
            seed=seed,
            attribute_name=attr,
        )

    if intersectional and len(primary_attrs) >= 2:
        present = [a for a in primary_attrs if a in audit_frame.columns]
        if len(present) >= 2:
            labels = _group_labels(audit_frame, present)
            reports["intersectional"] = group_fairness_report(
                y_true,
                proba,
                labels,
                threshold=threshold,
                min_cell_count=min_cell_count,
                n_bootstrap=n_bootstrap,
                seed=seed + 17,
                attribute_name="|".join(present),
            )
            reports["intersectional"]["note"] = (
                "Intersectional slices with n < min_cell_count are labeled underpowered; "
                "do not over-interpret noisy gaps."
            )
    return reports
