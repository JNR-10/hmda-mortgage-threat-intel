"""Explanation stability and faithfulness checks — Krati K-04."""

from __future__ import annotations

from typing import Any

import numpy as np

from mortgage_threat_intel.explainability.attribution import (
    select_background,
    select_explain_rows,
    tree_shap_values,
)
from mortgage_threat_intel.models.baselines import HistGradientBoostingBaseline


def top_k_features(mean_abs: np.ndarray, k: int = 5) -> list[int]:
    order = np.argsort(-np.asarray(mean_abs, dtype=float))
    return [int(i) for i in order[:k]]


def top_k_stability(
    rankings: list[list[int]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Jaccard stability of top-k feature index sets across seeds."""
    if not rankings:
        return {"k": k, "mean_pairwise_jaccard": float("nan"), "n_seeds": 0}
    sets = [set(r[:k]) for r in rankings]
    pairs: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            pairs.append(float(len(inter) / len(union)) if union else 1.0)
    return {
        "k": k,
        "n_seeds": len(rankings),
        "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else 1.0,
        "pairwise_jaccard": pairs,
        "top_k_by_seed": [list(map(int, r[:k])) for r in rankings],
    }


def faithfulness_ablation(
    model: HistGradientBoostingBaseline,
    X: np.ndarray,
    *,
    top_feature_indices: list[int],
    seed: int = 42,
    n_random_draws: int = 5,
) -> dict[str, Any]:
    """Compare score change after zeroing top-attributed features vs random features.

    Uses column-mean imputation (set to feature mean) as a bounded removal proxy
    compatible with already-transformed numeric matrices.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return {
            "mean_abs_score_change_top": float("nan"),
            "mean_abs_score_change_random": float("nan"),
            "faithfulness_ratio": float("nan"),
        }
    base = model.predict_proba(X)[:, 1]
    means = np.nanmean(X, axis=0)

    X_top = X.copy()
    for j in top_feature_indices:
        X_top[:, j] = means[j]
    top_change = float(np.mean(np.abs(base - model.predict_proba(X_top)[:, 1])))

    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    k = len(top_feature_indices)
    random_changes: list[float] = []
    for _ in range(n_random_draws):
        rand_idx = rng.choice(n_features, size=min(k, n_features), replace=False)
        X_rand = X.copy()
        for j in rand_idx:
            X_rand[:, j] = means[j]
        random_changes.append(float(np.mean(np.abs(base - model.predict_proba(X_rand)[:, 1]))))
    rand_mean = float(np.mean(random_changes)) if random_changes else float("nan")
    ratio = (
        float(top_change / rand_mean)
        if rand_mean and np.isfinite(rand_mean) and rand_mean > 0
        else float("nan")
    )
    return {
        "mean_abs_score_change_top": top_change,
        "mean_abs_score_change_random": rand_mean,
        "faithfulness_ratio": ratio,
        "k": k,
        "n_random_draws": n_random_draws,
        "note": (
            "Higher faithfulness_ratio means top-SHAP features change scores more "
            "than random features under mean-imputation ablation (behavioral check, "
            "not causal)."
        ),
    }


def multi_seed_shap_stability(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
    seeds: list[int],
    background_max: int = 100,
    explain_max: int = 1000,
    top_k: int = 5,
) -> dict[str, Any]:
    """Fit HGB per seed, compute TreeSHAP aggregates, stability, and faithfulness."""
    rankings: list[list[int]] = []
    global_tables: list[list[dict[str, Any]]] = []
    faithfulness_runs: list[dict[str, Any]] = []
    local_example: dict[str, Any] | None = None

    X_explain, _ = select_explain_rows(X_eval, max_rows=explain_max, seed=seeds[0])
    background = select_background(X_train, max_rows=background_max, seed=seeds[0])

    for seed in seeds:
        model = HistGradientBoostingBaseline().fit(X_train, y_train, seed=seed)
        shap_out = tree_shap_values(
            model, X_explain, background=background, feature_names=feature_names
        )
        mean_abs = shap_out["global_mean_abs"]
        top_idx = top_k_features(mean_abs, k=top_k)
        rankings.append(top_idx)
        global_tables.append(shap_out["global_ranking"])
        faith = faithfulness_ablation(model, X_explain, top_feature_indices=top_idx, seed=seed)
        faithfulness_runs.append(faith)
        if local_example is None and shap_out["shap_values"].shape[0] > 0:
            from mortgage_threat_intel.explainability.attribution import (
                synthetic_local_illustration,
            )

            local_example = synthetic_local_illustration(
                shap_out["shap_values"][0],
                feature_names,
                expected_value=shap_out["expected_value"],
            )

    stability = top_k_stability(rankings, k=top_k)
    return {
        "seeds": list(seeds),
        "background_rows": int(len(background)),
        "explained_rows": int(len(X_explain)),
        "top_k": top_k,
        "stability": stability,
        "faithfulness_by_seed": faithfulness_runs,
        "faithfulness_mean_ratio": (
            float(
                np.nanmean(
                    [
                        f["faithfulness_ratio"]
                        for f in faithfulness_runs
                        if np.isfinite(f.get("faithfulness_ratio", np.nan))
                    ]
                )
            )
            if any(np.isfinite(f.get("faithfulness_ratio", np.nan)) for f in faithfulness_runs)
            else float("nan")
        ),
        "global_ranking_seed0": global_tables[0] if global_tables else [],
        "synthetic_local_illustration": local_example,
        "aggregate_only": True,
        "row_level_outputs": False,
    }
