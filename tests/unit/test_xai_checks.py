"""Explainability stability/faithfulness unit checks — K-04."""

from __future__ import annotations

import numpy as np

from mortgage_threat_intel.explainability.checks import (
    faithfulness_ablation,
    top_k_features,
    top_k_stability,
)
from mortgage_threat_intel.models.baselines import HistGradientBoostingBaseline


def test_top_k_stability_and_faithfulness():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = HistGradientBoostingBaseline().fit(X, y, seed=0)
    # Fake mean |shap|
    mean_abs = np.array([0.5, 0.4, 0.1, 0.05, 0.02, 0.01])
    top = top_k_features(mean_abs, k=2)
    assert top == [0, 1]
    stab = top_k_stability([[0, 1, 2], [0, 1, 3], [1, 0, 4]], k=2)
    assert stab["mean_pairwise_jaccard"] == 1.0
    faith = faithfulness_ablation(model, X, top_feature_indices=top, seed=0, n_random_draws=3)
    assert "mean_abs_score_change_top" in faith
    assert "faithfulness_ratio" in faith
