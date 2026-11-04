"""Fairness metric edge cases — K-02."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mortgage_threat_intel.config import default_config_dir, load_fairness_config
from mortgage_threat_intel.fairness.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    group_fairness_report,
)
from mortgage_threat_intel.fairness.mitigation import (
    apply_group_thresholds,
    fit_equalized_odds_thresholds,
    reweight_by_group_label,
)


def test_load_fairness_config():
    cfg = load_fairness_config(default_config_dir() / "fairness.yaml")
    assert cfg.audit_model == "hist_gradient_boosting"
    assert cfg.groups.min_cell_count == 50
    assert "applicant_sex" in cfg.groups.primary


def test_demographic_parity_and_eo_basic():
    assert demographic_parity_difference([0.1, 0.4, 0.2]) == pytest.approx(0.3)
    assert equalized_odds_difference([0.8, 0.5], [0.1, 0.2]) == pytest.approx(0.3)
    assert np.isnan(demographic_parity_difference([0.5]))
    assert np.isnan(equalized_odds_difference([0.5], [0.2]))


def test_small_and_missing_groups_labeled_underpowered():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    group = pd.Series(["A", "A", "A", "A", "A", "A", "B", "B"])  # B has n=2
    report = group_fairness_report(y, p, group, min_cell_count=5, n_bootstrap=20, seed=0)
    assert report["groups"]["B"]["underpowered"] is True
    assert report["groups"]["A"]["underpowered"] is False
    assert report["n_underpowered_groups"] == 1
    assert "demographic_parity_difference" in report
    assert "equalized_odds_difference_ci" in report


def test_empty_group_rates_nan_safe():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    group = pd.Series(["only", "only", "only", "only"])
    report = group_fairness_report(y, p, group, min_cell_count=10, n_bootstrap=10)
    assert report["groups"]["only"]["underpowered"] is True
    assert np.isnan(report["demographic_parity_difference"])


def test_reweighting_mean_one_and_eo_thresholds():
    y = np.array([1, 1, 0, 0, 1, 0])
    group = pd.Series(["F", "F", "F", "M", "M", "M"])
    w = reweight_by_group_label(y, group)
    assert len(w) == 6
    assert float(np.mean(w)) == pytest.approx(1.0, rel=1e-6)

    p = np.array([0.9, 0.8, 0.2, 0.7, 0.6, 0.1])
    thr = fit_equalized_odds_thresholds(y, p, group)
    assert set(thr) == {"F", "M"}
    pred = apply_group_thresholds(p, group, thr)
    assert pred.shape == (6,)
    assert set(np.unique(pred)).issubset({0, 1})
