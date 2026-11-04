"""Robustness helpers — K-05."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mortgage_threat_intel.robustness.drift import feature_drift_statistic
from mortgage_threat_intel.robustness.perturbations import (
    mask_cells_as_missing,
    select_top_non_sensitive_features,
)


def test_feature_drift_statistic():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(100, 3))
    cur = ref + 1.0
    out = feature_drift_statistic(ref, cur)
    assert out["mean_abs_smd"] > 0
    assert out["n_features"] == 3


def test_mask_and_top_features():
    df = pd.DataFrame(
        {
            "loan_amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "income": [10.0] * 10,
            "applicant_sex": [1] * 10,
        }
    )
    masked = mask_cells_as_missing(df, ["loan_amount"], fraction=0.2, seed=0)
    assert masked["loan_amount"].isna().sum() == 2

    names = select_top_non_sensitive_features(
        [0.9, 0.1, 0.05],
        ["loan_amount", "income", "cat__loan_type_1"],
        allowlist=["loan_amount", "income", "loan_type"],
        protected=["applicant_sex"],
        k=2,
    )
    assert "loan_amount" in names
    assert "applicant_sex" not in names

    with pytest.raises(ValueError):
        mask_cells_as_missing(df, ["loan_amount"], fraction=0.0)
