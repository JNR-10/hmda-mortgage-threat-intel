"""Forbidden and protected columns must never enter model X."""

from __future__ import annotations

import pandas as pd
import pytest

from mortgage_threat_intel.config import default_config_dir, load_base_config
from mortgage_threat_intel.data.contracts import assert_no_forbidden_in_features
from mortgage_threat_intel.data.prepare import build_model_frame


def test_assert_rejects_forbidden_and_protected():
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_forbidden_in_features(
            ["loan_amount", "action_taken"],
            forbidden=["action_taken"],
            protected=["applicant_sex"],
        )
    with pytest.raises(ValueError, match="Protected"):
        assert_no_forbidden_in_features(
            ["loan_amount", "applicant_sex"],
            forbidden=["action_taken"],
            protected=["applicant_sex"],
        )


def test_build_model_frame_uses_allowlist_only():
    config = load_base_config(default_config_dir() / "base.yaml")
    df = pd.read_csv("tests/fixtures/synthetic_hmda.csv")
    X = build_model_frame(df, config)
    for col in config.features.forbidden:
        assert col not in X.columns
    for col in config.features.protected_audit_only:
        assert col not in X.columns
    assert list(X.columns) == list(config.features.allowlist)
