"""Config validation for base and privacy YAML."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mortgage_threat_intel.config import (
    SamplingConfig,
    default_config_dir,
    load_base_config,
    load_fairness_config,
    load_privacy_config,
)


def test_load_base_and_privacy_configs():
    base = load_base_config(default_config_dir() / "base.yaml")
    assert base.sampling.max_rows_per_year == 100_000
    assert base.years.test == 2025
    assert base.years.seal_test_outcomes is True
    assert "loan_amount" in base.features.allowlist
    assert "action_taken" in base.features.forbidden

    privacy = load_privacy_config(default_config_dir() / "privacy.yaml")
    assert privacy.dp_sgd.target_epsilon == 4.0
    assert privacy.dp_sgd.adjacency == "record_level"
    assert privacy.mia.attack == "confidence_score"

    fairness = load_fairness_config(default_config_dir() / "fairness.yaml")
    assert fairness.audit_model == "hist_gradient_boosting"
    assert fairness.mitigation.reweighting.fit_on == "train_2023"


def test_invalid_sample_cap_rejected():
    with pytest.raises(ValidationError):
        SamplingConfig(max_rows_per_year=0)
