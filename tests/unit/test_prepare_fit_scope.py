"""Transform fit scope: preprocessor fitted on train members only."""

from __future__ import annotations

import numpy as np

from mortgage_threat_intel.config import default_config_dir, load_base_config
from mortgage_threat_intel.data.pipeline import prepare_synthetic


def test_preprocessor_fitted_on_members_and_2025_absent():
    config = load_base_config(default_config_dir() / "base.yaml")
    prepared = prepare_synthetic(config)
    assert (prepared.split.train["activity_year"] != 2025).all()
    assert (prepared.split.validation["activity_year"] != 2025).all()
    assert 2025 not in set(prepared.mia.members["activity_year"])
    X = prepared.preprocessor.transform(prepared.mia.members)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(prepared.mia.members)
