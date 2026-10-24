"""MIA member/non-member disjointness and recorded class prior."""

from __future__ import annotations

import numpy as np

from mortgage_threat_intel.config import default_config_dir, load_base_config, load_privacy_config
from mortgage_threat_intel.data.pipeline import prepare_synthetic
from mortgage_threat_intel.models.baselines import LogisticBaseline
from mortgage_threat_intel.privacy.membership import run_confidence_mia


def test_mia_disjoint_and_prior_recorded():
    base = load_base_config(default_config_dir() / "base.yaml")
    privacy = load_privacy_config(default_config_dir() / "privacy.yaml")
    prepared = prepare_synthetic(base)
    assert prepared.mia.member_ids.isdisjoint(prepared.mia.non_member_ids)

    model = LogisticBaseline().fit(prepared.X_members, prepared.y_members, seed=base.seed)
    X_non = prepared.preprocessor.transform(prepared.mia.non_members)
    result = run_confidence_mia(
        model,
        prepared.X_members,
        X_non,
        member_ids=prepared.mia.member_ids,
        non_member_ids=prepared.mia.non_member_ids,
        seed=base.seed,
        n_bootstrap=20,
        evaluation_prior=privacy.mia.evaluation_prior,
    )
    assert result.member_nonmember_disjoint is True
    assert result.evaluation_prior == "balanced_1_to_1"
    assert result.n_members_eval == result.n_non_members_eval
    assert result.n_members_eval > 0
    assert 0.0 <= result.auc <= 1.0 or np.isnan(result.auc)
