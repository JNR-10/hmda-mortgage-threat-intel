"""Sampling load columns must not pull unused forbidden fields into memory."""

from mortgage_threat_intel.config import load_base_config
from mortgage_threat_intel.data.pipeline import _feature_load_cols


def test_feature_load_cols_excludes_denial_and_lei() -> None:
    cfg = load_base_config("configs/base.yaml")
    cols = set(_feature_load_cols(cfg))
    assert cfg.target.column in cols
    assert "activity_year" in cols
    for c in cfg.features.allowlist:
        assert c in cols
    for c in cfg.features.protected_audit_only:
        assert c in cols
    assert "denial_reason_1" not in cols
    assert "lei" not in cols
