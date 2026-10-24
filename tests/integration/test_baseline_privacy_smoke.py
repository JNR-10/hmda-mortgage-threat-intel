"""Synthetic baseline + privacy aggregate smoke (no row-level outputs)."""

from __future__ import annotations

from pathlib import Path

from mortgage_threat_intel.experiments.baseline import run_baselines
from mortgage_threat_intel.experiments.privacy import run_privacy


def test_baseline_and_privacy_aggregate_only(tmp_path: Path):
    base = run_baselines(output_dir=tmp_path / "baselines", use_synthetic=True)
    assert base["aggregate_only"] is True
    assert base["row_level_outputs"] is False
    assert base["manifest"]["used_2025"] is False
    assert "logistic_regression" in base["metrics"]
    assert "hist_gradient_boosting" in base["metrics"]

    priv = run_privacy(output_dir=tmp_path / "privacy", use_synthetic=True)
    assert priv["aggregate_only"] is True
    assert priv["row_level_outputs"] is False
    assert priv["manifest"]["used_2025_as_nonmember"] is False
    meta = priv["dp_privacy_metadata"]
    assert "achieved_epsilon" in meta
    assert "delta" in meta
    assert meta["target_epsilon"] == 4.0
    assert meta["achieved_epsilon"] != meta.get("PLACEHOLDER", None)
    assert priv["mia"]["mlp_nonprivate"]["evaluation_prior"] == "balanced_1_to_1"
    assert priv["mia"]["mlp_dp_sgd"]["member_nonmember_disjoint"] is True
