"""Synthetic fairness run produces aggregate-only artifacts (K-02/K-03)."""

from __future__ import annotations

from pathlib import Path

from mortgage_threat_intel.experiments.fairness import run_fairness


def test_fairness_synthetic_aggregate_only(tmp_path: Path):
    payload = run_fairness(
        output_dir=tmp_path / "fairness",
        use_synthetic=True,
        include_mitigations=True,
    )
    assert payload["aggregate_only"] is True
    assert payload["row_level_outputs"] is False
    assert payload["manifest"]["used_2025"] is False
    assert payload["manifest"]["protected_in_X"] is False
    names = [v["variant"] for v in payload["variants"]]
    assert names == ["unmitigated", "reweighting", "equalized_odds_postprocess"]
    for variant in payload["variants"]:
        assert "utility" in variant
        assert "fairness" in variant
        assert "auroc" in variant["utility"]
        assert "brier_score" in variant["utility"]
