"""Synthetic XAI + robustness smoke (2025 sealed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mortgage_threat_intel.experiments.robustness import run_robustness
from mortgage_threat_intel.experiments.xai import run_xai


def test_xai_synthetic_aggregate(tmp_path: Path):
    pytest.importorskip("shap")
    payload = run_xai(output_dir=tmp_path / "xai", use_synthetic=True, explain_max=40)
    assert payload["aggregate_only"] is True
    assert payload["row_level_outputs"] is False
    assert payload["manifest"]["used_2025"] is False
    assert "stability" in payload["results"]
    illust = payload["results"]["synthetic_local_illustration"]
    assert illust is not None
    assert "SYNTHETIC" in illust["record_label"]


def test_robustness_synthetic_keeps_2025_sealed(tmp_path: Path):
    payload = run_robustness(
        output_dir=tmp_path / "robustness",
        use_synthetic=True,
        open_2025=False,
    )
    assert payload["manifest"]["used_2025"] is False
    assert payload["temporal"].get("2025_sealed") is True
    assert payload["missingness_stress"]["fraction"] == 0.10
    assert payload["aggregate_only"] is True
