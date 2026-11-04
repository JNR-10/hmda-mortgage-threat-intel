"""K-01 threat-register schema and committed artifact checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mortgage_threat_intel.threats import (
    RiskScores,
    default_scales,
    find_risk,
    load_threat_register,
    score_lx_i,
    update_residual,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTER_PATH = REPO_ROOT / "reports" / "threat_register.yaml"
CONTEXT_PATH = REPO_ROOT / "reports" / "system_context.md"
EVIDENCE_PATH = REPO_ROOT / "reports" / "evidence_inventory.csv"


def test_score_lx_i_bounds():
    assert score_lx_i(4, 3) == 12
    with pytest.raises(ValueError):
        score_lx_i(0, 3)
    with pytest.raises(ValueError):
        score_lx_i(3, 6)


def test_risk_scores_fill_and_validate_product():
    scores = RiskScores(likelihood=2, impact=4, rationale="test")
    assert scores.product == 8
    with pytest.raises(ValidationError):
        RiskScores(likelihood=2, impact=4, product=9, rationale="bad product")


def test_default_scales_cover_1_to_5():
    scales = default_scales()
    assert [a.score for a in scales.likelihood] == [1, 2, 3, 4, 5]
    assert [a.score for a in scales.impact] == [1, 2, 3, 4, 5]


def test_committed_k01_artifacts_exist():
    assert REGISTER_PATH.is_file(), "reports/threat_register.yaml missing"
    assert CONTEXT_PATH.is_file(), "reports/system_context.md missing"
    assert EVIDENCE_PATH.is_file(), "reports/evidence_inventory.csv missing"
    text = CONTEXT_PATH.read_text(encoding="utf-8").lower()
    assert "desktop underwriter" in text
    assert "proxy" in text
    assert "not" in text


def test_load_committed_threat_register():
    register = load_threat_register(REGISTER_PATH)
    assert register.project == "mortgage-threat-intel"
    assert "HMDA" in register.proxy_boundary or "hmda" in register.proxy_boundary.lower()
    ids = [r.risk_id for r in register.risks]
    for required in register.required_risk_ids:
        assert required in ids

    mia = find_risk(register, "R1_membership_inference")
    assert mia.evidence_status == "measured"
    assert any(ref.kind == "project_observation" for ref in mia.evidence_refs)

    poison = find_risk(register, "R3_poisoning")
    assert poison.evidence_status == "untested"
    assert any(ref.kind == "untested" for ref in poison.evidence_refs)

    drift = find_risk(register, "R5_temporal_drift")
    assert drift.evidence_status == "deferred"


def test_update_residual_helper():
    register = load_threat_register(REGISTER_PATH)
    updated = update_residual(
        register,
        "R4_unequal_error_rates",
        likelihood=3,
        impact=3,
        rationale="synthetic residual update for unit test",
        evidence_confidence="medium",
        evidence_status="measured",
        notes="test-only",
    )
    risk = find_risk(updated, "R4_unequal_error_rates")
    assert risk.residual.product == 9
    assert risk.evidence_status == "measured"
    # Original object unchanged
    assert find_risk(register, "R4_unequal_error_rates").residual.product == 16
