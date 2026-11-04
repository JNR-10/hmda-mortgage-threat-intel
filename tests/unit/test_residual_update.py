"""K-06 residual update keeps poisoning untested and uses aggregates only."""

from __future__ import annotations

from pathlib import Path

from mortgage_threat_intel.threats.register import find_risk, load_threat_register
from mortgage_threat_intel.threats.residual import update_register_from_evidence

REPO = Path(__file__).resolve().parents[2]


def test_residual_update_poisoning_stays_untested():
    register = load_threat_register(REPO / "reports" / "threat_register.yaml")
    privacy = {
        "mia": {
            "mlp_dp_sgd": {"auc": 0.50},
            "mlp_nonprivate": {"auc": 0.50},
        }
    }
    fairness = {
        "variants": [
            {
                "variant": "unmitigated",
                "fairness": {
                    "primary": {
                        "applicant_sex": {
                            "demographic_parity_difference": 0.12,
                            "equalized_odds_difference": 0.08,
                        }
                    }
                },
            }
        ]
    }
    xai = {"results": {"stability": {"mean_pairwise_jaccard": 0.75}}}
    robustness = {
        "manifest": {"used_2025": False},
        "missingness_stress": {"utility_delta_stressed_minus_clean": {"auroc": -0.02}},
    }
    updated, changelog = update_register_from_evidence(
        register,
        privacy=privacy,
        fairness=fairness,
        xai=xai,
        robustness=robustness,
    )
    poison = find_risk(updated, "R3_poisoning")
    assert poison.evidence_status == "untested"
    assert any(c["risk_id"] == "R3_poisoning" for c in changelog)
    assert find_risk(updated, "R5_temporal_drift").evidence_status == "deferred"
    assert find_risk(updated, "R1_membership_inference").evidence_status == "measured"
