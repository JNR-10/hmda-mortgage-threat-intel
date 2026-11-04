"""Fairness metrics and mitigation — Krati K-02/K-03."""

from mortgage_threat_intel.fairness.metrics import (
    audit_attributes_fairness,
    demographic_parity_difference,
    equalized_odds_difference,
    group_fairness_report,
)
from mortgage_threat_intel.fairness.mitigation import (
    apply_group_thresholds,
    fit_equalized_odds_thresholds,
    reweight_by_group_label,
)

__all__ = [
    "apply_group_thresholds",
    "audit_attributes_fairness",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "fit_equalized_odds_thresholds",
    "group_fairness_report",
    "reweight_by_group_label",
]
