"""Explainability (TreeSHAP) — Krati K-04."""

from mortgage_threat_intel.explainability.attribution import (
    select_background,
    select_explain_rows,
    synthetic_local_illustration,
    tree_shap_values,
)
from mortgage_threat_intel.explainability.checks import (
    faithfulness_ablation,
    multi_seed_shap_stability,
    top_k_stability,
)

__all__ = [
    "faithfulness_ablation",
    "multi_seed_shap_stability",
    "select_background",
    "select_explain_rows",
    "synthetic_local_illustration",
    "top_k_stability",
    "tree_shap_values",
]
