"""Evidence-based residual-risk update — Krati K-06."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mortgage_threat_intel.threats.register import (
    ThreatRegister,
    find_risk,
    load_threat_register,
    update_residual,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _mia_near_chance(privacy: dict[str, Any] | None) -> bool | None:
    if not privacy:
        return None
    mia = privacy.get("mia") or {}
    aucs = []
    for key in ("mlp_dp_sgd", "mlp_nonprivate"):
        block = mia.get(key) or {}
        if "auc" in block:
            aucs.append(float(block["auc"]))
    if not aucs:
        return None
    return all(abs(a - 0.5) < 0.05 for a in aucs)


def _fairness_gap(fairness: dict[str, Any] | None) -> float | None:
    if not fairness:
        return None
    gaps: list[float] = []
    for variant in fairness.get("variants") or []:
        if variant.get("variant") != "unmitigated":
            continue
        primary = (variant.get("fairness") or {}).get("primary") or {}
        for report in primary.values():
            if not isinstance(report, dict):
                continue
            for key in (
                "demographic_parity_difference",
                "equalized_odds_difference",
            ):
                val = report.get(key)
                if isinstance(val, (int, float)) and val == val:  # not NaN
                    gaps.append(float(val))
    if not gaps:
        return None
    return max(gaps)


def _xai_stability(xai: dict[str, Any] | None) -> float | None:
    if not xai:
        return None
    results = xai.get("results") or {}
    stab = (results.get("stability") or {}).get("mean_pairwise_jaccard")
    return float(stab) if isinstance(stab, (int, float)) else None


def _missingness_auroc_drop(robustness: dict[str, Any] | None) -> float | None:
    if not robustness:
        return None
    delta = (
        (robustness.get("missingness_stress") or {}).get("utility_delta_stressed_minus_clean") or {}
    ).get("auroc")
    return float(delta) if isinstance(delta, (int, float)) else None


def update_register_from_evidence(
    register: ThreatRegister,
    *,
    privacy: dict[str, Any] | None,
    fairness: dict[str, Any] | None,
    xai: dict[str, Any] | None,
    robustness: dict[str, Any] | None,
) -> tuple[ThreatRegister, list[dict[str, Any]]]:
    """Return updated register and a changelog of residual score edits."""
    changelog: list[dict[str, Any]] = []
    updated = register

    # R1 — membership inference
    near = _mia_near_chance(privacy)
    if near is True:
        updated = update_residual(
            updated,
            "R1_membership_inference",
            likelihood=2,
            impact=3,
            rationale=(
                "Measured confidence-score MIA AUC near chance for the matched MLP pair "
                "under the balanced 1:1 prior and black-box score access. Residual likelihood "
                "reduced for this attack model only; not a claim against all attackers. "
                "DP reports accountant-achieved ε/δ with record-level adjacency."
            ),
            evidence_confidence="medium",
            evidence_status="measured",
            notes="Person-level privacy is not claimed from record-level DP.",
        )
        changelog.append(
            {
                "risk_id": "R1_membership_inference",
                "action": "residual_updated",
                "source": "reports/handoff/privacy_latest_privacy.json or outputs/privacy",
            }
        )
    elif near is False:
        updated = update_residual(
            updated,
            "R1_membership_inference",
            likelihood=4,
            impact=3,
            rationale=(
                "Measured MIA AUC materially above chance under the stated attack prior; "
                "residual remains elevated for this offline score-based attacker."
            ),
            evidence_confidence="medium",
            evidence_status="measured",
        )
        changelog.append({"risk_id": "R1_membership_inference", "action": "residual_elevated"})

    # R2 — input integrity from missingness stress
    drop = _missingness_auroc_drop(robustness)
    if drop is not None:
        # Larger absolute AUROC drop → higher residual impact/likelihood retention
        impact = 4 if drop <= -0.05 else 3
        updated = update_residual(
            updated,
            "R2_input_integrity",
            likelihood=3,
            impact=impact,
            rationale=(
                f"Controlled 10% missingness stress produced AUROC delta={drop:.4f} "
                "vs clean validation under fitted imputation. This is a controlled failure "
                "test, not a demonstrated live evasion attack."
            ),
            evidence_confidence="medium",
            evidence_status="measured",
        )
        changelog.append({"risk_id": "R2_input_integrity", "action": "residual_updated"})

    # R3 — poisoning remains untested
    poison = find_risk(updated, "R3_poisoning")
    updated = update_residual(
        updated,
        "R3_poisoning",
        likelihood=poison.residual.likelihood,
        impact=poison.residual.impact,
        rationale=(
            "Empirical poisoning resistance remains untested. Provenance/hash/schema "
            "checks are controls, not evidence of mitigation. Do not mark mitigated."
        ),
        evidence_confidence="low",
        evidence_status="untested",
        notes="Untested in core experiment grid.",
    )
    changelog.append({"risk_id": "R3_poisoning", "action": "confirmed_untested"})

    # R4 — fairness gaps
    gap = _fairness_gap(fairness)
    if gap is not None:
        impact = 4 if gap >= 0.1 else 3
        likelihood = 4 if gap >= 0.05 else 3
        updated = update_residual(
            updated,
            "R4_unequal_error_rates",
            likelihood=likelihood,
            impact=impact,
            rationale=(
                f"Unmitigated HGB audit showed max primary disparity≈{gap:.4f} "
                "(DP/EO differences). Interprets association with historical HMDA labels "
                "only — not discrimination, causation, or legal non-compliance."
            ),
            evidence_confidence="medium",
            evidence_status="measured",
        )
        changelog.append({"risk_id": "R4_unequal_error_rates", "action": "residual_updated"})

    # R5 — temporal: only update if 2025 was used; else keep deferred
    rob_manifest = (robustness or {}).get("manifest") or {}
    if rob_manifest.get("used_2025"):
        updated = update_residual(
            updated,
            "R5_temporal_drift",
            likelihood=3,
            impact=3,
            rationale=(
                "2025 temporal evaluation completed under frozen protocol. Treat metric "
                "shifts as reliability/drift unless malicious causation is evidenced."
            ),
            evidence_confidence="medium",
            evidence_status="measured",
        )
        changelog.append({"risk_id": "R5_temporal_drift", "action": "residual_updated"})
    else:
        updated = update_residual(
            updated,
            "R5_temporal_drift",
            likelihood=find_risk(updated, "R5_temporal_drift").residual.likelihood,
            impact=find_risk(updated, "R5_temporal_drift").residual.impact,
            rationale=(
                "2025 outcomes remain sealed or were not opened; temporal residual not "
                "updated from real holdout. Synthetic path does not change this score."
            ),
            evidence_confidence="low",
            evidence_status="deferred",
        )
        changelog.append({"risk_id": "R5_temporal_drift", "action": "kept_deferred"})

    # R6 — explanation quality
    jaccard = _xai_stability(xai)
    if jaccard is not None:
        likelihood = 2 if jaccard >= 0.6 else 4
        updated = update_residual(
            updated,
            "R6_explanation_quality",
            likelihood=likelihood,
            impact=3,
            rationale=(
                f"TreeSHAP top-k mean pairwise Jaccard≈{jaccard:.3f} across fixed seeds. "
                "Faithfulness uses ablation vs random features. Explanations are "
                "behavioral checks for the research model — not causal or legal reasons."
            ),
            evidence_confidence="medium",
            evidence_status="measured",
        )
        changelog.append({"risk_id": "R6_explanation_quality", "action": "residual_updated"})

    return updated, changelog


def run_residual_update(
    *,
    register_path: Path,
    output_path: Path,
    outputs_root: Path,
    handoff_root: Path,
) -> dict[str, Any]:
    """Load evidence artifacts, update residual scores, write YAML + changelog JSON."""
    register = load_threat_register(register_path)
    privacy = _read_json(handoff_root / "privacy_latest_privacy.json") or _read_json(
        outputs_root / "privacy" / "latest_privacy.json"
    )
    fairness = _read_json(handoff_root / "fairness_latest.json") or _read_json(
        outputs_root / "fairness" / "latest_fairness.json"
    )
    xai = _read_json(handoff_root / "xai_latest.json") or _read_json(
        outputs_root / "xai" / "latest_xai.json"
    )
    robustness = _read_json(handoff_root / "robustness_latest.json") or _read_json(
        outputs_root / "robustness" / "latest_robustness.json"
    )

    updated, changelog = update_register_from_evidence(
        register,
        privacy=privacy,
        fairness=fairness,
        xai=xai,
        robustness=robustness,
    )

    # Annotate monitoring signals reminder on each risk (already present); write YAML.
    payload = updated.model_dump(mode="json")
    payload["residual_update"] = {
        "ticket": "K-06",
        "changelog": changelog,
        "evidence_paths": {
            "privacy": bool(privacy),
            "fairness": bool(fairness),
            "xai": bool(xai),
            "robustness": bool(robustness),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    changelog_path = output_path.with_name("residual_risk_changelog.json")
    with changelog_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregate_only": True,
                "changelog": changelog,
                "register_out": str(output_path),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    return {
        "register_out": str(output_path),
        "changelog_out": str(changelog_path),
        "changelog": changelog,
        "aggregate_only": True,
    }
