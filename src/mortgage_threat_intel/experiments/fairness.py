"""Fairness audit + two mitigations on the HGB audit model (K-02/K-03)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mortgage_threat_intel.config import (
    BaseConfig,
    FairnessConfig,
    default_config_dir,
    load_base_config,
    load_fairness_config,
)
from mortgage_threat_intel.data.pipeline import prepare_hmda, prepare_synthetic
from mortgage_threat_intel.data.splits import assert_no_test_year_in_fit
from mortgage_threat_intel.evaluation.artifacts import new_run_manifest, write_json
from mortgage_threat_intel.evaluation.metrics import utility_metrics
from mortgage_threat_intel.fairness.metrics import audit_attributes_fairness, group_fairness_report
from mortgage_threat_intel.fairness.mitigation import (
    apply_group_thresholds,
    fit_equalized_odds_thresholds,
    mitigation_summary,
    reweight_by_group_label,
)
from mortgage_threat_intel.models.baselines import HistGradientBoostingBaseline
from mortgage_threat_intel.seeding import seed_everything


def _positive_proba(model: HistGradientBoostingBaseline, X: object) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.asarray(proba)[:, 1]


def _utility_from_scores(y_true: np.ndarray, p1: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Utility with an explicit hard-label vector (for threshold post-processing)."""
    base = utility_metrics(y_true, p1)
    # Replace balanced accuracy with labels from the mitigation thresholds.
    from sklearn.metrics import balanced_accuracy_score

    base["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    return base


def _mitigation_group(audit_frame: pd.DataFrame, fairness: FairnessConfig) -> pd.Series:
    attr = fairness.mitigation_group_attribute
    if attr not in audit_frame.columns:
        # Fall back to first configured primary attribute present in the frame.
        for cand in fairness.groups.primary:
            if cand in audit_frame.columns:
                attr = cand
                break
        else:
            raise KeyError(
                f"mitigation_group_attribute {fairness.mitigation_group_attribute!r} "
                "not in audit frame"
            )
    return audit_frame[attr].astype(str)


def run_fairness(
    *,
    base_config: BaseConfig | None = None,
    fairness_config: FairnessConfig | None = None,
    output_dir: Path | None = None,
    use_synthetic: bool = True,
    include_mitigations: bool = True,
) -> dict[str, Any]:
    """Evaluate unmitigated HGB fairness; optionally reweighting + EO post-process."""
    base_config = base_config or load_base_config(default_config_dir() / "base.yaml")
    fairness_config = fairness_config or load_fairness_config(
        default_config_dir() / "fairness.yaml"
    )
    if fairness_config.audit_model != "hist_gradient_boosting":
        raise ValueError(
            f"Fairness audit_model must be hist_gradient_boosting; "
            f"got {fairness_config.audit_model!r}"
        )

    seed_everything(base_config.seed)
    prepared = prepare_synthetic(base_config) if use_synthetic else prepare_hmda(base_config)
    assert_no_test_year_in_fit(
        [prepared.split.train, prepared.split.validation],
        test_year=base_config.years.test,
    )

    # Confirm protected attrs are not in model feature names.
    protected = set(base_config.features.protected_audit_only)
    leaked = protected.intersection(prepared.feature_names)
    if leaked:
        raise RuntimeError(f"Protected attributes leaked into model features: {sorted(leaked)}")

    hgb = HistGradientBoostingBaseline().fit(
        prepared.X_members, prepared.y_members, seed=base_config.seed
    )
    p_val = _positive_proba(hgb, prepared.X_validation)
    y_val = np.asarray(prepared.y_validation).astype(int)
    audit_val = prepared.audit_frame_validation.reset_index(drop=True)

    unmitigated_fairness = audit_attributes_fairness(
        y_val,
        p_val,
        audit_val,
        primary_attrs=list(fairness_config.groups.primary),
        intersectional=fairness_config.groups.intersectional,
        min_cell_count=fairness_config.groups.min_cell_count,
        threshold=fairness_config.decision_threshold,
        n_bootstrap=fairness_config.n_bootstrap,
        seed=base_config.seed,
    )
    unmitigated_utility = utility_metrics(y_val, p_val)

    variants: list[dict[str, Any]] = [
        mitigation_summary(
            name="unmitigated",
            utility=unmitigated_utility,
            fairness=unmitigated_fairness,
            extra={"decision_threshold": fairness_config.decision_threshold},
        )
    ]

    if include_mitigations:
        # --- Reweighting (fit weights on 2023 members only) ---
        if fairness_config.mitigation.reweighting.fit_on != "train_2023":
            raise ValueError("reweighting.fit_on must be train_2023 per T-01")
        train_group = _mitigation_group(prepared.audit_frame_members, fairness_config)
        weights = reweight_by_group_label(prepared.y_members, train_group)
        hgb_rw = HistGradientBoostingBaseline().fit(
            prepared.X_members,
            prepared.y_members,
            seed=base_config.seed,
            sample_weight=weights,
        )
        p_rw = _positive_proba(hgb_rw, prepared.X_validation)
        rw_fairness = audit_attributes_fairness(
            y_val,
            p_rw,
            audit_val,
            primary_attrs=list(fairness_config.groups.primary),
            intersectional=fairness_config.groups.intersectional,
            min_cell_count=fairness_config.groups.min_cell_count,
            threshold=fairness_config.decision_threshold,
            n_bootstrap=fairness_config.n_bootstrap,
            seed=base_config.seed + 3,
        )
        variants.append(
            mitigation_summary(
                name="reweighting",
                utility=utility_metrics(y_val, p_rw),
                fairness=rw_fairness,
                extra={
                    "fit_on": "train_2023_members",
                    "mitigation_group_attribute": fairness_config.mitigation_group_attribute,
                    "weight_mean": float(np.mean(weights)),
                    "weight_std": float(np.std(weights)),
                },
            )
        )

        # --- Equalized-odds post-processing (thresholds on 2024 only) ---
        if fairness_config.mitigation.equalized_odds_postprocess.select_thresholds_on != (
            "validation_2024"
        ):
            raise ValueError("EO thresholds must be selected on validation_2024")
        val_group = _mitigation_group(audit_val, fairness_config)
        thresholds = fit_equalized_odds_thresholds(y_val, p_val, val_group, seed=base_config.seed)
        pred_eo = apply_group_thresholds(p_val, val_group, thresholds)
        # Build a fairness report using EO hard labels via per-row effective scores:
        # map predictions back through a pseudo-probability for group_fairness_report
        # by evaluating rates directly.
        eo_attr_reports: dict[str, Any] = {"primary": {}, "intersectional": None}
        for attr in fairness_config.groups.primary:
            if attr not in audit_val.columns:
                continue
            # Use group_fairness_report with probabilities but override via threshold=0.5
            # after mapping pred to {0,1} as scores — cleaner: report from pred labels.
            labels = audit_val[attr].astype(str)
            # Construct scores that recover pred at 0.5 threshold.
            pseudo = pred_eo.astype(float)
            eo_attr_reports["primary"][attr] = group_fairness_report(
                y_val,
                pseudo,
                labels,
                threshold=0.5,
                min_cell_count=fairness_config.groups.min_cell_count,
                n_bootstrap=max(50, fairness_config.n_bootstrap // 2),
                seed=base_config.seed + 5,
                attribute_name=attr,
            )
        variants.append(
            mitigation_summary(
                name="equalized_odds_postprocess",
                utility=_utility_from_scores(y_val, p_val, pred_eo),
                fairness=eo_attr_reports,
                extra={
                    "select_thresholds_on": "validation_2024",
                    "mitigation_group_attribute": fairness_config.mitigation_group_attribute,
                    "group_thresholds": thresholds,
                    "note": (
                        "Thresholds selected on 2024 validation only; 2025 not used. "
                        "AUROC/AUPRC/Brier use original scores; balanced_accuracy uses "
                        "post-processed labels."
                    ),
                },
            )
        )

    run_id = str(uuid.uuid4())
    manifest = new_run_manifest(
        run_id=run_id,
        command="fairness",
        config_paths=["configs/base.yaml", "configs/fairness.yaml"],
        seed=base_config.seed,
        device=base_config.device,
        extra={
            "audit_model": "hist_gradient_boosting",
            "n_train_members": int(len(prepared.y_members)),
            "n_validation": int(len(y_val)),
            "used_2025": False,
            "synthetic": use_synthetic,
            "include_mitigations": include_mitigations,
            "protected_in_X": False,
            "interpretation": (
                "Disparities are relative to reported historical HMDA outcomes; "
                "not proof of discrimination, causation, or legal non-compliance."
            ),
        },
    )
    payload: dict[str, Any] = {
        "manifest": manifest,
        "variants": variants,
        "aggregate_only": True,
        "row_level_outputs": False,
    }

    out = output_dir or Path(base_config.paths.outputs) / "fairness"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{run_id}_fairness.json", payload)
    write_json(out / "latest_fairness.json", payload)
    return payload
