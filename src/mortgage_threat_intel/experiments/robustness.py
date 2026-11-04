"""Temporal + missingness robustness — Krati K-05."""

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
    resolve_data_raw,
)
from mortgage_threat_intel.data.ingest import discover_year_file
from mortgage_threat_intel.data.labels import map_action_taken
from mortgage_threat_intel.data.pipeline import (
    load_synthetic_fixture,
    prepare_hmda,
    prepare_synthetic,
    sample_year_eligible,
)
from mortgage_threat_intel.data.splits import assert_no_test_year_in_fit, deterministic_sample
from mortgage_threat_intel.evaluation.artifacts import new_run_manifest, write_json
from mortgage_threat_intel.explainability.attribution import (
    select_background,
    tree_shap_values,
)
from mortgage_threat_intel.models.baselines import HistGradientBoostingBaseline
from mortgage_threat_intel.robustness.drift import temporal_evaluation
from mortgage_threat_intel.robustness.perturbations import (
    missingness_stress_eval,
    select_top_non_sensitive_features,
)
from mortgage_threat_intel.seeding import seed_everything


def _protocol_freeze_recorded(repo_root: Path) -> bool:
    """Real 2025 evaluation requires an explicit freeze note."""
    path = repo_root / "docs" / "decisions" / "T-01-2025-protocol-freeze.md"
    return path.is_file()


def _synthetic_temporal_holdout(config: BaseConfig) -> pd.DataFrame:
    """Build a synthetic later-year slice when 2025 is sealed."""
    df = load_synthetic_fixture()
    years = set(df["activity_year"].astype(int))
    if config.years.test in years:
        hold = df[df["activity_year"].astype(int) == config.years.test].copy()
    else:
        hold = df[df["activity_year"].astype(int) == config.years.validation].copy()
        if hold.empty:
            hold = df[df["activity_year"].astype(int) == config.years.train].copy()
        hold = hold.copy()
        hold["activity_year"] = config.years.test
    labeled = map_action_taken(hold, config.target)
    return labeled.frame


def run_robustness(
    *,
    base_config: BaseConfig | None = None,
    fairness_config: FairnessConfig | None = None,
    output_dir: Path | None = None,
    use_synthetic: bool = True,
    open_2025: bool = False,
) -> dict[str, Any]:
    base_config = base_config or load_base_config(default_config_dir() / "base.yaml")
    fairness_config = fairness_config or load_fairness_config(
        default_config_dir() / "fairness.yaml"
    )
    seed_everything(base_config.seed)
    repo_root = Path(__file__).resolve().parents[3]

    prepared = prepare_synthetic(base_config) if use_synthetic else prepare_hmda(base_config)
    assert_no_test_year_in_fit(
        [prepared.split.train, prepared.split.validation],
        test_year=base_config.years.test,
    )

    hgb = HistGradientBoostingBaseline().fit(
        prepared.X_members, prepared.y_members, seed=base_config.seed
    )

    background = select_background(prepared.X_members, max_rows=50, seed=base_config.seed)
    try:
        shap_out = tree_shap_values(
            hgb,
            prepared.X_members[: min(200, len(prepared.X_members))],
            background=background,
            feature_names=list(prepared.feature_names),
        )
        importance = shap_out["global_mean_abs"]
    except Exception:
        est = hgb._model  # noqa: SLF001
        if hasattr(est, "feature_importances_"):
            importance = np.asarray(est.feature_importances_, dtype=float)
        else:
            importance = np.ones(len(prepared.feature_names), dtype=float)

    stress_cols = select_top_non_sensitive_features(
        importance,
        list(prepared.feature_names),
        allowlist=list(base_config.features.allowlist),
        protected=list(base_config.features.protected_audit_only),
        k=5,
    )

    val_frame = prepared.split.validation.reset_index(drop=True)
    missingness = missingness_stress_eval(
        model=hgb,
        preprocessor=prepared.preprocessor,
        eval_frame=val_frame,
        y_true=np.asarray(prepared.y_validation).astype(int),
        audit_frame=prepared.audit_frame_validation.reset_index(drop=True),
        stress_columns=stress_cols,
        primary_attrs=list(fairness_config.groups.primary),
        fraction=0.10,
        seed=base_config.seed,
        min_cell_count=fairness_config.groups.min_cell_count,
    )

    used_2025 = False
    freeze_ok = _protocol_freeze_recorded(repo_root)

    if open_2025 and not use_synthetic:
        if not freeze_ok:
            raise RuntimeError(
                "Refusing to open 2025 outcomes: missing "
                "docs/decisions/T-01-2025-protocol-freeze.md. "
                "Record the joint protocol freeze first, or run without --open-2025."
            )
        if base_config.years.seal_test_outcomes:
            raise RuntimeError(
                "configs/base.yaml still has years.seal_test_outcomes=true; "
                "set false only after the joint freeze is recorded."
            )
        raw = resolve_data_raw(base_config.paths)
        path_2025 = discover_year_file(raw, base_config.years.test)
        if path_2025 is None:
            raise FileNotFoundError("2025 HMDA file not found for temporal evaluation")
        frame_2025 = sample_year_eligible(
            path_2025,
            base_config.years.test,
            base_config,
            max_rows=base_config.sampling.max_rows_per_year,
            seed=base_config.sample_seed + int(base_config.years.test),
        )
        labeled = map_action_taken(frame_2025, base_config.target)
        test_frame = labeled.frame
        y_test = test_frame["y"].to_numpy()
        X_test = prepared.preprocessor.transform(test_frame)
        p_test = hgb.predict_proba(X_test)[:, 1]
        audit_cols = list(base_config.features.protected_audit_only)
        audit_test = test_frame[audit_cols].copy()
        p_val = hgb.predict_proba(prepared.X_validation)[:, 1]
        temporal = temporal_evaluation(
            y_ref=np.asarray(prepared.y_validation).astype(int),
            p_ref=p_val,
            audit_ref=prepared.audit_frame_validation.reset_index(drop=True),
            y_new=y_test,
            p_new=p_test,
            audit_new=audit_test.reset_index(drop=True),
            X_ref=np.asarray(prepared.X_validation),
            X_new=np.asarray(X_test),
            primary_attrs=list(fairness_config.groups.primary),
            min_cell_count=fairness_config.groups.min_cell_count,
            seed=base_config.seed,
        )
        used_2025 = True
        temporal["data_year"] = base_config.years.test
    else:
        hold_frame = _synthetic_temporal_holdout(base_config)
        hold_frame = deterministic_sample(
            hold_frame, max_rows=min(500, len(hold_frame)), seed=base_config.seed
        )
        y_hold = hold_frame["y"].to_numpy()
        X_hold = prepared.preprocessor.transform(hold_frame)
        p_hold = hgb.predict_proba(X_hold)[:, 1]
        audit_cols = [
            c for c in base_config.features.protected_audit_only if c in hold_frame.columns
        ]
        audit_hold = (
            hold_frame[audit_cols].copy() if audit_cols else pd.DataFrame(index=hold_frame.index)
        )
        p_val = hgb.predict_proba(prepared.X_validation)[:, 1]
        primary = [a for a in fairness_config.groups.primary if a in audit_hold.columns]
        temporal = temporal_evaluation(
            y_ref=np.asarray(prepared.y_validation).astype(int),
            p_ref=p_val,
            audit_ref=prepared.audit_frame_validation.reset_index(drop=True),
            y_new=y_hold,
            p_new=p_hold,
            audit_new=audit_hold.reset_index(drop=True),
            X_ref=np.asarray(prepared.X_validation),
            X_new=np.asarray(X_hold),
            primary_attrs=primary or list(fairness_config.groups.primary)[:1],
            min_cell_count=fairness_config.groups.min_cell_count,
            seed=base_config.seed,
        )
        temporal["data_year"] = "synthetic_holdout"
        temporal["2025_sealed"] = True
        temporal["note"] = (
            "2025 outcomes not opened. Synthetic/holdout path only. "
            f"protocol_freeze_recorded={freeze_ok}."
        )

    run_id = str(uuid.uuid4())
    manifest = new_run_manifest(
        run_id=run_id,
        command="robustness",
        config_paths=["configs/base.yaml", "configs/fairness.yaml"],
        seed=base_config.seed,
        device=base_config.device,
        extra={
            "audit_model": "hist_gradient_boosting",
            "used_2025": used_2025,
            "open_2025_requested": open_2025,
            "protocol_freeze_recorded": freeze_ok,
            "synthetic": use_synthetic,
            "stress_columns": stress_cols,
            "missingness_fraction": 0.10,
        },
    )
    payload = {
        "manifest": manifest,
        "missingness_stress": missingness,
        "temporal": temporal,
        "aggregate_only": True,
        "row_level_outputs": False,
    }
    out = output_dir or Path(base_config.paths.outputs) / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{run_id}_robustness.json", payload)
    write_json(out / "latest_robustness.json", payload)
    return payload
