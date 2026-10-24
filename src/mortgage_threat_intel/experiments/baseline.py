"""Train LR + HGB baselines and write aggregate metrics (J-04)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from mortgage_threat_intel.config import BaseConfig, default_config_dir, load_base_config
from mortgage_threat_intel.data.pipeline import prepare_hmda, prepare_synthetic
from mortgage_threat_intel.data.splits import assert_no_test_year_in_fit
from mortgage_threat_intel.evaluation.artifacts import new_run_manifest, write_json
from mortgage_threat_intel.evaluation.metrics import utility_metrics
from mortgage_threat_intel.models.baselines import HistGradientBoostingBaseline, LogisticBaseline
from mortgage_threat_intel.seeding import seed_everything


def run_baselines(
    *,
    config: BaseConfig | None = None,
    output_dir: Path | None = None,
    use_synthetic: bool = True,
) -> dict[str, Any]:
    config = config or load_base_config(default_config_dir() / "base.yaml")
    seed_everything(config.seed)
    prepared = prepare_synthetic(config) if use_synthetic else prepare_hmda(config)
    assert_no_test_year_in_fit(
        [prepared.split.train, prepared.split.validation],
        test_year=config.years.test,
    )

    lr = LogisticBaseline().fit(prepared.X_members, prepared.y_members, seed=config.seed)
    hgb = HistGradientBoostingBaseline().fit(
        prepared.X_members, prepared.y_members, seed=config.seed
    )

    results = {
        "logistic_regression": utility_metrics(
            prepared.y_validation, lr.predict_proba(prepared.X_validation)
        ),
        "hist_gradient_boosting": utility_metrics(
            prepared.y_validation, hgb.predict_proba(prepared.X_validation)
        ),
    }

    run_id = str(uuid.uuid4())
    manifest = new_run_manifest(
        run_id=run_id,
        command="baseline",
        config_paths=["configs/base.yaml"],
        seed=config.seed,
        device=config.device,
        extra={
            "n_train_members": int(len(prepared.y_members)),
            "n_validation": int(len(prepared.y_validation)),
            "n_mia_non_members": int(len(prepared.mia.non_members)),
            "feature_allowlist": list(config.features.allowlist),
            "audit_model": config.models.audit_model,
            "used_2025": False,
            "synthetic": use_synthetic,
        },
    )
    payload = {
        "manifest": manifest,
        "metrics": results,
        "aggregate_only": True,
        "row_level_outputs": False,
    }

    out = output_dir or Path(config.paths.outputs) / "baselines"
    write_json(out / f"{run_id}_metrics.json", payload)
    write_json(out / "latest_metrics.json", payload)
    return payload
