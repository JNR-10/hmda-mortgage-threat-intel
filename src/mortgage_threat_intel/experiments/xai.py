"""TreeSHAP stability/faithfulness experiment — Krati K-04."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from mortgage_threat_intel.config import BaseConfig, default_config_dir, load_base_config
from mortgage_threat_intel.data.pipeline import prepare_hmda, prepare_synthetic
from mortgage_threat_intel.data.splits import assert_no_test_year_in_fit
from mortgage_threat_intel.evaluation.artifacts import new_run_manifest, write_json
from mortgage_threat_intel.explainability.checks import multi_seed_shap_stability
from mortgage_threat_intel.seeding import seed_everything

DEFAULT_XAI_SEEDS = [42, 43, 44]


def run_xai(
    *,
    config: BaseConfig | None = None,
    output_dir: Path | None = None,
    use_synthetic: bool = True,
    seeds: list[int] | None = None,
    background_max: int = 100,
    explain_max: int = 1000,
    top_k: int = 5,
) -> dict[str, Any]:
    config = config or load_base_config(default_config_dir() / "base.yaml")
    seeds = list(seeds or DEFAULT_XAI_SEEDS)
    seed_everything(config.seed)
    prepared = prepare_synthetic(config) if use_synthetic else prepare_hmda(config)
    assert_no_test_year_in_fit(
        [prepared.split.train, prepared.split.validation],
        test_year=config.years.test,
    )

    results = multi_seed_shap_stability(
        X_train=prepared.X_members,
        y_train=prepared.y_members,
        X_eval=prepared.X_validation,
        feature_names=list(prepared.feature_names),
        seeds=seeds,
        background_max=background_max,
        explain_max=min(explain_max, len(prepared.X_validation)),
        top_k=top_k,
    )

    run_id = str(uuid.uuid4())
    manifest = new_run_manifest(
        run_id=run_id,
        command="xai",
        config_paths=["configs/base.yaml"],
        seed=config.seed,
        device=config.device,
        extra={
            "audit_model": "hist_gradient_boosting",
            "xai_seeds": seeds,
            "background_max": background_max,
            "explain_max": explain_max,
            "used_2025": False,
            "synthetic": use_synthetic,
            "method": "TreeSHAP",
            "no_lime_or_counterfactuals": True,
        },
    )
    payload = {
        "manifest": manifest,
        "results": results,
        "aggregate_only": True,
        "row_level_outputs": False,
    }
    out = output_dir or Path(config.paths.outputs) / "xai"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{run_id}_xai.json", payload)
    write_json(out / "latest_xai.json", payload)
    return payload
