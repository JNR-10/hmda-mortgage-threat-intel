"""Matched MLP DP-SGD + confidence-score MIA (J-05 / J-06)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from mortgage_threat_intel.config import (
    BaseConfig,
    PrivacyConfig,
    default_config_dir,
    load_base_config,
    load_privacy_config,
)
from mortgage_threat_intel.data.pipeline import prepare_hmda, prepare_synthetic
from mortgage_threat_intel.evaluation.artifacts import new_run_manifest, write_json
from mortgage_threat_intel.evaluation.metrics import utility_metrics
from mortgage_threat_intel.models.dp_mlp import train_matched_mlp_pair
from mortgage_threat_intel.privacy.accounting import serialize_privacy_metadata
from mortgage_threat_intel.privacy.membership import run_confidence_mia
from mortgage_threat_intel.seeding import seed_everything


def run_privacy(
    *,
    base_config: BaseConfig | None = None,
    privacy_config: PrivacyConfig | None = None,
    output_dir: Path | None = None,
    use_synthetic: bool = True,
) -> dict[str, Any]:
    base = base_config or load_base_config(default_config_dir() / "base.yaml")
    privacy = privacy_config or load_privacy_config(default_config_dir() / "privacy.yaml")
    seed_everything(base.seed)

    prepared = prepare_synthetic(base) if use_synthetic else prepare_hmda(base)
    # Transform non-members with the member-fitted preprocessor
    X_non = prepared.preprocessor.transform(prepared.mia.non_members)

    nonpriv, dpriv = train_matched_mlp_pair(
        prepared.X_members,
        prepared.y_members,
        privacy,
        seed=base.seed,
        device=privacy.dp_sgd.device,
    )

    util = {
        "mlp_nonprivate": utility_metrics(
            prepared.y_validation, nonpriv.predict_proba(prepared.X_validation)
        ),
        "mlp_dp_sgd": utility_metrics(
            prepared.y_validation, dpriv.predict_proba(prepared.X_validation)
        ),
    }

    mia_nonpriv = run_confidence_mia(
        nonpriv,
        prepared.X_members,
        X_non,
        member_ids=prepared.mia.member_ids,
        non_member_ids=prepared.mia.non_member_ids,
        seed=base.seed,
        n_bootstrap=privacy.mia.n_bootstrap,
        evaluation_prior=privacy.mia.evaluation_prior,
    )
    mia_dp = run_confidence_mia(
        dpriv,
        prepared.X_members,
        X_non,
        member_ids=prepared.mia.member_ids,
        non_member_ids=prepared.mia.non_member_ids,
        seed=base.seed,
        n_bootstrap=privacy.mia.n_bootstrap,
        evaluation_prior=privacy.mia.evaluation_prior,
    )

    if dpriv.privacy is None:
        raise RuntimeError("DP model missing privacy metadata")
    privacy_meta = serialize_privacy_metadata(dpriv.privacy)

    run_id = str(uuid.uuid4())
    manifest = new_run_manifest(
        run_id=run_id,
        command="privacy",
        config_paths=["configs/base.yaml", "configs/privacy.yaml"],
        seed=base.seed,
        device=privacy.dp_sgd.device,
        extra={
            "synthetic": use_synthetic,
            "n_train": int(len(prepared.y_members)),
            "delta_rule": privacy.dp_sgd.delta_rule,
            "target_epsilon_requested": privacy.dp_sgd.target_epsilon,
            "achieved_epsilon": privacy_meta["achieved_epsilon"],
            "achieved_delta": privacy_meta["delta"],
            "used_2025_as_nonmember": False,
            "mia_evaluation_prior": privacy.mia.evaluation_prior,
        },
    )

    payload: dict[str, Any] = {
        "manifest": manifest,
        "utility": util,
        "mia": {
            "mlp_nonprivate": mia_nonpriv.to_dict(),
            "mlp_dp_sgd": mia_dp.to_dict(),
        },
        "dp_privacy_metadata": privacy_meta,
        "aggregate_only": True,
        "row_level_outputs": False,
        "prevalence_train": float(np.mean(prepared.y_members)),
    }

    out = output_dir or Path(base.paths.outputs) / "privacy"
    write_json(out / f"{run_id}_privacy.json", payload)
    write_json(out / "latest_privacy.json", payload)
    return payload
