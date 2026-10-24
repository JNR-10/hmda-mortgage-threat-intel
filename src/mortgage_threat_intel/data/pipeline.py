"""End-to-end synthetic/HMDA prepare + split pipeline (J-03)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mortgage_threat_intel.config import BaseConfig, resolve_data_raw
from mortgage_threat_intel.data.ingest import discover_year_file
from mortgage_threat_intel.data.labels import map_action_taken
from mortgage_threat_intel.data.prepare import FittedPreprocessor, fit_preprocessor
from mortgage_threat_intel.data.splits import (
    MIAPartition,
    YearSplit,
    assert_no_test_year_in_fit,
    deterministic_sample,
    partition_mia_members,
    split_by_year,
)


@dataclass
class PreparedBundles:
    """Train-fitted bundles for baselines and privacy experiments."""

    split: YearSplit
    mia: MIAPartition
    preprocessor: FittedPreprocessor
    X_members: object
    y_members: object
    X_validation: object
    y_validation: object
    audit_frame_members: pd.DataFrame
    audit_frame_validation: pd.DataFrame
    feature_names: list[str]


def load_years_as_frame(config: BaseConfig, years: list[int]) -> pd.DataFrame:
    root = resolve_data_raw(config.paths)
    frames: list[pd.DataFrame] = []
    for year in years:
        path = discover_year_file(root, year)
        if path is None:
            raise FileNotFoundError(
                f"Missing HMDA file for {year} under {root}. "
                "Place hmda_YYYY.csv or set HMDA_DATA_DIR."
            )
        if year == config.years.test and config.years.seal_test_outcomes:
            raise RuntimeError(
                f"Refusing to load sealed test year {year} outcomes into the "
                "training/validation pipeline. Unseal only after joint protocol freeze."
            )
        part = pd.read_csv(path, low_memory=False)
        if "activity_year" not in part.columns:
            part = part.copy()
            part["activity_year"] = year
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_synthetic_fixture(path: Path | None = None) -> pd.DataFrame:
    fixture = path or (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "synthetic_hmda.csv"
    )
    return pd.read_csv(fixture)


def prepare_from_frame(df: pd.DataFrame, config: BaseConfig) -> PreparedBundles:
    """Label, sample, split, MIA-partition, and fit preprocessor on members only."""
    labeled = map_action_taken(df, config.target)
    frame = labeled.frame

    # Per-year sample cap on eligible labeled rows
    capped_parts: list[pd.DataFrame] = []
    for year, part in frame.groupby("activity_year", sort=True):
        capped_parts.append(
            deterministic_sample(
                part,
                max_rows=config.sampling.max_rows_per_year,
                seed=config.sample_seed + int(year),
            )
        )
    frame = pd.concat(capped_parts, ignore_index=True)

    split = split_by_year(frame, config, include_test=False)
    assert_no_test_year_in_fit([split.train, split.validation], test_year=config.years.test)
    mia = partition_mia_members(split.train, config)

    preprocessor = fit_preprocessor(mia.members, config)
    X_members = preprocessor.transform(mia.members)
    y_members = mia.members["y"].to_numpy()
    X_validation = preprocessor.transform(split.validation)
    y_validation = split.validation["y"].to_numpy()

    audit_cols = list(config.features.protected_audit_only)
    audit_members = (
        mia.members[audit_cols].copy()
        if all(c in mia.members.columns for c in audit_cols)
        else pd.DataFrame(index=mia.members.index)
    )
    audit_val = (
        split.validation[audit_cols].copy()
        if all(c in split.validation.columns for c in audit_cols)
        else pd.DataFrame(index=split.validation.index)
    )

    return PreparedBundles(
        split=split,
        mia=mia,
        preprocessor=preprocessor,
        X_members=X_members,
        y_members=y_members,
        X_validation=X_validation,
        y_validation=y_validation,
        audit_frame_members=audit_members,
        audit_frame_validation=audit_val,
        feature_names=list(preprocessor.feature_names),
    )


def prepare_synthetic(config: BaseConfig) -> PreparedBundles:
    return prepare_from_frame(load_synthetic_fixture(), config)
