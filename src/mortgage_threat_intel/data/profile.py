"""Safe aggregate HMDA profiles for development years (J-02).

2025 outcomes stay sealed: presence/schema/hash only via ingest.data_check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mortgage_threat_intel.config import BaseConfig, resolve_data_raw
from mortgage_threat_intel.data.ingest import discover_year_file
from mortgage_threat_intel.data.labels import UnknownActionCodeError, map_action_taken
from mortgage_threat_intel.evaluation.artifacts import write_json


@dataclass
class YearProfile:
    year: int
    source_path: str
    sha256: str | None
    n_rows_scanned: int
    action_taken_counts: dict[str, int]
    label_counts: dict[str, int] | None
    exclusion_counts: dict[str, int] | None
    missingness: dict[str, float]
    subgroup_counts: dict[str, dict[str, int]]
    outcomes_sealed: bool = False
    notes: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _missingness(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = max(len(df), 1)
    for col in columns:
        if col in df.columns:
            out[col] = float(df[col].isna().mean()) if n else 0.0
        else:
            out[col] = 1.0
    return out


def _subgroup_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].astype(str).value_counts(dropna=False).to_dict()
        result[col] = {str(k): int(v) for k, v in counts.items()}
    return result


def profile_development_year(
    path: Path,
    year: int,
    config: BaseConfig,
    *,
    seal_outcomes: bool = False,
) -> YearProfile:
    """Profile one annual file. If seal_outcomes, skip label/outcome aggregates."""
    digest = _sha256_file(path)
    # Read in chunks for large files; synthetic/small files load once.
    df = pd.read_csv(path, low_memory=False)
    n_rows = len(df)
    if "activity_year" in df.columns:
        # Prefer rows matching year when present
        year_mask = pd.to_numeric(df["activity_year"], errors="coerce") == year
        if year_mask.any():
            df = df.loc[year_mask].copy()
            n_rows = len(df)

    action_counts: dict[str, int] = {}
    if "action_taken" in df.columns and not seal_outcomes:
        vc = pd.to_numeric(df["action_taken"], errors="coerce").value_counts(dropna=False)
        action_counts = {str(int(k) if pd.notna(k) else "null"): int(v) for k, v in vc.items()}

    label_counts = None
    exclusion_counts = None
    notes: list[str] = []

    if seal_outcomes:
        notes.append("outcomes sealed: label and denial aggregates omitted")
    else:
        try:
            mapped = map_action_taken(df, config.target)
            label_counts = {
                "positive": mapped.n_positive,
                "negative": mapped.n_negative,
                "excluded": mapped.n_excluded,
            }
            exclusion_counts = {str(k): v for k, v in mapped.exclusion_counts.items()}
        except UnknownActionCodeError as exc:
            notes.append(f"label mapping failed: {exc}")
            raise

    cols_for_missing = list(config.features.allowlist) + list(config.features.protected_audit_only)
    missing = _missingness(df if not seal_outcomes else df.head(0), cols_for_missing)
    if seal_outcomes:
        # Still report header-level missingness without relying on outcome fields
        missing = _missingness(df, list(config.features.allowlist))

    subgroups = (
        {} if seal_outcomes else _subgroup_counts(df, list(config.features.protected_audit_only))
    )

    return YearProfile(
        year=year,
        source_path=str(path),
        sha256=digest,
        n_rows_scanned=n_rows,
        action_taken_counts=action_counts,
        label_counts=label_counts,
        exclusion_counts=exclusion_counts,
        missingness=missing,
        subgroup_counts=subgroups,
        outcomes_sealed=seal_outcomes,
        notes=notes,
    )


def build_data_profile(config: BaseConfig) -> dict[str, Any]:
    """Profile 2023/2024 fully; 2025 sealed (file metadata via caller/data-check)."""
    root = resolve_data_raw(config.paths)
    profiles: list[dict[str, Any]] = []
    errors: list[str] = []

    for year in (config.years.train, config.years.validation):
        path = discover_year_file(root, year)
        if path is None:
            errors.append(f"Missing development file for {year} under {root}")
            continue
        try:
            profiles.append(asdict(profile_development_year(path, year, config)))
        except UnknownActionCodeError as exc:
            errors.append(str(exc))
            raise

    test_path = discover_year_file(root, config.years.test)
    test_meta: dict[str, Any] = {
        "year": config.years.test,
        "outcomes_sealed": True,
        "exists": test_path is not None,
        "path": str(test_path) if test_path else None,
        "notes": ["presence/schema/hash only; see data-check for integrity"],
    }
    if test_path is not None:
        test_meta["sha256"] = _sha256_file(test_path)

    return {
        "data_root": str(root),
        "development_profiles": profiles,
        "test_year_meta": test_meta,
        "errors": errors,
        "aggregate_only": True,
    }


def write_data_profile(config: BaseConfig, output_path: Path) -> dict[str, Any]:
    profile = build_data_profile(config)
    write_json(output_path, profile)
    return profile
