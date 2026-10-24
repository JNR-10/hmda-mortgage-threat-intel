"""Safe aggregate HMDA profiles for development years (J-02).

2025 outcomes stay sealed: presence/schema/hash only via ingest.data_check.
Profiles stream in chunks so full national LAR files are not loaded at once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mortgage_threat_intel.config import BaseConfig, resolve_data_raw
from mortgage_threat_intel.data.ingest import discover_year_file
from mortgage_threat_intel.data.labels import UnknownActionCodeError, map_action_taken
from mortgage_threat_intel.evaluation.artifacts import write_json

CHUNKSIZE = 200_000


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


def profile_development_year(
    path: Path,
    year: int,
    config: BaseConfig,
    *,
    seal_outcomes: bool = False,
) -> YearProfile:
    """Stream-profile one annual file. If seal_outcomes, skip label aggregates."""
    digest = _sha256_file(path)
    notes: list[str] = []

    if seal_outcomes:
        notes.append("outcomes sealed: label and denial aggregates omitted")
        return YearProfile(
            year=year,
            source_path=str(path),
            sha256=digest,
            n_rows_scanned=0,
            action_taken_counts={},
            label_counts=None,
            exclusion_counts=None,
            missingness={},
            subgroup_counts={},
            outcomes_sealed=True,
            notes=notes,
        )

    header = list(pd.read_csv(path, nrows=0).columns)
    cols_for_missing = [
        c
        for c in list(config.features.allowlist) + list(config.features.protected_audit_only)
        if c in header
    ]
    usecols = sorted(
        set(cols_for_missing)
        | {config.target.column, "activity_year"}
        | set(config.features.protected_audit_only)
    )
    usecols = [c for c in usecols if c in header]

    action_counter: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    present_counts: Counter[str] = Counter()
    subgroup_counters: dict[str, Counter[str]] = {
        c: Counter() for c in config.features.protected_audit_only if c in header
    }
    n_rows = 0
    n_pos = n_neg = n_excl = 0
    exclusion_counts: Counter[int] = Counter()

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
        if "activity_year" in chunk.columns:
            chunk = chunk.loc[pd.to_numeric(chunk["activity_year"], errors="coerce") == year]
        n_rows += len(chunk)
        if chunk.empty:
            continue

        vc = pd.to_numeric(chunk[config.target.column], errors="coerce").value_counts(dropna=False)
        for k, v in vc.items():
            key = str(int(k)) if pd.notna(k) else "null"
            action_counter[key] += int(v)

        for col in cols_for_missing:
            if col in chunk.columns:
                missing_counts[col] += int(chunk[col].isna().sum())
                present_counts[col] += len(chunk)

        for col, counter in subgroup_counters.items():
            counts = chunk[col].astype(str).value_counts(dropna=False)
            for k, v in counts.items():
                counter[str(k)] += int(v)

        try:
            mapped = map_action_taken(chunk, config.target)
        except UnknownActionCodeError:
            raise
        n_pos += mapped.n_positive
        n_neg += mapped.n_negative
        n_excl += mapped.n_excluded
        for code, cnt in mapped.exclusion_counts.items():
            exclusion_counts[code] += cnt

    missingness = {
        col: (missing_counts[col] / present_counts[col]) if present_counts[col] else 1.0
        for col in cols_for_missing
    }
    subgroups = {col: dict(counter) for col, counter in subgroup_counters.items()}

    return YearProfile(
        year=year,
        source_path=str(path),
        sha256=digest,
        n_rows_scanned=n_rows,
        action_taken_counts=dict(action_counter),
        label_counts={"positive": n_pos, "negative": n_neg, "excluded": n_excl},
        exclusion_counts={str(k): int(v) for k, v in exclusion_counts.items()},
        missingness=missingness,
        subgroup_counts=subgroups,
        outcomes_sealed=False,
        notes=notes,
    )


def build_data_profile(config: BaseConfig) -> dict[str, Any]:
    """Profile 2023/2024 fully; 2025 sealed (file metadata only)."""
    root = resolve_data_raw(config.paths)
    profiles: list[dict[str, Any]] = []
    errors: list[str] = []

    for year in (config.years.train, config.years.validation):
        path = discover_year_file(root, year)
        if path is None:
            errors.append(f"Missing development file for {year} under {root}")
            continue
        profiles.append(asdict(profile_development_year(path, year, config)))

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
