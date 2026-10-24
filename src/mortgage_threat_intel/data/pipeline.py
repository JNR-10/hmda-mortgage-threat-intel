"""End-to-end synthetic/HMDA prepare + split pipeline (J-03)."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_threat_intel.config import BaseConfig, TargetConfig, resolve_data_raw
from mortgage_threat_intel.data.ingest import discover_year_file
from mortgage_threat_intel.data.labels import UnknownActionCodeError, map_action_taken
from mortgage_threat_intel.data.prepare import FittedPreprocessor, fit_preprocessor
from mortgage_threat_intel.data.splits import (
    MIAPartition,
    YearSplit,
    assert_no_test_year_in_fit,
    deterministic_sample,
    partition_mia_members,
    split_by_year,
)

# Smaller chunks keep peak RSS down on 8GB hosts while streaming gzip LAR.
CHUNKSIZE = 50_000


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


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _feature_load_cols(config: BaseConfig) -> list[str]:
    """Columns required for training/audit — never pull unused forbidden fields.

    ``action_taken`` is loaded as the target even though it is also listed under
    ``features.forbidden`` (forbidden means "not in model X", not "do not read").
    Denial reasons / LEI are intentionally omitted from the sample load.
    """
    cols = set(config.features.allowlist)
    cols.update(config.features.protected_audit_only)
    cols.add(config.target.column)
    cols.add("activity_year")
    return sorted(cols)


def _usecols(config: BaseConfig) -> list[str]:
    """Backward-compatible alias used by tests."""
    return _feature_load_cols(config)


def _eligible_mask(codes: pd.Series, target: TargetConfig) -> pd.Series:
    positive = set(target.positive_codes)
    negative = set(target.negative_codes)
    excluded = set(target.known_excluded_codes)
    known = positive | negative | excluded
    numeric = pd.to_numeric(codes, errors="coerce")
    if numeric.isna().any():
        raise UnknownActionCodeError(
            f"{int(numeric.isna().sum())} non-numeric/null {target.column} values"
        )
    unique = {int(c) for c in numeric.unique()}
    unknown = sorted(unique - known)
    if unknown:
        raise UnknownActionCodeError(
            f"Unknown {target.column} code(s) {unknown}; fail until reviewed"
        )
    return numeric.isin(positive | negative)


def _file_fingerprint(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "name": path.name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _cache_key(config: BaseConfig, year_paths: dict[int, Path]) -> str:
    payload = {
        "train": config.years.train,
        "validation": config.years.validation,
        "max_rows": config.sampling.max_rows_per_year,
        "sample_seed": config.sample_seed,
        "allowlist": list(config.features.allowlist),
        "protected": list(config.features.protected_audit_only),
        "target": config.target.model_dump(),
        "files": {str(y): _file_fingerprint(p) for y, p in sorted(year_paths.items())},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return digest


def _resolve_processed(config: BaseConfig) -> Path:
    return Path(config.paths.data_processed).expanduser().resolve()


def sample_year_eligible(
    path: Path,
    year: int,
    config: BaseConfig,
    *,
    max_rows: int,
    seed: int,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    """Two-pass deterministic uniform sample of eligible labeled rows for one year.

    Pass 1 counts eligible rows using only the action column (low memory).
    Pass 2 keeps a fixed-seed subset of size ``min(max_rows, n_eligible)``.
    Does not load the full LAR into memory.
    """
    header = pd.read_csv(path, nrows=0)
    header_cols = set(header.columns)
    target_col = config.target.column
    if target_col not in header_cols:
        raise KeyError(f"{target_col} missing from {path}")

    load_cols = [c for c in _feature_load_cols(config) if c in header_cols]
    count_cols = [target_col]
    if "activity_year" in header_cols:
        count_cols.append("activity_year")

    _log(f"[sample {year}] pass1 count eligible from {path.name} …")
    n_eligible = 0
    for chunk in pd.read_csv(
        path,
        usecols=count_cols,
        chunksize=chunksize,
        low_memory=True,
    ):
        if "activity_year" in chunk.columns:
            chunk = chunk.loc[pd.to_numeric(chunk["activity_year"], errors="coerce") == year]
        if chunk.empty:
            continue
        mask = _eligible_mask(chunk[target_col], config.target)
        n_eligible += int(mask.sum())

    if n_eligible == 0:
        raise ValueError(f"No eligible labeled rows for year {year} in {path}")

    take = min(max_rows, n_eligible)
    rng = np.random.default_rng(seed)
    chosen = np.zeros(n_eligible, dtype=bool)
    chosen[rng.choice(n_eligible, size=take, replace=False)] = True
    _log(f"[sample {year}] eligible={n_eligible:,} take={take:,}; pass2 collect …")

    parts: list[pd.DataFrame] = []
    eligible_i = 0
    # Force object→str-friendly dtypes up front to avoid per-chunk DtypeWarning spam
    # and mixed-type surprises on HMDA string/numeric hybrids.
    dtypes = {c: "string" for c in load_cols if c != target_col}
    for chunk in pd.read_csv(
        path,
        usecols=load_cols,
        chunksize=chunksize,
        low_memory=False,
        dtype=dtypes,
    ):
        if "activity_year" in chunk.columns:
            chunk = chunk.loc[pd.to_numeric(chunk["activity_year"], errors="coerce") == year]
        else:
            chunk = chunk.copy()
            chunk["activity_year"] = year
        if chunk.empty:
            continue
        mask = _eligible_mask(chunk[target_col], config.target)
        kept = chunk.loc[mask]
        if kept.empty:
            continue
        n = len(kept)
        local = chosen[eligible_i : eligible_i + n]
        eligible_i += n
        if local.any():
            parts.append(kept.iloc[np.flatnonzero(local)].copy())

    out = pd.concat(parts, ignore_index=True)
    out["activity_year"] = year
    out["row_id"] = [f"{year}-{i}" for i in range(len(out))]
    _log(f"[sample {year}] collected {len(out):,} rows")
    return out


def load_hmda_development_sample(config: BaseConfig, *, use_cache: bool = True) -> pd.DataFrame:
    """Load capped samples for train/validation years only; never opens sealed test."""
    root = resolve_data_raw(config.paths)
    years = [config.years.train, config.years.validation]
    year_paths: dict[int, Path] = {}
    for year in years:
        path = discover_year_file(root, year)
        if path is None:
            raise FileNotFoundError(
                f"Missing HMDA file for {year} under {root}. "
                "Place hmda_YYYY.csv[.gz] or set HMDA_DATA_DIR."
            )
        year_paths[year] = path

    processed = _resolve_processed(config)
    processed.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config, year_paths)
    cache_path = processed / f"dev_sample_{key}.csv.gz"
    meta_path = processed / f"dev_sample_{key}.meta.json"

    if use_cache and cache_path.is_file() and meta_path.is_file():
        _log(f"[hmda] loading cached development sample {cache_path.name}")
        return pd.read_csv(cache_path)

    frames: list[pd.DataFrame] = []
    for year, path in year_paths.items():
        frames.append(
            sample_year_eligible(
                path,
                year,
                config,
                max_rows=config.sampling.max_rows_per_year,
                seed=config.sample_seed + int(year),
            )
        )
    out = pd.concat(frames, ignore_index=True)
    if use_cache:
        out.to_csv(cache_path, index=False, compression="gzip")
        meta_path.write_text(
            json.dumps(
                {
                    "cache_key": key,
                    "n_rows": int(len(out)),
                    "years": years,
                    "files": {str(y): _file_fingerprint(p) for y, p in year_paths.items()},
                    "note": "Capped development sample only; gitignored under data/processed/",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _log(f"[hmda] wrote cache {cache_path.name} ({len(out):,} rows)")
    return out


def load_synthetic_fixture(path: Path | None = None) -> pd.DataFrame:
    fixture = path or (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "synthetic_hmda.csv"
    )
    return pd.read_csv(fixture)


def prepare_from_frame(df: pd.DataFrame, config: BaseConfig) -> PreparedBundles:
    """Label, sample, split, MIA-partition, and fit preprocessor on members only."""
    labeled = map_action_taken(df, config.target)
    frame = labeled.frame

    # Per-year sample cap (no-op if already capped at load time)
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


def prepare_hmda(config: BaseConfig) -> PreparedBundles:
    """Prepare development years from local HMDA files (2025 remains sealed)."""
    return prepare_from_frame(load_hmda_development_sample(config), config)
