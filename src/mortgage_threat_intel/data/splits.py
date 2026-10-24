"""Year splits and MIA member / non-member partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mortgage_threat_intel.config import BaseConfig


@dataclass(frozen=True)
class YearSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame | None


@dataclass(frozen=True)
class MIAPartition:
    members: pd.DataFrame
    non_members: pd.DataFrame
    member_ids: frozenset[str]
    non_member_ids: frozenset[str]


def _row_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Stable row id for disjointness checks (not a real applicant ID)."""
    if "row_id" in df.columns:
        return df["row_id"].astype(str)
    cols = [c for c in df.columns if c != "y"]
    return df[cols].astype(str).agg("|".join, axis=1)


def split_by_year(
    df: pd.DataFrame,
    config: BaseConfig,
    *,
    include_test: bool = False,
) -> YearSplit:
    """Split rows by activity_year. Test year is omitted unless include_test=True."""
    year_col = "activity_year"
    if year_col not in df.columns:
        raise KeyError(f"Missing {year_col}")

    train_year = config.years.train
    val_year = config.years.validation
    test_year = config.years.test

    train = df.loc[df[year_col] == train_year].copy()
    validation = df.loc[df[year_col] == val_year].copy()

    if include_test:
        if config.years.seal_test_outcomes and "y" in df.columns:
            # Allow schema/integrity use; callers must not train on these rows.
            pass
        test = df.loc[df[year_col] == test_year].copy()
    else:
        test = None
        leaked = df.loc[df[year_col] == test_year]
        if len(leaked) and not include_test:
            # Presence in the input is fine; they must not enter train/val returns.
            pass

    # Hard isolation: train/val must not contain test-year rows
    if (train[year_col] == test_year).any():
        raise RuntimeError("Test-year rows leaked into train split")
    if (validation[year_col] == test_year).any():
        raise RuntimeError("Test-year rows leaked into validation split")

    return YearSplit(train=train, validation=validation, test=test)


def assert_no_test_year_in_fit(
    frames: list[pd.DataFrame],
    test_year: int = 2025,
    year_col: str = "activity_year",
) -> None:
    for frame in frames:
        if year_col in frame.columns and (frame[year_col] == test_year).any():
            raise AssertionError(f"{test_year} rows must not appear in training/validation")


def partition_mia_members(
    train_year_df: pd.DataFrame,
    config: BaseConfig,
) -> MIAPartition:
    """Reserve a fraction of the train-year sample as MIA non-members.

    Non-members are never used for model fitting. Members and non-members are
    disjoint by row fingerprint.
    """
    if train_year_df.empty:
        raise ValueError("Cannot partition empty train-year frame")

    frac = config.mia.non_member_fraction
    if not 0.0 < frac < 1.0:
        raise ValueError("mia.non_member_fraction must be in (0, 1)")

    rng = np.random.default_rng(config.sample_seed)
    n = len(train_year_df)
    n_non = int(round(n * frac))
    n_non = min(max(n_non, 1), n - 1) if n > 1 else 0

    indices = np.arange(n)
    rng.shuffle(indices)
    non_idx = set(indices[:n_non].tolist())
    mem_idx = [i for i in indices if i not in non_idx]

    members = train_year_df.iloc[mem_idx].copy()
    non_members = train_year_df.iloc[list(non_idx)].copy()

    member_ids = frozenset(_row_fingerprint(members).tolist())
    non_member_ids = frozenset(_row_fingerprint(non_members).tolist())
    overlap = member_ids & non_member_ids
    if overlap:
        raise RuntimeError(f"MIA member/non-member overlap: {len(overlap)} ids")

    return MIAPartition(
        members=members,
        non_members=non_members,
        member_ids=member_ids,
        non_member_ids=non_member_ids,
    )


def deterministic_sample(
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Uniform sample without replacement, deterministic for fixed seed."""
    if len(df) <= max_rows:
        return df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_rows, replace=False)
    idx.sort()
    return df.iloc[idx].copy()
