"""Year split isolation and MIA partition disjointness."""

from __future__ import annotations

import pandas as pd
import pytest

from mortgage_threat_intel.config import default_config_dir, load_base_config
from mortgage_threat_intel.data.labels import map_action_taken
from mortgage_threat_intel.data.splits import (
    assert_no_test_year_in_fit,
    deterministic_sample,
    partition_mia_members,
    split_by_year,
)


@pytest.fixture
def labeled_fixture():
    config = load_base_config(default_config_dir() / "base.yaml")
    df = pd.read_csv("tests/fixtures/synthetic_hmda.csv")
    labeled = map_action_taken(df, config.target)
    return config, labeled.frame


def test_year_split_excludes_2025_from_train_val(labeled_fixture):
    config, frame = labeled_fixture
    split = split_by_year(frame, config, include_test=False)
    assert (split.train["activity_year"] == 2023).all()
    assert (split.validation["activity_year"] == 2024).all()
    assert split.test is None
    assert_no_test_year_in_fit([split.train, split.validation], test_year=2025)
    assert not (frame.loc[frame["activity_year"] == 2025].index.isin(split.train.index)).any()


def test_mia_members_non_members_disjoint(labeled_fixture):
    config, frame = labeled_fixture
    split = split_by_year(frame, config, include_test=False)
    part = partition_mia_members(split.train, config)
    assert part.member_ids.isdisjoint(part.non_member_ids)
    assert len(part.members) + len(part.non_members) == len(split.train)


def test_deterministic_sample_stable():
    df = pd.DataFrame({"x": range(100)})
    a = deterministic_sample(df, max_rows=20, seed=42)
    b = deterministic_sample(df, max_rows=20, seed=42)
    c = deterministic_sample(df, max_rows=20, seed=7)
    assert list(a["x"]) == list(b["x"])
    assert list(a["x"]) != list(c["x"])
