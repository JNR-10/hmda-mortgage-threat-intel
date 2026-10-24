"""HMDA outcome mapping and exclusion / unknown-code behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from mortgage_threat_intel.config import TargetConfig
from mortgage_threat_intel.data.labels import UnknownActionCodeError, map_action_taken


def test_maps_positive_and_negative_codes():
    df = pd.DataFrame({"action_taken": [1, 2, 3, 1]})
    result = map_action_taken(df, TargetConfig())
    assert list(result.frame["y"]) == [1, 1, 0, 1]
    assert result.n_positive == 3
    assert result.n_negative == 1
    assert result.n_excluded == 0


def test_excludes_known_non_primary_codes_and_counts():
    df = pd.DataFrame({"action_taken": [1, 3, 4, 5, 6, 7, 8]})
    result = map_action_taken(df, TargetConfig())
    assert len(result.frame) == 2
    assert result.n_excluded == 5
    assert result.exclusion_counts[4] == 1
    assert result.exclusion_counts[8] == 1


def test_unknown_code_fails():
    df = pd.DataFrame({"action_taken": [1, 99]})
    with pytest.raises(UnknownActionCodeError, match="Unknown"):
        map_action_taken(df, TargetConfig())


def test_null_action_taken_fails():
    df = pd.DataFrame({"action_taken": [1, None]})
    with pytest.raises(UnknownActionCodeError):
        map_action_taken(df, TargetConfig())
