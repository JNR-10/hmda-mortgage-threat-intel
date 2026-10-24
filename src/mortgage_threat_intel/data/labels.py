"""Map HMDA action_taken codes to the primary binary outcome."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mortgage_threat_intel.config import TargetConfig


class UnknownActionCodeError(ValueError):
    """Raised when an action_taken code is not in the reviewed set."""


@dataclass(frozen=True)
class LabelMappingResult:
    """Labeled frame plus exclusion accounting for the primary binary task."""

    frame: pd.DataFrame
    n_positive: int
    n_negative: int
    exclusion_counts: dict[int, int]
    n_excluded: int


def map_action_taken(
    df: pd.DataFrame,
    target: TargetConfig,
    *,
    label_column: str = "y",
) -> LabelMappingResult:
    """Map action_taken to binary y; exclude non-primary codes; fail on unknown.

    Positive: codes in ``target.positive_codes`` (default 1, 2).
    Negative: codes in ``target.negative_codes`` (default 3).
    Known excluded: ``target.known_excluded_codes`` (default 4–8).
    Any other code raises ``UnknownActionCodeError``.
    """
    col = target.column
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

    codes = pd.to_numeric(df[col], errors="coerce")
    if codes.isna().any():
        bad = int(codes.isna().sum())
        raise UnknownActionCodeError(
            f"{bad} non-numeric or null {col} value(s); review before mapping"
        )

    positive = set(target.positive_codes)
    negative = set(target.negative_codes)
    excluded = set(target.known_excluded_codes)
    known = positive | negative | excluded

    unique_codes = {int(c) for c in codes.unique()}
    unknown = sorted(unique_codes - known)
    if unknown:
        raise UnknownActionCodeError(
            f"Unknown {col} code(s) {unknown}; fail until reviewed. "
            f"Known positive={sorted(positive)}, negative={sorted(negative)}, "
            f"excluded={sorted(excluded)}"
        )

    exclusion_counts: dict[int, int] = {}
    for code in sorted(excluded):
        n = int((codes == code).sum())
        if n:
            exclusion_counts[code] = n

    keep_mask = codes.isin(positive | negative)
    out = df.loc[keep_mask].copy()
    mapped = codes.loc[keep_mask].map(lambda c: 1 if int(c) in positive else 0)
    out[label_column] = mapped.astype(int)

    n_pos = int((out[label_column] == 1).sum())
    n_neg = int((out[label_column] == 0).sum())
    n_excluded = int((~keep_mask).sum())

    return LabelMappingResult(
        frame=out,
        n_positive=n_pos,
        n_negative=n_neg,
        exclusion_counts=exclusion_counts,
        n_excluded=n_excluded,
    )
