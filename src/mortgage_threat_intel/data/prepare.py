"""Schema harmonization and train-fitted preprocessing transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mortgage_threat_intel.config import BaseConfig
from mortgage_threat_intel.data.contracts import assert_no_forbidden_in_features

# Snapshot LAR: CLTV is mostly continuous ratios (plus "Exempt"); one-hot would
# explode (~50k levels). DTI is a small set of bucket/code strings — keep categorical.
NUMERIC_CANDIDATES = (
    "loan_amount",
    "income",
    "property_value",
    "combined_loan_to_value_ratio",
)

CATEGORICAL_CANDIDATES = (
    "debt_to_income_ratio",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "state_code",
)


@dataclass
class FittedPreprocessor:
    feature_names: list[str]
    transformer: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]
    config: BaseConfig

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frame = build_model_frame(df, self.config)
        return self.transformer.transform(frame[self.feature_names])


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_model_frame(df: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Select allowlisted features; assert no forbidden/protected leakage."""
    allow = list(config.features.allowlist)
    assert_no_forbidden_in_features(
        allow,
        config.features.forbidden,
        config.features.protected_audit_only,
    )
    missing = [c for c in allow if c not in df.columns]
    if missing:
        raise KeyError(f"Allowlist columns missing from frame: {missing}")
    out = df[allow].copy()
    for col in allow:
        if col in NUMERIC_CANDIDATES:
            out[col] = _coerce_numeric(out[col])
        elif col in CATEGORICAL_CANDIDATES:
            # HMDA mixes numeric codes and strings (e.g. DTI buckets); unify as str.
            out[col] = out[col].map(lambda v: "NA" if pd.isna(v) else str(v))
    return out


def fit_preprocessor(train_df: pd.DataFrame, config: BaseConfig) -> FittedPreprocessor:
    """Fit imputers/encoders/scalers on training rows only."""
    frame = build_model_frame(train_df, config)
    numeric = [c for c in NUMERIC_CANDIDATES if c in frame.columns]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in frame.columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No features available to fit preprocessor")

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct.fit(frame)
    return FittedPreprocessor(
        feature_names=list(frame.columns),
        transformer=ct,
        numeric_features=numeric,
        categorical_features=categorical,
        config=config,
    )
