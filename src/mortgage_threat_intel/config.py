"""Validated configuration loading for experiment runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    outputs: str = "outputs"
    reports: str = "reports"


class YearsConfig(BaseModel):
    train: int = 2023
    validation: int = 2024
    test: int = 2025
    seal_test_outcomes: bool = True


class SamplingConfig(BaseModel):
    max_rows_per_year: int = 100_000
    strategy: str = "deterministic_uniform"

    @field_validator("max_rows_per_year")
    @classmethod
    def _cap(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_rows_per_year must be positive")
        return v


class TargetConfig(BaseModel):
    column: str = "action_taken"
    positive_codes: list[int] = Field(default_factory=lambda: [1, 2])
    negative_codes: list[int] = Field(default_factory=lambda: [3])
    known_excluded_codes: list[int] = Field(default_factory=lambda: [4, 5, 6, 7, 8])
    label_name: str = "reported_historical_application_outcome"


class MIASplitConfig(BaseModel):
    non_member_fraction: float = 0.2
    evaluation_prior: str = "balanced_1_to_1"


class FeaturesConfig(BaseModel):
    allowlist: list[str]
    protected_audit_only: list[str]
    forbidden: list[str]


class ModelsConfig(BaseModel):
    audit_model: str = "hist_gradient_boosting"
    reference_baseline: str = "logistic_regression"
    privacy_pair: str = "mlp"


class BaseConfig(BaseModel):
    project_name: str = "mortgage-threat-intel"
    seed: int = 42
    sample_seed: int = 42
    device: str = "cpu"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    years: YearsConfig = Field(default_factory=YearsConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    mia: MIASplitConfig = Field(default_factory=MIASplitConfig)
    features: FeaturesConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    utility_metrics: list[str] = Field(
        default_factory=lambda: ["auroc", "auprc", "balanced_accuracy", "brier_score"]
    )


class PrivacyMLPConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = 0.0
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0


class DPSGDConfig(BaseModel):
    target_epsilon: float = 4.0
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = None
    delta_rule: str = "one_over_n"
    adjacency: str = "record_level"
    device: str = "cpu"


class PrivacyMIAConfig(BaseModel):
    attack: str = "confidence_score"
    evaluation_prior: str = "balanced_1_to_1"
    n_bootstrap: int = 200
    member_years: list[int] = Field(default_factory=lambda: [2023])


class PrivacyConfig(BaseModel):
    mlp: PrivacyMLPConfig = Field(default_factory=PrivacyMLPConfig)
    dp_sgd: DPSGDConfig = Field(default_factory=DPSGDConfig)
    mia: PrivacyMIAConfig = Field(default_factory=PrivacyMIAConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_data_raw(paths: PathsConfig) -> Path:
    """Resolve raw data directory from config or HMDA_DATA_DIR env."""
    env = os.environ.get("HMDA_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(paths.data_raw).expanduser().resolve()


def load_base_config(path: str | Path) -> BaseConfig:
    raw = _load_yaml(Path(path))
    return BaseConfig.model_validate(raw)


def load_privacy_config(path: str | Path) -> PrivacyConfig:
    raw = _load_yaml(Path(path))
    return PrivacyConfig.model_validate(raw)


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"
