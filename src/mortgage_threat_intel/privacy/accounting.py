"""DP accountant metadata extraction and serialization (J-06)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PrivacyMetadata:
    """Record-level DP training metadata. Never treat target_epsilon as achieved."""

    target_epsilon: float
    achieved_epsilon: float
    delta: float
    sample_rate: float
    steps: int
    noise_multiplier: float
    max_grad_norm: float
    n_train: int
    adjacency: str
    device: str
    accountant_name: str
    note: str = (
        "achieved_epsilon is from the accountant at the stated delta; "
        "do not report target_epsilon as the privacy guarantee"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_accountant_metadata(
    *,
    target_epsilon: float,
    achieved_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    noise_multiplier: float,
    max_grad_norm: float,
    n_train: int,
    adjacency: str,
    device: str,
    accountant_name: str,
) -> PrivacyMetadata:
    return PrivacyMetadata(
        target_epsilon=float(target_epsilon),
        achieved_epsilon=float(achieved_epsilon),
        delta=float(delta),
        sample_rate=float(sample_rate),
        steps=int(steps),
        noise_multiplier=float(noise_multiplier),
        max_grad_norm=float(max_grad_norm),
        n_train=int(n_train),
        adjacency=adjacency,
        device=device,
        accountant_name=accountant_name,
    )


def serialize_privacy_metadata(meta: PrivacyMetadata) -> dict[str, Any]:
    payload = meta.to_dict()
    if payload["achieved_epsilon"] == payload["target_epsilon"]:
        # Still keep both fields; consumers must prefer achieved_*.
        payload["warning"] = "achieved_epsilon equals target_epsilon numerically; verify accountant"
    return payload
