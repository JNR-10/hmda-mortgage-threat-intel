"""Risk register schema and scoring helpers — Krati K-01.

Planning hypotheses from plan §7 / T-01. Residual scores are updated in K-06
after fairness, XAI, and robustness evidence exist. Measured privacy aggregates
from Jainil's handoff may be cited as observations without inventing new runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

EvidenceKind = Literal["public_claim", "project_observation", "hypothesis", "untested"]
EvidenceStatus = Literal["measured", "planned", "untested", "deferred"]


class ScaleAnchor(BaseModel):
    """Anchored 1–5 scale definition (plan §7)."""

    score: int
    meaning: str

    @field_validator("score")
    @classmethod
    def _one_to_five(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("scale score must be in 1..5")
        return v


class LikelihoodImpactScales(BaseModel):
    """Shared L×I anchors used for every register entry."""

    likelihood: list[ScaleAnchor]
    impact: list[ScaleAnchor]

    @model_validator(mode="after")
    def _require_full_scales(self) -> LikelihoodImpactScales:
        for name, items in (("likelihood", self.likelihood), ("impact", self.impact)):
            scores = sorted(a.score for a in items)
            if scores != [1, 2, 3, 4, 5]:
                raise ValueError(f"{name} must define anchors for scores 1..5")
        return self


class RiskScores(BaseModel):
    likelihood: int
    impact: int
    product: int | None = None
    rationale: str

    @field_validator("likelihood", "impact")
    @classmethod
    def _bounded(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("likelihood and impact must be in 1..5")
        return v

    @model_validator(mode="after")
    def _fill_product(self) -> RiskScores:
        expected = self.likelihood * self.impact
        if self.product is None:
            self.product = expected
        elif self.product != expected:
            raise ValueError(f"product {self.product} != likelihood×impact {expected}")
        return self


class EvidenceRef(BaseModel):
    """Separate public claims, project observations, and hypotheses."""

    kind: EvidenceKind
    summary: str
    source: str | None = None


class RiskEntry(BaseModel):
    """One threat-register row (six required by plan §7)."""

    risk_id: str
    title: str
    hypothesis: str
    asset_or_harm: str
    access_assumptions: str
    atlas_mapping: str | None = None
    atlas_note: str | None = None
    is_adversarial: bool = True
    inherent: RiskScores
    residual: RiskScores
    evidence_confidence: Literal["low", "medium", "high"]
    evidence_status: EvidenceStatus
    test_or_signal: str
    planned_control: str
    monitoring_signals: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    residual_update_ticket: str | None = "K-06"
    notes: str | None = None


class ThreatRegister(BaseModel):
    """Validated threat register document."""

    schema_version: str = "1.0"
    project: str = "mortgage-threat-intel"
    proxy_boundary: str
    scales: LikelihoodImpactScales
    risks: list[RiskEntry]
    required_risk_ids: list[str] = Field(
        default_factory=lambda: [
            "R1_membership_inference",
            "R2_input_integrity",
            "R3_poisoning",
            "R4_unequal_error_rates",
            "R5_temporal_drift",
            "R6_explanation_quality",
        ]
    )

    @model_validator(mode="after")
    def _require_six_core_risks(self) -> ThreatRegister:
        present = {r.risk_id for r in self.risks}
        missing = [rid for rid in self.required_risk_ids if rid not in present]
        if missing:
            raise ValueError(f"threat register missing required risk_ids: {missing}")
        return self


def score_lx_i(likelihood: int, impact: int) -> int:
    """Return L×I product after validating 1..5 bounds."""
    if likelihood < 1 or likelihood > 5 or impact < 1 or impact > 5:
        raise ValueError("likelihood and impact must be in 1..5")
    return likelihood * impact


def default_scales() -> LikelihoodImpactScales:
    """Plan §7 example anchors."""
    return LikelihoodImpactScales(
        likelihood=[
            ScaleAnchor(
                score=1,
                meaning="Requires privileged insider conditions or rare access.",
            ),
            ScaleAnchor(
                score=2,
                meaning="Needs specialized access or uncommon capability.",
            ),
            ScaleAnchor(
                score=3,
                meaning="Feasible for a motivated outsider with some resources.",
            ),
            ScaleAnchor(
                score=4,
                meaning="Readily available tooling and moderate exposure.",
            ),
            ScaleAnchor(
                score=5,
                meaning="Broadly available capability and exposed interface.",
            ),
        ],
        impact=[
            ScaleAnchor(
                score=1,
                meaning="Limited, local, and reversible effect.",
            ),
            ScaleAnchor(
                score=2,
                meaning="Narrow harm with low persistence.",
            ),
            ScaleAnchor(
                score=3,
                meaning="Material audit, privacy, or reliability degradation.",
            ),
            ScaleAnchor(
                score=4,
                meaning="Significant privacy, integrity, or unequal-error harm.",
            ),
            ScaleAnchor(
                score=5,
                meaning="Large-scale privacy, access, or financial harm.",
            ),
        ],
    )


def load_threat_register(path: Path) -> ThreatRegister:
    """Load and validate a threat-register YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Threat register not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Threat register must be a mapping: {path}")
    return ThreatRegister.model_validate(raw)


def register_to_dict(register: ThreatRegister) -> dict[str, Any]:
    """Serialize a validated register (aggregate / documentation use)."""
    return register.model_dump(mode="json")


def find_risk(register: ThreatRegister, risk_id: str) -> RiskEntry:
    for risk in register.risks:
        if risk.risk_id == risk_id:
            return risk
    raise KeyError(f"Unknown risk_id: {risk_id}")


def update_residual(
    register: ThreatRegister,
    risk_id: str,
    *,
    likelihood: int,
    impact: int,
    rationale: str,
    evidence_confidence: Literal["low", "medium", "high"],
    evidence_status: EvidenceStatus,
    notes: str | None = None,
) -> ThreatRegister:
    """Return a copy with one risk's residual scores updated (K-06 helper)."""
    risks: list[RiskEntry] = []
    found = False
    for risk in register.risks:
        if risk.risk_id != risk_id:
            risks.append(risk)
            continue
        found = True
        updated = risk.model_copy(
            update={
                "residual": RiskScores(
                    likelihood=likelihood,
                    impact=impact,
                    rationale=rationale,
                ),
                "evidence_confidence": evidence_confidence,
                "evidence_status": evidence_status,
                "notes": notes if notes is not None else risk.notes,
            }
        )
        risks.append(updated)
    if not found:
        raise KeyError(f"Unknown risk_id: {risk_id}")
    return register.model_copy(update={"risks": risks})
