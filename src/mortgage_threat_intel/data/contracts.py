"""Required columns and typed structures for HMDA-like frames."""

from __future__ import annotations

from dataclasses import dataclass

REQUIRED_CORE_COLUMNS: tuple[str, ...] = (
    "activity_year",
    "action_taken",
    "loan_amount",
    "income",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "property_value",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "state_code",
)

PROTECTED_AUDIT_COLUMNS: tuple[str, ...] = (
    "applicant_race_1",
    "applicant_ethnicity_1",
    "applicant_sex",
    "applicant_age",
)


@dataclass(frozen=True)
class SchemaCheckResult:
    ok: bool
    missing_required: tuple[str, ...]
    present_protected: tuple[str, ...]
    extra_notes: str = ""


def check_required_columns(columns: list[str] | tuple[str, ...]) -> SchemaCheckResult:
    present = set(columns)
    missing = tuple(c for c in REQUIRED_CORE_COLUMNS if c not in present)
    protected = tuple(c for c in PROTECTED_AUDIT_COLUMNS if c in present)
    return SchemaCheckResult(
        ok=len(missing) == 0,
        missing_required=missing,
        present_protected=protected,
        extra_notes="" if not missing else "Missing required columns for HMDA-like schema",
    )


def assert_no_forbidden_in_features(
    feature_names: list[str] | tuple[str, ...],
    forbidden: list[str] | tuple[str, ...],
    protected: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Raise if forbidden or protected attributes appear in model feature list."""
    feats = set(feature_names)
    bad_forbidden = sorted(feats.intersection(forbidden))
    if bad_forbidden:
        raise ValueError(f"Forbidden columns present in model features: {bad_forbidden}")
    if protected:
        bad_prot = sorted(feats.intersection(protected))
        if bad_prot:
            raise ValueError(
                f"Protected audit-only attributes present in model features: {bad_prot}"
            )
