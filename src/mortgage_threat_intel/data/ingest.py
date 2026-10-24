"""Local HMDA file discovery, hashing, and schema checks (no download)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from mortgage_threat_intel.config import BaseConfig, resolve_data_raw
from mortgage_threat_intel.data.contracts import REQUIRED_CORE_COLUMNS, check_required_columns


@dataclass
class YearFileStatus:
    year: int
    path: Path | None
    exists: bool
    sha256: str | None = None
    schema_ok: bool | None = None
    missing_columns: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class DataCheckReport:
    data_root: Path
    years: list[YearFileStatus] = field(default_factory=list)
    ok: bool = False
    message: str = ""


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def discover_year_file(data_root: Path, year: int) -> Path | None:
    """Find a local annual file: hmda_{year}.csv or {year}.csv (csv or csv.gz)."""
    candidates = [
        data_root / f"hmda_{year}.csv",
        data_root / f"hmda_{year}.csv.gz",
        data_root / f"{year}.csv",
        data_root / f"{year}.csv.gz",
        data_root / str(year) / f"hmda_{year}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def data_check(config: BaseConfig, *, peek_schema: bool = True) -> DataCheckReport:
    """Validate configured years: presence, hash, optional header schema.

    For the sealed test year, outcomes are not profiled here — only presence,
    integrity hash, and schema headers when the file exists.
    """
    root = resolve_data_raw(config.paths)
    years = [config.years.train, config.years.validation, config.years.test]
    statuses: list[YearFileStatus] = []
    all_ok = True

    if not root.exists():
        msg = (
            f"Raw data directory not found: {root}\n"
            f"Place annual HMDA CSV files as data/raw/hmda_YYYY.csv "
            f"(or set HMDA_DATA_DIR). Expected years: {years}."
        )
        return DataCheckReport(data_root=root, years=[], ok=False, message=msg)

    for year in years:
        path = discover_year_file(root, year)
        if path is None:
            all_ok = False
            statuses.append(
                YearFileStatus(
                    year=year,
                    path=None,
                    exists=False,
                    notes=(
                        f"Missing. Expected one of: hmda_{year}.csv, "
                        f"hmda_{year}.csv.gz under {root}"
                    ),
                )
            )
            continue

        digest = _sha256_file(path)
        missing: tuple[str, ...] = ()
        schema_ok: bool | None = None
        note = "file present"
        if peek_schema:
            try:
                import pandas as pd

                header = pd.read_csv(path, nrows=0)
                check = check_required_columns(list(header.columns))
                schema_ok = check.ok
                missing = check.missing_required
                if not check.ok:
                    all_ok = False
                    note = "schema missing required columns"
                elif year == config.years.test and config.years.seal_test_outcomes:
                    note = "test year: presence/schema/hash only; outcomes sealed"
            except Exception as exc:  # noqa: BLE001 — surface read errors in report
                all_ok = False
                schema_ok = False
                note = f"failed to read header: {exc}"

        statuses.append(
            YearFileStatus(
                year=year,
                path=path,
                exists=True,
                sha256=digest,
                schema_ok=schema_ok,
                missing_columns=missing,
                notes=note,
            )
        )

    message = (
        "All configured year files present."
        if all_ok
        else "One or more year files missing or invalid."
    )
    return DataCheckReport(data_root=root, years=statuses, ok=all_ok, message=message)


def format_data_check_report(report: DataCheckReport) -> str:
    lines = [
        f"HMDA data-check root: {report.data_root}",
        f"Status: {'OK' if report.ok else 'INCOMPLETE'}",
        report.message,
        "",
        "Expected placement: data/raw/hmda_YYYY.csv (or HMDA_DATA_DIR).",
        f"Required header columns include: {', '.join(REQUIRED_CORE_COLUMNS)}",
        "",
    ]
    for y in report.years:
        loc = str(y.path) if y.path else "(missing)"
        lines.append(f"  {y.year}: exists={y.exists} path={loc}")
        if y.sha256:
            lines.append(f"         sha256={y.sha256}")
        if y.schema_ok is not None:
            lines.append(f"         schema_ok={y.schema_ok} missing={list(y.missing_columns)}")
        if y.notes:
            lines.append(f"         notes={y.notes}")
    return "\n".join(lines)
