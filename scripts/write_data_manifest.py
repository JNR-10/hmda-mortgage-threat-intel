"""Write local HMDA provenance manifest (not committed with raw rows)."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = RAW / "docs"

FILES = {
    2023: {
        "local_path": "data/raw/hmda_2023.csv.gz",
        "source_zip_url": (
            "https://files.ffiec.cfpb.gov/static-data/snapshot/2023/2023_public_lar_csv.zip"
        ),
        "publication_page": (
            "https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023"
        ),
        "dataset": "CFPB/FFIEC Snapshot National Loan Level Dataset — Public LAR CSV",
        "freeze_date": "May 1, 2024",
        "source_zip_bytes": 624_535_331,
        "source_zip_last_modified": "Fri, 21 Nov 2025 18:27:50 GMT",
        "zip_member": "2023_public_lar_csv.csv",
        "download_completed_utc": "2026-10-23T23:31:00Z",
    },
    2024: {
        "local_path": "data/raw/hmda_2024.csv.gz",
        "source_zip_url": (
            "https://files.ffiec.cfpb.gov/static-data/snapshot/2024/2024_public_lar_csv.zip"
        ),
        "publication_page": (
            "https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024"
        ),
        "dataset": "CFPB/FFIEC Snapshot National Loan Level Dataset — Public LAR CSV",
        "freeze_date": "May 19, 2025",
        "source_zip_bytes": 664_242_987,
        "source_zip_last_modified": "Fri, 21 Nov 2025 18:27:51 GMT",
        "zip_member": "2024_public_lar_csv.csv",
        "download_completed_utc": "2026-10-23T23:41:00Z",
    },
    2025: {
        "local_path": "data/raw/hmda_2025.csv.gz",
        "source_zip_url": (
            "https://files.ffiec.cfpb.gov/static-data/snapshot/2025/2025_public_lar_csv.zip"
        ),
        "publication_page": (
            "https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2025"
        ),
        "dataset": "CFPB/FFIEC Snapshot National Loan Level Dataset — Public LAR CSV",
        "freeze_date": "June 2, 2026",
        "source_zip_bytes": 737_139_477,
        "source_zip_last_modified": "Tue, 23 Jun 2026 13:29:33 GMT",
        "zip_member": "2025_public_lar_csv.csv",
        "download_completed_utc": "2026-10-24T15:37:00Z",
        "outcomes_sealed": True,
    },
}

DOC_FILES = [
    {
        "local_path": "data/raw/docs/lar-data-fields.html",
        "source_url": (
            "https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields"
        ),
        "description": "Public HMDA Data Fields with Values and Definitions",
    },
    {
        "local_path": "data/raw/docs/public-lar-schema.html",
        "source_url": (
            "https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/public-lar-schema"
        ),
        "description": "Public LAR Schema",
    },
    {
        "local_path": "data/raw/docs/static-dataset-faq.html",
        "source_url": "https://ffiec.cfpb.gov/documentation/faq/static-dataset-faq",
        "description": "Static Dataset FAQ",
    },
]

CORE_COLS = [
    "activity_year",
    "action_taken",
    "loan_amount",
    "income",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "state_code",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def header_info(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
    missing = [c for c in CORE_COLS if c not in header]
    return {
        "n_columns": len(header),
        "missing_core_columns": missing,
        "has_action_taken": "action_taken" in header,
    }


def main() -> None:
    years = []
    for year, meta in FILES.items():
        path = ROOT / meta["local_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = dict(meta)
        entry["year"] = year
        entry["filename"] = path.name
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
        entry["format"] = "gzip-compressed CSV derived from official Snapshot LAR zip"
        entry["header_check"] = header_info(path)
        years.append(entry)

    docs = []
    for meta in DOC_FILES:
        path = ROOT / meta["local_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = dict(meta)
        entry["filename"] = path.name
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
        docs.append(entry)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "official_portal": "https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset",
        "file_server": "https://files.ffiec.cfpb.gov",
        "selection_rationale": (
            "Plan requires HMDA years 2023 (fit), 2024 (tune), 2025 (sealed test). "
            "Used official Snapshot National Loan Level public LAR CSV zips only."
        ),
        "local_layout": {
            "raw_lar": "data/raw/hmda_YYYY.csv.gz",
            "docs": "data/raw/docs/",
            "gitignored": True,
        },
        "years": years,
        "documentation": docs,
        "notes": [
            "Original zip archives were deleted after streaming conversion to .csv.gz to save disk.",
            "source_zip_bytes and source_zip_last_modified come from HTTP HEAD of the official zip.",
            "Do not commit loan-level files. 2025 outcomes remain sealed until protocol freeze.",
        ],
    }

    # Provenance only (no loan-level rows). Safe to keep under data/ and commit later if desired.
    out = ROOT / "data" / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    for y in years:
        hc = y["header_check"]
        print(
            f"year={y['year']} sha256={y['sha256'][:16]}... "
            f"bytes={y['bytes']} missing_core={hc['missing_core_columns']}"
        )


if __name__ == "__main__":
    main()
