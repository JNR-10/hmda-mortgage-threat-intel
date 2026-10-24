# Local HMDA data

Raw and processed HMDA files stay **local and git-ignored**. Do not commit loan-level rows.

## Expected placement

```text
data/raw/hmda_2023.csv.gz
data/raw/hmda_2024.csv.gz
data/raw/hmda_2025.csv.gz
data/raw/docs/                 # local HTML snapshots of official documentation
data/download_manifest.json    # provenance: URLs, dates, SHA-256 (no loan rows)
```

Plain `.csv` variants are also accepted by discovery. Override the root with `HMDA_DATA_DIR` (never hard-code a user home path in config).

## Official source (verified)

| Item | Value |
|---|---|
| Portal | [Snapshot National Loan Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset) |
| File host | `https://files.ffiec.cfpb.gov` |
| Dataset | Public LAR CSV inside yearly Snapshot zip |
| Years required | **2023** (model fit), **2024** (tuning), **2025** (sealed temporal test) |

Exact zip URLs (from CFPB `hmda-frontend` `snapshot-dataset.jsx`):

- 2023: `https://files.ffiec.cfpb.gov/static-data/snapshot/2023/2023_public_lar_csv.zip` (freeze May 1, 2024)
- 2024: `https://files.ffiec.cfpb.gov/static-data/snapshot/2024/2024_public_lar_csv.zip` (freeze May 19, 2025)
- 2025: `https://files.ffiec.cfpb.gov/static-data/snapshot/2025/2025_public_lar_csv.zip` (freeze June 2, 2026)

Documentation snapshots (local copies under `data/raw/docs/`):

- [LAR data fields](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields)
- [Public LAR schema](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/public-lar-schema)
- [Static dataset FAQ](https://ffiec.cfpb.gov/documentation/faq/static-dataset-faq)

## Provenance

See **`data/download_manifest.json`** for filename, byte size, SHA-256, source URL, freeze date, and download timestamp for each year. Loan-level files themselves are not listed in Git.

Local processing note: official zips were streamed to `hmda_YYYY.csv.gz` and the intermediate zips were removed to save disk. Checksums are for the local `.csv.gz` artifacts.

## Schema note (needs T-01 / J-02 follow-up)

`make data-check` finds files and hashes, but reports missing `loan_to_value_ratio`. Official Snapshot LAR uses **`combined_loan_to_value_ratio`** (and related fields) instead. Harmonize the feature allowlist after dictionary review before full training.

## Sealed 2025 outcomes

Until the joint test protocol is frozen, use 2025 only for presence, schema, and integrity checks. Do not open outcomes for training, tuning, or exploratory analysis.
