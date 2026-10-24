# Local HMDA data

Raw and processed HMDA files stay **local and git-ignored**. Do not commit loan-level rows.

## Expected placement

```text
data/raw/hmda_2023.csv.gz
data/raw/hmda_2024.csv.gz
data/raw/hmda_2025.csv.gz
data/raw/docs/                 # optional local HTML snapshots of official documentation
data/download_manifest.json    # provenance: URLs, dates, SHA-256 (committed; no loan rows)
data/processed/dev_sample_*.csv.gz   # auto cache after first sample (gitignored)
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

Documentation (bookmark; optional local copies under `data/raw/docs/`):

- [LAR data fields](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields)
- [Public LAR schema](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/public-lar-schema)
- [Static dataset FAQ](https://ffiec.cfpb.gov/documentation/faq/static-dataset-faq)

## Download recipe (macOS / Linux)

Cloudflare may return **403** to bare Python `urllib`; use a browser or `curl`.

```bash
mkdir -p data/raw
cd data/raw

# Example for one year — repeat for 2023, 2024, 2025
YEAR=2023
curl -L -o "${YEAR}_public_lar_csv.zip" \
  "https://files.ffiec.cfpb.gov/static-data/snapshot/${YEAR}/${YEAR}_public_lar_csv.zip"
unzip -o "${YEAR}_public_lar_csv.zip"
# Official member name is typically ${YEAR}_public_lar_csv.csv
gzip -c "${YEAR}_public_lar_csv.csv" > "hmda_${YEAR}.csv.gz"
rm -f "${YEAR}_public_lar_csv.zip" "${YEAR}_public_lar_csv.csv"

cd ../..
make data-check
```

Verify SHA-256 against `data/download_manifest.json` (or regenerate with `scripts/write_data_manifest.py` after a fresh download).

## Provenance

See **`data/download_manifest.json`** for filename, byte size, SHA-256, source URL, freeze date, and download timestamp for each year. Loan-level files themselves are not in Git.

## Schema notes (harmonized in config)

- Feature allowlist uses **`combined_loan_to_value_ratio`** (official Snapshot field), not `loan_to_value_ratio`.
- CLTV is mostly continuous (+ `Exempt`) → treated as **numeric** in preprocessing.
- DTI is bucket/code strings → treated as **categorical**.
- `make data-check` validates presence/schema/hash; it does not open 2025 for modeling.

## Processed cache (saves time)

The first real `make baseline` / `make privacy` / `prepare_hmda` builds a capped 100k/year development sample and writes `data/processed/dev_sample_<key>.csv.gz`. Later runs reuse it when seed, allowlist, and source fingerprints match. **Do not commit** this cache (real rows). Optional: share only within the team out-of-band to skip the first multi-pass scan.

## Sealed 2025 outcomes

Until the joint test protocol is frozen, use 2025 only for presence, schema, and integrity checks. Do not open outcomes for training, tuning, or exploratory analysis.
