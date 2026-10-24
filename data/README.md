# Local HMDA data

Raw and processed HMDA files stay **local and git-ignored**. Do not commit loan-level rows.

## Expected placement

```text
data/raw/hmda_2023.csv
data/raw/hmda_2024.csv
data/raw/hmda_2025.csv
```

Compressed `.csv.gz` variants are also accepted by discovery. Override the root with environment variable `HMDA_DATA_DIR` (never hard-code a user home path in config).

## Provenance to record (locally)

For each file, keep outside Git or in a private note: source URL, download timestamp, SHA-256 (printed by `make data-check`), and the data dictionary version used.

## Sealed 2025 outcomes

Until the joint test protocol is frozen, use 2025 only for presence, schema, and integrity checks. Do not open outcomes for training, tuning, or exploratory analysis.
