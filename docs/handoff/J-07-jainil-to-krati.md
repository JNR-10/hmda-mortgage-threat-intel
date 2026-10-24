# Jainil → Krati handoff (J-07)

**Branch:** `feat/jainil-hmda-pipeline-baselines-privacy`  
**Scope:** HMDA proxy pipeline, baselines, matched MLP privacy pair, confidence-score MIA, reproducibility + aggregate handoff.  
**Not in scope:** K-01–K-06 implementation; DU evaluation; row-level outputs.

## Start here (Krati)

1. Checkout this branch and set up the env (`README.md` → `make setup`).
2. Read `docs/decisions/T-00-T-01-shared-gates.md` and schedule **joint T-01 sign-off** with Jainil (contract still provisional).
3. For **K-01** (context / threat register): use plan §7 hypotheses + committed aggregates under `reports/handoff/` — **no HMDA download required**.
4. For **K-02+** (fairness / TreeSHAP / robustness): download Snapshot LAR once (below), then use the cached development sample so you do **not** re-scan national files every time.

## Avoid redundant work

| Need | Do this | Do **not** |
|---|---|---|
| Utility / MIA / DP numbers for the report | Use `reports/handoff/*.json` | Re-run privacy unless you change the MLP/DP config |
| 2023/2024 aggregate profile | Use `reports/handoff/data_profile.json` | Re-profile full national LAR unless schema changes |
| Fairness / TreeSHAP / robustness on HGB | Download data once → `make baseline` (uses cache) → consume `outputs/baselines/latest_audit_bundle.joblib` | Re-download zips; do not retrain the privacy MLPs |
| First-time national LAR sample | ~10–15+ min once; writes `data/processed/dev_sample_*.csv.gz` | Expect every experiment to re-read the raw `.csv.gz` |
| Optional team shortcut | Jainil can share the **gitignored** `data/processed/dev_sample_*.csv.gz` (~3MB) out-of-band | Commit that file to Git (it is real capped rows) |
| 2025 | Presence/schema/hash only until joint protocol freeze | Open outcomes for training, tuning, or K-05 until freeze |

Refresh committed aggregates after new Jainil runs: `make handoff-sync`.

## Setup

```bash
git checkout feat/jainil-hmda-pipeline-baselines-privacy
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make lint && make test && make smoke
```

Synthetic smoke proves the stack without HMDA.

## Download HMDA (once)

Official Snapshot National Loan Level **public LAR** zips → local `data/raw/hmda_YYYY.csv.gz`.

- Instructions + verified URLs + SHA-256: **`data/README.md`** and **`data/download_manifest.json`**
- After placement: `make data-check`
- Optional aggregate profile (2025 sealed): `make profile`

Minimal macOS/Linux pattern (see `data/README.md` for full notes):

```bash
mkdir -p data/raw
# download each yearly zip from files.ffiec.cfpb.gov (URLs in data/README.md),
# extract the public LAR CSV, gzip to hmda_YYYY.csv.gz, then:
make data-check
```

Cloudflare may block naive Python downloads; browser or `curl` usually works.

## Stable interfaces

| Interface | Location | Notes |
|---|---|---|
| Config | `configs/base.yaml`, `privacy.yaml`, `fairness.yaml` | T-01 provisional until joint sign-off |
| Labels / allowlist / splits | `src/mortgage_threat_intel/data/` | Forbidden + protected never in model `X` |
| Model contract | `fit(X,y,seed,config)` / `predict_proba(X)` | `models/baselines.py`, `models/dp_mlp.py` |
| Aggregate metrics | `evaluation/metrics.py`, `evaluation/artifacts.py` | Local `outputs/` + committed `reports/handoff/` |
| Audit bundle (local) | `outputs/baselines/latest_audit_bundle.joblib` | Preprocessor + LR + HGB after `make baseline` |
| CLI / Make | `baseline`, `privacy`, `data-check`, `profile`, `smoke` | `fairness` / `xai` / `robustness` reserved for you |

### Feature / prep notes (real LAR)

- Allowlist LTV field: **`combined_loan_to_value_ratio`** (not `loan_to_value_ratio`).
- CLTV is **numeric** (high cardinality; one-hot OOMs). DTI stays **categorical** buckets/codes.
- Strings like `Exempt` in numeric fields coerce to missing via preprocessing.

## Committed aggregate outputs

Under `reports/handoff/` (also described in that folder’s README):

- `baseline_latest_metrics.json` — LR + HGB utility on 2024 validation (real sample)
- `privacy_latest_privacy.json` — achieved DP ε/δ + MIA aggregates
- `data_profile.json` — 2023/2024 aggregates; 2025 presence/hash only

**No raw rows, membership labels, or row-level predictions.**

Local gitignored mirrors: `outputs/baselines/`, `outputs/privacy/`, `outputs/data_profile.json`.

## Frozen experiment choices

See `docs/decisions/T-00-T-01-shared-gates.md`.

- Audit model: `HistGradientBoostingClassifier` (preselected)
- Privacy: matched MLP, target ε=4, δ=1/N; report **achieved** ε/δ only
- MIA: confidence score; 2023 members vs same-year non-members; balanced 1:1 prior
- Cap: 100k eligible rows/year; 2023 MIA holdout 20% non-members
- 2025 sealed until joint protocol freeze

## Known limitations / open gates

1. **T-00:** Course rubric acceptance of HMDA proxy — unverified in-repo (Jainil + Krati).
2. **T-01:** Joint sign-off after real J-02 profile still required before reporting freeze.
3. DP path is **CPU-only**; MPS not enabled.
4. First LAR sample is slow; afterward `data/processed/` cache makes `make baseline` / privacy prep fast.
5. Do not open 2025 outcomes for K-05 until protocol freeze.

## What Krati should build next

1. **K-01** — system context + threat register from T-01 hypotheses; cite `reports/handoff/` for measured privacy/utility context.
2. **K-02/K-03** — fairness on the **boosted-tree** audit model (`configs/fairness.yaml`); use `latest_audit_bundle.joblib` + validation/audit frames from `prepare_hmda`.
3. **K-04** — TreeSHAP; aggregate summaries only; synthetic record for any published local illustration.
4. **K-05** — robustness / input stress (plan: mask 10% among top non-sensitive features chosen on 2023 train); 2025 only after protocol freeze.
5. **K-06** — residual-risk update from measured results.

Do **not** retrain privacy MLPs for fairness/XAI/robustness.
