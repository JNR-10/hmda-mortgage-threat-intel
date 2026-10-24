# Jainil → Krati handoff (J-07)

**Branch:** `feat/jainil-hmda-pipeline-baselines-privacy`  
**Scope:** HMDA proxy pipeline, baselines, matched MLP privacy pair, confidence-score MIA, reproducibility artifacts.  
**Not in scope:** K-01–K-06 implementation; DU evaluation; row-level outputs.

## Setup

See root `README.md`. Quick path:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make lint && make test && make smoke
```

## Stable interfaces

| Interface | Location | Notes |
|---|---|---|
| Config | `configs/base.yaml`, `privacy.yaml`, `fairness.yaml` | T-01 provisional; re-approve after J-02 real profile |
| Labels / allowlist / splits | `src/mortgage_threat_intel/data/` | Forbidden + protected never in `X` |
| Model contract | `fit(X,y,seed,config)` / `predict_proba(X)` | `models/baselines.py`, `models/dp_mlp.py` |
| Aggregate metrics | `evaluation/metrics.py`, `evaluation/artifacts.py` | JSON under `outputs/` only |
| CLI / Make | `baseline`, `privacy`, `data-check`, `profile`, `smoke` | Fairness/xai/robustness reserved for Krati |

## Aggregate outputs (local, git-ignored)

After `make smoke` / `make baseline` / `make privacy`:

- `outputs/baselines/latest_metrics.json`
- `outputs/privacy/latest_privacy.json`

These contain run manifests, utility metrics, MIA aggregates, and DP accountant metadata. **No raw rows, membership labels, or row-level predictions.**

## Frozen experiment choices (see `docs/decisions/T-00-T-01-shared-gates.md`)

- Audit model: `HistGradientBoostingClassifier` (preselected)
- Privacy: matched MLP, target ε=4, δ=1/N; report **achieved** ε/δ
- MIA: confidence score; 2023 members vs same-year non-members; 1:1 prior
- 2025 sealed until joint protocol freeze

## Known limitations / Jainil must verify before close

1. **T-00:** Course rubric acceptance of HMDA proxy — unverified in-repo.
2. **T-01:** Joint sign-off after real J-02 HMDA profile (allowlist/geography/missing codes).
3. Full HMDA CLI paths currently require `--synthetic` for automated smoke; wire explicit HMDA `prepare` once local files exist (`data/raw/hmda_YYYY.csv`).
4. DP path is **CPU-only**; MPS not enabled pending Opacus smoke on Apple Silicon.
5. Do not open 2025 outcomes for K-05 until protocol freeze.

## What Krati should build next

Start at **K-01** using T-01 threat hypotheses. Consume boosted-tree interfaces/outputs from J-04; do not retrain privacy MLPs for fairness/XAI/robustness.
