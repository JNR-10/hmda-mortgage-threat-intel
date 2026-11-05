# T-01 / T-02 — 2025 temporal protocol freeze

**Status:** Frozen for final temporal evaluation (T-02 closeout).  
**Date:** 2026-11-05  
**Authors:** Jainil Rana, Krati Jain (team closeout; AI Threat Intelligence Track B proxy study)

## Decisions frozen

| Item | Frozen value |
|---|---|
| Feature allowlist | `configs/base.yaml` (`combined_loan_to_value_ratio`; CLTV numeric; DTI categorical) |
| Audit model | `HistGradientBoostingClassifier` (preselected; not chosen after peeking at 2025) |
| Privacy pair | Matched MLP; target ε=4; report **achieved** accountant ε/δ only |
| MIA | Confidence-score attack; 2023 members vs same-year non-members; balanced 1:1 prior |
| Fairness metrics | Demographic-parity + equalized-odds differences; thresholds/reweighting selected on **2024 only** |
| Robustness stress | Mask/impute **10%** of cells among **five** top non-sensitive features chosen using 2023 training attributions/importance |
| Sample | Deterministic uniform ≤100k eligible labeled rows/year; seeds in `configs/base.yaml` |
| 2025 use | **Final temporal holdout only** after this freeze — not for feature selection, threshold tuning, or model choice |

## Checklist

- [x] Feature allowlist frozen (`configs/base.yaml`)
- [x] Model settings frozen (HGB + fairness mitigations)
- [x] Attack definitions frozen (MIA protocol unchanged)
- [x] Fairness metrics / thresholds selection rules frozen (2024 only)
- [x] Robustness stress definition frozen (10% missingness on five top non-sensitive features from 2023 train)
- [x] Jainil + Krati sign-off recorded below

## Sign-off

| Member | Date | Initials |
|---|---|---|
| Jainil Rana | 2026-11-05 | JR |
| Krati Jain | 2026-11-05 | KJ |

After this freeze, `years.seal_test_outcomes` may be set `false` and `cli robustness --no-synthetic --open-2025` may evaluate 2025 outcomes for temporal drift only.
