# System context — HMDA research proxy (not Desktop Underwriter)

**Owner:** Krati Jain (K-01)  
**Status:** Context and threat framing only; no new experiments  
**Sources:** plan §2 / §7, `docs/decisions/T-00-T-01-shared-gates.md`, Jainil handoff aggregates under `reports/handoff/`

## 1. Proxy boundary (read first)

| Layer | What it is | What it is not |
|---|---|---|
| **Deployment context** | Fannie Mae Desktop Underwriter (DU) is a publicly documented automated mortgage underwriting system used as *use-case context* for Track B. | Not accessed, queried, reproduced, or evaluated in this project. |
| **Empirical subject** | An offline model trained on public HMDA Loan/Application Register (LAR) fields that predicts a *reported historical application outcome* (`action_taken` 1/2 vs 3). | Not creditworthiness, not an approval recommendation, not a lender production model, not DU. |

Public DU facts and project observations must stay separated in the evidence inventory and threat register. This study does **not** claim legal compliance, causal discrimination, person-level privacy from DP metrics, or that HMDA labels are objective truth.

## 2. Stakeholders and assets

| Stakeholder | Interest / asset at risk |
|---|---|
| Applicants / borrowers (represented in public HMDA) | Privacy of participation patterns; unequal error burden if a similar model were misused |
| Lenders / underwriting ops (contextual) | Integrity of automated screening inputs; auditability of model behavior |
| Regulators / auditors (contextual) | Traceable evaluation of privacy, fairness, explanation, and drift risks |
| Research team | Reproducible offline evidence; clear proxy labeling to avoid DU confusion |

**Primary research assets:** frozen feature allowlist and splits; boosted-tree audit model; matched non-private/DP MLP pair; aggregate metrics and run manifests. Raw LAR rows, row-level predictions, membership labels, and row-level explanations stay local and out of Git.

## 3. Dataflow (offline)

```text
Public HMDA Snapshot LAR (local, git-ignored)
        │
        ▼
Schema / hash checks ──► deterministic ≤100k/year sample
        │
        ▼
Label map (1|2 → positive, 3 → negative; other codes excluded)
        │
        ├── 2023: 80% model-fit members / 20% MIA non-members
        ├── 2024: tuning / threshold selection only
        └── 2025: sealed until joint protocol freeze
        │
        ▼
Train-fitted preprocessing (allowlist only; protected attrs audit-only)
        │
        ├── Logistic regression (utility reference)
        ├── HistGradientBoostingClassifier (preselected audit model)
        └── Matched MLP pair (non-private vs DP-SGD) for privacy/MIA only
        │
        ▼
Aggregate metrics → outputs/ (local) and reports/handoff/ (committed snapshots)
```

No lender APIs, no authenticated scraping, no live model queries, no real customer records beyond public HMDA.

## 4. Public deployment context vs project observations

**Public claims (context only)**

- Fannie Mae describes Desktop Underwriter as its automated mortgage loan underwriting system ([public DU page](https://singlefamily.fanniemae.com/applications-technology/desktop-underwriter-desktop-originator)).
- HMDA public loan-level files are privacy-modified for publication ([FFIEC/CFPB data publication](https://ffiec.cfpb.gov/data-publication/)).
- Adverse-action / specific-reasons context for lending notifications is regulatory background (e.g. 12 CFR §1002.9), not a compliance test of this proxy.

**Project observations (from Jainil handoff aggregates; not DU)**

- Boosted-tree audit model utility on 2024 validation (capped sample): AUROC ≈ 0.853, AUPRC ≈ 0.934 (`reports/handoff/baseline_latest_metrics.json`, run `88915b24-…`).
- Logistic regression reference: AUROC ≈ 0.795 on the same split/features.
- Matched DP MLP: accountant-achieved ε ≈ 3.994 at δ = 1/N (`1.25e-05`), record-level adjacency; target ε=4 was a request, not the reported guarantee (`reports/handoff/privacy_latest_privacy.json`, run `8da6a76a-…`).
- Confidence-score MIA (balanced 1:1 prior, 2023 members vs same-year non-members): near-random AUC ≈ 0.50 for both non-private and DP MLPs under this attack model/access assumption.

**Hypotheses (threat register; residual update in K-06)**

Six T-01-frozen hypotheses: membership inference, input integrity (missingness), poisoning (untested empirically), unequal group error rates, temporal drift, explanation instability/unfaithfulness. See `reports/threat_register.yaml`.

## 5. NIST AI RMF crosswalk (compact)

| Function | How this project uses it |
|---|---|
| **Govern** | Proxy boundary, ownership split (J/K/T tickets), no production deployment claim |
| **Map** | System context, stakeholders, dataflow, ATLAS-informed hypotheses |
| **Measure** | Utility, MIA/DP (done); fairness, TreeSHAP, temporal/missingness (K-02–K-05) |
| **Manage** | Planned controls and residual-risk update (K-06); monitoring signals in the register |

## 6. Out of scope (explicit)

- Live DU or lender-model evaluation; production access; applicant advice.
- Shadow-model MIA, ε sweeps, LIME, counterfactuals, poisoning experiments (stretch or untested).
- Choosing the audit model after seeing test results; opening 2025 outcomes before protocol freeze.

## 7. Open shared gates

- **T-00:** Current course rubric acceptance of an HMDA proxy with DU-as-context remains **unverified** in-repo.
- **T-01:** Joint Jainil+Krati sign-off after the real data profile is still required before treating the contract as frozen for reporting.
- **2025:** Outcomes remain sealed until the joint test protocol is frozen.
