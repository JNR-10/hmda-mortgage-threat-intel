# T-00 / T-01 Shared Gate Decisions

**Date:** 2026-10-23  
**Authors:** Jainil Rana, Krati Jain  
**Sources:** `revised_mortgage_ai_threat_intelligence_implementation_plan.md`, `AGENTS.md` only  
**Status:** **Closed for T-02 closeout (2026-11-05).**  
**Update (2026-10-24):** Real Snapshot LAR downloaded; J-02 profile + real baseline/privacy aggregates under `reports/handoff/`.  
**Update (2026-11-05):** Joint T-00/T-01 closeout for AI Threat Intelligence Track B. HMDA research proxy with DU as public context only is the team’s locked system boundary. T-01 contract signed after real J-02 profile. 2025 protocol freeze recorded in `docs/decisions/T-01-2025-protocol-freeze.md`.

---

## Document conflict resolved here

| Topic | Conflicting text | Resolution |
|---|---|---|
| Annual sample cap | Plan §5 change table mentions 250,000; plan §2 glance table, §6.1, §14.3, and `AGENTS.md` specify 100,000 | **Use 100,000** eligible labeled rows per year (deterministic uniform sample). More specific contract sections and `AGENTS.md` agree. |
| Course rubric vs plan | `AGENTS.md` says current course rubric takes precedence over older course files summarized in the plan | Rubric/calendar/proxy acceptance **not available in this repository**. Marked unverified below; Jainil must check course materials before treating T-00 as closed. |

No other plan vs `AGENTS.md` conflicts found for experiment design. Ownership and offline-proxy boundaries are consistent.

---

## T-00 — Confirm current course requirements

### Selected approach

Treat the revised plan as the **technical** source of truth for the offline HMDA proxy study. Treat Fannie Mae Desktop Underwriter (DU) as **public deployment context only** — not an empirical subject. Do **not** claim the project evaluates DU or any lender production model.

### Closeout decision (2026-11-05)

**Selected system for AI Threat Intelligence:** an offline public-HMDA research proxy that predicts *reported historical application outcomes*, with Fannie Mae Desktop Underwriter used only as documented public deployment context. The project does **not** access, reproduce, or evaluate DU.

The exact Track B rubric PDF is not stored in this repository. For course closeout the team locks the plan/`AGENTS.md` proxy boundary as the technical contract and structures the report/demo around NIST AI RMF + MITRE ATLAS threat intelligence (evidence inventory, risk register, measured residual risks). If the instructor requires a different system boundary, amend this note rather than silently reinterpreting results as DU evaluations.

### Rationale and tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| **A (selected for closeout):** Lock HMDA proxy + DU-as-context; complete Measure/Manage artifacts | Matches plan, ethics, and feasibility; produces real threat-intel evidence | Instructor could still prefer experiments on a live production system |
| B: Block closeout until rubric PDF is committed | Strict documentary purity | Prevents finishing an otherwise complete pipeline |
| C: Claim the study evaluates DU | Appears “more realistic” | False; violates plan/`AGENTS.md` |

### Concrete deliverables

1. This decisions note (proxy wording, experiment boundaries).
2. Final report under `reports/final_report.md` and demo script under `docs/demo.md` (plan §12 narrative; course page-limit may require trimming).
3. No old week numbers treated as current.

### Acceptance criteria (plan)

- [x] Short decisions note records proxy interpretation and experiment boundaries.
- [x] Deliverable format for closeout: markdown report + reproducible Make/CLI demo path (rubric PDF not in-repo).
- [x] HMDA proxy locked as the Track B system under study (DU = context only).
- [x] No old week number treated as current.

### Dependencies / risks / unresolved

- **Dependency:** Instructor may still ask for rubric-specific packaging (page count, slide template).
- **Risk:** Instructor rejects a proxy-only design if the course requires experiments on the exact deployed system (plan §2).
- **Instructor note:** Results must never be presented as DU scores, creditworthiness, legal compliance, or person-level privacy guarantees.

---

## T-01 — Freeze the data and experiment contract

### Timing note

Plan: finalize T-01 **after J-02 data profile** and **before J-03 model training**.  
This note freezes **plan-default** contract values so configs and code can be written. After J-02, Jainil + Krati must re-approve (or amend) fields, sample seed, and any profile-driven schema notes. Do not treat this as a substitute for that joint review.

### Selected approach

Freeze one binary outcome task, one feature allowlist, one sampling/split protocol, one audit model, one privacy ablation, and the six threat hypotheses from plan §7. Prefer a single controlled comparison grid over a model×control factorial (plan §1–§2).

### Rationale and tradeoffs

| Choice | Selected | Alternative rejected | Why |
|---|---|---|---|
| Target | `action_taken` ∈ {1,2} → positive; `3` → negative; other codes excluded | Multi-class or “creditworthy” label | Matches plan/AGENTS; avoids creditworthiness framing |
| Sample cap | 100k/year deterministic uniform | 250k (stale §5 row) or full files | Memory/runtime on Apple Silicon; AGENTS + §6.1/§14.3 |
| Audit model | Preselected `HistGradientBoostingClassifier` | Pick best after seeing results | Avoids post-hoc model shopping for risk narrative |
| Privacy subject | Matched non-private / DP-SGD MLP only | DP on trees or multi-ε sweep | DP-SGD fits differentiable models; one ε keeps scope feasible |
| MIA | Confidence-score attack; 2023 member vs same-year non-member; 1:1 prior | Shadow models; 2025 as non-members | Plan core; year shift would confound membership |
| 2025 | Sealed until joint protocol freeze | Early peek for “debugging” | Prevents leakage into thresholds/features |

### Frozen contract (provisional pending post–J-02 sign-off)

**Target and exclusions**

- Positive: originated (`1`) or approved but not accepted (`2`).
- Negative: denied (`3`).
- Exclude all other `action_taken` codes from the primary task; log exclusion counts; **fail on unknown codes** until reviewed.
- Label wording: “reported historical application outcome” only.

**Feature allowlist (predictive `X`)**

- `loan_amount`, `income`, `debt_to_income_ratio`, `combined_loan_to_value_ratio`, `property_value`
  - **Schema note (post–download):** official Snapshot LAR uses `combined_loan_to_value_ratio`, not `loan_to_value_ratio`.
- `loan_type`, `loan_purpose`, `occupancy_type`, `lien_status`
- Coarse geography: `state_code` (county/tract retained only if jointly confirmed after J-02 dictionary review)

**Forbidden from `X`**

- Target and `action_taken`; denial-reason fields; post-decision / post-closing fields; institution ID as a predictive feature (may be retained for audit only).

**Sensitive attributes (audit only, never in main `X`)**

- Race/ethnicity, sex, age (and derived audit groupings).

**Sampling and splits**

- Cap: 100,000 eligible labeled rows per year; if fewer, use all.
- Seed: `sample_seed = 42` (freeze before opening 2025 outcomes).
- 2023: 80% model-fit members / 20% MIA non-members (disjoint row fingerprints); all models fit on the 80% only.
- 2024: tuning and threshold selection only.
- 2025: sealed final temporal test until joint protocol freeze.

**Models**

- Logistic regression: reference utility baseline.
- `HistGradientBoostingClassifier`: preselected audit model (fairness / TreeSHAP / robustness — Krati after handoff).
- Matched MLP pair (non-private vs DP-SGD): privacy comparison only.

**Privacy / MIA**

- One DP-SGD target: requested ε = 4; δ = 1/N for frozen MLP training sample size N.
- Report **accountant-achieved** ε/δ and all accountant inputs; never report target ε as achieved.
- Record-level adjacency; do not claim person-level privacy without contribution bounds.
- MIA: max predicted-class probability score; balanced 1:1 evaluation prior (analysis assumption); MLP pair only.

**Utility metrics (baselines)**

- AUROC, AUPRC, balanced accuracy, Brier score; paired bootstrap CIs where applicable.

**Fairness metrics / subgroup rules (for T-01 freeze; K implements)**

- Primary groups: race/ethnicity and sex; intersections only when cell support supports useful uncertainty; else count + underpowered label.
- Metrics: demographic-parity difference; equalized-odds difference (max of TPR/FPR gaps).

**Initial threat hypotheses (plan §7; Krati evidences in K-01)**

1. Black-box membership inference (training-data membership).
2. Missing/invalid input fields change results (input integrity).
3. Training data or preprocessing altered (poisoning — **untested** in core).
4. Unequal error rates across groups.
5. Later market conditions shift (temporal drift).
6. Explanations unstable or not behaviorally faithful.

### Concrete deliverables

1. This note under `docs/decisions/`.
2. Matching defaults in `configs/base.yaml` and `configs/privacy.yaml`.
3. Joint re-approval checklist after J-02 (below).

### Acceptance criteria

- [x] Target, exclusions, allowlist, sensitive fields, sample strategy, splits, metrics, MIA protocol, and threat hypotheses recorded.
- [x] Post–J-02 joint approval (2026-11-05): allowlist uses `combined_loan_to_value_ratio`; 100k/year cap + seeds retained; profile aggregates in `reports/handoff/data_profile.json`.
- [x] 2025 sealed until freeze; freeze recorded in `docs/decisions/T-01-2025-protocol-freeze.md` for T-02 temporal eval.

### Post–J-02 joint checklist

1. [x] Allowlist confirmed against Snapshot LAR (`combined_loan_to_value_ratio`).
2. [x] Sample seed/cap and exclusion counts confirmed from profile/handoff.
3. [x] This note signed for reporting freeze (2026-11-05).
4. [x] 2025 opened only after protocol freeze file exists (temporal eval only).

### Geography / missing codes (locked for closeout)

- Geography grain remains **`state_code`** (no county/tract in model `X`).
- Special strings such as `Exempt` in numeric fields coerce to missing; DTI bucket strings stay categorical.

---

## What this does *not* authorize

- Access to DU, lender APIs, or real applicant records.
- Claims of legal compliance, causal discrimination, creditworthiness, or person-level DP.
- Starting Krati’s K-01–K-06 implementation during Jainil’s J phase.
- Expanding into shadow-model MIA, ε sweeps, or a model/control factorial.
