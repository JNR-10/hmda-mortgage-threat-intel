# T-00 / T-01 Shared Gate Decisions

**Date:** 2026-10-23  
**Authors:** Jainil Rana, Krati Jain (provisional; joint sign-off pending)  
**Sources:** `revised_mortgage_ai_threat_intelligence_implementation_plan.md`, `AGENTS.md` only  
**Status:** Working decisions for implementation. Course-rubric acceptance of the HMDA proxy is **unverified**.  
**Update (2026-10-24):** Real Snapshot LAR downloaded; J-02 aggregate profile written to `outputs/data_profile.json`. Feature allowlist LTV field corrected to `combined_loan_to_value_ratio`. Real-data baseline + privacy completed on capped 2023/2024 samples with 2025 sealed (`outputs/baselines/latest_metrics.json`, `outputs/privacy/latest_privacy.json`). CLTV treated as numeric (high-cardinality); DTI remains categorical buckets. Development sample cached under gitignored `data/processed/`. **Joint T-01 sign-off with Krati still required** before treating the contract as frozen for reporting. **T-00 course-rubric check still unverified** (rubric not in repo).

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

Because the current Track B rubric and calendar are **not present** in this repository, T-00 cannot fully close here. Record provisional boundaries from the plan and require Jainil to verify against current course materials before instructor-facing claims.

### Rationale and tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| **A (selected):** Proceed with plan/AGENTS boundaries; mark rubric/proxy acceptance unverified | Unblocks scaffold and J work without inventing course facts | T-00 incomplete until Jainil checks rubric |
| B: Block all coding until rubric is in-repo | Strict to ticket wording | Contradicts user request to scaffold and implement J tasks now |
| C: Assume proxy is accepted | Fast | Invents a course fact; forbidden |

### Concrete deliverables

1. This decisions note (proxy wording, experiment boundaries, open verification items).
2. Explicit statement that due dates and deliverable format are **unknown** until checked against current course instructions.
3. No old week numbers treated as current (plan §5 / T-00 acceptance).

### Acceptance criteria (plan)

- [x] Short decisions note records proxy interpretation and experiment boundaries.
- [ ] Current rubric / dates / deliverable format recorded — **blocked: materials not in repo**.
- [ ] HMDA proxy accepted under current course system requirement — **unverified; Jainil must confirm**.
- [x] No old week number treated as current.

### Dependencies / risks / unresolved

- **Dependency:** Current CMPE 253 Track B rubric, calendar, and deliverable format (external).
- **Risk:** Instructor rejects a proxy-only design if the course requires experiments on the exact deployed system (plan §2).
- **Unresolved for Jainil:** Confirm proxy acceptability; record due dates and report/demo format; note any instructor questions.
- **Instructor question (draft):** Does an offline HMDA research proxy, with DU used only as documented public context, satisfy the Track B “system” requirement?

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
- [ ] Post–J-02 joint approval by Jainil + Krati before treating J-03 as unlocked for full HMDA runs.
- [x] 2025 remains sealed until final protocol freeze.

### Dependencies / risks / unresolved

- **Depends on:** T-00 verification (external); J-02 profile for schema/missingness confirmation.
- **Risk:** Dictionary review may force allowlist edits; sample seed/cap must not change after 2025 is opened.
- **Unresolved after J-02:** Exact geography grain; encoding of special HMDA missing codes; confirm natural prevalence and subgroup counts under the 100k cap; runtime pilot if 100k is infeasible on available hardware.

### Post–J-02 joint checklist (do not skip)

1. Confirm allowlist against downloaded data dictionary.
2. Confirm sample seed/cap and exclusion counts from the profile.
3. Sign off this note (or amend and re-commit) before J-03 full training on real HMDA.
4. Keep 2025 outcomes unread until T-02 protocol freeze.

---

## What this does *not* authorize

- Access to DU, lender APIs, or real applicant records.
- Claims of legal compliance, causal discrimination, creditworthiness, or person-level DP.
- Starting Krati’s K-01–K-06 implementation during Jainil’s J phase.
- Expanding into shadow-model MIA, ε sweeps, or a model/control factorial.
