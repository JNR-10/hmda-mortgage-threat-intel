# AI Threat Intelligence Report — HMDA Mortgage Research Proxy

**Course:** CMPE 253 — AI Threat Intelligence (Track B)  
**Team:** Jainil Rana, Krati Jain  
**Date:** 2026-11-05  
**Status:** T-02 closeout draft (trim to course page limit as needed)

---

## 1. Problem, contribution, and proxy boundary

**Research question.** For an offline HMDA proxy that predicts *reported historical mortgage application outcomes*, what privacy–utility change does a fixed DP-SGD intervention produce in a matched MLP pair, and what fairness, explanation-reliability, and robustness risks are measurable in a preselected gradient-boosted-tree audit model?

**Contribution.** An end-to-end, reproducible threat-intelligence pipeline: public evidence inventory, ATLAS-informed risk register, measured privacy/MIA, fairness (+ mitigations), TreeSHAP quality checks, temporal + missingness robustness, and residual-risk update.

**Proxy boundary (non-negotiable).**

| Layer | Is | Is not |
|---|---|---|
| Deployment context | Fannie Mae Desktop Underwriter (public docs) | Accessed, reproduced, or evaluated |
| Empirical subject | Public HMDA LAR outcome prediction (`action_taken` 1\|2 vs 3) | Creditworthiness, approval advice, lender production model |

We do **not** claim legal compliance, causal discrimination, or person-level privacy from DP metrics.

---

## 2. Context, stakeholders, and dataflow

See `reports/system_context.md` and `reports/evidence_inventory.csv`.

Stakeholders: applicants (privacy / unequal error burden if misused), lenders/ops (integrity, auditability), regulators/auditors (traceable evaluation), research team (reproducibility + clear proxy labeling).

Offline dataflow: Snapshot LAR → schema/hash checks → ≤100k/year deterministic sample → label map → 2023 fit (80% MIA members) / 2024 tune / 2025 temporal holdout → train-fitted preprocessing on allowlist only → LR + HGB + matched MLP pair → aggregate metrics only.

---

## 3. Data contract

| Item | Decision |
|---|---|
| Source | CFPB/FFIEC Snapshot National Loan Level public LAR (2023–2025); provenance in `data/download_manifest.json` |
| Target | Reported historical application outcome: codes 1\|2 positive, 3 negative; other known codes excluded |
| Cap | 100,000 eligible labeled rows/year |
| Features | Allowlist in `configs/base.yaml` (`combined_loan_to_value_ratio` numeric; DTI categorical); protected attrs audit-only |
| Seeds | `seed=42`, `sample_seed=42` |

**Limitations.** HMDA labels are reported historical actions, not ground-truth “correct” underwriting. Public-use privacy modifications limit person-level claims. `Exempt` and bucket strings require explicit coercion.

---

## 4. Threat model and risk register

Six T-01 hypotheses mapped with provisional MITRE ATLAS notes (`reports/threat_register.yaml`): membership inference, input integrity (missingness), poisoning (**untested** empirically), unequal group error rates, temporal drift, explanation instability/unfaithfulness.

Likelihood×impact scales and monitoring signals live in the register. Residual update: `reports/threat_register_residual.yaml` + `reports/residual_risk_changelog.json`.

---

## 5. Experimental protocol

| Dimension | Protocol |
|---|---|
| Audit model | Preselected `HistGradientBoostingClassifier` |
| Utility reference | Logistic regression |
| Privacy | Matched MLP; target ε=4; δ=1/N; report **achieved** accountant ε/δ |
| MIA | Confidence-score; 2023 members vs same-year non-members; balanced 1:1 prior |
| Fairness | DP + EO differences; bootstrap CIs (≤20k rows/draw); reweight + EO post-process thresholds on **2024 only** |
| XAI | TreeSHAP; 3 seeds; top-k stability; faithfulness via top vs random ablation |
| Robustness | 10% missingness on five top non-sensitive features (from 2023 train); 2025 temporal after freeze |

Configs: `configs/base.yaml`, `privacy.yaml`, `fairness.yaml`. Freeze: `docs/decisions/T-01-2025-protocol-freeze.md`.

---

## 6. Results (aggregate artifacts only)

Sources under `reports/handoff/`. Run IDs in manifests.

### 6.1 Utility (2024 validation, capped sample)

| Model | AUROC | AUPRC | Balanced acc. | Brier |
|---|---:|---:|---:|---:|
| Logistic regression | 0.795 | 0.902 | 0.679 | 0.132 |
| HistGradientBoosting (audit) | 0.853 | 0.934 | 0.728 | 0.116 |

Run: `88915b24-…` (`baseline_latest_metrics.json`).

### 6.2 Privacy / MIA (matched MLP)

| Quantity | Value |
|---|---|
| Target ε (request only) | 4.0 |
| **Achieved ε** (PRV accountant) | ≈ 3.994 |
| δ | 1.25×10⁻⁵ (=1/N), record-level adjacency |
| Non-private MLP AUROC | ≈ 0.834 |
| DP MLP AUROC | ≈ 0.807 |
| MIA AUC (non-private / DP) | ≈ 0.499 / 0.500 |

Run: `8da6a76a-…` (`privacy_latest_privacy.json`). Near-chance MIA under this attack/access model is **not** proof against all adversaries.

### 6.3 Fairness (HGB audit; real HMDA)

Unmitigated validation AUROC ≈ 0.853 (same protocol as baseline).

| Attribute | DP difference (all / powered) | EO difference (all / powered) | Underpowered groups |
|---|---:|---:|---:|
| `applicant_race_1` | 0.75 / **0.21** | 1.00 / **0.38** | 7 / 19 |
| `applicant_sex` | 0.16 / 0.16 | 0.44 / 0.44 | 0 / 5 |

Mitigations (2024 thresholds / 2023 reweighting):

- **Reweighting:** sex EO ≈ 0.28 (down from 0.44); utility AUROC ≈ 0.852.
- **EO post-process:** sex EO ≈ 0.062; sex DP ≈ 0.051; race gaps remain large when tiny cells dominate the “all groups” statistic — report **powered-group** gaps for interpretation.

Run: `d544a852-…` (`fairness_latest.json`). Disparities are associations with *reported historical outcomes*, not proof of discrimination or causation.

### 6.4 Explainability (TreeSHAP)

| Check | Result |
|---|---|
| Top-5 stability (3 seeds) | mean pairwise Jaccard = **1.0** |
| Method | TreeSHAP only (no LIME / counterfactuals in core) |
| Local illustration | Synthetic labeled record only |

Run: `efdad606-…` (`xai_latest.json`).

### 6.5 Robustness

**Missingness stress (10% on top non-sensitive features):**  
features ≈ `{combined_loan_to_value_ratio, debt_to_income_ratio, loan_amount, income, loan_type}`; AUROC Δ ≈ **−0.032** vs clean 2024 validation.

**Temporal (2025 opened after freeze):** AUROC 2024→2025 ≈ 0.853 → 0.845 (Δ ≈ **−0.008**). Labeled as reliability/drift unless malicious causation is evidenced.

Run: `bad97364-…` (`robustness_latest.json`); `used_2025=true`.

---

## 7. Uncertainty, sensitivity, and trade-offs

- Fairness CIs use bootstrap with ≤20k rows per draw for CPU feasibility on 100k validation; tiny race codes are labeled underpowered.
- MIA evaluation prior is balanced 1:1 by design (analysis assumption).
- DP shows a modest utility drop vs matched non-private MLP at achieved ε≈4.
- EO post-processing can shrink sex gaps while race “all-group” maxima remain driven by sparse codes — always report counts.

---

## 8. Controls, residual risks, governance

| Risk | Evidence status | Residual note |
|---|---|---|
| R1 Membership inference | measured | Likelihood reduced under confidence-score attack; other attacks untested |
| R2 Input integrity | measured | Controlled missingness degrades utility; monitoring on missing rates |
| R3 Poisoning | **untested** | Provenance/hash controls ≠ empirical mitigation |
| R4 Unequal error rates | measured | Material gaps remain; mitigations help sex more than sparse race cells |
| R5 Temporal drift | measured (2025) | Small AUROC shift; continue monitoring year-over-year |
| R6 Explanation quality | measured | Stable top-k; faithfulness is behavioral, not causal/legal |

Governance: Govern/Map/Measure/Manage (NIST AI RMF). Do not deploy this proxy as a lending tool.

---

## 9. Reproducibility

```bash
make setup && source .venv/bin/activate
make lint && make test && make smoke
# With local Snapshot LAR (data/README.md):
make data-check
make baseline privacy fairness-hmda
.venv/bin/python -m mortgage_threat_intel.cli xai --config configs/base.yaml --no-synthetic
.venv/bin/python -m mortgage_threat_intel.cli robustness --config configs/base.yaml \
  --fairness-config configs/fairness.yaml --no-synthetic --open-2025
make handoff-sync && make residual-update
```

Demo walkthrough: `docs/demo.md`. Decisions: `docs/decisions/`.

---

## References

Plan §13 sources (Fannie Mae DU public page, CFPB/FFIEC HMDA, MITRE ATLAS, NIST AI RMF, Opacus, SHAP TreeExplainer, scikit-learn HGB) plus committed aggregates under `reports/handoff/`.
