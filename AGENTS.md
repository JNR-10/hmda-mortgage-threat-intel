# Coding-Agent Instructions — Mortgage Threat Intelligence Project

## Source of truth

Read `revised_mortgage_ai_threat_intelligence_implementation_plan.md` before coding. It is the project specification for scope, research questions, data definitions, experiments, interpretation, and milestone acceptance checks. The current course rubric takes precedence over the older course files summarized in the plan. If a course requirement conflicts with the plan, stop and report the exact conflict for team review.

## Project boundary

- Build an offline research pipeline using public HMDA data.
- Fannie Mae Desktop Underwriter is context only. This project does not access, reproduce, test, or make claims about DU or any lender's production model.
- The model predicts a reported historical HMDA application outcome, not creditworthiness and not an approval recommendation.
- Do not connect to lender APIs, scrape authenticated services, query live models, or use real applicant/customer records.
- Do not claim legal compliance, causal discrimination, or person-level privacy from model metrics.

## Work in reviewed milestones

Implement only the milestone requested by Jainil or Krati. Begin with Milestone A from the plan: scaffold, config, CLI, synthetic fixtures, and tests. Do not jump to model training or fetch full data before Milestone A passes review. After each milestone, run its acceptance commands, summarize changed files and test results, and list decisions that need team input.

## Ownership and execution order

- Complete T-00 together before implementation. Complete T-01 together after Jainil's initial data profile and before J-03 model training.
- Jainil starts from the empty repository and owns J-01 through J-07: repository scaffold, HMDA intake, data/preprocessing/splits, baselines, membership inference, DP-SGD, and reproducibility artifacts.
- After J-07 passes its acceptance checks, Jainil hands off the repository, stable interfaces, frozen configs, run manifests, and aggregate outputs. Do not include raw records or row-level predictions in the handoff.
- Krati then owns K-01 through K-06: system context and threat register, fairness, explainability, robustness, and residual-risk update. Do not start Krati's K tickets during Jainil's implementation phase.
- After both individual phases pass, Jainil and Krati complete T-02 together: integration, final analysis, report, and demo.
- Keep 2025 outcome data sealed until the test protocol is jointly frozen. Do not let the handoff or intermediate review reveal row-level data or change the frozen evaluation protocol.

## Repository layout

Use the layout in Section 14.1 of the plan. Keep implementation under `src/mortgage_threat_intel/`, tests under `tests/`, experiment settings under `configs/`, raw/processed HMDA records under ignored local `data/`, and context/evidence/threat artifacts under `reports/`. Keep models and generated outputs under ignored local `outputs/`. Keep final logic in importable modules and CLI commands; notebooks are for exploration only.

## Data and experiment rules

- Normalize annual HMDA files to a documented schema and retain year and source fingerprint.
- For the primary binary task, map `action_taken` 1 or 2 to positive and 3 to negative. Exclude other action codes from this task, report exclusion counts, and fail on unknown codes until reviewed.
- Cap each annual eligible sample at 100,000 rows using deterministic uniform sampling. For 2023, reserve 20% as MIA non-members and fit every model on the remaining 80%; use 2024 for tuning and keep 2025 outcomes sealed for the final temporal test.
- Train logistic regression as the reference baseline and `HistGradientBoostingClassifier` as the preselected audit model. Use the boosted tree for fairness, TreeSHAP, and robustness. Use a matched non-private/DP MLP pair only for the privacy comparison; do not expand into a model/control factorial.
- For MIA, compare only the matched MLP pair with a confidence-score attack using disjoint 2023 members/non-members. Record the balanced 1:1 evaluation prior. Do not use 2025 as the non-member set.
- Train one DP-SGD target (ε=4, δ=1/N for the frozen MLP training sample). Record the accountant's achieved ε/δ and all accountant inputs; do not add an epsilon sweep to the core.
- Use one reviewed feature allowlist. Exclude the target, action-taken field, denial reasons, post-decision fields, and protected attributes from the main model inputs. Protected attributes are retained for audits only.
- Fit preprocessing on training data only. Record all filters, imputers, encoders, scalers, seeds, and sampling decisions.
- Full HMDA processing must require an explicit config. Unit and integration tests use synthetic data and must pass without HMDA files.
- Evaluate the boosted-tree model on 2025 only after the joint test protocol is frozen; the core input-stress test masks 10% of cells among five top non-sensitive features selected using 2023 training data.

## Privacy and output rules

- Keep HMDA source files, derived rows, checkpoints, row-level predictions, membership labels, and row-level explanations out of Git and out of reports.
- Use local data paths from configuration or an environment variable; never hard-code a user-specific absolute path.
- Publish only aggregate metrics, subgroup counts that are safe to report, and figures derived from aggregate result artifacts.
- If a report or demo needs an individual TreeSHAP illustration, use a clearly labeled synthetic HMDA-like record; never publish an explanation for a real row.
- For DP-SGD, report the achieved accountant ε and δ, sampling rate, number of steps, noise multiplier, clipping norm, and record-level adjacency assumption. Never report the requested target epsilon as if it were the achieved guarantee.
- Treat DP-SGD as a record-level guarantee. Do not describe it as person-level protection unless the data unit and contribution bounds support that claim.

## Coding standards

- Target Python 3.11 or newer. Keep the project CPU-runnable on macOS Apple Silicon and do not require CUDA. Begin DP-SGD on CPU; add MPS for the DP path only after a smoke test validates the exact Opacus training path and accountant output.
- Pin dependencies in one lockfile. Avoid adding libraries unless they implement a required method or replace meaningful custom code.
- Use type hints for public functions, validated config/result objects, small single-purpose modules, explicit inputs/outputs, and actionable errors. Avoid global mutable state and hidden defaults.
- Seed Python, NumPy, scikit-learn, and PyTorch from the run config. Record seed and device in every run manifest.
- Log structured run metadata. Do not catch broad exceptions and continue with incomplete results.
- Generate all tables and plots from saved aggregate result artifacts; never hand-edit values in a figure.
- Use Ruff for lint/format checks and pytest for tests. Test edge cases and leakage prevention, not only the happy path.

## Minimum tests

Write tests for:

1. HMDA outcome mapping and exclusion/unknown-code behavior.
2. Forbidden columns and protected attributes never entering model `X`.
3. Year split and transform fit scope; no 2025 records in training or validation.
4. Deterministic results for fixed seeds where the library supports determinism.
5. Fairness metric edge cases, including missing or small groups.
6. MIA member/non-member disjointness and recorded class prior.
7. DP accountant metadata and achieved ε/δ serialization.
8. A synthetic end-to-end smoke run producing aggregate metrics without writing record-level outputs.

## Stable command interface

Provide these Make targets with behavior described in the plan:

```bash
make setup
make lint
make test
make data-check
make baseline
make privacy
make fairness
make xai
make robustness
make all
```

`make lint`, `make test`, and a synthetic smoke run must not require HMDA data. `make data-check` must report missing files and expected placement clearly. Full experiments must not start as a side effect of setup, tests, or imports.

## Required handoff after each task

Report:

- what changed and why;
- exact commands run and their pass/fail results;
- any assumptions or unresolved decisions;
- the next milestone only after the current acceptance gate passes.

At the J-07 to K-01 handoff, also provide a concise README section or handoff note with setup steps, stable interfaces, config locations, available aggregate outputs, and known limitations. Keep raw files and row-level artifacts local.

