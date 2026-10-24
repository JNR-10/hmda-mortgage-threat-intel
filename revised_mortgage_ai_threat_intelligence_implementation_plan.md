# Revised Implementation Plan: Threat-Informed, Trustworthy Mortgage Outcome Modeling

**Team:** Jainil Rana and Krati Jain  
**Course:** CMPE 253 — AI Threat Intelligence  
**Plan status:** Revised implementation baseline, 23 October 2026

> This plan uses the supplied course bundle to understand the project and Track B, but treats it as an older course-material version. Its technical expectations are useful planning context; checkpoint dates, page limits, and any changed rubric must come from the current course instructions.

## 1. Project assessment

The project is a strong Track B choice. Mortgage decision modeling gives the team a realistic, high-impact setting in which privacy, disparate error rates, explanation quality, and changes over time all have practical consequences. Public HMDA data make a substantial offline study feasible, and the proposal already recognizes two essential limits: HMDA records historical lender actions rather than objective creditworthiness, and the resulting model must not be presented as an automated mortgage approval system.

The main improvement is to make the AI Threat Intelligence contribution visible. In its current form, the proposal reads primarily as a responsible-ML comparison. The implementation should connect each measured risk to an adversary or failure hypothesis, the asset at risk, observable evidence, a test, a control, and residual risk. This adds a compact threat-intelligence layer to Track B without turning the project into Track A or adding live-system testing.

The second issue is scope. The proposal combines multiple models, several privacy budgets, fairness controls, multiple explainers, counterfactuals, temporal shift, and perturbation tests. A full Cartesian product would be expensive and hard to interpret. The recommended scope keeps the Track B dimensions but assigns each one a clear subject: logistic regression is the reference baseline; gradient-boosted trees are the preselected model for fairness, explanation, and robustness; a matched non-private/DP MLP pair is used only for the privacy experiment. This avoids choosing a different model for every result and keeps the comparisons understandable.

## 2. Revised framing and scope

### Recommended title

**Threat-Informed Evaluation of Privacy, Fairness, Explainability, and Robustness in an HMDA-Based Mortgage Outcome Model**

This title makes the course focus explicit while preserving the proposal's research subject.

### System context and empirical subject

Use Fannie Mae's Desktop Underwriter (DU) as a **publicly documented deployment context** for automated mortgage underwriting. Fannie Mae describes DU as its automated mortgage loan underwriting system [1]. The empirical subject remains an independently trained, offline HMDA-based research proxy. It is **not DU**, does not reproduce DU's internals, and will not be used to make or recommend lending decisions. No DU interface, production model, or user data will be accessed.

This distinction addresses the older Track B material's expectation for a known, deployed or well-documented use case while avoiding an unsupported claim that public HMDA data can reproduce a proprietary underwriting system. If current course instructions require experiments on the exact deployed system rather than a transparent proxy in its use-case context, that requirement must be resolved against the current rubric before implementation.

### Revised main research question

> For an offline HMDA proxy that predicts reported historical mortgage outcomes, what privacy–utility change does a fixed DP-SGD intervention produce in a matched MLP pair, and what fairness, explanation-reliability, and robustness risks are measurable in a preselected gradient-boosted-tree model?

### Research questions

1. **Baseline utility:** How do logistic regression and gradient-boosted trees perform on the same frozen data contract, with the boosted-tree model fixed in advance as the main audit subject?
2. **Privacy/threat:** On a matched MLP pair, how does one predeclared DP-SGD setting change black-box membership-inference results and predictive utility?
3. **Fairness:** On the fixed boosted-tree model, how do the selected group metrics change under reweighting and equalized-odds post-processing, alongside utility and calibration?
4. **Explainability and robustness:** Are global/local TreeSHAP explanations stable and behaviorally faithful, and how does the boosted-tree model change under the 2025 temporal holdout and one bounded missingness stress test?

### Chosen scope at a glance

| Area | Core scope |
|---|---|
| System and data | Public HMDA-based proxy in the documented mortgage-underwriting context; no claim that the project evaluates DU. Sample up to 100,000 labeled records per year. |
| Data split | 2023 model fit with a 20% same-year MIA non-member holdout; 2024 for tuning/thresholds; 2025 sealed until final testing. |
| Utility baselines | Logistic regression reference plus `HistGradientBoostingClassifier`, fixed in advance as the audit model. |
| Privacy | Matched non-private/DP MLP pair at one target ε = 4, with δ = 1/N; one confidence-score MIA comparing the pair. |
| Fairness | Two group metrics and two mitigations on the boosted-tree audit model only. |
| Explanations | Global and local TreeSHAP on that model, with three-seed stability and a faithfulness check. Any displayed local example is synthetic. |
| Robustness | One 2025 temporal evaluation and one 10% missingness stress on the boosted-tree audit model. |
| Threat intelligence | A six-risk register with evidence, access assumptions, controls, and residual risk; measure MIA and input failure, and record provenance checks. Treat poisoning as an untested risk. |
| Out of scope | Live DU access, real applicant data, shadow-model MIA, poisoning experiments, LIME, counterfactuals, extra privacy budgets, and a full model/control factorial. |

### Explicit boundaries

- **In scope:** public, privacy-modified HMDA data; offline model training; synthetic or held-out evaluation; documented public information about the deployment context; fairness, privacy, explanation, and robustness metrics; proposed controls and monitoring signals.
- **Out of scope:** actual DU or lender-model evaluation; production access; live API queries; real applicant or customer data; a deployable approval/rejection system; claims of lender compliance, causal discrimination, or creditworthiness.
- **Interpretation:** the target is a historical HMDA action outcome. Results describe how a research model learns and behaves relative to those reported outcomes. They do not establish whether the historical decisions were correct or fair.

## 3. How Track B fits

The supplied Track B deck and checklist describe an end-to-end, reproducible privacy/fairness/xAI pipeline with at least two baselines, control ablations, multi-criteria measurements, uncertainty reporting, and robustness checks. The proposal fits those expectations; the work packages below make each one auditable.

| Track B expectation in the supplied older materials | Project implementation and evidence |
|---|---|
| Realistic, known or documented use case | Public DU documentation is context; the report labels the HMDA-trained model as an independent research proxy. |
| Ethical and feasible data | Public HMDA only; record the year, download date, source, hash, filtering rules, and privacy modifications. |
| Clear target and leakage controls | Explicit binary target; exclude outcome/denial fields and fit every transform on training data only. |
| At least two model baselines | Logistic regression and gradient-boosted trees on identical splits and preprocessing; the boosted-tree model is the predeclared audit subject. |
| Privacy measurement | One score-based membership-inference attack and one DP-SGD target setting on a matched MLP pair; no shadow-model sweep. |
| At least two fairness metrics | Demographic-parity difference and equalized-odds difference on the boosted-tree audit model, with group counts and uncertainty. |
| Fairness mitigation | Reweighting and equalized-odds post-processing on that same model, with settings fixed on 2024. |
| Global and local explanation | TreeSHAP global and local explanations for the boosted-tree audit model; test stability and faithfulness. Do not add LIME or counterfactuals to the core. |
| Robustness | 2025 temporal holdout plus one predeclared missingness stress test on the boosted-tree audit model. |
| Statistical rigor | Paired confidence intervals, appropriate paired tests, multiple seeds for selected experiments, and a subgroup sensitivity/power note. |
| Reproducibility | Pinned environment, fixed seeds, data/run manifests, scripts, configurations, metrics, and plots. |

## 4. Coursework integration that contributes to the result

The plan integrates **11 course themes**. Each is tied to a test, decision, or deliverable rather than added as background decoration.

| Course theme | Practical integration | Resulting artifact or evidence |
|---|---|---|
| AI threat intelligence and adversary TTPs | Write attack hypotheses with actor, capability, entry point, target asset, impact, and evidence. Map only applicable techniques to MITRE ATLAS; do not claim an attack occurred. | Threat register and concise attack-path narratives. |
| Threat modeling and risk scoring | Define anchored 1–5 likelihood and impact scales; score inherent and residual risk with a rationale and evidence confidence. | Risk register with control and residual-risk fields. |
| AI governance / NIST AI RMF | Use Govern, Map, Measure, and Manage to structure context, measurement, controls, and monitoring. | Governance-to-experiment crosswalk and limitations. |
| Privacy and data protection | Minimize retained fields, document HMDA's public-use modifications, keep record-level outputs local, and release aggregated results only. | Data card, field allowlist, and handling notes. |
| Membership inference | Simulate a black-box attacker using model scores to infer whether a record was in training. | Attack AUC, advantage, precision/recall under a stated attack prior. |
| Differential privacy | Train a matched non-private/DP-SGD MLP pair at one predeclared target ε and report actual accountant output, δ, sampling assumptions, and utility. | Privacy comparison and DP configuration manifest. |
| Fairness definitions and measurement | Report selection-rate and error-rate gaps on one fixed audit model by race/ethnicity and sex; report intersections only when counts support useful uncertainty. | Group metrics, counts, and confidence intervals. |
| Fairness mitigation | Compare reweighting and equalized-odds threshold post-processing on that same model. | Before/after disparity and utility table. |
| Explainable AI | Use one model-specific method (TreeSHAP) for global and local explanations of the fixed audit model. | Attribution summaries plus stability and faithfulness checks. |
| Adversarial robustness, OOD, and drift | Evaluate one later-year holdout and one bounded missingness stress; distinguish input corruption from ordinary market drift. | Temporal and missingness results, with explicit interpretation. |
| Auditability, statistical practice, and reproducible engineering | Pin dependencies, log seeds/configs, preserve a held-out test set, quantify uncertainty, and state what the data cannot establish. | Reproducible run package, evidence table, and sensitivity/power note. |

## 5. Small project changes and why they are made

| Change | Why it is needed |
|---|---|
| Add a one-paragraph DU deployment-context note while keeping HMDA as the proxy's training/evaluation data. | Makes the deployed use case explicit, as Track B expects, without implying access to or evaluation of DU. |
| Add a compact ATLAS-informed threat register and map the privacy and input-integrity tests to threat hypotheses. | Makes the project clearly about AI Threat Intelligence while preserving Track B's pipeline focus. |
| Make the research question explicitly include one membership-inference test, one DP-SGD comparison, fairness controls, and temporal/input stress. | Aligns the question with the actual experiments and avoids a broad “responsible AI” claim. |
| Preselect gradient-boosted trees as the single audit model; keep logistic regression as the reference baseline and use an MLP pair only for privacy. | Gives fairness, explanation, and robustness results one consistent subject, limits model/control combinations, and avoids post-hoc model selection driving the risk narrative. |
| Restrict DP-SGD to the MLP, with a matched non-private MLP control. | DP-SGD is a training algorithm for differentiable models; this avoids implying the same DP method applies directly to every estimator. |
| Use one DP-SGD target setting (ε = 4; set δ after the training sample is frozen) and one matched non-private control. | A single controlled privacy ablation is enough for the core question and is more feasible than a multi-budget sweep. Report the achieved accountant ε and δ. If the target is infeasible or utility is unusable, report that result rather than expanding the sweep. |
| Use a default cap of 250,000 labeled records per year with deterministic uniform sampling when source files exceed that cap; retain the natural outcome prevalence and log sampling details. | Bounds Apple Silicon memory and runtime while keeping the split and experiments repeatable. Revisit the cap only after a resource pilot and before the test protocol is frozen. |
| Run fairness, explanations, and robustness only on the preselected boosted-tree model; do not cross these with DP and every baseline. | Preserves the Track B comparisons while controlling compute and interpretation risk. The MLP privacy experiment remains a separate matched comparison. |
| Keep one TreeSHAP method and one bounded missingness stress test; leave LIME, counterfactuals, shadow-model MIA, and extra corruption scenarios as stretch work. | These additions are not required to answer the core questions and would compete with subgroup uncertainty and reproducibility work. |
| Treat record-level DP as a record-level guarantee and discuss the possibility of multiple applications per person. | Prevents the team from overstating an individual-level guarantee that the dataset structure may not support. |
| Keep counterfactual explanations exploratory, not a core deliverable. | Feasible counterfactuals for lending variables are difficult to establish from public HMDA fields and can be misread as advice to applicants. TreeSHAP stability and faithfulness checks address explanation quality more directly. |
| Update the proposal's regulatory citation from CFPB Circular 2022-03 to current Regulation B §1002.9, and describe it as context rather than a compliance test. | CFPB lists Circular 2022-03 as withdrawn on May 12, 2025 [5]; current §1002.9 remains the appropriate primary regulatory text for the specific-reasons context [6]. |
| Use relative sprints, not the old bundle's week numbers. | The supplied files disagree about early checkpoints (Week 6, Week 8, and Week 13 references). The current course calendar should determine dates. |

## 6. Data and modeling design

### 6.1 Data contract

- **Source:** CFPB/FFIEC HMDA public data portal. HMDA's public loan-level files are modified to protect applicant and borrower privacy [2]. Save the source URL, data year, download timestamp, file hash, and a copy of the data dictionary used.
- **Years and split:** sample each year independently. For 2023, reserve a fixed 20% of the sample as non-members for the MIA and train every model on the remaining 80%; use 2024 for tuning and threshold selection; keep 2025 as the final temporal holdout. Do not inspect 2025 outcomes until features, model settings, attack definitions, and thresholds are frozen.
- **Primary cohort:** completed applications with a clean binary outcome. Positive: originated or approved but not accepted. Negative: denied. Exclude withdrawn, incomplete, purchased, and preapproval-only records from this primary task; preserve counts and exclusion reasons in a flow table.
- **Target caveat:** label the outcome “reported historical application outcome,” never “creditworthy,” “good borrower,” or equivalent.
- **Feature allowlist:** use only fields plausibly available at application/decision time, such as loan amount, applicant income, debt-to-income and loan-to-value categories, property value, loan type/purpose, occupancy, lien status, and selected coarse geography. Freeze the exact field list after a data dictionary review.
- **Excluded fields:** action taken and all target-derived fields; denial reason; fields only known after the decision or loan closing; institution ID as a predictive feature. Institution IDs may be retained for audit and grouped sensitivity analysis.
- **Sensitive attributes:** race/ethnicity, sex, and age are held for fairness auditing, not used as predictive inputs in the main models. Report that other variables may still act as proxies.
- **Preprocessing:** harmonize annual schema changes and document special codes and missingness. Fit imputers, encoders, scalers, and any feature selection only on the 2023 model-fit records; apply the frozen transforms to 2024 and, after the protocol is frozen, 2025. Keep preprocessing identical for logistic regression and boosted trees.
- **Sampling:** use a deterministic uniform sample capped at 100,000 labeled records per year; if the eligible cohort is smaller, use all of it. For 2025, select the sample only after the protocol is frozen. Scan full files for provenance/schema checks without loading all rows into memory. Freeze the sampling seed before outcomes are opened, retain natural prevalence, and report group counts. Revisit the cap only after a small runtime/memory pilot and before T-01 is approved.
- **Integrity controls:** keep raw source files read-only; verify file hashes; record schema, row counts, filters, and data-version identifiers. Any corrupted-input stress test acts only on an experimental copy.

### 6.2 Models and staged ablation

| Configuration | Role |
|---|---|
| Logistic regression | Reference utility baseline; no fairness/XAI/robustness variant grid. |
| `HistGradientBoostingClassifier` from scikit-learn | Nonlinear baseline and preselected primary audit model. It supports the sample-weight mitigation and TreeSHAP workflow; keep its configuration modest and tune only on 2024 [9], [10]. |
| MLP without DP-SGD | Matched control for the privacy experiment only. |
| MLP with DP-SGD | Privacy intervention at one target ε = 4, with δ set to 1/N for the frozen MLP training sample; report the accountant's achieved values. |
| Boosted-tree model with reweighting | Fairness mitigation A; reweighting is fitted on 2023 training data. |
| Boosted-tree model with equalized-odds post-processing | Fairness mitigation B; thresholds are selected on 2024 validation data. |

Use identical frozen data splits and input features for the logistic and boosted-tree utility baselines. The boosted-tree model is designated as the fairness, explanation, and robustness subject before results are observed; do not choose it after comparing test results. The MLP pair is a separate privacy ablation. Do not cross DP, fairness, and explanation variants into a full factorial. Use 2024 only for bounded hyperparameter/threshold selection; never tune on 2025.

### 6.3 Membership-inference experiment

- **Adversary:** an external black-box user who can observe model confidence scores for candidate records, but has no access to parameters, training files, or a production lender API.
- **Probe:** use one confidence-based attack score (maximum predicted class probability) against candidates from the 2023 training distribution. Compare records in the 80% model-fit partition with the held-out 20% non-member partition; non-members are never used for fitting or validation. Do not use 2025 as the non-member set because year shift would confound membership with distribution. Do not build shadow models or train a separate attack classifier in the core.
- **Evaluation:** draw equal-sized member and non-member candidate sets using a fixed seed, then report attack AUC, maximum attack advantage (TPR − FPR), bootstrap uncertainty, and precision under the explicit 1:1 evaluation prior. State that this balanced prior is an analysis assumption, not an estimate of real-world attacker prevalence. The score-based attack requires no fitted attack model.
- **Comparison:** run the same attack protocol on the matched non-private and DP MLPs only. Do not extend MIA to logistic regression or boosted trees in the core.
- **Interpretation:** a score near random is evidence only for this offline attack, model, output access, and sampled data; it is not proof of privacy against every attacker.

### 6.4 Fairness measurements and controls

- **Primary groups:** race/ethnicity and sex. Include intersectional slices when each cell has enough support to produce useful uncertainty; otherwise show the count and label the estimate underpowered.
- **Metrics:** demographic-parity difference and equalized-odds difference (maximum of TPR and FPR gaps). Include group-specific TPR/FPR and the underlying counts. Add group calibration as a secondary diagnostic if time permits.
- **Controls:** compare the unmitigated boosted-tree audit model, sample reweighting, and equalized-odds threshold post-processing. Fit reweighting on training data and select post-processing thresholds using 2024 only. This is an offline audit experiment, not a proposal to use protected attributes at inference.
- **Trade-offs:** show predictive utility and calibration beside every fairness result. Interpret disparities relative to the historical label, not as proof of discrimination, legal compliance, or causation.

### 6.5 Explainability and robustness

- **Global and local explanation:** use TreeSHAP for global summaries and per-record local explanations from the boosted-tree audit model. Use a fixed 100-row background sample and explain at most 1,000 validation records per seed to bound runtime. Keep actual row-level explanations local and report aggregate attribution patterns only. If the report/demo needs a local waterfall plot, use a clearly labeled synthetic HMDA-like record; do not present a real applicant's explanation or frame any explanation as applicant advice.
- **Explanation checks:** measure top-k stability across three fixed model seeds and faithfulness by comparing score changes after removing the most-attributed features with a random-feature baseline. Treat these as model-behavior checks, not causal explanations. Do not add LIME or a separate randomization study to the core.
- **Temporal robustness:** evaluate the frozen boosted-tree model on 2025. Report changes in AUROC, AUPRC, balanced accuracy, calibration, and the two primary fairness metrics; include one feature-distribution drift statistic.
- **Input robustness:** run one predeclared missingness stress test on a copy of the 2025 evaluation sample: set 10% of cells among the five highest-impact non-sensitive features (identified from 2023 training data) to missing, then apply the already-fitted preprocessing/imputation path. Report utility and fairness changes with uncertainty. This is a controlled input-failure test, not a claim of a feasible borrower attack.
- **Optional work:** additional corruption levels, OOD detection, counterfactuals, LIME, and shadow-model MIA are stretch items only after every core result is reproducible.

## 7. Threat-intelligence layer

Use MITRE ATLAS as a living AI-adversary knowledge base [3]. The mapping below is an initial hypothesis list; confirm exact technique names and IDs against ATLAS when writing the final register. Do not map ordinary fairness gaps or temporal market drift to an adversary technique unless the mechanism actually fits.

| Hypothesis | Asset / possible harm | Test or observable signal | Planned control / residual-risk discussion |
|---|---|---|---|
| A black-box user infers whether a record was in training (ATLAS: infer training-data membership). | Privacy of represented records; possible disclosure of participation. | Score-based MIA AUC/advantage against the stated attack prior. | DP-SGD, output minimization as a proposed deployment control, access monitoring; residual risk depends on ε, sampling assumptions, and the attack model. |
| Missing or invalid input fields change a model result (input-integrity failure hypothesis). | Prediction integrity and downstream decision quality. | One 10% missingness stress test on a copy of the final evaluation sample. | Input validation, range/consistency checks, human review for invalid records; this is a controlled failure test, not evidence of a live evasion attack. |
| Training data or preprocessing is altered (data-integrity / poisoning hypothesis). | Model integrity and auditability. | No poisoning implementation in the core plan; inspect provenance, hashes, schema and distribution checks. | Immutable source snapshots, reviewable transformation code, dataset checks, and retraining approval; state that empirical poisoning resistance is untested. |
| The model has unequal error rates across groups. | Unequal error burden and audit blind spots. | Group fairness metrics on the boosted-tree audit model. | Review thresholds, error patterns, and data limitations; this is a socio-technical risk and not necessarily malicious activity. |
| Later market conditions shift from the training period. | Reliability degradation and increased errors. | 2025 temporal metrics and feature-drift statistic. | Monitoring thresholds and revalidation; treat drift as a reliability failure unless there is evidence of malicious cause. |
| Explanations are unstable or do not track model behavior. | Auditability and ability to investigate a result. | TreeSHAP stability across three fixed boosted-tree seeds and a top-feature ablation check. | Explanation validation and model review; post-hoc explanations do not establish causality or satisfy a real lender's obligations by themselves. |

Define likelihood and impact before assigning scores. Example anchors: **likelihood 1** requires privileged insider conditions; **5** requires broadly available capability and exposure; **impact 1** is limited and reversible; **5** is large-scale privacy, access, or financial harm. Give each score a written rationale, evidence confidence, planned control, and residual score. These are planning hypotheses until supported by project evidence.

## 8. Measurement and analysis plan

| Dimension | Core reported measures | Uncertainty / interpretation |
|---|---|---|
| Utility | AUROC, AUPRC, balanced accuracy, and Brier score for logistic regression and the boosted-tree model; utility/calibration for the MLP pair. | 95% paired bootstrap CIs on held-out predictions; report class prevalence. |
| Privacy | Confidence-score MIA AUC and advantage; DP ε and δ from the matched MLP accountant. | Bootstrap CIs; state the balanced attack prior, record-level adjacency, and black-box score access. |
| Fairness | Demographic-parity difference and equalized-odds difference on the boosted-tree audit model, with group TPR/FPR and counts. | Bootstrap CIs; flag low-powered intersectional slices. |
| Explainability | TreeSHAP top-k stability across seeds and score change under top-feature removal versus random removal. | Use only the boosted-tree audit model; report aggregate summaries and use synthetic records for any local illustration. No causal interpretation. |
| Robustness | Boosted-tree metric change on 2025 and under the single 10% missingness stress; one drift statistic. | Paired intervals; call temporal change drift and missingness a controlled input failure, not a demonstrated attack. |
| Risk | Initial and residual L×I scores with evidence confidence and control coverage. | Qualitative judgment is labeled as such and tied to the defined scales. |

Run three fixed seeds for the boosted-tree audit model and the matched MLP privacy comparison; logistic regression can be reported as a deterministic reference fit. Use paired bootstrap for metric differences on the same test records; consider McNemar's test for paired classification errors when appropriate. The small number of predeclared comparisons should be listed explicitly; label any added analyses exploratory. Include a subgroup sensitivity note with group counts and uncertainty. Do not call overlapping/noisy estimates “equal”; report uncertainty and limits.

## 9. Implementation work packages and exit gates

Use relative sprints and map them to the current course schedule after checking the current instructions.

| Sprint | Work | Suggested lead | Exit gate / artifact |
|---|---|---|---|
| 0 — Shared scope gate | Review current Track B rubric; freeze research question, proxy wording, target, year split, primary audit model, minimum experiments, and report/demo expectations. Confirm whether a proxy is acceptable. | Shared | T-00 decision note and initial experiment boundaries; no coding before any critical course-selection issue is resolved. |
| 1 — Jainil starts from scratch | Create repository and synthetic scaffold; validate HMDA sources/schema; profile 2023/2024 development data and prepare the data contract. | Jainil | J-01/J-02 acceptance checks; 2025 outcomes stay sealed. |
| 2 — Shared protocol gate | After the data profile, freeze target, features, sample cap/seed, model split, MIA protocol, fairness metrics, and thresholds. | Shared | T-01 recorded before J-03 model training. |
| 3 — Jainil data/ML phase | Build preprocessing/splits, logistic and boosted-tree baselines, matched MLP privacy pair, MIA, and reproducible run artifacts. | Jainil | J-03–J-07 pass; handoff package includes stable interfaces and aggregate results only. |
| 4 — Krati analysis phase | Build public-use context and threat register, audit fairness and mitigations, run TreeSHAP checks, evaluate temporal/missingness robustness, and update residual risks. | Krati | K-01–K-06 artifacts trace to frozen configs and approved outputs. |
| 5 — Shared closeout | Integrate the two workstreams, verify statistics and claims, complete limitations and final evidence, write the report, and prepare the demo. | Shared | T-02 complete; every reported metric traces to a run/config. |

The ownership split below is the working team assignment: Jainil owns the HMDA/data and ML pipeline, baselines, privacy, and membership-inference work; Krati owns system context and the threat register, fairness, explainability, and robustness. The main execution is sequential: Jainil starts from the empty repository, completes and hands off his data/ML work, Krati then completes her work on the shared interfaces and outputs, and both finish integration and reporting together. T-00 and the minimum T-01 decisions are shared gates that must happen before model implementation; final integration, analysis, the report, and the demo happen after both individual workstreams. Each owner should get a second-person review of data leakage, metric definitions, and interpretation.

### Suggested repository and artifact layout

```text
project/
  README.md
  configs/                 # frozen experiment configurations
  data/README.md            # source, hashes, and retrieval steps; no raw data committed
  src/                      # loading, filtering, modeling, attacks, metrics
  notebooks/                # exploration only; final metrics run from scripts
  reports/                  # figures, tables, evidence inventory
  runs/                     # manifests and aggregated outputs
  environment.yml           # or pinned requirements/lock file
  Makefile                  # data-check, baseline, privacy, fairness, xai, report targets
```

Prefer a small script-driven pipeline over a collection of manually executed notebooks. Use MLflow or a plain structured JSON/CSV manifest for experiment tracking; adopt DVC only if the team already knows it. Keep raw HMDA files out of Git and write only aggregate metrics to the report outputs.

## 10. Minimum viable results and stretch work

### Required core

- Logistic regression reference and preselected boosted-tree audit model on the same frozen splits.
- Non-private MLP vs. DP-SGD MLP at one target ε, with actual ε/δ and one confidence-score membership-inference comparison.
- Two fairness metrics, subgroup counts/uncertainty, and two mitigation methods on the boosted-tree audit model.
- Global and local TreeSHAP explanations for that model, with three-seed stability and one faithfulness check.
- One 2025 temporal evaluation and one 10% missingness stress test on the boosted-tree audit model.
- Confidence intervals, a sensible paired comparison, and a power/sensitivity note.
- ATLAS-informed threat hypotheses, defined L×I scales, controls, residual-risk reasoning, and evidence table.
- Reproducible scripts, pinned dependencies, seeds, configuration and artifact logs.

### Stretch only after the core is stable

- Additional privacy budgets, shadow-model MIA, group calibration, a third fairness control, LIME, counterfactuals, additional corruption levels, a formal OOD detector, or institution-clustered sensitivity analysis.

## 11. Risks and responses

| Risk | Response |
|---|---|
| Proxy is confused with the actual DU model | Label the research proxy in the title, diagrams, figures, and conclusions; separate public system facts from experiment observations. |
| Current course instructions differ from the older files | Recheck current rubric, deadlines, and deliverable format before Sprint 0 is closed; retain relative sprint plan until then. |
| HMDA historical labels are treated as truth | Use “reported historical outcome”; repeat target caveat in abstract, results, and conclusion. |
| Target leakage from action/denial/outcome-dependent columns | Maintain a reviewed feature allowlist; add automated checks that target and denial fields never enter model matrices. |
| Public data are privacy-modified | Document modification status and do not make claims about raw applicant privacy from model DP results. |
| Repeated applications weaken person-level DP interpretation | Define record adjacency and discuss that multiple records per person can require group-privacy analysis. |
| Small intersectional groups yield unstable estimates | Report counts and CIs; mark low-power results as descriptive or suppress a conclusion. |
| DP-SGD damages utility or exceeds compute budget | Run one small feasibility pilot, then attempt the single predeclared ε = 4 comparison on the frozen sample. If it is infeasible or utility collapses, report that finding; do not quietly change ε, δ, sample, or expand into an unplanned sweep. |
| Fairness metric appears to improve only because model utility falls | Pair fairness metrics with AUROC/AUPRC, balanced accuracy, and calibration. |
| Temporal drift is mislabeled as an attack | Keep it as a reliability shift scenario unless evidence supports malicious causation. |
| Post-hoc explanation is misrepresented as a reason for a real lending action | State that explanations describe the research model and are not a legal notice or compliance validation. |

## 12. Report and demo structure

Follow the **current** report template and page limit. A coherent narrative is:

1. Problem, project contribution, and proxy boundary.
2. Deployment context, public evidence, stakeholders, and dataflow.
3. HMDA data contract, target, exclusions, and limitations.
4. Threat model, ATLAS mapping, likelihood/impact scales, and risk register.
5. Experimental protocol, splits, baselines, controls, and metrics.
6. Results: utility, privacy/MIA, fairness, explanations, and robustness.
7. Statistical uncertainty, sensitivity/power, and trade-offs.
8. Controls, residual risks, governance/monitoring implications, and limitations.
9. Reproducibility instructions and references.

The demo should show one end-to-end run path, the main baseline/control comparison, the multi-criteria results table, and one example of how a result changes the risk assessment. Avoid presenting the prototype as a lending tool.

## 13. Current evidence sources

Course-specific Track B interpretation is based on the supplied `AI_Threat_Intelligence_Track_A_vs_Track_B.pptx`, `Initial_Report_Checklists_TrackA_TrackB.pdf`, `AI_Threat_Intelligence_CMPE_253_Fall_2026.pdf`, and the team's `Mortgage_Responsible_ML_Project_Proposal_IEEE.docx`. These documents are older-version planning inputs.

1. Fannie Mae, [Desktop Underwriter and Desktop Originator](https://singlefamily.fanniemae.com/applications-technology/desktop-underwriter-desktop-originator) — public description of DU as an automated mortgage underwriting system.
2. CFPB/FFIEC, [HMDA Data Publication](https://ffiec.cfpb.gov/data-publication/) and [Modified Loan/Application Register](https://ffiec.cfpb.gov/data-publication/modified-lar) — public data access and privacy-modification context.
3. MITRE, [ATLAS](https://atlas.mitre.org/) — AI adversary techniques and tactics; use current technique names in the final register.
4. NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — voluntary AI risk management resource and lifecycle framing.
5. CFPB, [Withdrawn Guidance](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/) — lists Circular 2022-03 as withdrawn May 12, 2025.
6. eCFR, [12 CFR §1002.9, Notifications](https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.9) — current regulatory text for adverse-action notices and specific reasons.

7. PyTorch, [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) — current device documentation for Apple Silicon's Metal backend.
8. Opacus, [Introduction](https://opacus.ai/docs/introduction.html) — DP-SGD integration and model-compatibility context; verify the exact device/model path used by the project.
9. SHAP, [TreeExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) — TreeSHAP method and supported model families.
10. scikit-learn, [HistGradientBoostingClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) — CPU-oriented histogram gradient boosting with sample-weight support.

## 14. Coding-agent handoff specification

The research plan above defines what to study. This section turns it into a buildable project structure and code contract for a repository created from scratch. There is no existing codebase to extend: create a new repository, put this plan and the companion `AGENTS.md` at its root, and implement one milestone at a time. Jainil owns the initial scaffold and all J-01–J-07 data/ML tickets. After J-07, he hands the repository, stable interfaces, run manifests, and aggregate results to Krati for K-01–K-06. The team completes T-02 together. T-00 and the pre-model decisions in T-01 are joint gates before J-03 so the modeling protocol is fixed before experiments begin.

### 14.1 Repository layout

```text
mortgage-threat-intel/
  AGENTS.md
  README.md
  pyproject.toml              # runtime and dev dependencies, tool configuration
  Makefile                    # repeatable setup, test, and experiment commands
  .gitignore                  # virtualenvs, local data, checkpoints, generated outputs
  configs/
    base.yaml                 # paths, years, target, feature allowlist, seeds
    privacy.yaml              # matched MLP settings, clipping/noise, δ and one ε target
    fairness.yaml             # audited groups, metrics, mitigation settings
  src/mortgage_threat_intel/
    __init__.py
    cli.py                    # command-line entry point; dispatch only
    config.py                 # validated config loading
    data/
      contracts.py            # required columns and typed output structures
      labels.py               # action_taken -> binary target and exclusions
      ingest.py               # local HMDA discovery and read/query
      prepare.py              # schema harmonization and train-fitted transforms
      splits.py               # year split and MIA member/non-member partition
    models/
      interface.py            # common fit/predict-probability contract
      baselines.py            # logistic regression and boosted trees
      dp_mlp.py               # MLP and DP-SGD adapter
    privacy/
      membership.py           # score-based membership attack and metrics
      accounting.py           # ε/δ extraction and privacy metadata
    fairness/
      metrics.py               # group and intersectional metrics
      mitigation.py            # reweighting and equalized-odds post-processing
    explainability/
      attribution.py           # global/local TreeSHAP for the boosted-tree audit model
      checks.py                # stability and faithfulness checks
    robustness/
      drift.py                 # temporal feature and performance shift
      perturbations.py          # one 10% missingness stress test
    evaluation/
      metrics.py               # utility and calibration metrics
      uncertainty.py           # paired bootstrap and comparison tests
      artifacts.py             # aggregate result tables and run manifests
    threats/
      register.py              # risk register schema and scoring helpers
  tests/
    unit/                      # labels, filters, transforms, metrics, config
    integration/               # synthetic end-to-end run; no downloaded HMDA needed
    fixtures/                  # tiny synthetic HMDA-like records only
  data/
    raw/                       # local source files; always git-ignored
    processed/                 # local derived data; always git-ignored
  reports/                     # context, evidence, threat register, figures, and tables
  outputs/                     # local models, aggregate results, figures; git-ignored
```

Keep final logic in importable modules and CLI commands. Notebooks may be used for exploration, but a final result must be reproducible from a script/Make target and configuration file.

### 14.2 Coding guidelines

1. **Runtime and portability:** target Python 3.11 or newer, use a single lockfile, and keep the project CPU-runnable on macOS Apple Silicon. Start DP-SGD on CPU. A PyTorch MPS path may be added after a correctness smoke test for the exact DP training path; ordinary PyTorch MPS availability does not by itself prove every Opacus operation works there [7], [8]. Do not require CUDA.
2. **Small, typed modules:** use type hints for public functions, dataclasses or validated models for configs/results, and functions with explicit inputs/outputs. Avoid global mutable state and hidden configuration.
3. **Determinism:** seed Python, NumPy, scikit-learn, and PyTorch from config. Log every seed and the device used. Make nondeterministic operations explicit in the README and run manifest.
4. **No silent data changes:** target mapping, filtering, code normalization, missingness treatment, and feature inclusion must be named, tested, and logged. Unknown HMDA codes should fail or be counted and reported; do not silently coerce them.
5. **Leakage prevention:** one reviewed feature allowlist feeds all models. Assert that the target, action-taken field, denial reasons, and post-decision fields are absent from `X`. Fit preprocessing on 2023 training data only; apply the frozen transforms to later years.
6. **Privacy-aware outputs:** raw/processed records, per-record predictions, checkpoints, and membership attack data stay local and out of Git. Reports and committed artifacts contain aggregate results only. Do not put record IDs, row-level explanations, or source rows in logs.
7. **Consistent model interface:** every model adapter exposes `fit(X, y, seed, config)` and `predict_proba(X)`. Evaluation code consumes probabilities and labels through the same path; no model-specific metric logic hidden in training scripts.
8. **Fixed experiment scope:** logistic regression is the reference baseline; gradient-boosted trees are the preselected model for fairness, TreeSHAP, and robustness; the matched MLP pair is used only for one DP/MIA privacy comparison. Do not build a full factorial across models and controls.
9. **Explicit experiment metadata:** each run records config hash, data hashes/years, code revision if available, package versions, filters, feature list, split, seed, model settings, DP sampling/clipping/noise/steps, accountant ε/δ, and output paths.
10. **Errors and logging:** validate config and data at the boundary with actionable errors. Use structured logging; do not catch broad exceptions and continue with partial results.
11. **Tests before scale:** first use tiny synthetic fixtures. Add tests for label mapping/exclusion, target leakage, year split, transform fit scope, deterministic seed behavior, group metric edge cases, disjoint MIA sets, and DP-accountant metadata. Add one synthetic end-to-end smoke test before reading full data.
12. **Code quality:** format/lint with Ruff and use pytest. Keep dependency versions pinned. Avoid adding libraries unless they implement a required measurement or reduce meaningful code.
13. **Evidence over plots:** figures must be generated from saved aggregate result tables; do not hand-edit reported values. Each figure/table has a source run ID and config.

### 14.3 Data and model contracts

- Normalize HMDA annual schemas into one documented internal schema before splitting. Preserve the original year and source file fingerprint.
- Use a deterministic uniform sample capped at 100,000 eligible labeled rows per year. For 2023, reserve 20% as MIA non-members and use the other 80% to fit all model comparisons. Keep 2024 for selection and 2025 sealed for final evaluation.
- Implement the proposal's binary target explicitly: `action_taken` 1 or 2 maps to positive, 3 maps to negative; all other values are excluded from this primary task. A unit test must cover every known action code present in each input year and fail on an unseen code until reviewed.
- Split by year: 2023 model fit, 2024 validation, 2025 final test. Tuning/threshold selection may use 2024 only. Include a test that makes it impossible for rows marked 2025 to enter training or validation.
- Compare logistic regression and boosted trees for utility; use the boosted-tree model for fairness, TreeSHAP, and robustness. Use only the matched non-private/DP MLP pair for MIA.
- Partition the 2023 sample into 80% model-fit members and 20% non-members using a fixed seed and disjoint row fingerprints. The non-member partition cannot enter any fit or validation step.
- Protected attributes remain outside `X` in the main models and inside the audit dataset. Explicitly check this at model entry.
- All model and evaluation functions accept the validated configuration and the same feature names. Return typed aggregate results; keep row-level outputs in ephemeral local objects unless a privacy review explicitly approves storage.
- Train one DP-SGD MLP target (ε = 4, δ = 1/N for the frozen MLP training sample) against a matched non-private MLP. DP metadata is not inferred from the target epsilon: save the accountant's achieved ε at the chosen δ together with the sampling rate, number of steps, noise multiplier, clipping norm, and adjacency statement.

### 14.4 Repeatable commands

The first implementation should provide these Make targets (exact internal command syntax may vary, but the target names and behavior should be stable):

```bash
make setup          # create/install the pinned environment
make lint           # ruff check and format check
make test           # unit and integration tests on synthetic fixtures
make data-check     # validate configured HMDA files, years, schema, and hashes
make baseline       # train LR/boosted-tree baselines and save aggregate metrics
make privacy        # one MIA and DP-SGD comparison for the matched MLP pair
make fairness       # metrics and two mitigation variants on the boosted-tree audit model
make xai             # TreeSHAP checks on the boosted-tree audit model
make robustness     # temporal and input-stress evaluation
make all            # execute the frozen required experiment sequence
```

`make test`, `make lint`, and a synthetic `make all` smoke run must work without HMDA files. `make data-check` should explain exactly where to put the files and what years are missing. Full HMDA experiments must require an explicit config and never start accidentally as a side effect of installation or tests.

### 14.5 Milestone acceptance checks

**Milestone A — scaffold and data contracts**

- `README.md`, `AGENTS.md`, `pyproject.toml`, lockfile, `.gitignore`, configs, package structure, CLI, and Makefile exist.
- Synthetic fixtures exercise target mapping, exclusions, feature allowlist, temporal split, and config validation.
- `make setup`, `make lint`, and `make test` pass on a clean checkout without HMDA data.

**Milestone B — data contract and utility baselines (Jainil)**

- Data validation prints development-year row/label/subgroup counts and source fingerprints; for 2025 it checks only file presence, schema, and integrity.
- Logistic regression and the preselected boosted-tree audit model run on the same split/features; metrics are emitted with run IDs and config metadata.
- A deliberate leakage test fails if a forbidden feature is introduced.
- 2025 is not accessed by training or model selection.

**Milestone C — privacy, MIA, and handoff (Jainil)**

- MIA member and non-member candidates are disjoint; the 1:1 evaluation prior is recorded.
- DP results report achieved ε/δ and accountant inputs; non-private and private MLPs share architecture and data protocol.
- The matched MLP pair reports one DP target's achieved ε/δ, utility, MIA AUC/advantage, attack prior, and uncertainty.

**Milestone D — context, threat register, and fairness (Krati)**

- Context and threat artifacts distinguish public DU evidence from the HMDA proxy; the risk register uses T-01's frozen hypotheses.
- Fairness metrics and both mitigation methods run only on the boosted-tree audit model with counts and uncertainty.

**Milestone E — explainability, robustness, and risk update (Krati)**

- Global/local TreeSHAP, three-seed stability, the faithfulness check, 2025 evaluation, and the one 10% missingness stress can be rerun from config.
- Paired uncertainty and subgroup sensitivity notes are generated from saved aggregate outputs.

**Milestone F — integration and final reproducibility (shared)**

- The team checks that each result traces to a run/config and prepares the final report and demo.
- A clean checkout can reproduce the synthetic test path and documented commands; final figures trace to run IDs.

### 14.6 First prompt for Claude or Cursor

> This is a new project with no existing source code. Read `revised_mortgage_ai_threat_intelligence_implementation_plan.md` and `AGENTS.md` first; assume the repository contains only those two planning files. Jainil owns the first implementation phase. Build **Milestone A only**: create the package scaffold, pinned environment, configs, CLI/Make targets, synthetic fixtures, and tests for target mapping, exclusions, forbidden-feature checks, year split, and config validation. Do not download HMDA data or implement model training, DP-SGD, MIA, fairness mitigation, or explainability yet. Keep all real data paths local and git-ignored. Run `make lint` and `make test`, fix failures, and report the exact commands and results plus any assumptions that need a team decision. Stop after Milestone A for team review. Continue only with Jainil's J-02–J-07 tickets when explicitly asked, then prepare a handoff package for Krati. Do not implement Krati's K tickets in Jainil's phase; keep their interfaces and expected output paths stable.

This staged prompt is intentional: it gives a coding agent a bounded first task with tests and a review gate, instead of asking it to generate the whole research pipeline in one pass.

## 15. Individual work tickets for Jainil and Krati

The assignments below use the team's intended ownership split. Jainil is starting the implementation from an empty repository and owns the data/ML work; Krati owns the context/threat, fairness, explainability, and robustness work. The ticket owner produces the code or research artifact, adds the required tests/evidence, and requests review from the other member before merging.

| Workstream | Owner | Scope |
|---|---|---|
| HMDA intake, data contracts, preprocessing/splits, model interfaces, utility baselines, membership inference, DP-SGD, and run reproducibility | Jainil | J-01–J-07 |
| Deployment/use-case context, evidence inventory, threat register, fairness metrics and mitigations, explanation checks, robustness evaluation, and residual-risk update | Krati | K-01–K-06 |
| Scope and rubric decisions, data/experiment contract approval, cross-review, integration, final analysis, report, and demo | Shared | T-00–T-02 |

### Shared kickoff tickets

| Ticket | Owner | Work | Depends on | Done when |
|---|---|---|---|---|
| T-00 — Confirm current course requirements | Jainil + Krati | Before implementation, compare the current Track B rubric/calendar with the older files; decide whether the HMDA proxy satisfies the system requirement; record due dates and deliverable format. | None; complete before J-01. | A short decisions note records the current rubric, proxy interpretation, dates, and any instructor question. No old week number is treated as current. |
| T-01 — Freeze the data and experiment contract | Jainil + Krati | After J-02's profile, agree on the target mapping, exclusions, feature allowlist, sensitive audit fields, sample strategy, splits, metrics, subgroup rules, and MIA protocol. Freeze initial threat hypotheses from this plan and the proposal; Krati will document and evidence them in K-01 after handoff. | T-00 and J-02. | Decisions are committed to config/docs and jointly approved before J-03 modeling; the 2025 test stays sealed until the final protocol is frozen. |
| T-02 — Integrate, report, and demo | Jainil + Krati | Run the frozen pipeline, cross-check results, update the measured risk register, write the report, and prepare the demo. | All core tickets. | Clean synthetic smoke run passes; each reported number traces to a run/config; final limitations and proxy boundary are explicit. |

### Jainil-owned tickets

| Ticket | Work and deliverable | Depends on | Acceptance checks |
|---|---|---|---|
| J-01 — Repository scaffold | Create a new Git repository from the two planning files; create `src/`, configs, tests, `pyproject.toml`/lockfile, `Makefile`, `.gitignore`, CLI, and README. | T-00; first implementation ticket. | `make setup`, `make lint`, `make test`, and a synthetic smoke command work without HMDA files. Raw data and outputs are ignored by Git. |
| J-02 — HMDA intake and data profile | Implement file discovery, year/schema validation, source hashes, action-code counts, missingness, label counts, and subgroup counts for 2023/2024. For 2025, validate presence, schema, and integrity only; keep raw files local. | J-01. | `make data-check` reports inputs and discrepancies; unknown codes fail visibly; the profile contains safe aggregates and source fingerprints; 2025 outcomes remain sealed. |
| J-03 — Target, preprocessing, and split pipeline | Implement the target mapping, exclusions, train-fitted transforms, feature allowlist, 100k/year deterministic sample cap, and year split. Reserve 20% of 2023 as MIA non-members; train all models on the remaining 80%. | T-01 and J-02. | Tests cover known/unknown codes, forbidden features, transform fit scope, sample determinism, year isolation, fixed seeds, and disjoint MIA partitions. |
| J-04 — Baselines and shared model interface | Train logistic regression and the preselected gradient-boosted-tree audit model; implement the shared `fit`/`predict_proba` interface and report utility/calibration metrics. Tune only on 2024; do not choose the audit model after seeing results. | J-03. | Same records/features feed both baselines; results include run IDs/config; tests prove 2025 was not used for training/tuning. Publish stable interfaces for Krati's modules. |
| J-05 — Membership-inference probe | Implement one confidence-score black-box attack for the matched non-private/DP MLP pair; do not add a shadow model or attack classifier. | J-03 and J-06. | Reports AUC, advantage, precision under an explicit 1:1 evaluation prior, and uncertainty; tests verify member/non-member disjointness. |
| J-06 — DP-SGD MLP | Implement the matched non-private/DP MLP pair with one ε=4 target and δ=1/N for the frozen training sample; save actual accountant output and privacy metadata. | J-03. | Both models use the same architecture/data split; achieved ε/δ, sampling, steps, noise, clipping, and record-level adjacency are logged; no target epsilon is reported as achieved privacy. |
| J-07 — Reproducibility and handoff artifacts | Ensure commands, configs, seeds, package versions, hashes, and aggregate result tables are captured consistently; prepare the handoff note for Krati. | J-04 through J-06. | A clean synthetic run reproduces the same outputs under fixed settings; handoff contains stable interfaces and aggregate artifacts but no raw or row-level records. |

### Krati-owned tickets

| Ticket | Work and deliverable | Depends on | Acceptance checks |
|---|---|---|---|
| K-01 — System context, evidence, and threat register | After Jainil's handoff, document DU as public context and the HMDA model as a separate proxy; build the dataflow/stakeholder view, evidence inventory, ATLAS mappings, and evidence-backed risk register from the hypotheses frozen in T-01. | Handoff after J-07; T-01 hypotheses. | `reports/system_context.md`, `reports/evidence_inventory.csv`, and `reports/threat_register.yaml` are complete; claims, observations, and hypotheses are separated; all six risks have assets, assumptions, L×I rationale, test/control idea, and confidence. No new experiment is added without joint review. |
| K-02 — Fairness audit metrics | Implement subgroup/intersectional counts, demographic-parity difference, equalized-odds difference, group TPR/FPR, and uncertainty on the boosted-tree audit model. | Handoff after J-07; J-03/J-04 interfaces and outputs. | Synthetic tests cover edge cases; baseline results show group counts and CIs; low-powered slices are clearly labeled. |
| K-03 — Fairness mitigation | Apply reweighting and equalized-odds post-processing to the boosted-tree audit model and compare against its unmitigated version. | K-02 and J-04. | Before/after metrics include utility and calibration; thresholds/settings come from 2024, not 2025; protected attributes are not main model inputs. |
| K-04 — Explainability and quality checks | Implement global/local TreeSHAP for the boosted-tree audit model; measure stability across three seeds and faithfulness via top-feature removal versus random removal. | Handoff after J-07; J-04 predictions/model interface. | Reported explanations identify model/config; no LIME or counterfactuals in the core; publish aggregate explanations only and use a synthetic record for any local illustration. |
| K-05 — Temporal and input robustness | After joint protocol freeze, evaluate the boosted-tree model on 2025 and run one 10% missingness stress over the five top non-sensitive features identified on 2023 training data. | Handoff after J-07; J-03/J-04; T-01 freeze. | 2025 is opened only for final evaluation; temporal drift and the controlled missingness failure are labeled separately; report utility/fairness changes with uncertainty. |
| K-06 — Evidence-based residual-risk update | Revisit K-01's risk register using actual MIA, fairness, XAI, and robustness evidence; write residual risk and monitoring signals. | J-05/J-06, K-02 through K-05. | Every changed score has a rationale and confidence note; untested risks (such as poisoning) remain explicitly untested rather than marked mitigated. |

### Sequential execution and handoff points

1. **Shared kickoff gate:** complete T-00 together to verify the current rubric, proxy-system requirement, and dates. Agree on initial threat hypotheses and experiment boundaries from the plan/proposal before modeling; finalize T-01's data-specific settings after J-02's profile.
2. **Jainil's build-from-scratch phase:** create the new Git repository, place this plan and `AGENTS.md` at its root, and complete J-01/Milestone A. Use synthetic fixtures only in this milestone. After review, continue through J-02–J-07: data profile and contract, preprocessing/splits, baselines, membership inference, DP-SGD, and reproducibility. Complete T-01 with Krati after J-02 and before J-03.
3. **Formal handoff to Krati:** after J-07 passes its acceptance checks, give Krati the repository state, setup/command instructions, stable data/model/prediction interfaces, frozen configs, run manifests, and aggregate outputs. Do not include raw records or row-level artifacts in the handoff.
4. **Krati's analysis phase:** Krati completes K-01–K-06 in order: context/evidence/threat register; fairness metrics; fairness mitigation; explanation checks; robustness; residual-risk update. Jainil answers interface questions and reviews outputs, but does not take over Krati's implementation tickets.
5. **Protect the final test:** the 2025 protocol must be frozen jointly before anyone opens final-test outcomes. Krati runs temporal and subgroup evaluation under that protocol; no threshold or model selection changes follow.
6. **Shared closeout:** after both individual phases pass, complete T-02 together: integrate result artifacts, run the final reproducibility check, cross-check claims and uncertainty, finish limitations, write the report, and prepare the demo.

### Suggested issue workflow

Create one issue per ticket ID (for example, `J-03 — Target, preprocessing, and split pipeline`). Each issue should contain the ticket's acceptance checks, expected output paths, and dependency links. Keep the work in three stages: Jainil's J tickets, Krati's K tickets after the handoff, then shared T-02 closeout. T-00/T-01 are required shared gates before modeling. Require the other member to review metric definitions and data-leakage assumptions, and close an issue only after its acceptance checks pass. Use no calendar due dates until T-00 records the current course schedule.

