# Demo script — AI Threat Intelligence (HMDA proxy)

**Audience:** course demo / peer review  
**Boundary:** Offline public-HMDA research proxy. **Not** Desktop Underwriter. **Not** a lending tool.

## 1. One-minute framing

1. Show `reports/system_context.md` §1 (proxy boundary table).
2. State the research question (plan §2): privacy–utility of a matched DP MLP; fairness / explanation / robustness risks on a preselected boosted-tree audit model.
3. Point to NIST AI RMF Map → Measure → Manage and MITRE ATLAS-informed register.

## 2. End-to-end path (synthetic smoke; no HMDA required)

```bash
make setup && source .venv/bin/activate
make lint && make test
make smoke          # contracts + synthetic baselines/privacy
make fairness xai robustness residual-update
```

## 3. Main comparisons (use committed aggregates)

Open `reports/handoff/`:

| Artifact | What to say |
|---|---|
| `baseline_latest_metrics.json` | LR vs HGB utility on 2024 validation (capped sample); HGB is the audit model |
| `privacy_latest_privacy.json` | Achieved ε/δ (not target ε); MIA AUC near chance under confidence-score attack |
| `fairness_latest.json` | Group DP/EO gaps + mitigations (reweight / EO thresholds on 2024) |
| `xai_latest.json` | TreeSHAP top-k stability + faithfulness ablation; synthetic local illustration only |
| `robustness_latest.json` | 10% missingness stress; 2025 temporal after protocol freeze |

## 4. Multi-criteria table

Use the summary table in `reports/final_report.md` §6. Emphasize uncertainty and underpowered slices.

## 5. Risk-register change example

1. Open `reports/threat_register.yaml` (inherent) vs `reports/threat_register_residual.yaml`.
2. Walk **R1** (MIA near chance → residual likelihood down) and **R3** (poisoning remains **untested** — never mark mitigated).
3. Show `reports/residual_risk_changelog.json`.

## 6. Real HMDA commands (local data)

```bash
make data-check
make fairness-hmda
.venv/bin/python -m mortgage_threat_intel.cli xai --config configs/base.yaml --no-synthetic
.venv/bin/python -m mortgage_threat_intel.cli robustness \
  --config configs/base.yaml --fairness-config configs/fairness.yaml \
  --no-synthetic --open-2025
make handoff-sync
make residual-update
```

Requires Snapshot LAR under `data/raw/` (see `data/README.md`) and `docs/decisions/T-01-2025-protocol-freeze.md`.
