# Committed aggregate handoff artifacts

Aggregate-only snapshots from real-HMDA runs (safe to commit). No loan-level rows, membership labels, or row-level explanations.

| File | Source (local, gitignored) |
|---|---|
| `baseline_latest_metrics.json` | `outputs/baselines/latest_metrics.json` |
| `privacy_latest_privacy.json` | `outputs/privacy/latest_privacy.json` |
| `data_profile.json` | `outputs/data_profile.json` |
| `fairness_latest.json` | `outputs/fairness/latest_fairness.json` |
| `xai_latest.json` | `outputs/xai/latest_xai.json` |
| `robustness_latest.json` | `outputs/robustness/latest_robustness.json` |

Refresh after new **real** runs (`manifest.synthetic=false`):

```bash
make handoff-sync
make residual-update
```

See `docs/handoff/J-07-jainil-to-krati.md` and `reports/final_report.md`.
