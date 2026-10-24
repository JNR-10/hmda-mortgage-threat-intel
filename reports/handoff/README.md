# Committed aggregate handoff artifacts

These JSON files are **aggregate-only** snapshots from Jainil’s real-HMDA runs (2025 sealed). Safe to commit; no loan-level rows, membership labels, or row-level predictions.

| File | Source (local, gitignored) |
|---|---|
| `baseline_latest_metrics.json` | `outputs/baselines/latest_metrics.json` |
| `privacy_latest_privacy.json` | `outputs/privacy/latest_privacy.json` |
| `data_profile.json` | `outputs/data_profile.json` |

Refresh after new runs:

```bash
make handoff-sync
```

For Krati’s next steps, see `docs/handoff/J-07-jainil-to-krati.md`.
