# T-01 / T-02 — 2025 temporal protocol freeze (TEMPLATE)

**Status:** Not frozen. Copy to `T-01-2025-protocol-freeze.md` and complete jointly before opening 2025 outcomes.

## Checklist

- [ ] Feature allowlist frozen (`configs/base.yaml`)
- [ ] Model settings frozen (HGB + fairness mitigations)
- [ ] Attack definitions frozen (MIA protocol unchanged)
- [ ] Fairness metrics / thresholds selection rules frozen (2024 only)
- [ ] Robustness stress definition frozen (10% missingness on five top non-sensitive features from 2023 train)
- [ ] Jainil + Krati sign-off dates recorded below

## Sign-off

| Member | Date | Initials |
|---|---|---|
| Jainil Rana | | |
| Krati Jain | | |

After this file exists as `T-01-2025-protocol-freeze.md` **and** `years.seal_test_outcomes` is set `false` in config, `cli robustness --no-synthetic --open-2025` may evaluate 2025.
