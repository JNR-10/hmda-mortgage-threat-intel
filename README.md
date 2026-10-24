# HMDA Mortgage Threat Intelligence

Offline research pipeline for a **public HMDA-based proxy** that predicts *reported historical application outcomes*. Fannie Mae Desktop Underwriter (DU) is **deployment context only** — this project does not access, reproduce, test, or evaluate DU or any lender production model.

Team: Jainil Rana (HMDA/data/ML, privacy) and Krati Jain (context/threat, fairness, XAI, robustness).

Authoritative scope: `revised_mortgage_ai_threat_intelligence_implementation_plan.md` and `AGENTS.md`. Shared gate decisions: `docs/decisions/T-00-T-01-shared-gates.md`.

## Requirements

- Python **3.11+**
- CPU-only by default (macOS Apple Silicon supported). CUDA is not required.

## Create, activate, and deactivate `.venv`

### macOS / Linux

```bash
cd /path/to/hmda-mortgage-threat-intel
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
# optional lock snapshot after install:
pip freeze > requirements.lock
```

Deactivate:

```bash
deactivate
```

Or use Make (creates `.venv` and installs editable + dev tools):

```bash
make setup
source .venv/bin/activate
```

### Windows (cmd)

```bat
cd \path\to\hmda-mortgage-threat-intel
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip freeze > requirements.lock
```

Deactivate:

```bat
deactivate
```

### Windows (PowerShell)

```powershell
cd \path\to\hmda-mortgage-threat-intel
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Common commands

```bash
make lint          # ruff check + format check (no HMDA data)
make test          # pytest on synthetic fixtures (no HMDA data)
make smoke         # synthetic CLI smoke run (no HMDA data)
make data-check    # reports missing local HMDA files / expected placement
make all           # synthetic smoke path only on main scaffold
```

CLI:

```bash
python -m mortgage_threat_intel.cli version
python -m mortgage_threat_intel.cli smoke --config configs/base.yaml
python -m mortgage_threat_intel.cli data-check --config configs/base.yaml
```

## Local HMDA data (never commit)

Place annual files under `data/raw/` (or set `HMDA_DATA_DIR`):

- `data/raw/hmda_2023.csv`
- `data/raw/hmda_2024.csv`
- `data/raw/hmda_2025.csv` (presence/schema/hash only until protocol freeze)

See `data/README.md`. Raw/processed rows, checkpoints, and row-level predictions stay git-ignored under `data/` and `outputs/`.

## Repository layout

See plan §14.1. Implementation lives in `src/mortgage_threat_intel/`. Experiment settings in `configs/`. Tests use synthetic fixtures only.

## Ownership

| Phase | Owner | Tickets |
|---|---|---|
| Scaffold / HMDA / baselines / MIA / DP-SGD / handoff | Jainil | J-01–J-07 |
| Context, fairness, XAI, robustness, residual risk | Krati | K-01–K-06 |
| Rubric gate, experiment freeze, closeout | Shared | T-00–T-02 |

## PyTorch note (Apple Silicon)

Install uses the default `torch` wheel from PyPI (CPU / MPS-capable builds on macOS). Training paths default to **CPU**. An MPS path for DP-SGD is only considered after a dedicated Opacus smoke test; do not assume CUDA.
