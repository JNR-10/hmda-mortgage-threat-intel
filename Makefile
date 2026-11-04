# Repeatable setup / lint / test / experiment targets.
# Full HMDA experiments require explicit configs and local data; never start on import.

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup lint test data-check smoke baseline privacy fairness fairness-hmda xai robustness residual-update all profile handoff-sync

setup:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) freeze | grep -v '^-e ' | grep -v '^##' > requirements.lock

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest

data-check:
	$(PYTHON) -m mortgage_threat_intel.cli data-check --config configs/base.yaml

profile:
	$(PYTHON) -m mortgage_threat_intel.cli profile --config configs/base.yaml --output outputs/data_profile.json

smoke:
	$(PYTHON) -m mortgage_threat_intel.cli smoke --config configs/base.yaml --with-models

baseline:
	$(PYTHON) -m mortgage_threat_intel.cli baseline --config configs/base.yaml --no-synthetic

privacy:
	$(PYTHON) -m mortgage_threat_intel.cli privacy --config configs/base.yaml --privacy-config configs/privacy.yaml --no-synthetic

baseline-synthetic:
	$(PYTHON) -m mortgage_threat_intel.cli baseline --config configs/base.yaml --synthetic

privacy-synthetic:
	$(PYTHON) -m mortgage_threat_intel.cli privacy --config configs/base.yaml --privacy-config configs/privacy.yaml --synthetic

handoff-sync:
	$(PYTHON) scripts/sync_handoff_aggregates.py

fairness:
	$(PYTHON) -m mortgage_threat_intel.cli fairness --config configs/base.yaml --fairness-config configs/fairness.yaml --synthetic

fairness-hmda:
	$(PYTHON) -m mortgage_threat_intel.cli fairness --config configs/base.yaml --fairness-config configs/fairness.yaml --no-synthetic

xai:
	$(PYTHON) -m mortgage_threat_intel.cli xai --config configs/base.yaml --synthetic

robustness:
	$(PYTHON) -m mortgage_threat_intel.cli robustness --config configs/base.yaml --fairness-config configs/fairness.yaml --synthetic

residual-update:
	$(PYTHON) -m mortgage_threat_intel.cli residual-update --register reports/threat_register.yaml --output reports/threat_register_residual.yaml

all: smoke fairness xai robustness residual-update
	@echo "Synthetic Krati+Jainil smoke sequence complete (no HMDA / 2025 sealed)."
