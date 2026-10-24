# Repeatable setup / lint / test / experiment targets.
# Full HMDA experiments require explicit configs and local data; never start on import.

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup lint test data-check smoke baseline privacy fairness xai robustness all profile

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

fairness:
	@echo "fairness: reserved for Krati (K-02/K-03); not started in Jainil phase"
	@exit 1

xai:
	@echo "xai: reserved for Krati (K-04); not started in Jainil phase"
	@exit 1

robustness:
	@echo "robustness: reserved for Krati (K-05); not started in Jainil phase"
	@exit 1

all: smoke
	@echo "Synthetic required sequence complete (baseline+privacy via smoke --with-models)."
