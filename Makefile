# Repeatable setup / lint / test / experiment targets.
# Full HMDA experiments require explicit configs and local data; never start on import.

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PACKAGE := mortgage-threat-intel

.PHONY: setup lint test data-check smoke baseline privacy fairness xai robustness all

setup:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) freeze > requirements.lock

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest

data-check:
	$(PYTHON) -m mortgage_threat_intel.cli data-check --config configs/base.yaml

smoke:
	$(PYTHON) -m mortgage_threat_intel.cli smoke --config configs/base.yaml

# J-04+ targets — implemented on feat/jainil-hmda-pipeline-baselines-privacy
baseline:
	@echo "baseline: run on Jainil feature branch after J-04 wiring"
	@exit 1

privacy:
	@echo "privacy: run on Jainil feature branch after J-05/J-06 wiring"
	@exit 1

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
	@echo "Synthetic smoke complete. Full experiment sequence requires HMDA configs on feature branch."
