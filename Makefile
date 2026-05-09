# afro-health-qa — common commands.
# Run `make help` for the list.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
VENV ?= .venv
CONFIG ?= configs/training/qlora_default.yaml
RUN ?=

.PHONY: help setup audit baseline train evaluate submit verify test lint format clean

help:  ## Show this help.
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create venv and install pinned deps.
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/$(PIP) install --upgrade pip==24.3.1
	$(VENV)/bin/$(PIP) install -r requirements.txt
	$(VENV)/bin/$(PIP) install -e .
	@echo "Activate with: source $(VENV)/bin/activate   (Windows: $(VENV)/Scripts/activate)"

audit:  ## Tokeniser audit — chars/token per language per model.
	$(PYTHON) -m afro_health_qa.data.tokeniser_audit --output docs/TOKENISER_AUDIT.md

baseline:  ## Run zero-shot Aya baseline and write a submission CSV.
	$(PYTHON) scripts/run_baseline.sh || bash scripts/run_baseline.sh

train:  ## Fine-tune. Pass CONFIG=configs/training/<name>.yaml
	$(PYTHON) -m afro_health_qa.training.run --config $(CONFIG)

evaluate:  ## Score a prediction file. Pass RUN=submissions/<file>.csv
	@test -n "$(RUN)" || (echo "RUN=submissions/<file>.csv is required"; exit 1)
	$(PYTHON) -m afro_health_qa.evaluation.combined --predictions $(RUN) --split val

submit:  ## Build and validate a Zindi-format submission CSV. Pass RUN=<predictions>.
	@test -n "$(RUN)" || (echo "RUN=<predictions> is required"; exit 1)
	$(PYTHON) -m afro_health_qa.submission.format --input $(RUN) --output submissions/$$(date +%Y-%m-%d_%H%M)_$(notdir $(basename $(RUN))).csv
	$(PYTHON) -m afro_health_qa.submission.validate --path submissions/

verify:  ## End-to-end reproducibility check — re-runs pipeline, hashes output.
	$(PYTHON) scripts/verify_reproducibility.py

test:  ## Run pytest.
	$(PYTHON) -m pytest -ra

lint:  ## Ruff check.
	$(PYTHON) -m ruff check src tests scripts

format:  ## Ruff format.
	$(PYTHON) -m ruff format src tests scripts

clean:  ## Remove caches (not data/models).
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
