# afro-health-qa — common commands.
# Run `make help` for the list. Override the interpreter with e.g.
#   make test PYTHON=.venv-marimo/bin/python

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
VENV ?= .venv
RUN ?=
REFS ?= data/processed/held_out.csv

.PHONY: help setup evaluate sanity submit test lint format clean

help:  ## Show this help.
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create venv and install pinned deps.
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/$(PIP) install --upgrade pip==24.3.1
	$(VENV)/bin/$(PIP) install -r requirements.txt
	$(VENV)/bin/$(PIP) install -e .
	@echo "Activate with: source $(VENV)/bin/activate   (Windows: $(VENV)/Scripts/activate)"

evaluate:  ## Score predictions per subset (whitespace ROUGE, 0.37/0.37/0.26). RUN=<preds.csv> [REFS=<refs.csv>]
	@test -n "$(RUN)" || (echo "RUN=<predictions.csv> is required (REFS defaults to $(REFS))"; exit 1)
	PYTHONPATH=src $(PYTHON) -m afro_health_qa.evaluation.scorer --predictions $(RUN) --references $(REFS)

sanity:  ## CPU smoke test of the autoresearch harness (synthetic data, no model).
	$(PYTHON) autoresearch_nlp/tools/synth_sanity.py

submit:  ## Build and validate a Zindi-format submission CSV. Pass RUN=<predictions>.
	@test -n "$(RUN)" || (echo "RUN=<predictions> is required"; exit 1)
	PYTHONPATH=src $(PYTHON) -m afro_health_qa.submission.format --input $(RUN) --output submissions/$$(date +%Y-%m-%d_%H%M)_$(notdir $(basename $(RUN))).csv
	PYTHONPATH=src $(PYTHON) -m afro_health_qa.submission.validate --path submissions/

test:  ## Run pytest.
	PYTHONPATH=src $(PYTHON) -m pytest -q tests

lint:  ## Ruff check.
	$(PYTHON) -m ruff check src tests scripts

format:  ## Ruff format.
	$(PYTHON) -m ruff format src tests scripts

clean:  ## Remove caches (not data/models).
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
