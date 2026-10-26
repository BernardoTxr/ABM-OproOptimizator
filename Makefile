# Use the interpreter selected by the caller's active environment. A copied or
# incomplete .venv is not a portable runtime.
PYTHON ?= python3

.PHONY: install install-experiment install-followup-foundry test smoke simulate validate reproduce-plan reproduce reproduce-reference summarize

install:
	$(PYTHON) -m pip install -r requirements/icaif.lock.txt
	$(PYTHON) -m pip install -r requirements/component-aware.lock.txt
	$(PYTHON) -m pip install -e .

install-experiment:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/icaif.lock.txt
	$(PYTHON) -m pip install -r requirements/component-aware.lock.txt
	$(PYTHON) -m pip install -e .

install-followup-foundry:
	$(PYTHON) -m pip install -r requirements/anthropic-foundry.lock.txt

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m workflows.validation.smoke

simulate:
	$(PYTHON) -m lob_sim run --config configs/examples/scenario.yaml

validate:
	$(PYTHON) -m workflows.validation.environment
	$(PYTHON) -m workflows.validation.environment --lock configs/paper/component-aware-environment-lock.yaml
	$(PYTHON) -m workflows.validation.heuristics
	$(PYTHON) -m pytest -q

reproduce-plan:
	$(PYTHON) -m workflows.benchmark.reproduce_published_study --phase plan

reproduce:
	$(PYTHON) -m workflows.benchmark.reproduce_published_study --phase all --mode portable

reproduce-reference:
	$(PYTHON) -m workflows.benchmark.reproduce_published_study --phase all --mode reference

summarize:
	$(PYTHON) -m workflows.benchmark.published_summary
