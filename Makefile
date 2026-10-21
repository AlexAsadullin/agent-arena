.PHONY: setup lint format typecheck test reproduce report clean

UV ?= uv
SEASON ?= config/seasons/synthetic.yaml

setup:
	test -d .venv || $(UV) venv --python 3.13 .venv
	$(UV) sync --frozen

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest --cov=agent_arena --cov-report=term

# Offline end-to-end demo: scripted agents + baselines against the in-process fake I9 gateway.
reproduce:
	$(UV) run arena run-season --season $(SEASON) --gateway fake

# Experiments behind docs/report.md; tables go to $$TRADEQ_DATA_DIR/agent-arena/reports/.
report:
	$(UV) run python -m agent_arena.experiments --season $(SEASON)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
