# Thin wrappers over the uv CLI and the web build. `make help` lists them.
# `make ci` runs what .gitea/workflows/ci.yml runs, in the same order.
.PHONY: help test lint integration validate ci doctor experiments check spec web

VALIDATION = docs/cytochrome-p450-derivation/validation
SIMULATED = f1_partial_sums f2_individuation f3_recurrence f4_ladders

help:
	@echo "test         unit tests"
	@echo "lint         ruff over src, tests and experiments"
	@echo "integration  the experiment route end to end (needs bash, flock, python3)"
	@echo "validate     rerun the simulated validation experiments and compare with the committed results"
	@echo "ci           lint, test, integration, validate, spec, web"
	@echo "doctor       lattice, the forge, and which repositories lattice can reach"
	@echo "experiments  every experiment and where it stands"
	@echo "check E=id   validate one experiment and everything a wrap needs"
	@echo "spec         regenerate the component reference in components-graph.md"
	@echo "web          check the graph data and build the front page"

test:
	uv run --group dev pytest -q

lint:
	uv run --group dev ruff check src tests experiments

integration:
	uv run --group dev pytest -q -m integration

validate:
	cd $(VALIDATION) && uv run python run_all.py f && uv run python check_reproduced.py $(SIMULATED)

ci: lint test integration validate
	python docs/specifications/render_reference.py --check
	$(MAKE) web

doctor:
	uv run dcat-ap-plus-p450 experiment doctor

experiments:
	uv run dcat-ap-plus-p450 experiment list

check:
	uv run dcat-ap-plus-p450 experiment check $(E)

spec:
	python docs/specifications/render_reference.py

web:
	cd web && npm run check && npm run build
