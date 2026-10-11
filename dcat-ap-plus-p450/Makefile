# Thin wrappers over the uv CLI and the web build. `make help` lists them.
.PHONY: help test doctor experiments check spec web

help:
	@echo "test         run the test suite"
	@echo "doctor       lattice, the forge, and which repositories lattice can reach"
	@echo "experiments  every experiment and where it stands"
	@echo "check E=id   validate one experiment and everything a wrap needs"
	@echo "spec         regenerate the component reference in components-graph.md"
	@echo "web          check the graph data and build the front page"

test:
	uv run --group dev pytest -q

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
