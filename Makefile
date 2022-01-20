file_finder = find . -type f \( $(1) \) -not \( -path '*/venv/*' -o -path '*/build/*' -o -path './paper/figures/*' \)

ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CRUFT_DIR = $(ROOT_DIR)/paper/build/format
LATEX_SETTINGS = $(ROOT_DIR)/paper/.latexindentrc.yaml

LATEX_FILES = $(shell $(call file_finder,-name "*.tex"))

.PHONY: test
test:
	python3 -m unittest discover

coverage:
	coverage run --source=vibromaf -m unittest discover
	coverage report -m

coverage_reports: coverage
	coverage xml
	coverage html

latexindent:
	mkdir -p $(CRUFT_DIR)
	@for src in $(LATEX_FILES); do \
		echo "-- Formatting LaTeX file $$src" && \
		latexindent -w -m -s --cruft=$(CRUFT_DIR) --local=$(LATEX_SETTINGS) $$src; \
	done

serve_docs:
	mkdocs serve

.PHONY: docs
docs:
	mkdocs build

package:
	python3 -m build

check_dist:
	twine check --strict dist/*

# TODO we first need to create an accout here
deploy: package check_dist
	python3 -m twine upload dist/*
