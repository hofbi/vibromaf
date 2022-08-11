.PHONY: test
test:
	python3 -m unittest discover

coverage:
	coverage run --source=vibromaf -m unittest discover
	coverage report -m

coverage_reports: coverage
	coverage xml
	coverage html

pages: docs coverage_reports
	cp -r htmlcov site/coverage

serve_docs:
	mkdocs serve

.PHONY: docs
docs:
	mkdocs build

package:
	python3 -m build

check_dist:
	twine check --strict dist/*

deploy: package check_dist
	python3 -m twine upload dist/*

clean:
	rm -rf site htmlcov dist vibromaf.egg-info

smoke_test:
	mv dist/vibromaf-*.tar.gz dist/vibromaf.tar.gz || true
	pip3 install -U dist/vibromaf.tar.gz
	cd examples && python3 white_noise.py

smoke_test_clean: clean package smoke_test
