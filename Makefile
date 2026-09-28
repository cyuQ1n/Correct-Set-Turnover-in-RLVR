.PHONY: build commit license quality style test

check_targets := examples scripts tests verl setup.py
check_dirs := $(strip $(foreach path,$(check_targets),$(if $(wildcard $(path)),$(path),)))

code_targets := scripts tests verl setup.py
code_dirs := $(strip $(foreach path,$(code_targets),$(if $(wildcard $(path)),$(path),)))

build:
	python3 setup.py sdist bdist_wheel

commit:
	pre-commit install
	pre-commit run --all-files

license:
	@if [ -f tests/check_license.py ]; then \
		python3 tests/check_license.py $(code_dirs); \
	else \
		echo "tests/check_license.py not found; skipping license check."; \
	fi

quality:
	ruff check --select E9,F63,F7,F82 $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	@if [ -d tests ]; then \
		python -m unittest discover -s tests -v; \
	else \
		echo "tests/ not found; skipping pytest."; \
	fi
