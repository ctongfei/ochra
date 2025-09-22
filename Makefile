# Project maintenance targets

.PHONY: ruff ruff-fix pyright test tests mkdocs-build mkdocs-serve docs check all clean

# Python source locations
SRC_DIR := src
TEST_DIR := tests

# Ruff - lint
ruff:
	ruff check $(SRC_DIR) $(TEST_DIR)

# Ruff - autofix
ruff-fix:
	ruff check --fix $(SRC_DIR) $(TEST_DIR)

# Pyright - static typing
pyright:
	pyright

# Pytest - unit tests (configured via pyproject.toml)
.test-run:
	pytest

test: .test-run

# MkDocs - docs
mkdocs-build:
	mkdocs build --strict

mkdocs-serve:
	mkdocs serve --dev-addr=127.0.0.1:8000

docs: mkdocs-build

# Aggregate checks
check: ruff pyright test

# Run everything
all: check

# Clean common artifacts
clean:
	rm -rf .pytest_cache .ruff_cache .pyright \ \
		site \ \
		**/__pycache__ **/*.pyc **/*.pyo

