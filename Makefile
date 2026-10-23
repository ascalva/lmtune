.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test cov check install install-dev uninstall

help: ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv with dev tools
	uv sync

lint: ## Lint (ruff)
	uv run ruff check src tests
	uv run ruff format --check src tests

format: ## Auto-format and fix lint
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck: ## mypy --strict
	uv run mypy

test: ## Unit tests (no GPU, no network, no launchd)
	uv run pytest

cov: ## Tests with coverage report
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

install: ## Install `lmtune` on PATH (copied, not editable: launchd agents can't read ~/Documents)
	uv tool install --force --reinstall .

install-dev: ## Editable install for development (launchd-run `lmtune` will hang on macOS privacy prompts)
	uv tool install --editable --force .

uninstall: ## Remove `lmtune` from PATH
	uv tool uninstall lmtune
