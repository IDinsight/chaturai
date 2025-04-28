#!make

.PHONY: clean fresh-env help lint poetry-export poetry-lock

# Put it first so that "make" without argument is like "make help".
help: ## Display available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-20s\033[0m %s\n", $$1, $$2}'

########## GLOBALS ##########
SHELL := /bin/bash
PROJECT_NAME = naukriwaala

# Color codes for output.
RED := $(shell tput setaf 1)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
RESET := $(shell tput sgr0)

########## HELPERS ##########
ifeq ($(UPDATE), no)
	UPDATE_FLAG = --no-update
else
	UPDATE_FLAG =
endif

########## CLEAN UP ##########
# Delete all compiled Python files.
clean: ## Remove compiled Python files and caches
	@echo "$(BLUE)Cleaning backend project files...$(RESET)"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@echo "$(GREEN)Cleanup complete.$(RESET)"

########## BACKEND ENVIRONMENT SETUP ##########
fresh-env: clean poetry-export ## Create a fresh backend environment
	@echo "$(YELLOW)Creating fresh backend environment: $(PROJECT_NAME)$(RESET)"
	poetry env remove python || true
	poetry install --with dev
	@echo "$(GREEN)Backend environment setup complete.$(RESET)"

########## LINTING ##########
lint: lint-isort lint-black lint-ruff lint-interrogate lint-mypy lint-pylint lint-cloc ## Run all linters and code checks

lint-isort: ## Sort imports
	@echo "$(BLUE)Running isort on backend...$(RESET)"
	isort src/ tests/

lint-black: ## Format code with Black
	@echo "$(BLUE)Running Black on backend...$(RESET)"
	black src/ tests/

lint-ruff: ## Run Ruff checks
	@echo "$(BLUE)Running Ruff on backend...$(RESET)"
	ruff check src/
	ruff check tests/

lint-interrogate: ## Run Interrogate for docstring coverage
	@echo "$(BLUE)Running Interrogate on backend...$(RESET)"
	interrogate -v src/ tests/

lint-mypy: ## Run MyPy for type checking
	@echo "$(BLUE)Running MyPy on backend...$(RESET)"
	mypy src/ --ignore-missing-imports --show-error-codes
	mypy tests/ --ignore-missing-imports --show-error-codes

lint-pylint: ## Run Pylint for code quality checks
	@echo "$(BLUE)Running Pylint on backend...$(RESET)"
	pylint src/
	pylint tests/

lint-cloc: ## Run CLOC for code statistics
	@echo "$(BLUE)Running CLOC on backend...$(RESET)"
	cloc src/ tests/

########## POETRY ##########
# Include UPDATE=no on the command line if package upgrades should NOT be searched for
# and updated in the lock file (e.g., make poetry.lock UPDATE=no).
poetry-lock: pyproject.toml ## Lock and update dependencies as specified in pyproject.toml
	@echo "$(YELLOW)Locking dependencies...$(RESET)"
	poetry lock $(UPDATE_FLAG)
	@echo "$(GREEN)Dependencies locked.$(RESET)"

poetry-export: poetry-lock ## Export dependencies to requirements files
	@echo "$(YELLOW)Exporting dependencies to requirements files...$(RESET)"
	poetry export -f requirements.txt --without-hashes --output ../cicd/requirements/requirements.txt
	poetry export -f requirements.txt --without-hashes --with dev --output ../cicd/requirements/requirements_dev.txt
	@echo "$(GREEN)Dependencies exported.$(RESET)"
