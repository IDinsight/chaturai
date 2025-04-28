#!make

.PHONY: clean fresh-env help lint-fix run-tests-node run-tests-jsdom tests

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

########## CLEAN UP ##########
# Delete all compiled Python files.
clean: ## Remove compiled files and caches
	@echo "$(BLUE)Cleaning frontend project files...$(RESET)"
	rm -rf dist build coverage
	rm -rf .turbo .parcel-cache .next .vite .svelte-kit
	rm -rf .jest_cache .jest .cache
	rm -rf playwright/.cache .playwright-cache
	rm -rf .eslintcache .stylelintcache
	pnpm jest --clearCache
	pnpm store prune
	@echo "$(GREEN)Cleanup complete.$(RESET)"

########## FRONTEND ENVIRONMENT SETUP ##########
fresh-env: clean ## Create a fresh frontend environment
	@echo "$(YELLOW)Creating fresh frontend environment: $(PROJECT_NAME)$(RESET)"
	rm -rf node_modules pnpm-lock.yaml
	pnpm install
	@echo "$(GREEN)Frontend environment setup complete.$(RESET)"

########## LINTING ##########
lint-fix: ## Fix lint and style issues
	@echo "$(BLUE)Running code linters and style checks on frontend...$(RESET)"
	pnpm stylelint --config .stylelintrc.js --ignore-path .gitignore --fix "**/*.css"
	pnpm eslint --config eslint.config.js --fix "src/**/*.{js,ts,jsx,tsx,mjs,cjs}" --format stylish
	pnpm eslint --config eslint.config.js --fix "tests/**/*.{js,ts,jsx,tsx,mjs,cjs}" --format stylish
	pnpm prettier --write "src/**/*.{js,ts,jsx,tsx,mjs,cjs}"
	pnpm prettier --write "tests/**/*.{js,ts,jsx,tsx,mjs,cjs}"
	@echo "$(GREEN)Linting and formatting complete.$(RESET)"

########## TESTING ##########
tests: run-tests-node run-tests-jsdom ## Run all tests

run-tests-node: ## Run Node-based tests
	pnpm node --experimental-vm-modules node_modules/jest/bin/jest.js --config jest.config.ts --passWithNoTests --testMatch "tests/**/!(*.jsdom).test.ts"

run-tests-jsdom: ## Run JSDOM-based tests
	pnpm node --experimental-vm-modules node_modules/jest/bin/jest.js --config jest.config.ts --passWithNoTests --testMatch "tests/**/*.jsdom.test.ts" --testEnvironment jsdom
