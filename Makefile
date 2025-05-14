#!make

.PHONY: clean-docker down help restart-docker-compose-dev start-docker-compose-dev stop-docker-compose-dev up

# Put it first so that "make" without argument is like "make help".
help: ## Display available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-20s\033[0m %s\n", $$1, $$2}'

########## GLOBALS ##########
SHELL := /bin/bash
PROJECT_NAME = chaturai

# Docker image versions.
LITELLM_IMAGE = ghcr.io/berriai/litellm:main-v1.67.0-stable
PG_VECTOR_IMAGE = pgvector/pgvector:pg16
REDIS_IMAGE = redis:6.0-alpine
SCRAPER_IMAGE = naps-scraper

# Color codes for output.
RED := $(shell tput setaf 1)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
RESET := $(shell tput sgr0)

########## AWS ##########

# ECR
ecr-login: ## Log in to AWS ECR (Elastic Container Registry)
	@echo "$(GREEN)Logging into AWS ECR...$(RESET)"
	@aws ecr get-login-password --profile ${AWS_PROFILE} --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

ecr-build-push: ## Build and push the Docker image to AWS ECR
	@echo "$(GREEN)Building Docker image for scraper...$(RESET)"
	@docker buildx build --platform linux/amd64 -f backend/Dockerfile.scraper -t $(SCRAPER_IMAGE) ./backend --load
	@echo "$(GREEN)Tagging Docker image for ECR push...$(RESET)"
	@docker tag $(SCRAPER_IMAGE) ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/$(SCRAPER_IMAGE):latest
	@echo "$(GREEN)Pushing Docker image to AWS ECR...$(RESET)"
	@docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/$(SCRAPER_IMAGE):latest

# AWS Scraper
run-scraper-aws: ## Run the scraper task on AWS ECS (Fargate)
	@echo "$(GREEN)Running scraper task on AWS ECS (Fargate)...$(RESET)"
	@aws ecs run-task \
		--cluster naps-scraper-cluster \
		--task-definition naps-scraper \
		--launch-type FARGATE \
		--network-configuration "awsvpcConfiguration={subnets=[${AWS_SUBNET}],securityGroups=[${AWS_SG}],assignPublicIp=ENABLED}"

########## DEV SETUP ##########
up: up-litellm up-pgvector up-redis ## Set up the development environment by starting all containers

up-litellm: ## Set up LiteLLM container
	$(call clean-docker-container,litellm-proxy)
	@sleep 2
	@echo "$(GREEN)Starting a new LiteLLM container...$(RESET)"
	@docker run \
		--name litellm-proxy \
		--rm \
		-v "$(CURDIR)/cicd/litellm/litellm_config.yaml":/app/config.yaml \
		-v "$(CURDIR)/secrets/gcp_credentials.json":/app/gcp_credentials.json \
		--env-file "$(CURDIR)/cicd/litellm/.env" \
		-p 4000:4000 \
		-d $(LITELLM_IMAGE) \
		--config /app/config.yaml --detailed_debug --telemetry False

up-pgvector: ## Set up pg-vector container
	$(call clean-docker-container,pg-vector-local)
	@sleep 2
	@echo "$(GREEN)Starting a new pg-vector container...$(RESET)"
	@docker run \
		--name pg-vector-local \
		--env-file "$(CURDIR)/backend/.env" \
		-p 5432:5432 \
		-v pgvector_data:/var/lib/postgresql/data \
		-d $(PG_VECTOR_IMAGE)
	@set -a && source "$(CURDIR)/backend/.env" && set +a && cd backend && python -m alembic upgrade head

up-redis: ## Set up Redis container
	$(call clean-docker-container,redis-local)
	@sleep 2
	@echo "$(GREEN)Starting a new Redis container...$(RESET)"
	@docker run \
		--name redis-local \
     	-p 6379:6379 \
	 	-d $(REDIS_IMAGE)

########## DEV TEARDOWN ##########
down: down-litellm down-pgvector down-redis ## Tear down all development containers

down-litellm: ## Tear down LiteLLM container
	$(call clean-docker-container,litellm-proxy)

down-pgvector: ## Tear down pg-vector container
	$(call clean-docker-container,pg-vector-local)

down-redis: ## Tear down Redis container
	$(call clean-docker-container,redis-local)

########## DOCKER ##########
clean-docker: ## Clean up Docker volumes and networks
	@echo "$(RED)Cleaning up Docker volumes and networks...$(RESET)"
	@docker volume prune -f
	@docker network prune -f

clean-docker-container = \
	@echo "$(RED)Stopping and removing container: $(1)...$(RESET)"; \
	docker stop $(1) || true; \
	docker rm $(1) || true; \
	docker system prune -f

restart-docker-compose-dev: stop-docker-compose-dev start-docker-compose-dev ## Restart the Docker Compose dev environment

# Dev
start-docker-compose-dev: ## Start Docker Compose dev environment
	@echo "$(GREEN)Spinning up dev Docker containers...$(RESET)"
	@docker compose -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.yml -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.dev.yml -p chaturai-dev up --build -d --remove-orphans
	@docker system prune -f

stop-docker-compose-dev: ## Stop Docker Compose dev environment
	@echo "$(RED)Spinning down dev Docker containers...$(RESET)"
	@docker compose -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.yml -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.dev.yml -p chaturai-dev down

# Prod
run-prod:  ## Start Docker Compose prod environment
	@echo "$(GREEN)Spinning up prod Docker containers...$(RESET)"
	@docker compose -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.yml -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.testing.yml -p chaturai up --build -d --remove-orphans
	@docker system prune -f

stop-prod:  ## Stop Docker Compose prod environment
	@echo "$(RED)Spinning down prod Docker containers...$(RESET)"
	@docker compose -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.yml -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.testing.yml -p chaturai down
	@docker system prune -f

# Scraper
scrape:  ## Run the scraper Docker container
	@echo "$(GREEN)Spinning up scraper Docker containers...$(RESET)"
	@docker compose -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.scrape.yml -p chaturai-scrape build scraper
	@docker compose -f ${CURDIR}/cicd/deployment/docker-compose/docker-compose.scrape.yml -p chaturai-scrape run --rm scraper
