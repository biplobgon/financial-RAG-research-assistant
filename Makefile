# ==============================================================================
# Financial RAG Research Assistant — Makefile
# ==============================================================================

.PHONY: help install dev test lint format docker-build docker-up docker-down \
        k8s-deploy k8s-delete ingest-data clean

PYTHON := python3
PIP := pip
DOCKER_COMPOSE := docker compose
KUBECTL := kubectl
NAMESPACE := financial-rag

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ─────────────────────────────────────────────────
# Development Setup
# ─────────────────────────────────────────────────
install: ## Install all dependencies
	$(PIP) install -r requirements.txt

dev: ## Start development server with auto-reload
	$(PYTHON) -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

setup-env: ## Copy .env template
	cp .env.example .env
	@echo "Edit .env with your API keys and settings"

# ─────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────
test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ -v -m integration

test-coverage: ## Generate coverage report
	$(PYTHON) -m pytest tests/ --cov=app --cov-report=html --cov-fail-under=80

# ─────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────
lint: ## Run ruff linter
	$(PYTHON) -m ruff check app/ tests/ main.py

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format app/ tests/ main.py

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy app/ main.py --ignore-missing-imports

# ─────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t financial-rag-api:latest --target runtime .

docker-up: ## Start full stack with Docker Compose
	$(DOCKER_COMPOSE) up -d
	@echo "Services starting..."
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  ChromaDB:   http://localhost:8001"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"
	@echo "  Jaeger:     http://localhost:16686"

docker-down: ## Stop Docker Compose stack
	$(DOCKER_COMPOSE) down

docker-logs: ## Follow API container logs
	$(DOCKER_COMPOSE) logs -f api

docker-shell: ## Open shell in API container
	$(DOCKER_COMPOSE) exec api /bin/bash

# ─────────────────────────────────────────────────
# Kubernetes
# ─────────────────────────────────────────────────
k8s-deploy: ## Deploy to Kubernetes
	$(KUBECTL) apply -f infrastructure/kubernetes/
	$(KUBECTL) rollout status deployment/financial-rag-api -n $(NAMESPACE)

k8s-delete: ## Remove Kubernetes deployment
	$(KUBECTL) delete -f infrastructure/kubernetes/

k8s-status: ## Show Kubernetes status
	$(KUBECTL) get pods,services,ingress -n $(NAMESPACE)

helm-deploy: ## Deploy using Helm chart
	helm upgrade --install financial-rag infrastructure/helm/financial-rag/ \
		--namespace $(NAMESPACE) --create-namespace --values infrastructure/helm/financial-rag/values.yaml

helm-uninstall: ## Remove Helm release
	helm uninstall financial-rag --namespace $(NAMESPACE)

# ─────────────────────────────────────────────────
# Data Ingestion
# ─────────────────────────────────────────────────
ingest-data: ## Ingest synthetic financial data
	$(PYTHON) -m scripts.ingest_data --source data/synthetic/ --collection sec_filings

# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────
clean: ## Clean generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov/ dist/ build/
	@echo "Cleaned generated files"

health-check: ## Check API health
	curl -s http://localhost:8000/health | python3 -m json.tool
