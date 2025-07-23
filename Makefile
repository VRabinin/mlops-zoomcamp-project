# MLOps Platform Makefile

.PHONY: help setup install clean test lint format data train serve monitor deploy architecture

# Default target
help: ## Show this help message
	@echo "MLOps Platform Commands:"
	@echo "========================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
setup: ## Set up development environment
	@echo "Setting up development environment..."
	@echo "Checking Python version..."
	@python_version=$$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2); \
	if [ "$$python_version" != "3.11" ]; then \
		echo "⚠️  Warning: Python 3.11 is recommended, but found Python $$python_version"; \
		echo "   Consider installing Python 3.11 for best compatibility"; \
	else \
		echo "✅ Python 3.11 detected - perfect!"; \
	fi
	python -m venv .venv
	@echo "✅ Virtual environment created at .venv/"
	@echo "To activate: source .venv/bin/activate"

install-venv: ## Install dependencies directly in .venv (no activation required)
	@echo "Installing dependencies in virtual environment..."
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install pytest pytest-cov black isort flake8 mypy
	@echo "Dependencies installed successfully in .venv!"

# Environment Management
create-dirs: ## Create necessary directories
	@echo "Creating project directories..."
	mkdir -p data/raw data/processed data/features
	mkdir -p models artifacts logs
	mkdir -p reports #notebooks
	mkdir -p app_data/postgres app_data/minio
	touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep

env-file: ## Create environment file from template
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "Created .env file. Please update with your settings."; \
	else \
		echo ".env file already exists."; \
	fi

# Architecture
architecture-start: ## View architecture diagrams
	@echo "Starting architecture viewer..."
	cd architecture && docker compose up -d structurizr
	@echo "Architecture available at: http://localhost:$${STRUCTURIZR_PORT:-8080}"

architecture-stop: ## Stop architecture viewer
	@echo "Stopping architecture viewer..."
	cd architecture && docker compose down

# Data Pipeline - direct start
data-download: ## Download CRM dataset from Kaggle
	@echo "Downloading CRM dataset..."
	python -m src.data.ingestion.crm_ingestion

data-validate: ## Validate downloaded data
	@echo "Validating data quality..."
	python -m src.data.validation.run_validation

data-process: ## Process raw data into features
	@echo "Processing data..."
	python -m src.data.preprocessing.feature_engineering

data-pipeline: data-download data-validate data-process ## Run complete data pipeline

# MinIO S3 Management
minio-ui: ## Open MinIO web console
	@echo "MinIO Console available at: http://localhost:9001"
	@echo "Username: minioadmin"
	@echo "Password: minioadmin"

minio-buckets: ## List all MinIO buckets
	@echo "Listing MinIO buckets..."
	@docker run --rm --network mlops-zoomcamp-project_mlops-network \
		--entrypoint /bin/sh \
		minio/mc:RELEASE.2024-07-11T18-01-28Z \
		-c "mc alias set minio http://minio:9000 minioadmin minioadmin && mc ls minio/"

minio-list-data: ## List contents of data-lake bucket
	@echo "Listing data-lake bucket contents..."
	@docker run --rm --network mlops-zoomcamp-project_mlops-network \
		--entrypoint /bin/sh \
		minio/mc:RELEASE.2024-07-11T18-01-28Z \
		-c "mc alias set minio http://minio:9000 minioadmin minioadmin && mc ls minio/data-lake/ --recursive"

minio-list-mlflow: ## List MLflow artifacts
	@echo "Listing MLflow artifacts..."
	@docker run --rm --network mlops-zoomcamp-project_mlops-network \
		--entrypoint /bin/sh \
		minio/mc:RELEASE.2024-07-11T18-01-28Z \
		-c "mc alias set minio http://minio:9000 minioadmin minioadmin && mc ls minio/mlflow-artifacts/ --recursive"

minio-status: ## Check MinIO status
	@echo "Checking MinIO status..."
	@docker run --rm --network mlops-zoomcamp-project_mlops-network \
		--entrypoint /bin/sh \
		minio/mc:RELEASE.2024-07-11T18-01-28Z \
		-c "mc alias set minio http://minio:9000 minioadmin minioadmin && mc admin info minio"


# ================= Validated till here ================


# Model Training
train: ## Train ML models
	@echo "Training models..."
	python -m src.models.train

train-experiment: ## Run training experiment with MLFlow
	@echo "Running training experiment..."
	python -m src.models.experiment

evaluate: ## Evaluate trained models
	@echo "Evaluating models..."
	python -m src.models.evaluate

# Model Serving
serve-model: ## Start model serving API
	@echo "Starting model serving API..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-ui: ## Start Streamlit UI
	@echo "Starting Streamlit UI..."
	streamlit run src/streamlit_app/main.py

# MLFlow
mlflow-ui: ## Start MLFlow UI
	@echo "Starting MLFlow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-server: ## Start MLFlow tracking server
	@echo "Starting MLFlow tracking server..."
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000




# Prefect
prefect-deploy-crm: ## Deploy CRM ingestion flow with S3 storage
	@echo "Deploying CRM ingestion flow with S3 storage..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) .venv/bin/python src/pipelines/deploy_crm_pipeline.py

prefect-deploy-s3: ## Deploy CRM flow with S3 storage (MinIO)
	@echo "Deploying CRM flow with S3 storage (MinIO)..."
	@echo "Ensuring MinIO S3 bucket exists..."
	@docker exec mlops-minio-setup mc mb minio/data-lake --ignore-existing 2>/dev/null || true
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) .venv/bin/python -c "\
from src.pipelines.deploy_crm_pipeline import deploy_with_s3_storage; \
deploy_with_s3_storage()"

prefect-run-crm: ## Run CRM ingestion flow directly (for testing)
	@echo "Running CRM ingestion flow locally..."
	@echo "Setting Prefect API URL..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_crm_pipeline.py

prefect-deployments: ## List all Prefect deployments
	@echo "Listing Prefect deployments..."
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect deployment ls

prefect-run-deployment: ## Run the CRM deployment manually
	@echo "Running CRM deployment manually..."
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect deployment run crm_data_ingestion_flow/crm-data-ingestion

prefect-flows: ## List all flow runs
	@echo "Listing Prefect flow runs..."
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect flow-run ls

prefect-ui: ## Open Prefect UI in browser
	@echo "Opening Prefect UI..."
	@open http://localhost:4200

prefect-help: ## Show all Prefect commands
	@echo "=== Available Prefect Commands ==="
	@echo "prefect-deploy-crm:      Deploy CRM flow with S3 storage"
	@echo "prefect-deploy-s3:       Deploy CRM flow with MinIO S3 storage"
	@echo "prefect-run-crm:         Run CRM ingestion flow directly"	
	@echo "prefect-deployments:     List all deployments"
	@echo "prefect-flows:           List recent flow runs"
	@echo "prefect-run-deployment:  Run CRM deployment manually"
	@echo "prefect-status-all:      Show comprehensive status"
	@echo "prefect-ui:              Open Prefect UI in browser"
	@echo "prefect-help:            Show this help message"
	@echo ""
	@echo "=== MinIO S3 Storage ==="
	@echo "minio-ui:                Open MinIO web console"
	@echo "minio-buckets:           List all MinIO buckets"
	@echo "minio-list-data:         List data-lake bucket contents"
	@echo "minio-list-mlflow:       List MLflow artifacts"
	@echo "minio-status:            Check MinIO service status"

prefect-status-all: ## Show comprehensive Prefect status
	@echo "=== Prefect Status ==="
	@echo "Server Health:"
	@curl -s http://localhost:4200/api/health 2>/dev/null && echo " ✅ Server is running" || echo " ❌ Server is not responding"
	@echo "\nWork Pools:"
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect work-pool ls 2>/dev/null || echo "❌ Could not list work pools"
	@echo "\nDeployments:"
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect deployment ls 2>/dev/null || echo "❌ Could not list deployments"
	@echo "\nRecent Flow Runs:"
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect flow-run ls --limit 5 2>/dev/null || echo "❌ Could not list flow runs"

# Code Quality
lint: ## Run linting
	@echo "Running linting..."
	flake8 src/
	mypy src/

format: ## Format code
	@echo "Formatting code..."
	black src/
	isort src/

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	black --check src/
	isort --check-only src/

# Testing
test: ## Run tests
	@echo "Running tests..."
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Documentation
docs-build: ## Build documentation
	@echo "Building documentation..."
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	mkdocs serve

# Docker
application-start: ## Run application in Docker
	@echo "Running application in Docker..."
	docker compose up -d

application-stop: ## Stop Docker containers
	@echo "Stopping Docker containers..."
	docker compose down

application-restart: application-stop application-start ## Restart Docker containers

application-status: ## Check application status in Docker
	@echo "Checking application status in Docker..."
	docker compose ps

# Infrastructure
infra-plan: ## Plan Terraform infrastructure
	@echo "Planning infrastructure..."
	cd infrastructure && terraform plan

infra-apply: ## Apply Terraform infrastructure
	@echo "Applying infrastructure..."
	cd infrastructure && terraform apply

infra-destroy: ## Destroy Terraform infrastructure
	@echo "Destroying infrastructure..."
	cd infrastructure && terraform destroy

# Local Development
dev-setup: setup install-venv create-dirs env-file ## Complete development setup
	@echo "🚀 MLOps development environment setup complete!"
	@echo ""
	@echo "🎉 Development environment ready!"
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source .venv/bin/activate"
	@echo "2. Update .env file with your Kaggle credentials"
	@echo "3. Start services: docker compose up -d"
	@echo "4. Run data pipeline: make data-pipeline"

dev-start: mlflow-server ## Start development services
	@echo "Starting Prefect services..."
	@docker compose up -d prefect-server prefect-setup

dev-stop: ## Stop development services
	@echo "Stopping development services..."
	pkill -f "mlflow"
	pkill -f "prefect"
	@echo "Development services stopped"

# Cleaning
clean: ## Clean temporary files
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-data: ## Clean processed data (keep raw data)
	@echo "Cleaning processed data..."
	rm -rf data/processed/*
	rm -rf data/features/*
	touch data/processed/.gitkeep data/features/.gitkeep

clean-models: ## Clean trained models
	@echo "Cleaning trained models..."
	rm -rf models/*
	rm -rf mlruns/*

clean-all: clean clean-data clean-models ## Clean everything

# CI/CD
ci-test: install-venv test lint ## Run CI tests

ci-build: clean install-venv test lint docker-build ## Complete CI build

# Quick Commands
quick-start: dev-setup data-pipeline train ## Quick start for new developers

quick-flow: dev-setup prefect-deploy-crm prefect-agent ## Quick start with Prefect flows

status: ## Show project status
	@echo "MLOps Platform Status"
	@echo "===================="
	@echo "Python version: $(shell python --version)"
	@echo "MLFlow tracking URI: $(shell echo $${MLFLOW_TRACKING_URI:-http://localhost:5000})"
	@echo "Data directory size: $(shell du -sh data/ 2>/dev/null || echo 'No data directory')"
	@echo "Models directory size: $(shell du -sh models/ 2>/dev/null || echo 'No models directory')"
