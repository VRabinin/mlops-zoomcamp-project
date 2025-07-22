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

#install: ## Install Python dependencies (requires activated virtual environment)
#	@echo "Installing dependencies..."
#	@if [ -z "$$VIRTUAL_ENV" ]; then \
#		echo "⚠️  Warning: No virtual environment detected."; \
#		echo "   Please activate your virtual environment first: source .venv/bin/activate"; \
#		echo "   Or use 'make dev-setup' for complete setup."; \
#		exit 1; \
#	fi
#	pip install --upgrade pip
#	pip install -r requirements.txt
#	@echo "Dependencies installed successfully!"

#install-dev: ## Install development dependencies (requires activated virtual environment)
#	@echo "Installing development dependencies..."
#	@if [ -z "$$VIRTUAL_ENV" ]; then \
#		echo "⚠️  Warning: No virtual environment detected."; \
#		echo "   Please activate your virtual environment first: source .venv/bin/activate"; \
#		exit 1; \
#	fi
#	pip install --upgrade pip
#	pip install -r requirements.txt
#	pip install pytest pytest-cov black isort flake8 mypy
#	@echo "Development dependencies installed successfully!"

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
	mkdir -p app_data/postgres app_data/localstack
	touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep

env-file: ## Create environment file from template
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "Created .env file. Please update with your settings."; \
	else \
		echo ".env file already exists."; \
	fi


# =================. Validated till here ================


# Data Pipeline
data-download: ## Download CRM dataset from Kaggle
	@echo "Downloading CRM dataset..."
	python -m src.data.ingestion.crm_ingestion

data-validate: ## Validate downloaded data
	@echo "Validating data quality..."
	python -m src.data.validation.run_validation

data-process: ## Process raw data into features
	@echo "Processing data..."
	python -m src.data.preprocessing.process_data

data-pipeline: data-download data-validate data-process ## Run complete data pipeline

data-pipeline-flow: ## Run data pipeline as Prefect flow
	@echo "Running data pipeline as Prefect flow..."
	make prefect-run-crm

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
prefect-server: ## Start Prefect server
	@echo "Starting Prefect server..."
	prefect server start --host 0.0.0.0 --port 4200

prefect-agent: ## Start Prefect agent
	@echo "Starting Prefect agent..."
	prefect agent start -q default

prefect-deploy: ## Deploy Prefect flows
	@echo "Deploying Prefect flows..."
	python -m src.pipelines.deploy_flows

prefect-deploy-crm: ## Deploy CRM ingestion flow
	@echo "Deploying CRM ingestion flow..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_prefect_deployment.py

prefect-run-crm: ## Run CRM ingestion flow directly (for testing)
	@echo "Running CRM ingestion flow locally..."
	@echo "Setting Prefect API URL..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_crm_ingestion.py

prefect-status: ## Check status of Prefect and related services
	@echo "Checking Prefect pipeline status..."
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/check_status.py

prefect-test: ## Test Prefect flow setup
	@echo "Testing Prefect flow setup..."
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/test_setup.py

prefect-view-flows: ## View registered Prefect flows
	@echo "Viewing registered Prefect flows..."
	prefect deployment ls

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

# Architecture
architecture-start: ## View architecture diagrams
	@echo "Starting architecture viewer..."
	cd architecture && docker compose up -d structurizr
	@echo "Architecture available at: http://localhost:$${STRUCTURIZR_PORT:-8080}"

architecture-stop: ## Stop architecture viewer
	@echo "Stopping architecture viewer..."
	cd architecture && docker compose down

# Docker
docker-build: ## Build Docker images
	@echo "Building Docker images..."
	docker build -t mlops-platform:latest .

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

dev-start: mlflow-server prefect-server prefect-agent ## Start development services

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
