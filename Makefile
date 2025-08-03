# MLOps Platform Makefile

.PHONY: help setup install clean test lint format data train serve monitor deploy architecture

# Default target
help: ## Show this help message
	@echo "🚀 MLOps Platform Commands"
	@echo "============================"
	@echo "📊 Current Implementation: CRM Sales Opportunities (Phases 1-6 Complete)"
	@echo "🔧 Stack: Prefect 3.x + MLflow + PostgreSQL + MinIO + Evidently AI"
	@echo ""
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
data-acquisition: ## Download CRM dataset from Kaggle
	@echo "Downloading CRM dataset..."
	python -m src.data.ingestion.crm_acquisition

data-ingestion: ## Download CRM dataset from Kaggle
	@echo "Downloading CRM dataset..."
	python -m src.data.ingestion.crm_ingestion

data-validation: ## Validate downloaded data
	@echo "Validating data quality..."
	python -m src.data.validation.run_validation

data-preprocess: ## Process raw data into features
	@echo "Processing data..."
	python -m src.data.preprocessing.feature_engineering

data-pipeline: data-acquisition data-ingestion data-validation data-preprocess ## Run complete data pipeline (excluding acquisition)
	@echo "✅ Complete data pipeline executed successfully!"
	python -m src.data.pipeline.run_pipeline

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


train-monthly-win: ## Train monthly win probability model
	@echo "Training monthly win probability model..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_monthly_win_training.py

evaluate: ## Evaluate trained models
	@echo "Evaluating models..."
	python -m src.models.evaluate

# Model Serving
serve-model: ## Start model serving API
	@echo "Starting model serving API..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-ui: ## Start Streamlit UI
	@echo "Starting Streamlit UI..."
	streamlit run src_app/app.py

# MLFlow
mlflow-ui: ## Start MLFlow UI
	@echo "Starting MLFlow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-server: ## Start MLFlow tracking server
	@echo "Starting MLFlow tracking server..."
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000


# Prefect
prefect-run-acquisition: ## Run CRM ingestion flow directly (for testing)
	@echo "Running CRM ingestion flow locally..."
	@echo "Setting Prefect API URL..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_crm_acquisition.py

prefect-run-ingestion: ## Run CRM ingestion flow directly (for testing)
	@echo "Running CRM ingestion flow locally..."
	@echo "Setting Prefect API URL..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_crm_ingestion.py
prefect-run-monthly-training: ## Run monthly win probability training flow
	@echo "Running monthly win probability training flow..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/run_monthly_win_training.py

prefect-run-reference-creation: ## Run reference data creation flow
	@echo "Running reference data creation flow..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) .venv/bin/python src/pipelines/run_reference_data_creation.py

prefect-run-drift-monitoring: ## Run drift monitoring flow (requires current_month parameter)
	@echo "Running drift monitoring flow..."
	@read -p "Enter current month (e.g., 2017-06): " current_month; \
	export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) .venv/bin/python src/pipelines/run_drift_monitoring.py $$current_month

prefect-run: prefect-run-acquisition prefect-run-ingestion prefect-run-monthly-training prefect-run-reference-creation prefect-run-drift-monitoring ## Run all Prefect flows in sequence

prefect-deploy: ## Deploy CRM ingestion flow with S3 storage
	@echo "Deploying CRM ingestion flow with S3 storage..."
	@export PREFECT_API_URL=http://localhost:4200/api && \
	PYTHONPATH=$${PYTHONPATH}:$(shell pwd) python src/pipelines/deploy_crm_pipelines.py

prefect-deployments: ## List all Prefect deployments
	@echo "Listing Prefect deployments..."
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect deployment ls

prefect-flows: ## List all flow runs
	@echo "Listing Prefect flow runs..."
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect flow-run ls

prefect-ui: ## Open Prefect UI in browser
	@echo "Opening Prefect UI..."
	@open http://localhost:4200

prefect-help: ## Show all Prefect commands
	@echo "=== 🔄 Prefect Workflows (Operational) ==="
	@echo "prefect-deploy:                  Deploy all CRM flows with S3 storage"
	@echo "prefect-run-acquisition:         Run CRM data acquisition flow"
	@echo "prefect-run-ingestion:           Run CRM data ingestion flow"
	@echo "prefect-run-monthly-training:    Run monthly win probability training flow"
	@echo "prefect-run-reference-creation:  Create reference data for monitoring"
	@echo "prefect-run-drift-monitoring:    Run drift monitoring analysis"
	@echo "prefect-deployments:             List all deployments"
	@echo "prefect-flows:                   List recent flow runs"
	@echo "prefect-status-all:              Show comprehensive Prefect status"
	@echo "prefect-ui:                      Open Prefect UI in browser"
	@echo ""
	@echo "=== 🤖 ML Training (Complete) ==="
	@echo "train-monthly-win:               Train monthly win probability model locally"
	@echo ""
	@echo "=== 📦 MinIO S3 Storage (Active) ==="
	@echo "minio-ui:                        Open MinIO web console"
	@echo "minio-buckets:                   List all MinIO buckets"
	@echo "minio-list-data:                 List data-lake bucket contents"
	@echo "minio-list-mlflow:               List MLflow artifacts"
	@echo "minio-status:                    Check MinIO service status"

prefect-status-all: ## Show comprehensive Prefect status
	@echo "=== 🔄 Prefect Orchestration Status ==="
	@echo "Server Health:"
	@curl -s http://localhost:4200/api/health 2>/dev/null && echo " ✅ Server is running" || echo " ❌ Server is not responding"
	@echo "\nWork Pools:"
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect work-pool ls 2>/dev/null || echo "❌ Could not list work pools"
	@echo "\nDeployments:"
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect deployment ls 2>/dev/null || echo "❌ Could not list deployments"
	@echo "\nRecent Flow Runs:"
	@export PREFECT_API_URL=http://localhost:4200/api && .venv/bin/prefect flow-run ls --limit 5 2>/dev/null || echo "❌ Could not list flow runs"

# Code Quality
precommit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	.venv/bin/pre-commit install
	.venv/bin/pre-commit install --hook-type pre-push
	@echo "✅ Pre-commit hooks installed!"

precommit-run: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks on all files..."
	.venv/bin/pre-commit run --all-files

precommit-update: ## Update pre-commit hook versions
	@echo "Updating pre-commit hooks..."
	.venv/bin/pre-commit autoupdate

precommit-uninstall: ## Uninstall pre-commit hooks
	@echo "Uninstalling pre-commit hooks..."
	.venv/bin/pre-commit uninstall
	.venv/bin/pre-commit uninstall --hook-type pre-push

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

precommit-help: ## Show all pre-commit commands
	@echo "=== 🛡️  Code Quality & Security (Active) ==="
	@echo "precommit-install:           Install pre-commit hooks"
	@echo "precommit-run:               Run pre-commit hooks on all files"
	@echo "precommit-update:            Update pre-commit hook versions"
	@echo "precommit-uninstall:         Uninstall pre-commit hooks"
	@echo "format:                      Format code with black and isort"
	@echo "format-check:                Check code formatting without fixing"
	@echo "lint:                        Run linting checks"
	@echo "security-check:              Run security vulnerability scans"
	@echo ""
	@echo "=== ⚙️  Pre-commit Hook Details ==="
	@echo "The following hooks are configured:"
	@echo "- trailing-whitespace:       Remove trailing whitespace"
	@echo "- end-of-file-fixer:         Ensure files end with newline"
	@echo "- check-yaml:                Validate YAML syntax"
	@echo "- check-json:                Validate JSON syntax"
	@echo "- check-merge-conflict:      Detect merge conflict markers"
	@echo "- check-added-large-files:   Prevent large file commits (>10MB)"
	@echo "- mixed-line-ending:         Fix line endings to LF"
	@echo "- black:                     Python code formatting"
	@echo "- isort:                     Python import sorting"
	@echo "- flake8:                    Python linting (permissive)"
	@echo "- nbqa-black:                Format Jupyter notebooks"
	@echo ""
	@echo "See docs/PRE_COMMIT_GUIDE.md for detailed documentation"

security-check: ## Run security checks
	@echo "Running security checks..."
	bandit -r src/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	@echo "Security reports generated: bandit-report.json, safety-report.json"

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
	@echo "🚀 Starting MLOps Platform (PostgreSQL + MLflow + Prefect + MinIO)..."
	docker compose up -d

application-stop: ## Stop Docker containers
	@echo "Stopping Docker containers..."
	docker compose down

application-restart: application-stop application-start ## Restart Docker containers

application-status: ## Check application status in Docker
	@echo "📊 Checking MLOps Platform service status..."
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
dev-setup: setup install-venv create-dirs env-file precommit-install ## Complete development setup
	@echo "🚀 MLOps Platform development environment setup complete!"
	@echo ""
	@echo "✅ Current Implementation: CRM Sales Opportunities (Phases 1-6 Operational)"
	@echo "   📊 Data Pipeline (Kaggle CRM → Feature Engineering)"
	@echo "   🤖 ML Training (Monthly Win Probability Models)"
	@echo "   🌐 Streamlit Web App (Interactive Predictions)"
	@echo "   📈 Model Monitoring (Evidently AI Drift Detection)"
	@echo ""
	@echo "🎯 Next steps:"
	@echo "1. Activate virtual environment: source .venv/bin/activate"
	@echo "2. Update .env file with your Kaggle credentials"
	@echo "3. Start services: make application-start"
	@echo "4. Run complete pipeline: make prefect-run"
	@echo "5. Launch Streamlit app: make streamlit-app"

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
ci-test: install-venv test lint security-check ## Run CI tests

ci-build: clean install-venv test lint security-check docker-build ## Complete CI build

# Quick Commands
quick-start: dev-setup data-pipeline train ## Quick start for new developers

quick-flow: dev-setup prefect-deploy-crm prefect-agent ## Quick start with Prefect flows

# Streamlit Application
streamlit-app: ## Start Streamlit application (local)
	@echo "🚀 Starting CRM Win Probability Predictor..."
	@echo "📊 App will be available at: http://localhost:8501"
	@echo "🔧 Make sure MLflow server is running at http://localhost:5005"
	@echo ""
	PYTHONPATH=$${PYTHONPATH}:$(PWD) python scripts/start_streamlit.py

streamlit-dev: ## Start Streamlit in development mode with auto-reload (local)
	@echo "🚀 Starting Streamlit in development mode..."
	@echo "📊 App will be available at: http://localhost:8501"
	@echo "🔧 Make sure MLflow server is running at http://localhost:5005"
	@echo ""
	PYTHONPATH=$${PYTHONPATH}:$(PWD) streamlit run src_app/app.py \
		--server.port 8501 \
		--server.address 0.0.0.0 \
		--theme.base light \
		--server.runOnSave true

streamlit-logs: ## Show Streamlit container logs
	docker compose logs -f streamlit

streamlit-rebuild: ## Rebuild and restart Streamlit container
	@echo "🔄 Rebuilding Streamlit container..."
	docker compose build --no-cache streamlit
	docker compose up -d streamlit
	@echo "✅ Streamlit container rebuilt and restarted"


status: ## Show project status
	@echo "🚀 MLOps Platform Status"
	@echo "========================"
	@echo "📊 Current Implementation: CRM Sales Opportunities (Phases 1-6 Operational)"
	@echo "🐍 Python version: $(shell python --version)"
	@echo "🔬 MLFlow tracking URI: $(shell echo $${MLFLOW_TRACKING_URI:-http://localhost:5005})"
	@echo "📁 Data directory size: $(shell du -sh data/ 2>/dev/null || echo 'No data directory')"
	@echo "🤖 Models directory size: $(shell du -sh models/ 2>/dev/null || echo 'No models directory')"
	@echo ""
	@echo "🌐 Streamlit Web App Status:"
	@docker compose ps streamlit 2>/dev/null || echo "   Container not running"
	@curl -s -o /dev/null -w "   Health Check: %{http_code}" http://localhost:8501/_stcore/health 2>/dev/null || echo "   Not accessible"
	@echo ""
	@echo "🔗 Service URLs:"
	@echo "   • Streamlit App: http://localhost:8501"
	@echo "   • MLflow UI: http://localhost:5005"
	@echo "   • Prefect UI: http://localhost:4200"
	@echo "   • MinIO Console: http://localhost:9001"
