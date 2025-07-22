# Copilot Instructions for MLOps Platform

## 🎯 Project Overview
This is a **CRM Sales Opportunities MLOps Platform** - an end-to-end machine learning system for predicting sales outcomes using microservices architecture. The project follows MLOps best practices with Prefect orchestration, MLflow tracking, and multi-environment deployment (local Docker → AWS production).

## 🏗️ Architecture & Components

### Core Technology Stack
- **Orchestration**: Prefect 2.14+ for workflow management
- **Experiment Tracking**: MLflow 2.7+ with PostgreSQL backend  
- **Data Source**: Kaggle CRM dataset (`innocentmfa/crm-sales-opportunities`)
- **Services**: Docker Compose with PostgreSQL, Redis, LocalStack (AWS emulation)
- **Infrastructure**: Terraform + HashiCorp Nomad for container orchestration
- **CI/CD**: GitHub Actions with security scanning and multi-stage builds

### Project Structure (Early Stage)
```
├── .github/workflows/          # CI/CD with pytest, flake8, bandit, safety
├── architecture/               # C4 model diagrams (Structurizr DSL)
├── config/development.yaml     # YAML-based configuration management
├── docker-compose.yml          # Multi-service local development stack
├── Makefile                    # 30+ commands for development workflow
├── tests/                      # pytest with fixtures for data pipeline testing
├── notebooks/                  # EDA and experimentation
└── [src/]                      # Source code (planned, see NEXT_STEPS.md)
    ├── data/                   # Data pipeline modules
    │   ├── ingestion/         # CRM dataset download & validation
    │   ├── validation/        # Schema validation & quality checks
    │   ├── preprocessing/     # Feature engineering & data cleaning
    │   └── schemas/           # Data schema definitions (CRMDataSchema)
    ├── models/                # ML training & evaluation
    │   ├── training/          # Model training logic
    │   ├── evaluation/        # Model performance assessment
    │   └── hyperparameter_tuning/  # HPO workflows
    ├── pipelines/             # Prefect workflow definitions
    │   ├── run_crm_ingestion.py     # CRM data pipeline flow
    │   ├── deploy_flows.py    # Batch flow deployment
    │   └── deploy_crm_pipeline.py   # Flow deployment scripts
    ├── api/                   # FastAPI model serving
    ├── streamlit_app/         # Web interface
    └── config/                # Configuration management
        └── config.py          # Config classes (DataConfig, MLflowConfig)
```

## 🔧 Development Patterns

### Essential Makefile Commands
```bash
make dev-setup                  # Complete environment setup (venv + deps + dirs)
make application-start          # Start all Docker services
make data-pipeline-flow         # Run data ingestion as Prefect flow
make prefect-deploy-crm         # Deploy CRM flow to Prefect server
make test                       # Run pytest suite with coverage
make architecture-start         # View C4 diagrams at localhost:8080
```

### Configuration Management
- **Environment**: `.env` file from `.env.template` (requires Kaggle API credentials)
- **App Config**: `config/development.yaml` for structured settings
- **Database**: PostgreSQL with MLflow backend store
- **Artifact Storage**: LocalStack S3 (dev) → AWS S3 (prod)
- **Multi-Environment Pattern**: 
  - Base config in `development.yaml`
  - Override with environment variables for staging/prod
  - Use `get_config()` function for runtime config access
  - Database URLs: `postgresql://mlops_user:mlops_password@localhost:5432/mlops` (local)

### Prefect Workflow Patterns
- **Flow Definition**: Use `@flow` decorator with descriptive names (`crm_data_ingestion_flow`)
- **Task Structure**: Break flows into reusable `@task` functions with error handling
- **Deployment Pattern**: 
  ```python
  # 1. Define flow in src/pipelines/
  # 2. Deploy with: make prefect-deploy-crm  
  # 3. Run with agent: make prefect-agent
  ```
- **Work Pools**: Use `default-agent-pool` for local development
- **Error Handling**: Implement retry logic and failure notifications in tasks
- **PYTHONPATH**: Always set `PYTHONPATH=${PYTHONPATH}:$(pwd)` before running flows

### Data Pipeline Architecture
- **Ingestion**: Kaggle API → raw data validation → feature engineering
- **Orchestration**: Prefect flows with error handling and monitoring
- **Storage**: Tiered approach (data/raw, data/processed, data/features)
- **Validation**: Schema-based validation with quality scoring
- **Schema Validation Pattern**:
  ```python
  # 1. Define schema in src/data/schemas/crm_schema.py (CRMDataSchema)
  # 2. Use required_columns list and column_types dict
  # 3. Implement validate_data() with quality scoring (0.0-1.0)
  # 4. Return tuple: (is_valid: bool, issues: List[str])
  # 5. Quality score considers: missing values, duplicates, schema compliance
  ```
- **Target Column**: Always use `deal_stage` as prediction target
- **Feature Engineering**: Separate feature columns from target using `get_feature_columns()`

## 🧪 Testing & Quality

### Testing Strategy
- **pytest** with `sample_dataframe` fixtures for data testing
- **Coverage**: `pytest --cov=src --cov-report=html`
- **Linting**: flake8 + mypy for type checking
- **Security**: bandit + safety in CI pipeline
- **Integration**: Services tested against Docker stack

### Code Quality Standards
- **Formatting**: black + isort (automatic in `make format`)
- **Type Hints**: Required for public APIs
- **Module Structure**: Follow `src/data/ingestion/`, `src/models/`, etc.
- **Import Patterns**: Use `from src.config.config import get_config` for configuration
- **Class Design**: Follow test patterns like `CRMDataIngestion(config)` initialization
- **Error Handling**: Return tuples for validation: `(success: bool, data/errors)`

## 🚀 Development Workflow

### Daily Commands
```bash
# Start environment
make dev-start                  # MLflow UI + Prefect server + agent

# Development cycle
make data-pipeline              # Download → validate → process
make test lint                  # Quality checks
make mlflow-ui                  # http://localhost:5000
make prefect-server             # http://localhost:4200
```

### Service URLs (Local)
- **MLflow UI**: http://localhost:5000 (experiment tracking)
- **Prefect UI**: http://localhost:4200 (workflow management)
- **Architecture**: http://localhost:8080 (C4 diagrams)
- **PostgreSQL**: localhost:5432 (mlops/mlops_user/mlops_password)

## 📋 Key Implementation Notes

### Current Phase: Data Pipeline Development
- Project is in **Phase 2** - data pipeline implementation (see NEXT_STEPS.md)
- Tests expect `src/` modules but source code is still being developed
- Makefile references `src.data.ingestion.crm_ingestion` module structure
- Configuration supports both local Docker and AWS production environments

### Critical Dependencies
- **Python 3.11** (strict requirement for MLOps tool compatibility)
- **Kaggle API**: Required for dataset access (`~/.kaggle/kaggle.json`)
- **Docker**: Core development environment with compose v2+

### Important Patterns
- **PYTHONPATH management**: `PYTHONPATH=${PYTHONPATH}:$(pwd)` in Makefile
- **Multi-environment config**: development.yaml → production overrides
- **Prefect deployment**: Separate deployment scripts for flow registration
- **Data validation**: Quality scoring with schema compliance checks

## 🔍 Integration Points

### External Systems
- **Kaggle API**: Dataset ingestion with credential management
- **MLflow**: Backend store on PostgreSQL, artifacts on S3/LocalStack
- **Prefect**: API-based deployment with work pool management
- **GitHub Actions**: Multi-stage pipeline with security and build phases

### Cross-Component Communication
- **Prefect ↔ MLflow**: Experiment logging during orchestrated runs
- **MLflow ↔ Model Serving**: Model registry for deployment
- **Streamlit ↔ Model API**: Web interface for predictions (planned)

When implementing new features, follow the established patterns in tests/, respect the Makefile workflow commands, and ensure compatibility with both local Docker and AWS production environments.

## 📚 **Related Documentation**
- **README.md**: Quick start, troubleshooting, and service URLs
- **NEXT_STEPS.md**: Current sprint tasks and immediate action items  
- **ROADMAP.md**: Long-term project phases and business context
- **architecture/README.md**: Detailed technical architecture and data flows
