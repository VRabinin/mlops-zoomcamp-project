# MLOps Platform Architecture

This directory contains the solution architecture for the CRM Sales Opportunities MLOps Platform using C4 modeling notation and Structurizr DSL.

## 📁 Project Structure

```
mlops-zoomcamp-project/
├── .github/                        # CI/CD automation
│   ├── workflows/                  # GitHub Actions workflows
│   │   ├── ci.yml                 # Continuous integration pipeline
│   │   └── [other workflows]      # Additional automation
│   └── copilot-instructions.md     # AI assistant configuration
├── .pre-commit-config.yaml         # Code quality hooks
├── architecture/ 📍               # This directory - Solution architecture
│   ├── structurizr/               # Architecture diagrams as code
│   │   └── workspace.dsl          # C4 model in Structurizr DSL
│   ├── README.md                  # This file - Architecture documentation
│   └── docker-compose.yml         # Architecture viewer setup
├── config/                         # Configuration management
│   ├── development.yaml           # Local development config
│   ├── production.yaml            # Production environment config
│   └── staging.yaml               # Staging environment config
├── data/                          # Data storage (local development)
│   ├── raw/                       # Original datasets from Kaggle
│   ├── processed/                 # Cleaned and transformed data
│   ├── features/                  # ML-ready feature datasets
│   └── monitoring/                # Drift monitoring data
├── docs/                          # Documentation
│   ├── MODEL_DRIFT_MONITORING.md  # Monitoring system guide
│   ├── STREAMLIT_QUICKSTART.md    # Web app quick start
│   ├── STREAMLIT_DOCKER_SETUP.md  # Container deployment
│   └── [other guides]             # Additional documentation
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_processing_pipeline.ipynb      # EDA and data analysis
│   └── 02_monthly_win_probability_prediction.ipynb  # Model development
├── src/                           # Source code ✅ OPERATIONAL
│   ├── data/                      # Data pipeline modules ✅
│   │   ├── ingestion/             # Data acquisition and ingestion
│   │   │   ├── crm_ingestion.py          # Monthly snapshot processing ✅
│   │   │   └── crm_acquisition.py        # Enhanced data acquisition ✅
│   │   ├── validation/            # Data quality assurance
│   │   │   ├── run_validation.py         # Schema validation ✅
│   │   │   └── quality_checks.py         # Data quality metrics ✅
│   │   ├── preprocessing/         # Feature engineering ✅
│   │   │   ├── feature_engineering.py    # 23 ML features ✅
│   │   │   ├── data_cleaning.py          # Data cleaning utilities ✅
│   │   │   └── data_transformations.py   # Data transformations ✅
│   │   └── schemas/               # Data schema definitions ✅
│   │       └── crm_schema.py             # CRM data schema ✅
│   ├── models/                    # ML training modules ✅
│   │   └── training/              # Model training implementation ✅
│   │       └── monthly_win_probability.py    # Complete ML pipeline ✅
│   ├── monitoring/                # Model monitoring system ✅
│   │   ├── drift_monitor.py              # Evidently AI integration ✅
│   │   ├── evidently_metrics_calculation.py  # Metrics calculation ✅
│   │   └── streamlit_dashboard.py        # Monitoring dashboard ✅
│   ├── pipelines/                 # Prefect workflow definitions ✅
│   │   ├── run_crm_ingestion.py          # Monthly processing flow ✅
│   │   ├── run_crm_acquisition.py        # Data acquisition flow ✅
│   │   ├── run_monthly_win_training.py   # ML training flow ✅
│   │   ├── run_drift_monitoring.py       # Drift monitoring flow ✅
│   │   ├── run_reference_data_creation.py # Reference data flow ✅
│   │   ├── deploy_crm_pipelines.py       # S3-based deployment ✅
│   │   └── [other pipeline files]        # Additional workflows ✅
│   ├── utils/                     # Utility modules ✅
│   │   ├── storage.py                    # Intelligent S3/local storage ✅
│   │   └── prefect_client.py             # Prefect management utilities ✅
│   └── config/                    # Configuration classes ✅
│       ├── __init__.py                   # Configuration package ✅
│       └── config.py                     # Config management ✅
├── src_app/                       # Streamlit web application ✅
│   ├── app.py                     # Complete web interface ✅
│   ├── Dockerfile                 # Containerized deployment ✅
│   └── README.md                  # Application documentation ✅
├── tests/                         # Test suites ✅
│   ├── conftest.py               # Test configuration and fixtures ✅
│   ├── test_data_pipeline.py     # Data pipeline tests ✅
│   ├── test_model_training.py    # ML training tests ✅
│   ├── test_config.py            # Configuration tests ✅
│   ├── test_data_validation.py   # Data validation tests ✅
│   ├── test_prefect_client.py    # Prefect client tests ✅
│   └── test_storage_manager.py   # Storage management tests ✅
├── app_data/                      # Application data (Docker volumes)
│   ├── minio/                     # MinIO S3 storage data
│   ├── mlflow_data/              # MLflow artifacts and metadata
│   └── postgres/                 # PostgreSQL database files
├── docker-compose.yml             # Local development services ✅
├── Makefile                       # Development automation (30+ commands) ✅
├── requirements.txt               # Python dependencies ✅
├── pytest.ini                    # Test configuration ✅
├── prefect.yaml                   # Prefect configuration ✅
├── .env.template                  # Environment variables template ✅
├── README.md                      # Project overview and quick start
├── NEXT_STEPS.md                  # Development roadmap and current status
├── ROADMAP.md                     # Long-term project phases
└── .gitignore                     # Git ignore configuration
```

### Key Directory Explanations

**🏗️ Architecture (`architecture/`)**
- Contains C4 model diagrams and architectural documentation
- Structurizr DSL for architecture-as-code
- This comprehensive README with system design

**🔧 Source Code (`src/`)**
- **`data/`**: Complete data pipeline with ingestion, validation, and preprocessing
- **`models/`**: ML training modules with automated model selection
- **`monitoring/`**: Evidently AI-based drift detection and monitoring
- **`pipelines/`**: Prefect workflow orchestration for all ML operations
- **`utils/`**: Shared utilities including intelligent storage management
- **`config/`**: Centralized configuration management

**🌐 Web Application (`src_app/`)**
- Streamlit application providing complete business interface
- Multi-tab interface: predictions, monitoring, insights, pipeline overview
- Docker containerization for deployment

**🧪 Testing (`tests/`)**
- Comprehensive test suite covering all major components
- pytest-based testing with fixtures and mocks
- CI/CD integration for automated testing

**📊 Data Storage (`data/` and `app_data/`)**
- **`data/`**: Local development data storage
- **`app_data/`**: Docker volume mounts for persistent services

**📚 Documentation (`docs/`, `notebooks/`)**
- Detailed guides for monitoring, deployment, and usage
- Jupyter notebooks for exploration and model development
- Architecture documentation and development guides

### Status Legend
- ✅ **OPERATIONAL**: Fully implemented and tested
- 🚧 **IN PROGRESS**: Currently under development
- 📋 **PLANNED**: Scheduled for future implementation
- 🔮 **FUTURE**: Optional enhancement for later phases

## Architecture Overview

The solution follows a microservices-based architecture that supports both local development/testing and cloud production deployment. The platform is designed to handle the complete machine learning lifecycle from data ingestion to model monitoring.

## Architecture Components

### 1. User Interface Layer
- **Streamlit Web App**: Primary user interface for predictions and monitoring
  - Prediction Interface: Input forms for sales opportunity data
  - Visualization Dashboard: Model performance charts and business insights
  - Monitoring Dashboard: Real-time model performance monitoring

### 2. Model Serving Layer
- **MLFlow Model Serving**: Production model inference service
  - Model API: REST endpoints for predictions
  - Model Registry: Version management and model metadata
  - Model Loader: Efficient model caching and loading

### 3. ML Pipeline Layer ✅ **OPERATIONAL**
- **Enhanced Data Acquisition Pipeline**: Advanced CRM data processing ✅
  - Kaggle Dataset Integration: Automated download and validation
  - Multi-Month Simulation: Time-series data enhancement
  - Data Quality Validation: 0.93 quality score achievement
  - S3 Storage Integration: MinIO-based data lake (7.5MB+ features)

- **Monthly Processing Pipeline**: Feature engineering and validation ✅
  - Feature Engineering: 23 ML-ready features from 8 original columns
  - Data Validation: Comprehensive quality checks and schema compliance
  - Storage Management: Intelligent local/S3 backend selection
  - Pipeline Orchestration: Prefect 3.x workflow automation

- **Training Pipeline**: ML model training workflow 🚧 **NEXT PHASE**
  - Model Training: ML model training with hyperparameter tuning
  - Model Evaluation: Performance assessment and validation
  - Experiment Tracking: MLflow integration for model versioning

### 4. Orchestration Layer ✅ **OPERATIONAL**
- **Prefect 3.x Server**: Workflow orchestration and scheduling ✅
  - Flow Management: Dual pipeline architecture (acquisition + processing)
  - Task Scheduling: Automated workflow execution
  - S3-Based Deployment: Complete source code stored in MinIO
  - Monitoring Dashboard: Real-time workflow status and execution history

- **Storage Orchestration**: Intelligent data management ✅
  - MinIO S3 Storage: Production-ready object storage (8.8MB+ data)
  - Automatic Backend Selection: Environment-aware storage routing
  - Bucket Management: Organized data lake with versioning
  - Data Lifecycle: Raw → Processed → Features → Models
- **Prefect Workflow Orchestrator**: Manages ML workflows
  - Scheduler: Automated training and batch prediction scheduling
  - Task Executor: Individual pipeline task execution
  - Workflow Monitor: Pipeline execution monitoring and alerting

### 5. Experiment Tracking
- **MLFlow Tracking**: Experiment and model lifecycle management
  - Experiment Logger: Parameters, metrics, and artifact logging
  - Artifact Store: Model artifacts and dataset versioning
  - Metrics Tracker: Performance metrics across experiments

### 6. Monitoring Layer
- **Evidently AI Monitoring**: Model performance and data quality monitoring
  - Data Drift Detector: Input data distribution monitoring
  - Performance Monitor: Model accuracy and business metrics tracking
  - Alerting System: Automated alerts for model degradation

### 7. Data Storage
- **Multi-tier Storage**: Optimized data storage strategy
  - Feature Store: Curated features for training and inference
  - Data Lake: Raw and processed data with versioning
  - Model Artifacts: Trained models and associated metadata

## Infrastructure

### Local Development Environment
- **Docker**: Containerization for consistent environments
- **MinIO**: S3-compatible object storage for development
- **Docker Compose**: Multi-service orchestration

### Production Environment
- **AWS Cloud**: Production deployment platform
  - EC2: Scalable compute instances
  - S3: Object storage for data and models
  - RDS: Managed database services
  - ALB: Application load balancing

### Container Orchestration
- **HashiCorp Nomad**: Container orchestration and scheduling
- **Consul**: Service discovery and configuration management

### Infrastructure as Code
- **Terraform**: Infrastructure provisioning and management
  - Modular design for reusability
  - State management for consistency
  - Multi-environment support

## CI/CD Pipeline

### GitHub Actions Workflows
- **Build Pipeline**: Code compilation, testing, and containerization
- **Deployment Pipeline**: Automated deployment to different environments
- **Testing Pipeline**: Unit tests, integration tests, and model validation

## Data Flow

1. **Data Ingestion**: CRM system data is extracted and validated
2. **Data Processing**: Raw data is cleaned, transformed, and feature-engineered
3. **Model Training**: ML models are trained using processed features
4. **Model Evaluation**: Models are validated and compared against baselines
5. **Model Deployment**: Best performing models are deployed to serving infrastructure
6. **Prediction Serving**: Real-time and batch predictions are served via API
7. **Monitoring**: Model performance and data quality are continuously monitored
8. **Feedback Loop**: Monitoring insights trigger retraining workflows

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI Framework | Streamlit | Web application interface |
| Experiment Tracking | MLFlow | Model lifecycle management |
| Workflow Orchestration | Prefect | Pipeline automation |
| Model Serving | MLFlow | Model deployment and serving |
| Monitoring | Evidently AI | Model and data monitoring |
| CI/CD | GitHub Actions | Automation and deployment |
| IaC | Terraform | Infrastructure management |
| Container Orchestration | HashiCorp Nomad | Container management |
| Database | PostgreSQL | Structured data storage |
| Object Storage | AWS S3 / MinIO | Unstructured data and artifacts |
| Containerization | Docker | Application packaging |

## Deployment Strategies

### Local Development
1. Start Docker services with Docker Compose
2. Deploy MinIO for S3-compatible object storage
3. Use Docker containers for service orchestration
4. Connect to local databases and storage

### Production Deployment
1. Provision AWS infrastructure using Terraform
2. Deploy services using HashiCorp Nomad
3. Configure monitoring and alerting
4. Set up CI/CD pipelines for automated deployments

## Security Considerations

- API authentication and authorization
- Data encryption at rest and in transit
- Network segmentation and security groups
- Secret management for credentials
- Audit logging for compliance

## Monitoring and Observability

- Application performance monitoring
- Infrastructure monitoring
- Model performance tracking
- Data quality monitoring
- Business metrics tracking
- Alerting and notification systems

## Scalability Design

- Horizontal scaling for compute-intensive workloads
- Auto-scaling based on demand
- Load balancing for high availability
- Caching strategies for performance optimization
- Database optimization and read replicas

This architecture provides a robust, scalable, and maintainable foundation for the MLOps platform while supporting both local development and cloud production environments.
