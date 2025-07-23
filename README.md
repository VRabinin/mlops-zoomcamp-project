# CRM Sales Opportunities MLOps Platform

An end-to-end machine learning platform for predicting sales opportunities using CRM data. This project implements a complete MLOps pipeline with modern tools and practices for both local development and cloud production deployment.

## 🎯 Objective

Build a production-ready machine learning system that:
- Predicts sales opportunity outcomes using CRM data
- Provides real-time predictions through a web interface
- Monitors model performance and data quality
- Supports continuous integration and deployment
- Scales from local development to cloud production

## 🏗️ Solution Architecture

The platform follows a microservices-based architecture with the following key components:

### Core Technologies
- **User Interface**: Streamlit
- **Experiment Tracking**: MLFlow 2.7+ (PostgreSQL backend)
- **Workflow Orchestration**: Prefect 3.x (fully operational with CRM data pipeline)
- **Model Serving**: MLFlow
- **Monitoring**: Evidently AI
- **CI/CD**: GitHub Actions
- **Infrastructure as Code**: Terraform
- **Container Orchestration**: HashiCorp Nomad

### Environment Support
- **Local Development**: Docker + MinIO
- **Production Deployment**: AWS Cloud

## 📁 Project Structure

```
├── .github/workflows/      # CI/CD pipelines
├── architecture/           # Solution architecture documentation
│   ├── structurizr/       # Structurizr DSL Directory
│   │   └── workspace.dsl  # C4 model in Structurizr DSL
│   ├── README.md          # Architecture documentation
│   └── docker-compose.yml # Structurizr local setup
├── config/                # Configuration files
│   └── development.yaml   # Development environment config
├── data/                  # Data directories
│   ├── raw/              # Raw data storage
│   ├── processed/        # Processed data
│   └── features/         # Feature store
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/             # Data pipeline modules
│   │   ├── ingestion/    # Data ingestion (Kaggle CRM dataset) ✅ OPERATIONAL
│   │   ├── validation/   # Data quality validation ✅ OPERATIONAL
│   │   ├── preprocessing/ # Feature engineering ✅ OPERATIONAL
│   │   └── schemas/      # Data schema definitions ✅ OPERATIONAL
│   ├── pipelines/        # Prefect workflow definitions ✅ OPERATIONAL
│   │   ├── run_crm_pipeline.py     # CRM data pipeline flow
│   │   └── deploy_crm_pipeline.py  # Flow deployment scripts
│   └── config/           # Configuration management ✅ OPERATIONAL
├── tests/                 # Test suites
├── docker-compose.yml     # Local development services
├── Makefile              # Development commands
├── requirements.txt      # Python dependencies
└── .env.template         # Environment configuration template
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**: Required for optimal compatibility with MLOps tools
- **Docker & Docker Compose**: For local development services (v2+ with compose plugin)
- **Git**: For version control
- **Kaggle Account**: For accessing the CRM dataset

### 1. Installation

**Check Python Version:**
```bash
python --version  # Should be 3.11.x
```

**Install Python 3.11 (if needed):**
```bash
# macOS with Homebrew
brew install python@3.11

# Or download from https://www.python.org/downloads/
```

**Clone and Setup:**
```bash
# Clone the repository
git clone https://github.com/VRabinin/mlops-zoomcamp-project.git
cd mlops-zoomcamp-project

# Complete development setup (creates venv, installs deps, creates directories)
make dev-setup

# Activate virtual environment
source .venv/bin/activate
```

### 2. Configure Environment

**Set up Kaggle API (required for dataset):**
```bash
# 1. Get your API credentials from https://www.kaggle.com/account
# 2. Download kaggle.json
# 3. Place credentials:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Configure environment variables:**
```bash
# Edit .env file with your Kaggle credentials and other settings
```

### 3. Start Development Services

```bash
# Start all local infrastructure (PostgreSQL, MLFlow, Redis, etc.) on Docker
make application-start

# Verify services are running
make application-status
```

### 4. Run Data Pipeline

```bash
# 🎯 FULLY OPERATIONAL: Complete CRM data pipeline with Prefect orchestration

# Download and process CRM dataset from Kaggle (standalone)
# Data is stored in subfolders of directory ./data
make data-pipeline

# Or run steps individually:
make data-download    # Download CRM dataset
make data-validate    # Validate data quality  
make data-process     # Process raw data into features


# Run CRM flow on Prefect server directly
make prefect-run-crm  # Run flow without deploying

# Deploy CRM flow to Prefect server
#Data is stored in the bucket on MinIO
make prefect-deploy-crm  # Deploy flow for scheduled/manual execution

# Monitor workflow execution
make prefect-status-all  # Comprehensive status (server, deployments, runs)
make prefect-ui         # Open Prefect dashboard (http://localhost:4200)

# Manual workflow execution
make prefect-run-deployment  # Trigger deployed CRM flow manually

# Additional Prefect commands
make prefect-deployments    # List all deployments
make prefect-flows         # Show recent flow runs
make prefect-help          # Show all Prefect commands
```

**✅ Current Status**: The CRM data pipeline is fully operational and processes 8,800+ CRM records with 23 engineered features. The pipeline achieves a 0.93 validation score and supports both standalone and Prefect-orchestrated execution.

### 5. Storage Configuration

**🗃️ Intelligent Storage Management**: The pipeline automatically selects the appropriate storage backend based on execution environment:

```bash
# 🔍 Storage Detection Logic:
# ✅ Local Mode (Direct execution): Uses ./data directories
# ✅ S3 Mode (Prefect orchestration): Uses MinIO buckets  
# ✅ Docker Mode (Container execution): Uses S3/MinIO storage
# ✅ Forced Mode: USE_S3_STORAGE=true environment variable

# Test storage modes
./test-storage-modes.sh

# Local filesystem storage (development)
python src/pipelines/run_crm_pipeline.py

# S3/MinIO storage (orchestrated/production)
make prefect-run-deployment  # Uses S3 automatically

# Force S3 storage for testing
USE_S3_STORAGE=true python src/pipelines/run_crm_pipeline.py
```

**📦 Storage Locations:**
- **Local Mode**: `./data/raw/`, `./data/processed/`, `./data/features/`
- **S3 Mode**: `s3://data-lake/raw/`, `s3://data-lake/processed/`, `s3://data-lake/features/`
- **MinIO Web UI**: http://localhost:9001 (minioadmin/minioadmin)

**🎯 Benefits:**
- **Seamless Transition**: Same code works locally and in production
- **Development Efficiency**: Local files for quick iteration
- **Production Ready**: S3-compatible storage for scalability
- **Container Compatible**: Automatic S3 mode in Docker environments

### 6. Explore and Train

```bash
# Start MLFlow UI (experiment tracking)
make mlflow-ui
# Open http://localhost:5000

# Run initial data exploration
python notebooks/01_exploratory_data_analysis.py

# Train baseline models (coming soon)
# make train
```

### 7. View Architecture

```bash
# Start architecture viewer
make architecture-start
# Open http://localhost:8080 to view C4 diagrams (port configurable via STRUCTURIZR_PORT)

# Stop the architecture viewer 
make architecture-stop
```

## 📊 Key Features

- **Real-time Predictions**: Web interface for immediate sales opportunity scoring
- **Automated Training**: Scheduled model retraining with new data
- **Model Monitoring**: Continuous monitoring of model performance and data drift
- **Experiment Tracking**: Complete ML experiment lifecycle management
- **Scalable Infrastructure**: Auto-scaling based on demand
- **Multi-environment Support**: Seamless local-to-cloud deployment

## 🔧 Development Workflow

### Daily Development Commands

```bash
# Start development environment with Prefect orchestration
make prefect-start      # Start Prefect server + agent (replaces dev-start)

# Alternative: Start individual services  
make dev-start          # Start MLFlow + basic services
make prefect-server     # Start Prefect server only
make prefect-agent      # Start Prefect agent only

# Check project status
make status
make prefect-status-all # Enhanced status with Prefect info

# Run data pipeline (multiple options)
make data-pipeline              # Standalone execution
make data-pipeline-flow         # Prefect-orchestrated execution  
make prefect-run-deployment     # Manual deployment trigger

# Deploy and manage Prefect workflows
make prefect-deploy-crm         # Deploy CRM flow to server
make prefect-deployments        # List all deployments
make prefect-flows             # Show recent flow runs

# Development and testing
make test               # Run test suite
make lint format        # Code quality checks

# Monitoring and debugging
make prefect-ui         # Open Prefect dashboard
make mlflow-ui          # Open MLFlow dashboard
make prefect-help       # Show all Prefect commands

# Clean up
make clean
make prefect-stop       # Stop Prefect services
```

### Full Development Cycle

1. **Data Ingestion**: ✅ **OPERATIONAL** - Extract CRM data from Kaggle with validation using Prefect 3.x flows
2. **Feature Engineering**: ✅ **OPERATIONAL** - Transform raw data into 23 ML-ready features  
3. **Model Training**: 🚧 **IN PROGRESS** - Train and evaluate ML models with Prefect orchestration
4. **Experiment Tracking**: ✅ **OPERATIONAL** - Log experiments and artifacts with MLFlow (PostgreSQL backend)
5. **Model Deployment**: 📋 **PLANNED** - Deploy best models to serving infrastructure
6. **Monitoring**: 📋 **PLANNED** - Track model performance with Evidently AI
7. **CI/CD**: ✅ **OPERATIONAL** - Automated testing and deployment with GitHub Actions

**Pipeline Achievements:**
- **Data Volume**: Processing 8,800+ CRM records successfully
- **Feature Engineering**: 23 engineered features from 8 original columns  
- **Data Quality**: 0.93 validation score with comprehensive quality checks
- **Orchestration**: Prefect 3.x workflows with scheduling and monitoring
- **Infrastructure**: Docker Compose with PostgreSQL, Redis, and MinIO

### Available Make Commands

| Command | Description | Status |
|---------|-------------|--------|
| `make help` | Show all available commands | ✅ |
| `make dev-setup` | Complete development environment setup | ✅ |
| `make prefect-start` | **Start Prefect server + agent (recommended)** | ✅ |
| `make dev-start` | Start MLFlow + basic services | ✅ |
| `make data-pipeline` | Run complete data ingestion pipeline | ✅ |
| `make data-pipeline-flow` | **Run data pipeline as Prefect flow** | ✅ |
| `make prefect-deploy-crm` | **Deploy CRM ingestion flow to Prefect** | ✅ |
| `make prefect-run-deployment` | **Manually trigger CRM deployment** | ✅ |
| `make prefect-status-all` | **Show comprehensive Prefect status** | ✅ |
| `make prefect-deployments` | **List all Prefect deployments** | ✅ |
| `make prefect-flows` | **Show recent flow runs** | ✅ |
| `make prefect-ui` | **Open Prefect dashboard** | ✅ |
| `make prefect-help` | **Show all Prefect commands** | ✅ |
| `make test` | Run test suite | ✅ |
| `make lint` | Run code quality checks | ✅ |
| `make architecture` | View architecture diagrams | ✅ |
| `make clean` | Clean temporary files | ✅ |

**✅ New Prefect Commands**: 11 comprehensive workflow management commands added.

See `make help` for the complete list of 30+ commands.

## 📈 Monitoring & Observability

- Model performance metrics and alerts
- Data quality and drift detection
- Infrastructure and application monitoring
- Business impact tracking
- Automated retraining triggers

## 🔒 Security & Compliance

- API authentication and authorization
- Data encryption at rest and in transit
- Audit logging and compliance reporting
- Network security and isolation
- Secret management

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

- [Architecture Overview](architecture/README.md) - Detailed system architecture
- [Next Steps Guide](NEXT_STEPS.md) - Development roadmap and priorities  
- [Development Roadmap](ROADMAP.md) - Complete project timeline
- [Configuration Guide](.env.template) - Environment setup

### Service URLs (Local Development)

| Service | URL | Purpose |
|---------|-----|---------|
| MLFlow UI | http://localhost:5000 | Experiment tracking and model registry |
| Prefect UI | http://localhost:4200 | Workflow orchestration dashboard |
| Architecture Viewer | http://localhost:8080* | C4 model diagrams |
| PostgreSQL | localhost:5432 | Database (mlops/mlops_user/mlops_password) |
| Redis | localhost:6379 | Caching and message broker |
| MinIO | http://localhost:9001 | S3-compatible object storage |

*Port configurable via `STRUCTURIZR_PORT` environment variable

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose | Status |
|-----------|------------|---------|---------|--------|
| **Runtime** | Python | 3.11+ | Core programming language | ✅ |
| **Frontend** | Streamlit | Latest | Web application interface | 📋 |
| **ML Tracking** | MLFlow | 2.7+ | Experiment and model management | ✅ |
| **Orchestration** | Prefect | 3.x | Workflow automation | ✅ |
| **Monitoring** | Evidently AI | 0.4+ | Model and data monitoring | 📋 |
| **CI/CD** | GitHub Actions | - | Automation pipeline | ✅ |
| **Infrastructure** | Terraform | Latest | Infrastructure as code | 📋 |
| **Containers** | Docker + Nomad | Latest | Container orchestration | ✅ |
| **Database** | PostgreSQL | 15 | Structured data storage | ✅ |
| **Cache** | Redis | 7 | Caching and message broker | ✅ |
| **Storage** | MinIO/S3 | Latest | Object storage for artifacts | ✅ |
| **API Framework** | FastAPI | 0.104+ | Model serving API | 📋 |

**Legend**: ✅ Operational | 🚧 In Progress | 📋 Planned

## 🚨 Troubleshooting

### Common Issues

**Python Version Issues:**
```bash
# Check your Python version
python --version

# If not 3.11, install it:
brew install python@3.11  # macOS
# or download from python.org
```

**Kaggle API Issues:**
```bash
# Verify Kaggle setup
kaggle datasets list
# Should show available datasets

# If authentication fails:
# 1. Check ~/.kaggle/kaggle.json exists
# 2. Verify file permissions: chmod 600 ~/.kaggle/kaggle.json
# 3. Check credentials are correct
```

**Docker Issues:**
```bash
# Check Docker is running
docker info

# Restart services if needed
docker compose down
docker compose up -d

# Check service logs
docker compose logs mlflow
docker compose logs postgres
```

**Port Conflicts:**
```bash
# Check what's using ports
lsof -i :5000  # MLFlow
lsof -i :4200  # Prefect
lsof -i :5432  # PostgreSQL
lsof -i :9000  # MinIO API
lsof -i :9001  # MinIO Console

# Kill processes if needed
sudo kill -9 <PID>
```

**MinIO S3 Storage Issues:**
```bash
# Check MinIO containers
docker ps | grep minio

# Check MinIO status
make minio-status

# List MinIO buckets
make minio-buckets

# Check MinIO logs
docker logs mlops-minio
docker logs mlops-minio-setup

# Access MinIO web UI
open http://localhost:9001
# Login: minioadmin / minioadmin

# Clear MinIO data if needed
make minio-clear-data
```

### Getting Help

1. **Check service status**: `make status`
2. **View logs**: `docker compose logs <service-name>`
3. **Restart services**: `docker compose restart`
4. **Clean restart**: `make clean && make dev-setup`
5. **Check issues**: [GitHub Issues](https://github.com/VRabinin/mlops-zoomcamp-project/issues)

---

Built with ❤️ for the MLOps community