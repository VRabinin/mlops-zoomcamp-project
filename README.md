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
- **Storage**: MinIO S3-compatible storage (✅ **OPERATIONAL** - 7.5MB+ CRM features stored)
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
├── .pre-commit-config.yaml # Pre-commit hooks configuration
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
│   │   │   ├── crm_ingestion.py     # Monthly snapshot processing ✅
│   │   │   └── crm_acquisition.py   # Enhanced data acquisition ✅
│   │   ├── validation/   # Data quality validation ✅ OPERATIONAL (0.93 score)
│   │   ├── preprocessing/ # Feature engineering ✅ OPERATIONAL (23 features)
│   │   └── schemas/      # Data schema definitions ✅
│   ├── pipelines/        # Prefect workflow definitions ✅ OPERATIONAL
│   │   ├── run_crm_ingestion.py     # Monthly CRM processing flow ✅
│   │   ├── run_crm_acquisition.py   # Enhanced acquisition flow ✅
│   │   └── deploy_crm_pipelines.py  # S3-based deployment ✅
│   ├── utils/            # Utility modules ✅
│   │   └── storage.py    # Intelligent S3/local storage manager ✅
│   └── config/           # Configuration management ✅
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

Get the MLOps platform running locally with Docker in under 10 minutes.

### Prerequisites

- **Docker & Docker Compose**: v2+ with compose plugin
- **Python 3.11**: Required for optimal compatibility
- **Git**: For repository cloning
- **Kaggle Account**: For CRM dataset access

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/VRabinin/mlops-zoomcamp-project.git
cd mlops-zoomcamp-project

# Complete environment setup (Python venv + dependencies + directories)
make dev-setup

# Activate virtual environment
source .venv/bin/activate
```

### Step 2: Configure Kaggle API

```bash
# Get API credentials from https://www.kaggle.com/account
# Download kaggle.json and place it:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Create environment file from template
cp .env.template .env
# Edit .env with your Kaggle username and API key and adjust other variables if needed
```

### Step 3: Start Docker Services

```bash
# Start all MLOps infrastructure services
make application-start

# This starts:
# - PostgreSQL (MLflow backend)
# - MinIO (S3-compatible storage)
# - Redis (caching)
# - MLflow Server
# - Prefect Server

# Verify all services are running
make application-status
```

### Step 4: Run the Data Pipeline

```bash
#Deploy all pipeline parts to Prefect Server
make prefect-deploy

# Run the complete CRM data pipeline
make prefect-run

# This will:
# 1. Download CRM data from Kaggle (8,800+ records)
# 2. Validate data quality (0.93 score expected)
# 3. Engineer 23 ML features
# 4. Store results in MinIO S3 storage
```

### Step 5: Access the Platform

Open these services in your browser:

| Service | URL | Purpose |
|---------|-----|---------|
| **Prefect UI** | http://localhost:4200 | Workflow orchestration dashboard |
| **MLflow UI** | http://localhost:5000 | Experiment tracking & model registry |
| **MinIO Console** | http://localhost:9001 | S3 storage management |
| **Streamlit App** | http://localhost:8501 | ML predictions interface |

**Default Credentials:**
- MinIO: `minioadmin` / `minioadmin`
- PostgreSQL: `mlops_user` / `mlops_password`

### Step 6: Explore the Features

```bash
# View processed data in MinIO
make minio-ui
# Navigate to 'data-lake' bucket to see processed CRM features

# Check pipeline status
make prefect-status-all

# Launch the prediction web app
make streamlit-app
# Visit http://localhost:8501 for interactive ML predictions

# Start MLFlow UI for experiment tracking
make mlflow-ui
# Visit http://localhost:5000 for model registry and experiments
```

### That's It! 🎉

You now have a complete MLOps platform running locally with:
- ✅ **Data Pipeline**: Automated CRM data processing
- ✅ **Experiment Tracking**: MLflow with PostgreSQL backend
- ✅ **Workflow Orchestration**: Prefect for pipeline automation
- ✅ **Object Storage**: MinIO S3-compatible storage
- ✅ **Web Interface**: Streamlit app for ML predictions
- ✅ **Monitoring**: Service health and pipeline status

### Step 7: View Architecture

```bash
# Start architecture viewer
make architecture-start
# Open http://localhost:8080 to view C4 diagrams (port configurable via STRUCTURIZR_PORT)

# Stop the architecture viewer
make architecture-stop
```

### Next Steps

```bash
# Explore the data
make jupyter-start  # Launch Jupyter notebooks

# Train ML models
make model-train    # Train win probability models

# Deploy flows for scheduling
make prefect-deploy-crm  # Deploy to Prefect server

# Monitor data quality
make monitoring-start    # Launch Evidently AI monitoring

# View comprehensive help
make help  # See all 30+ available commands
```

## 📊 Key Features

- **Real-time Predictions**: Web interface for immediate sales opportunity scoring
- **Automated Training**: Scheduled model retraining with new data
- **Model Monitoring**: Continuous monitoring of model performance and data drift
- **Experiment Tracking**: Complete ML experiment lifecycle management
- **Scalable Infrastructure**: Auto-scaling based on demand
- **Multi-environment Support**: Seamless local-to-cloud deployment

## 🔧 Development Workflow

### Code Quality & Pre-commit Hooks

**✅ Automated Code Quality**: This project uses pre-commit hooks to ensure consistent code quality:

```bash
# Pre-commit hooks are automatically installed with `make dev-setup`
# They run automatically before each commit to:
# - Format Python code with Black
# - Sort imports with isort
# - Run basic linting with flake8
# - Check YAML/JSON syntax
# - Remove trailing whitespace
# - Format Jupyter notebooks

# Manual pre-commit commands:
make precommit-run          # Run hooks on all files
make format                 # Format code manually
make lint                   # Run linting checks
make precommit-help         # Show all pre-commit options

# The hooks are configured to be developer-friendly:
# - Permissive linting rules to ease adoption
# - Automatic code formatting
# - Gradual quality improvement over time
```

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
make precommit-run      # Run pre-commit hooks manually

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
| `make dev-setup` | Complete development environment setup + pre-commit hooks | ✅ |
| `make prefect-start` | **Start Prefect server + agent (recommended)** | ✅ |
| `make dev-start` | Start MLFlow + basic services | ✅ |
| `make streamlit-app` | **Start Streamlit web application** | ✅ |
| `make streamlit-dev` | **Start Streamlit in development mode** | ✅ |
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
| `make format` | Format code with Black and isort | ✅ |
| `make precommit-run` | Run pre-commit hooks on all files | ✅ |
| `make precommit-help` | Show all pre-commit commands | ✅ |
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

This project uses pre-commit hooks to maintain code quality. When you run `make dev-setup`, pre-commit hooks are automatically installed and will run before each commit to:

- **Format code**: Black for Python formatting, isort for import sorting
- **Quality checks**: Flake8 linting with developer-friendly rules
- **File validation**: YAML/JSON syntax, trailing whitespace removal
- **Notebook formatting**: Black formatting for Jupyter notebooks

The hooks are configured to be permissive initially to ease adoption. See `docs/PRE_COMMIT_GUIDE.md` for detailed information.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

### Core Documentation
- [Architecture Overview](architecture/README.md) - Detailed system architecture
- [Next Steps Guide](NEXT_STEPS.md) - Development roadmap and priorities
- [Development Roadmap](ROADMAP.md) - Complete project timeline
- [Configuration Guide](.env.template) - Environment setup

### Development Guides
- [Pre-commit Guide](docs/PRE_COMMIT_GUIDE.md) - Code quality and pre-commit hooks
- [Prefect Workflow Integration](docs/PREFECT_WORKFLOW_INTEGRATION.md) - Workflow orchestration setup
- [Streamlit Docker Setup](docs/STREAMLIT_DOCKER_SETUP.md) - Containerized web app deployment
- [Streamlit Quickstart](docs/STREAMLIT_QUICKSTART.md) - Web application development guide

### Configuration References
- [Configurable Data Paths](docs/CONFIGURABLE_DATA_PATHS.md) - Data storage configuration
- [Storage Configuration](docs/STORAGE_CONFIGURATION.md) - S3/MinIO storage setup
- [Model Drift Monitoring](docs/MODEL_DRIFT_MONITORING.md) - Model performance monitoring

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

**Pre-commit Hook Issues:**
```bash
# Check if pre-commit hooks are installed
ls -la .git/hooks/pre-commit

# Reinstall pre-commit hooks if needed
make precommit-install

# Run hooks manually to test
make precommit-run

# Skip hooks temporarily (emergency only)
git commit --no-verify -m "message"

# Update hook versions
make precommit-update

# Show all pre-commit options
make precommit-help
```

### Getting Help

1. **Check service status**: `make status`
2. **View logs**: `docker compose logs <service-name>`
3. **Restart services**: `docker compose restart`
4. **Clean restart**: `make clean && make dev-setup`
5. **Check issues**: [GitHub Issues](https://github.com/VRabinin/mlops-zoomcamp-project/issues)

---

Built with ❤️ for the MLOps community
