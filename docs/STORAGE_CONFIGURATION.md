# Storage Configuration Guide

## Overview

The MLOps platform now supports fully configurable storage paths and buckets through environment variables. This allows for flexible deployment across different environments (development, staging, production) without code changes.

## Configuration Hierarchy

The system follows this configuration hierarchy (higher priority overrides lower):

1. **Environment Variables** (highest priority)
2. **YAML Configuration Files** (`config/development.yaml`)
3. **Default Values** (lowest priority)

## Environment Variables

### Bucket Names

These variables control the S3/MinIO bucket names used by different services:

```bash
# S3/MinIO Bucket Configuration  
MLFLOW_ARTIFACTS_BUCKET=mlflow-artifacts    # MLflow experiment artifacts
DATA_LAKE_BUCKET=data-lake                  # Raw, processed, and feature data
MODEL_ARTIFACTS_BUCKET=model-artifacts      # Trained models and model artifacts
```

### Data Paths Within Buckets

These variables control the directory structure within buckets:

```bash
# S3/MinIO Data Path Configuration (paths within buckets)
S3_RAW_DATA_PATH=raw                        # Raw data from sources (e.g., Kaggle)
S3_PROCESSED_DATA_PATH=processed            # Cleaned and validated data
S3_FEATURES_DATA_PATH=features              # Feature-engineered datasets
S3_MODELS_PATH=models                       # Model binaries and metadata
S3_EXPERIMENTS_PATH=experiments             # Experiment tracking data
S3_PREFECT_FLOWS_PATH=prefect-flows         # Prefect flow definitions and logs
```

### Local File System Paths

For local development (non-containerized):

```bash
# Local Data Configuration
RAW_DATA_PATH=data/raw                      # Local raw data directory
PROCESSED_DATA_PATH=data/processed          # Local processed data directory
FEATURE_STORE_PATH=data/features            # Local feature store directory
```

### Storage Connection Settings

```bash
# MinIO/S3 Connection Configuration
MINIO_ENDPOINT=http://localhost:9000        # MinIO endpoint URL
MINIO_ROOT_USER=minioadmin                  # MinIO access key
MINIO_ROOT_PASSWORD=minioadmin              # MinIO secret key
AWS_DEFAULT_REGION=us-east-1                # S3 region
```

## Storage Structure Examples

### Development Environment (Default)

```
MinIO Buckets:
├── mlflow-artifacts/           # MLflow experiments and models
│   ├── experiments/           
│   └── models/               
├── data-lake/                 # Primary data storage
│   ├── raw/                  # Raw datasets (Kaggle, APIs, etc.)
│   ├── processed/            # Cleaned and validated data
│   ├── features/             # Feature-engineered datasets
│   └── prefect-flows/        # Prefect flow artifacts
└── model-artifacts/          # Model registry and serving
    ├── models/              
    └── metadata/            
```

### Production Environment Example

```bash
# Production bucket names
export MLFLOW_ARTIFACTS_BUCKET=prod-mlflow-artifacts
export DATA_LAKE_BUCKET=prod-data-lake
export MODEL_ARTIFACTS_BUCKET=prod-model-registry

# Custom data organization
export S3_RAW_DATA_PATH=ingestion/raw
export S3_PROCESSED_DATA_PATH=pipeline/processed
export S3_FEATURES_DATA_PATH=ml/features
export S3_MODELS_PATH=registry/models
```

Resulting structure:
```
S3 Buckets:
├── prod-mlflow-artifacts/
├── prod-data-lake/
│   ├── ingestion/raw/
│   ├── pipeline/processed/
│   ├── ml/features/
│   └── prefect-flows/
└── prod-model-registry/
    └── registry/models/
```

## Usage Examples

### 1. Using Environment Variables

Create a `.env` file:
```bash
# Custom bucket names for staging
MLFLOW_ARTIFACTS_BUCKET=staging-mlflow
DATA_LAKE_BUCKET=staging-data
MODEL_ARTIFACTS_BUCKET=staging-models

# Custom data paths
S3_RAW_DATA_PATH=raw-data
S3_FEATURES_DATA_PATH=ml-features
```

### 2. YAML Configuration

Update `config/development.yaml`:
```yaml
storage:
  buckets:
    mlflow_artifacts: "custom-mlflow-bucket"
    data_lake: "custom-data-bucket"
    model_artifacts: "custom-model-bucket"
  data_paths:
    raw: "ingestion/raw"
    processed: "pipeline/clean"
    features: "ml/engineered"
```

### 3. Programmatic Access

```python
from src.config.config import get_config
from src.utils.storage import StorageManager

# Get configuration
config = get_config()
storage = StorageManager(config.__dict__)

# Save DataFrame with automatic path management
storage.save_dataframe_by_type(df, 'raw', 'sales_data.csv')
# → Saves to: s3://data-lake/raw/sales_data.csv

# Load DataFrame with automatic path management
df = storage.load_dataframe_by_type('features', 'crm_features.csv')
# → Loads from: s3://data-lake/features/crm_features.csv

# Use different data types
storage.save_dataframe_by_type(model_metrics, 'experiments', 'metrics.csv')
# → Saves to: s3://mlflow-artifacts/experiments/metrics.csv
```

## Docker Compose Integration

The system automatically configures Docker services using environment variables:

```yaml
# MLflow uses configurable bucket
environment:
  - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://${MLFLOW_ARTIFACTS_BUCKET:-mlflow-artifacts}/

# MinIO creates configurable buckets
environment:
  - MINIO_DEFAULT_BUCKETS=${MLFLOW_ARTIFACTS_BUCKET:-mlflow-artifacts},${DATA_LAKE_BUCKET:-data-lake},${MODEL_ARTIFACTS_BUCKET:-model-artifacts}
```

## Migration from Legacy Configuration

### Old Way (Hardcoded)
```python
# ❌ Old hardcoded approach
df.to_csv('data/raw/sales.csv')
s3_key = 'raw/sales.csv'
bucket = 'data-lake'
```

### New Way (Configurable)
```python
# ✅ New configurable approach
storage.save_dataframe_by_type(df, 'raw', 'sales.csv')
# Automatically handles local vs S3, bucket selection, and path structure
```

## Environment-Specific Configurations

### Development
- Uses default bucket names
- Local storage for direct Python execution
- S3 storage for Docker containers

### Staging
```bash
export MLFLOW_ARTIFACTS_BUCKET=staging-mlflow-artifacts
export DATA_LAKE_BUCKET=staging-data-lake
export MODEL_ARTIFACTS_BUCKET=staging-model-artifacts
```

### Production
```bash
export MLFLOW_ARTIFACTS_BUCKET=prod-mlflow-artifacts
export DATA_LAKE_BUCKET=prod-data-lake
export MODEL_ARTIFACTS_BUCKET=prod-model-artifacts
export S3_RAW_DATA_PATH=production/raw
export S3_FEATURES_DATA_PATH=production/features
```

## Best Practices

1. **Use Environment Variables for Deployment**: Set bucket names and paths via environment variables rather than hardcoding.

2. **Consistent Naming Convention**: Follow a clear naming pattern for buckets across environments:
   - `{env}-{service}-{purpose}` (e.g., `prod-mlflow-artifacts`)

3. **Logical Data Organization**: Group related data types in appropriate buckets:
   - Raw/processed/features → `data-lake`
   - Experiments/tracking → `mlflow-artifacts`
   - Models/registry → `model-artifacts`

4. **Use Typed Storage Methods**: Prefer `save_dataframe_by_type()` over manual path construction.

5. **Environment Isolation**: Use different bucket names for each environment to prevent cross-contamination.

## Troubleshooting

### Issue: Bucket Not Found
- Check bucket name environment variables
- Verify MinIO/S3 connection settings
- Ensure Docker services are running

### Issue: Path Not Found
- Verify data path environment variables
- Check if using correct data type parameter
- Ensure file exists in expected location

### Issue: Permission Denied
- Check MinIO/S3 credentials
- Verify bucket policies
- Ensure proper AWS/MinIO access rights

## Testing Configuration

Test your configuration:

```bash
# Test environment variables
echo $DATA_LAKE_BUCKET
echo $S3_RAW_DATA_PATH

# Test storage connectivity
make minio-status

# Test bucket creation
make docker-setup
```

This flexible configuration system enables seamless deployment across environments while maintaining data organization and access patterns.
