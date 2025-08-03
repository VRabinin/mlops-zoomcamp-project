# Configurable Data Paths Implementation Guide

## 🎯 Overview

The MLOps platform now supports fully configurable data paths for both local development and cloud production environments. This dual-path system allows seamless deployment across different environments without code changes.

## 🏗️ Configuration Architecture

### Two-Tier Path System

1. **Local Paths** - For direct Python execution (development, testing)
2. **S3 Paths** - For containerized/cloud execution (staging, production)

### Configuration Sources (Priority Order)

1. **Environment Variables** (highest priority)
2. **YAML Configuration Files** (medium priority)
3. **Default Values** (lowest priority)

## 📁 Data Path Categories

### Local Filesystem Paths

Used when running Python directly (non-containerized):

```yaml
data:
  local_paths:
    raw_data_path: "data/raw"              # Raw data downloads
    processed_data_path: "data/processed"  # Cleaned datasets
    feature_store_path: "data/features"    # Engineered features
```

### S3/MinIO Paths

Used within containerized environments (Docker, Kubernetes):

```yaml
storage:
  s3_paths:
    raw: "raw"                    # Raw data in S3 bucket
    processed: "processed"        # Processed data in S3 bucket
    features: "features"          # Feature store in S3 bucket
    models: "models"              # Model artifacts
    experiments: "experiments"    # MLflow experiment data
    prefect_flows: "prefect-flows" # Prefect workflow definitions
```

## 🌍 Environment Variables

### Local Path Overrides

```bash
# Local filesystem paths (for direct execution)
RAW_DATA_PATH=custom/local/raw
PROCESSED_DATA_PATH=custom/local/processed
FEATURE_STORE_PATH=custom/local/features
```

### S3 Path Overrides

```bash
# S3/MinIO paths within buckets
S3_RAW_DATA_PATH=ingestion/raw
S3_PROCESSED_DATA_PATH=pipeline/processed
S3_FEATURES_DATA_PATH=ml/features
S3_MODELS_PATH=registry/models
S3_EXPERIMENTS_PATH=tracking/experiments
S3_PREFECT_FLOWS_PATH=orchestration/flows
```

### Bucket Configuration

```bash
# S3/MinIO bucket names
MLFLOW_ARTIFACTS_BUCKET=prod-mlflow-artifacts
DATA_LAKE_BUCKET=prod-data-lake
MODEL_ARTIFACTS_BUCKET=prod-model-registry
```

## 📋 Configuration Examples

### Development (development.yaml)

```yaml
data:
  local_paths:
    raw_data_path: "data/raw"
    processed_data_path: "data/processed"
    feature_store_path: "data/features"

storage:
  buckets:
    data_lake: "data-lake"
    mlflow_artifacts: "mlflow-artifacts"
    model_artifacts: "model-artifacts"
  s3_paths:
    raw: "raw"
    processed: "processed"
    features: "features"
```

**Results in:**
- Local: `data/raw/sales.csv`
- S3: `s3://data-lake/raw/sales.csv`

### Staging (staging.yaml)

```yaml
data:
  local_paths:
    raw_data_path: "staging/data/raw"
    processed_data_path: "staging/data/processed"
    feature_store_path: "staging/data/features"

storage:
  buckets:
    data_lake: "staging-data-lake"
    mlflow_artifacts: "staging-mlflow-artifacts"
  s3_paths:
    raw: "staging/raw"
    processed: "staging/processed"
    features: "staging/features"
```

**Results in:**
- Local: `staging/data/raw/sales.csv`
- S3: `s3://staging-data-lake/staging/raw/sales.csv`

### Production (production.yaml)

```yaml
data:
  local_paths:
    raw_data_path: "/opt/mlops/data/raw"
    processed_data_path: "/opt/mlops/data/processed"
    feature_store_path: "/opt/mlops/data/features"

storage:
  buckets:
    data_lake: "prod-data-lake"
    mlflow_artifacts: "prod-mlflow-artifacts"
  s3_paths:
    raw: "ingestion/raw"
    processed: "pipeline/processed"
    features: "ml/features"
```

**Results in:**
- Local: `/opt/mlops/data/raw/sales.csv`
- S3: `s3://prod-data-lake/ingestion/raw/sales.csv`

## 🚀 Usage Patterns

### Environment-Specific Deployment

**Development:**
```bash
# Use defaults - no environment variables needed
python src/pipelines/run_crm_pipeline.py
```

**Staging:**
```bash
export RAW_DATA_PATH=staging/raw
export S3_RAW_DATA_PATH=staging/data/raw
export DATA_LAKE_BUCKET=staging-data-lake
python src/pipelines/run_crm_pipeline.py
```

**Production:**
```bash
export RAW_DATA_PATH=/opt/mlops/data/raw
export S3_RAW_DATA_PATH=production/ingestion/raw
export DATA_LAKE_BUCKET=prod-data-lake
export S3_FEATURES_DATA_PATH=production/ml/features
python src/pipelines/run_crm_pipeline.py
```

### Programmatic Usage

```python
from src.config.config import get_config
from src.utils.storage import StorageManager

# Load configuration (respects environment variables)
config = get_config()

# For local development
local_raw_path = config.data.raw_data_path  # "data/raw"
local_features_path = config.data.feature_store_path  # "data/features"

# For S3/containerized deployment
storage = StorageManager({'storage': config.storage.__dict__})

# Type-aware storage operations
storage.save_dataframe_by_type(df, 'raw', 'sales.csv')
# → Local: saves to data/raw/sales.csv
# → S3: saves to s3://data-lake/raw/sales.csv

storage.save_dataframe_by_type(features, 'features', 'crm_features.csv')
# → Local: saves to data/features/crm_features.csv
# → S3: saves to s3://data-lake/features/crm_features.csv
```

## 🔄 Migration Guide

### From Hardcoded Paths

**Before:**
```python
# ❌ Hardcoded paths
df.to_csv('data/raw/sales.csv')
features.to_csv('data/features/crm_features.csv')
```

**After:**
```python
# ✅ Configurable paths
from src.config.config import get_config
from src.utils.storage import StorageManager

config = get_config()
storage = StorageManager({'storage': config.storage.__dict__})

storage.save_dataframe_by_type(df, 'raw', 'sales.csv')
storage.save_dataframe_by_type(features, 'features', 'crm_features.csv')
```

### Environment Variable Migration

**Development → Staging:**
```bash
# Development (defaults)
# No environment variables needed

# Staging
export RAW_DATA_PATH=staging/data/raw
export S3_RAW_DATA_PATH=staging/raw
export DATA_LAKE_BUCKET=staging-data-lake
```

**Staging → Production:**
```bash
# Production
export RAW_DATA_PATH=/opt/mlops/data/raw
export S3_RAW_DATA_PATH=production/ingestion/raw
export DATA_LAKE_BUCKET=prod-data-lake
export S3_FEATURES_DATA_PATH=production/ml/features
```

## 🎛️ Advanced Configuration

### Custom Data Organization

```bash
# Custom enterprise structure
export S3_RAW_DATA_PATH=sources/external/kaggle
export S3_PROCESSED_DATA_PATH=pipeline/etl/cleaned
export S3_FEATURES_DATA_PATH=ml/feature-store/engineered
export S3_MODELS_PATH=ml/model-registry/artifacts
export S3_EXPERIMENTS_PATH=ml/tracking/experiments
```

### Multi-Tenant Setup

```bash
# Tenant-specific paths
export S3_RAW_DATA_PATH=tenant-${TENANT_ID}/raw
export S3_FEATURES_DATA_PATH=tenant-${TENANT_ID}/features
export DATA_LAKE_BUCKET=${TENANT_ID}-data-lake
```

### Compliance Separation

```bash
# Separate sensitive data
export DATA_LAKE_BUCKET=company-general-data
export S3_RAW_DATA_PATH=non-sensitive/raw
export SENSITIVE_DATA_BUCKET=company-sensitive-data
export S3_SENSITIVE_PATH=pii/processed
```

## 🧪 Testing & Validation

### Configuration Testing

```python
from src.config.config import get_config

# Test with custom environment
import os
os.environ['RAW_DATA_PATH'] = 'test/raw'
os.environ['S3_RAW_DATA_PATH'] = 'test/s3/raw'

config = get_config()
assert config.data.raw_data_path == 'test/raw'
assert config.storage.s3_paths['raw'] == 'test/s3/raw'
```

### Storage Manager Testing

```python
from src.utils.storage import StorageManager

storage = StorageManager({'storage': config.storage.__dict__})

# Test path generation
assert storage.get_s3_path('raw', 'test.csv') == 'test/s3/raw/test.csv'
assert storage.get_bucket_for_data_type('raw') == 'data-lake'
```

## 📊 Data Flow Examples

### Development Flow

```
Kaggle API → data/raw/sales.csv → data/processed/clean.csv → data/features/engineered.csv
```

### Production Flow

```
Kaggle API → s3://prod-data-lake/ingestion/raw/sales.csv
           → s3://prod-data-lake/pipeline/processed/clean.csv
           → s3://prod-data-lake/ml/features/engineered.csv
```

### Multi-Environment Flow

```
Development:  data/raw → data/features
Staging:      s3://staging-data-lake/staging/raw → s3://staging-data-lake/staging/features
Production:   s3://prod-data-lake/ingestion/raw → s3://prod-data-lake/ml/features
```

## ✅ Benefits

1. **Environment Flexibility**: Same code works across dev/staging/prod
2. **Data Organization**: Logical separation by environment and data type
3. **Security Compliance**: Separate buckets and paths for different data sensitivity
4. **Multi-Tenancy**: Support for tenant-specific data isolation
5. **Zero Code Changes**: Environment changes via configuration only
6. **Backward Compatibility**: Existing code continues to work
7. **Enterprise Ready**: Supports complex organizational data structures

## 🔧 Troubleshooting

### Path Resolution Issues

```python
# Debug path resolution
from src.config.config import get_config
config = get_config()

print("Local paths:", {
    'raw': config.data.raw_data_path,
    'processed': config.data.processed_data_path,
    'features': config.data.feature_store_path
})

print("S3 paths:", config.storage.s3_paths)
print("Buckets:", config.storage.buckets)
```

### Environment Variable Debugging

```bash
# Check environment variables
echo "RAW_DATA_PATH: $RAW_DATA_PATH"
echo "S3_RAW_DATA_PATH: $S3_RAW_DATA_PATH"
echo "DATA_LAKE_BUCKET: $DATA_LAKE_BUCKET"
```

This comprehensive configuration system provides enterprise-grade flexibility while maintaining simplicity for development workflows.
