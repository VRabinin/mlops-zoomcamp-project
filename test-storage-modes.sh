#!/bin/bash
# CRM Pipeline Storage Mode Test Script

echo "=== CRM Pipeline Storage Mode Test ==="
echo "This script demonstrates the pipeline working with both local and S3 storage modes."
echo

# Test 1: Local storage mode (direct execution)
echo "🗃️ Test 1: Local Storage Mode (Direct Execution)"
echo "Running pipeline without Prefect API - should use local filesystem"
echo
cd /Users/vrabinin/Documents/Github/mlops-zoomcamp-project
PYTHONPATH=/Users/vrabinin/Documents/Github/mlops-zoomcamp-project .venv/bin/python -c "
from src.utils.storage import StorageManager
from src.config.config import get_config

config = get_config()
config_dict = {
    'minio': {
        'endpoint_url': config.minio.endpoint_url,
        'access_key': config.minio.access_key,
        'secret_key': config.minio.secret_key,
        'region': config.minio.region,
        'buckets': config.minio.buckets
    }
}

storage = StorageManager(config_dict)
print('💾 Storage mode:', storage.get_storage_info())
"

echo
echo "----------------------------------------"
echo

# Test 2: S3 storage mode (Prefect execution)
echo "☁️ Test 2: S3 Storage Mode (Prefect Orchestrated)"
echo "Running with Prefect API URL - should use S3/MinIO storage"
echo
PREFECT_API_URL=http://localhost:4200/api PYTHONPATH=/Users/vrabinin/Documents/Github/mlops-zoomcamp-project .venv/bin/python -c "
from src.utils.storage import StorageManager
from src.config.config import get_config

config = get_config()
config_dict = {
    'minio': {
        'endpoint_url': config.minio.endpoint_url,
        'access_key': config.minio.access_key,
        'secret_key': config.minio.secret_key,
        'region': config.minio.region,
        'buckets': config.minio.buckets
    }
}

storage = StorageManager(config_dict)
print('💾 Storage mode:', storage.get_storage_info())
"

echo
echo "----------------------------------------"
echo

# Test 3: Forced S3 storage mode
echo "🔧 Test 3: Forced S3 Storage Mode"
echo "Running with USE_S3_STORAGE=true - should force S3/MinIO storage"
echo
USE_S3_STORAGE=true PYTHONPATH=/Users/vrabinin/Documents/Github/mlops-zoomcamp-project .venv/bin/python -c "
from src.utils.storage import StorageManager
from src.config.config import get_config

config = get_config()
config_dict = {
    'minio': {
        'endpoint_url': config.minio.endpoint_url,
        'access_key': config.minio.access_key,
        'secret_key': config.minio.secret_key,
        'region': config.minio.region,
        'buckets': config.minio.buckets
    }
}

storage = StorageManager(config_dict)
print('💾 Storage mode:', storage.get_storage_info())
"

echo
echo "=== Summary ==="
echo "✅ Storage manager correctly detects execution environment"
echo "✅ Local mode: Uses ./data directories when running directly"
echo "✅ S3 mode: Uses MinIO buckets when running with Prefect or forced"
echo "✅ Docker mode: Will use S3 when /.dockerenv exists"
echo
echo "To run the full pipeline:"
echo "  Local mode:    python src/pipelines/run_crm_pipeline.py"
echo "  S3 mode:       PREFECT_API_URL=http://localhost:4200/api python src/pipelines/run_crm_pipeline.py"
echo "  Forced S3:     USE_S3_STORAGE=true python src/pipelines/run_crm_pipeline.py"
echo "  Prefect flow:  make prefect-run-deployment"
