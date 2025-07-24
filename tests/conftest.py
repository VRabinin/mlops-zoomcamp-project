"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.config.config import Config


@pytest.fixture
def temp_config():
    """Fixture providing a Config instance with temporary directories."""
    temp_dir = tempfile.mkdtemp()
    config = Config()
    config.data_path.raw = f"{temp_dir}/raw"
    config.data_path.processed = f"{temp_dir}/processed"
    config.data_path.features = f"{temp_dir}/features"
    
    yield config
    
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_crm_dataframe():
    """Fixture providing a sample CRM DataFrame for testing."""
    import pandas as pd
    
    return pd.DataFrame({
        'opportunity_id': ['OPP001', 'OPP002', 'OPP003'],
        'sales_agent': ['Agent1', 'Agent2', 'Agent1'],
        'account': ['CLIENT001', 'CLIENT002', 'CLIENT003'],
        'product': ['Product A', 'Product B', 'Product A'],
        'deal_stage': ['Prospecting', 'Won', 'Lost'],
        'engage_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'close_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'close_value': [1000.0, 2000.0, 1500.0]
    })


@pytest.fixture
def s3_config():
    """Fixture providing a Config instance configured for S3/MinIO testing."""
    config = Config()
    config.storage.endpoint_url = "http://localhost:9000"
    config.storage.access_key = "test_access"
    config.storage.secret_key = "test_secret"
    config.storage.region = "us-east-1"
    config.storage.buckets = {
        "data_lake": "test-data-lake",
        "mlflow_artifacts": "test-mlflow",
        "model_artifacts": "test-models"
    }
    config.storage.s3_paths = {
        "raw": "raw-data",
        "processed": "processed-data",
        "features": "feature-data",
        "models": "model-data"
    }
    return config
