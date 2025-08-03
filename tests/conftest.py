"""
Pytest configuration and shared fixtures.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from src.config.config import Config, DataPathConfig, MLflowConfig, StorageConfig

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "prefect: mark test as prefect-related")
    config.addinivalue_line("markers", "storage: mark test as storage-related")
    config.addinivalue_line("markers", "ml: mark test as ML model-related")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test file names."""
    for item in items:
        # Add markers based on test file location
        if "test_prefect" in str(item.fspath):
            item.add_marker(pytest.mark.prefect)
        if "test_storage" in str(item.fspath):
            item.add_marker(pytest.mark.storage)
        if "test_model" in str(item.fspath):
            item.add_marker(pytest.mark.ml)
        if "integration" in str(item.fspath).lower() or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        if item.name.startswith("test_end_to_end") or "integration" in item.name:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration for all tests."""
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="mlops_test_")

    # Create test configuration
    config = Config(
        data_path=DataPathConfig(
            raw=f"{temp_dir}/raw",
            processed=f"{temp_dir}/processed",
            features=f"{temp_dir}/features",
            models=f"{temp_dir}/models",
            logs=f"{temp_dir}/logs",
            reports=f"{temp_dir}/reports",
        ),
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5005",
            experiment_name="test_experiment",
            model_name="test_model",
        ),
        storage=StorageConfig(
            endpoint_url="http://localhost:9000",
            access_key="test_access",
            secret_key="test_secret",
            region="us-east-1",
            buckets={
                "data_lake": "test-data-lake",
                "mlflow_artifacts": "test-mlflow",
                "model_artifacts": "test-models",
            },
        ),
    )

    yield config

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Disable S3 storage by default for tests
    monkeypatch.setenv("USE_S3_STORAGE", "false")
    # Set test environment
    monkeypatch.setenv("ENVIRONMENT", "test")
    # Disable MLflow UI auto-launch
    monkeypatch.setenv("MLFLOW_DISABLE_ENV_CREATION", "true")


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    from unittest.mock import MagicMock, patch

    import mlflow

    with patch.object(mlflow, "start_run") as mock_start_run:
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)

        with patch.object(mlflow, "log_metric") as mock_log_metric:
            with patch.object(mlflow, "log_param") as mock_log_param:
                with patch.object(mlflow.sklearn, "log_model") as mock_log_model:
                    yield {
                        "start_run": mock_start_run,
                        "log_metric": mock_log_metric,
                        "log_param": mock_log_param,
                        "log_model": mock_log_model,
                    }


@pytest.fixture
def mock_prefect_api():
    """Mock Prefect API for testing."""
    from unittest.mock import MagicMock, patch

    with patch("requests.Session") as mock_session:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        mock_session.return_value.get.return_value = mock_response
        mock_session.return_value.post.return_value = mock_response

        yield mock_session


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
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_samples = 20

    return pd.DataFrame(
        {
            "opportunity_id": [f"OPP{i:03d}" for i in range(1, n_samples + 1)],
            "sales_agent": np.random.choice(
                ["Agent1", "Agent2", "Agent3", "Agent4"], n_samples
            ),
            "account": [f"CLIENT{i:03d}" for i in np.random.randint(1, 10, n_samples)],
            "product": np.random.choice(
                ["Product A", "Product B", "Product C"], n_samples
            ),
            "deal_stage": np.random.choice(
                [
                    "Prospecting",
                    "Qualification",
                    "Proposal/Price Quote",
                    "Negotiation/Review",
                    "Closed Won",
                    "Closed Lost",
                ],
                n_samples,
            ),
            "engage_date": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "close_date": pd.date_range("2024-01-15", periods=n_samples, freq="D"),
            "close_value": np.random.uniform(500, 5000, n_samples),
        }
    )


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
        "model_artifacts": "test-models",
    }
    config.storage.s3_paths = {
        "raw": "raw-data",
        "processed": "processed-data",
        "features": "feature-data",
        "models": "model-data",
    }
    return config


# Custom test markers for pytest-html reports
pytest_html_report_title = "MLOps CRM Win Probability - Test Report"
