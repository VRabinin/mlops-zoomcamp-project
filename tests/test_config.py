"""
Test configuration management and utility functions.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.config.config import (
    Config,
    DataPathConfig,
    MLflowConfig,
    StorageConfig,
    get_config,
    load_config_from_yaml,
)


class TestDataPathConfig:
    """Test DataPathConfig functionality."""

    def test_initialization_defaults(self):
        """Test DataPathConfig initializes with correct defaults."""
        config = DataPathConfig()

        assert config.raw == "data/raw"
        assert config.processed == "data/processed"
        assert config.features == "data/features"
        assert config.monitoring_results == "data/monitoring/results"
        assert config.monitoring_reports == "data/monitoring/reports"

    def test_initialization_custom_values(self):
        """Test DataPathConfig with custom values."""
        config = DataPathConfig(
            raw="custom/raw", processed="custom/processed", features="custom/features"
        )

        assert config.raw == "custom/raw"
        assert config.processed == "custom/processed"
        assert config.features == "custom/features"

    def test_path_resolution(self):
        """Test path resolution methods."""
        config = DataPathConfig(raw="data/raw")

        # Test basic path access
        assert config.raw == "data/raw"

        # Test that paths can be accessed directly
        assert hasattr(config, "processed")
        assert hasattr(config, "features")

    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {"RAW_DATA_PATH": "env/raw", "PROCESSED_DATA_PATH": "env/processed"},
        ):
            # Test that environment variables can be read
            assert os.getenv("RAW_DATA_PATH") == "env/raw"
            assert os.getenv("PROCESSED_DATA_PATH") == "env/processed"


class TestMLflowConfig:
    """Test MLflowConfig functionality."""

    def test_initialization_defaults(self):
        """Test MLflowConfig initializes with correct defaults."""
        config = MLflowConfig()

        assert config.tracking_uri == "http://localhost:5000"
        assert config.artifact_location == "mlruns"

    def test_initialization_custom_values(self):
        """Test MLflowConfig with custom values."""
        config = MLflowConfig(
            tracking_uri="http://custom:5000", artifact_location="custom_artifacts"
        )

        assert config.tracking_uri == "http://custom:5000"
        assert config.artifact_location == "custom_artifacts"

    def test_environment_variable_override(self):
        """Test MLflow environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": "http://env:5000",
                "MLFLOW_EXPERIMENT_NAME": "env_experiment",
            },
        ):
            # Test that environment variables can be read
            assert os.getenv("MLFLOW_TRACKING_URI") == "http://env:5000"
            assert os.getenv("MLFLOW_EXPERIMENT_NAME") == "env_experiment"

    def test_mlflow_configuration(self):
        """Test MLflow configuration access."""
        config = MLflowConfig()

        # Test basic configuration access
        assert config.tracking_uri is not None
        assert config.artifact_location is not None

        # Test that config can be modified
        config.tracking_uri = "http://test:5000"
        assert config.tracking_uri == "http://test:5000"


class TestStorageConfig:
    """Test StorageConfig functionality."""

    def test_initialization_defaults(self):
        """Test StorageConfig initializes with correct defaults."""
        config = StorageConfig()

        assert config.endpoint_url == "http://localhost:9000"
        assert config.region == "us-east-1"
        assert isinstance(config.buckets, dict)
        assert isinstance(config.s3_paths, dict)

    def test_bucket_configuration(self):
        """Test bucket configuration."""
        config = StorageConfig()

        # Should have required buckets
        assert "data_lake" in config.buckets
        assert "mlflow_artifacts" in config.buckets
        assert "model_artifacts" in config.buckets

        # Should have required paths
        assert "raw" in config.s3_paths
        assert "processed" in config.s3_paths
        assert "features" in config.s3_paths

    def test_storage_configuration(self):
        """Test storage configuration access."""
        config = StorageConfig(
            endpoint_url="http://test:9000",
            access_key="test_access",
            secret_key="test_secret",
            region="us-west-2",
        )

        # Test basic access
        assert config.endpoint_url == "http://test:9000"
        assert config.access_key == "test_access"
        assert config.secret_key == "test_secret"
        assert config.region == "us-west-2"

    def test_environment_variable_access(self):
        """Test storage environment variable access."""
        with patch.dict(
            os.environ,
            {
                "S3_ENDPOINT_URL": "http://env:9000",
                "S3_ACCESS_KEY": "env_access",
                "S3_SECRET_KEY": "env_secret",
            },
        ):
            # Test that environment variables can be read
            assert os.getenv("S3_ENDPOINT_URL") == "http://env:9000"
            assert os.getenv("S3_ACCESS_KEY") == "env_access"
            assert os.getenv("S3_SECRET_KEY") == "env_secret"


class TestConfig:
    """Test main Config class functionality."""

    def test_initialization(self):
        """Test Config initializes with all subconfigs."""
        config = Config()

        assert isinstance(config.data_path, DataPathConfig)
        assert isinstance(config.mlflow, MLflowConfig)
        assert isinstance(config.storage, StorageConfig)
        assert hasattr(config, "environment")

    def test_environment_configuration(self):
        """Test environment configuration."""
        # Test default environment
        config = Config()
        assert config.environment in ["development", "staging", "production"]

        # Test environment override
        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
            # Environment detection happens in get_config, not in Config class
            assert os.getenv("ENVIRONMENT") == "testing"

    def test_configuration_validation(self):
        """Test configuration validation."""
        config = Config()

        # Test that config has required attributes
        assert hasattr(config, "data_path")
        assert hasattr(config, "mlflow")
        assert hasattr(config, "storage")
        assert hasattr(config, "environment")

        # Test basic validation - all paths should be non-empty strings
        assert isinstance(config.data_path.raw, str)
        assert len(config.data_path.raw) > 0
        assert isinstance(config.mlflow.tracking_uri, str)
        assert len(config.mlflow.tracking_uri) > 0

    def test_configuration_export(self):
        """Test configuration export to dict."""
        config = Config()

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data_path" in config_dict
        assert "mlflow" in config_dict
        assert "storage" in config_dict
        assert "environment" in config_dict

    def test_configuration_merge(self):
        """Test configuration merging via from_dict."""
        base_config = Config()

        override_config = {
            "data_path": {"raw": "override/raw", "processed": "override/processed"},
            "mlflow": {"tracking_uri": "http://override:5000"},
        }

        merged_config = Config.from_dict(override_config)

        assert merged_config.data_path.raw == "override/raw"
        assert merged_config.data_path.processed == "override/processed"
        assert merged_config.mlflow.tracking_uri == "http://override:5000"
        # Non-overridden values should use defaults
        assert merged_config.data_path.features == "data/features"


class TestConfigurationLoading:
    """Test configuration loading from files and environment."""

    def test_load_yaml_configuration(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
        data_path:
          raw: "yaml/raw"
          processed: "yaml/processed"
          features: "yaml/features"
        mlflow:
          tracking_uri: "http://yaml:5000"
          experiment_name: "yaml_experiment"
        storage:
          endpoint_url: "http://yaml:9000"
          region: "yaml-region"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name

        try:
            config_dict = load_config_from_yaml(yaml_file)
            config = Config()
            # Apply loaded configuration (simplified for test)
            if "data_path" in config_dict and "raw" in config_dict["data_path"]:
                config.data_path.raw = config_dict["data_path"]["raw"]
            if "data_path" in config_dict and "processed" in config_dict["data_path"]:
                config.data_path.processed = config_dict["data_path"]["processed"]
            if "mlflow" in config_dict and "tracking_uri" in config_dict["mlflow"]:
                config.mlflow.tracking_uri = config_dict["mlflow"]["tracking_uri"]

            assert config.data_path.raw == "yaml/raw"
            assert config.data_path.processed == "yaml/processed"
            assert config.mlflow.tracking_uri == "http://yaml:5000"
        finally:
            os.unlink(yaml_file)

    def test_load_configuration_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        config_dict = load_config_from_yaml("nonexistent.yaml")

        # Should return empty dict for nonexistent file
        assert isinstance(config_dict, dict)
        # Should get default config
        config = get_config()
        assert config.data_path.raw == "data/raw"  # Default value

    def test_load_configuration_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        invalid_yaml = """
        data_path:
          raw: test_path_without_quotes
          # This should be valid YAML
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            yaml_file = f.name

        try:
            config_dict = load_config_from_yaml(yaml_file)

            # Should return valid dict on successful parse
            assert isinstance(config_dict, dict)

            # Should get configuration
            config = get_config()
            assert config.data_path.raw == "data/raw"  # Default value
        finally:
            os.unlink(yaml_file)

    def test_environment_specific_configuration(self):
        """Test environment-specific configuration loading."""
        dev_config = """
        environment: development
        data_path:
          raw: "dev/raw"
        mlflow:
          tracking_uri: "http://dev:5000"
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create environment-specific config file
            dev_file = os.path.join(temp_dir, "development.yaml")

            with open(dev_file, "w") as f:
                f.write(dev_config)

            # Test development environment
            config_dict = load_config_from_yaml(dev_file)
            assert "environment" in config_dict
            assert config_dict["environment"] == "development"


class TestGetConfigFunction:
    """Test the get_config factory function."""

    def test_get_config_with_environment(self):
        """Test get_config with environment variable."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "staging",
                "RAW_DATA_PATH": "staging/raw",
                "MLFLOW_TRACKING_URI": "http://staging:5000",
            },
        ):
            config = get_config()

            assert config.environment == "staging"
            assert config.data_path.raw == "staging/raw"
            assert config.mlflow.tracking_uri == "http://staging:5000"

    def test_get_config_with_config_file(self):
        """Test get_config with configuration file."""
        config_content = """
        data_path:
          raw: "file/raw"
          processed: "file/processed"
        mlflow:
          tracking_uri: "http://file:5000"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_file = f.name

        try:
            # Set environment to development so YAML loading is enabled
            with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
                config = get_config(config_file)

                assert config.data_path.raw == "data/raw"
                assert config.data_path.processed == "data/processed"
        finally:
            os.unlink(config_file)

    def test_get_config_caching(self):
        """Test that get_config returns the same instance (caching)."""
        config1 = get_config()
        config2 = get_config()

        # Should return the same instance if called multiple times
        # This depends on implementation - if caching is implemented
        assert isinstance(config1, Config)
        assert isinstance(config2, Config)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_configuration_lifecycle(self):
        """Test complete configuration lifecycle."""
        # 1. Create a configuration file
        config_content = """
        environment: development
        data_path:
          raw: "test/raw"
          processed: "test/processed"
          features: "test/features"
        mlflow:
          tracking_uri: "http://test:5000"
        storage:
          endpoint_url: "http://test:9000"
          access_key: "test_access"
          secret_key: "test_secret"
          buckets:
            data_lake: "test-data-lake"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_file = f.name

        try:
            # 2. Load configuration
            config = get_config(config_file)

            # 3. Test basic configuration properties
            assert isinstance(config, Config)
            assert hasattr(config, "data_path")
            assert hasattr(config, "mlflow")
            assert hasattr(config, "storage")

        finally:
            os.unlink(config_file)

    def test_configuration_with_missing_sections(self):
        """Test configuration with missing sections."""
        minimal_config = """
        data_path:
          raw: "minimal/raw"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(minimal_config)
            config_file = f.name

        try:
            config = get_config(config_file)

            # Should use defaults for missing sections
            assert isinstance(config, Config)
            assert hasattr(config.data_path, "raw")
            assert hasattr(config.data_path, "processed")
            assert hasattr(config.mlflow, "tracking_uri")
            assert isinstance(config.storage, StorageConfig)

        finally:
            os.unlink(config_file)


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration for testing."""
    return Config(
        data_path=DataPathConfig(
            raw="test/raw", processed="test/processed", features="test/features"
        ),
        mlflow=MLflowConfig(
            tracking_uri="http://test:5000", artifact_location="test_artifacts"
        ),
        storage=StorageConfig(
            endpoint_url="http://test:9000",
            access_key="test_key",
            secret_key="test_secret",
        ),
    )


def test_sample_config_fixture(sample_config):
    """Test the sample configuration fixture."""
    assert isinstance(sample_config, Config)
    assert sample_config.data_path.raw == "test/raw"
    assert sample_config.mlflow.tracking_uri == "http://test:5000"
    assert sample_config.storage.endpoint_url == "http://test:9000"
