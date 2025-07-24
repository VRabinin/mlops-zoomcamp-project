"""
Test StorageManager functionality comprehensively.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import shutil

from src.utils.storage import StorageManager, create_storage_manager
from src.config.config import Config, get_config


class TestStorageManagerLocal:
    """Test StorageManager in local storage mode."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.data_path.raw = f"{self.temp_dir}/raw"
        self.config.data_path.processed = f"{self.temp_dir}/processed"
        self.config.data_path.features = f"{self.temp_dir}/features"
    
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_local_mode(self):
        """Test StorageManager initializes correctly in local mode."""
        storage = StorageManager(self.config)
        
        assert not storage.use_s3
        assert storage.config.data_path.raw == f"{self.temp_dir}/raw"
        assert storage.config.data_path.processed == f"{self.temp_dir}/processed"
        assert storage.config.data_path.features == f"{self.temp_dir}/features"
    
    def test_resolve_path_without_filename(self):
        """Test path resolution without filename."""
        storage = StorageManager(self.config)
        
        raw_path = storage.resolve_path('raw')
        assert str(raw_path) == f"{self.temp_dir}/raw"
        assert isinstance(raw_path, Path)
    
    def test_resolve_path_with_filename(self):
        """Test path resolution with filename."""
        storage = StorageManager(self.config)
        
        file_path = storage.resolve_path('processed', 'data.csv')
        expected_path = f"{self.temp_dir}/processed/data.csv"
        assert str(file_path) == expected_path
        assert isinstance(file_path, Path)
    
    def test_ensure_path_exists(self):
        """Test directory creation."""
        storage = StorageManager(self.config)
        
        # Directory shouldn't exist initially
        raw_dir = Path(f"{self.temp_dir}/raw")
        assert not raw_dir.exists()
        
        # ensure_path_exists should create it
        result_path = storage.ensure_path_exists('raw')
        assert raw_dir.exists()
        assert raw_dir.is_dir()
        assert str(result_path) == str(raw_dir)
    
    def test_get_full_path(self):
        """Test get_full_path method."""
        storage = StorageManager(self.config)
        
        full_path = storage.get_full_path('features', 'test.csv')
        expected_path = f"{self.temp_dir}/features/test.csv"
        
        assert full_path.endswith(expected_path)
        assert os.path.isabs(full_path)
        
        # Directory should be created
        features_dir = Path(f"{self.temp_dir}/features")
        assert features_dir.exists()
    
    def test_save_and_load_dataframe(self):
        """Test saving and loading DataFrames using smart methods."""
        storage = StorageManager(self.config)
        
        # Create test data
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.7]
        })
        
        # Save data
        saved_path = storage.save_dataframe(test_df, 'processed', 'test_data.csv')
        assert 'processed/test_data.csv' in saved_path
        
        # Verify file exists
        expected_file = Path(f"{self.temp_dir}/processed/test_data.csv")
        assert expected_file.exists()
        
        # Load data back
        loaded_df = storage.load_dataframe('processed', 'test_data.csv')
        
        # Verify data integrity
        pd.testing.assert_frame_equal(test_df, loaded_df)
    
    def test_file_exists(self):
        """Test file existence checking."""
        storage = StorageManager(self.config)
        
        # File doesn't exist initially
        assert not storage.file_exists('raw', 'nonexistent.csv')
        
        # Create a file
        test_df = pd.DataFrame({'col': [1, 2, 3]})
        storage.save_dataframe(test_df, 'raw', 'exists.csv')
        
        # Now it should exist
        assert storage.file_exists('raw', 'exists.csv')
    
    def test_list_files(self):
        """Test file listing with patterns."""
        storage = StorageManager(self.config)
        
        # Create some test files
        test_df = pd.DataFrame({'col': [1, 2, 3]})
        storage.save_dataframe(test_df, 'features', 'data1.csv')
        storage.save_dataframe(test_df, 'features', 'data2.csv')
        
        # Create a non-CSV file
        features_dir = Path(f"{self.temp_dir}/features")
        (features_dir / 'readme.txt').write_text('test content')
        
        # List all files
        all_files = storage.list_files('features', '*')
        assert len(all_files) == 3
        
        # List only CSV files
        csv_files = storage.list_files('features', '*.csv')
        assert len(csv_files) == 2
        assert all(f.endswith('.csv') for f in csv_files)
    
    def test_get_working_directory(self):
        """Test getting working directory."""
        storage = StorageManager(self.config)
        
        work_dir = storage.get_working_directory('raw')
        expected_dir = Path(f"{self.temp_dir}/raw")
        
        assert str(work_dir) == str(expected_dir)
        assert expected_dir.exists()
        assert expected_dir.is_dir()


class TestStorageManagerS3:
    """Test StorageManager in S3 mode."""
    
    def setup_method(self):
        """Set up test environment for S3 tests."""
        self.config = Config()
        self.config.storage.endpoint_url = "http://localhost:9000"
        self.config.storage.access_key = "test_access"
        self.config.storage.secret_key = "test_secret"
        self.config.storage.region = "us-east-1"
        self.config.storage.buckets = {
            "data_lake": "test-data-lake",
            "mlflow_artifacts": "test-mlflow",
            "model_artifacts": "test-models"
        }
        self.config.storage.s3_paths = {
            "raw": "raw-data",
            "processed": "processed-data", 
            "features": "feature-data",
            "models": "model-data"
        }
    
    @patch.dict(os.environ, {'USE_S3_STORAGE': 'true'})
    @patch('src.utils.storage.boto3.client')
    def test_initialization_s3_mode(self, mock_boto_client):
        """Test StorageManager initializes correctly in S3 mode."""
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        
        storage = StorageManager(self.config)
        
        assert storage.use_s3
        assert storage.config.storage.buckets["data_lake"] == "test-data-lake"
        assert storage.config.storage.s3_paths['raw'] == "raw-data"
        assert storage.config.storage.s3_paths['processed'] == "processed-data"
    
    @patch.dict(os.environ, {'USE_S3_STORAGE': 'true'})
    @patch('src.utils.storage.boto3.client')
    def test_resolve_path_s3(self, mock_boto_client):
        """Test path resolution in S3 mode."""
        mock_boto_client.return_value = MagicMock()
        storage = StorageManager(self.config)
        
        # Test without filename
        path = storage.resolve_path('raw')
        assert path == "raw-data"
        
        # Test with filename
        file_path = storage.resolve_path('processed', 'data.csv')
        assert file_path == "processed-data/data.csv"
    
    @patch.dict(os.environ, {'USE_S3_STORAGE': 'true'})
    @patch('src.utils.storage.boto3.client')
    def test_get_full_path_s3(self, mock_boto_client):
        """Test get_full_path in S3 mode."""
        mock_boto_client.return_value = MagicMock()
        storage = StorageManager(self.config)
        
        full_path = storage.get_full_path('features', 'test.csv')
        expected = "s3://test-data-lake/feature-data/test.csv"
        assert full_path == expected
    
    @patch.dict(os.environ, {'USE_S3_STORAGE': 'true'})
    @patch('src.utils.storage.boto3.client')
    def test_bucket_selection_by_data_type(self, mock_boto_client):
        """Test bucket selection for different data types."""
        mock_boto_client.return_value = MagicMock()
        storage = StorageManager(self.config)
        
        assert storage.get_bucket_for_data_type('raw') == "test-data-lake"
        assert storage.get_bucket_for_data_type('processed') == "test-data-lake"
        assert storage.get_bucket_for_data_type('features') == "test-data-lake"
        assert storage.get_bucket_for_data_type('models') == "test-models"
        assert storage.get_bucket_for_data_type('experiments') == "test-mlflow"


class TestStorageManagerConfiguration:
    """Test various configuration scenarios."""
    
    def test_backward_compatibility_direct_paths(self):
        """Test backward compatibility with direct path configuration."""
        config = Config()
        config.data_path.raw = "legacy/raw"
        config.data_path.processed = "legacy/processed"
        config.data_path.features = "legacy/features"
        
        storage = StorageManager(config)
        
        assert storage.config.data_path.raw == "legacy/raw"
        assert storage.config.data_path.processed == "legacy/processed"
        assert storage.config.data_path.features == "legacy/features"
    
    @patch.dict(os.environ, {
        'RAW_DATA_PATH': 'env/raw',
        'PROCESSED_DATA_PATH': 'env/processed',
        'FEATURE_STORE_PATH': 'env/features'
    })
    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        config = get_config()  # This will apply environment overrides
        
        storage = StorageManager(config)
        
        # Environment variables should override config
        assert storage.config.data_path.raw == "env/raw"
        assert storage.config.data_path.processed == "env/processed"
        assert storage.config.data_path.features == "env/features"
    
    def test_default_fallbacks(self):
        """Test default value fallbacks."""
        config = Config()  # Use defaults
        
        storage = StorageManager(config)
        
        # Should use defaults
        assert storage.config.data_path.raw == "data/raw"
        assert storage.config.data_path.processed == "data/processed"
        assert storage.config.data_path.features == "data/features"
    
    def test_invalid_data_type_error(self):
        """Test error handling for invalid data types."""
        config = Config()
        config.data_path.raw = "test/raw"
        
        storage = StorageManager(config)
        
        with pytest.raises(ValueError, match="Unknown data type"):
            storage.resolve_path('invalid_type')


class TestFactoryFunction:
    """Test the factory function."""
    
    def test_create_storage_manager(self):
        """Test the create_storage_manager factory function."""
        config = Config()
        config.data_path.raw = "test/raw"
        
        storage = create_storage_manager(config)
        
        assert isinstance(storage, StorageManager)
        assert storage.config.data_path.raw == "test/raw"


class TestStorageManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_filename_handling(self):
        """Test handling of empty filenames."""
        config = Config()
        config.data_path.raw = "test/raw"
        
        storage = StorageManager(config)
        
        # Should work with None
        path = storage.resolve_path('raw', None)
        assert str(path) == "test/raw"
        
        # Should work with empty string
        path = storage.resolve_path('raw', '')
        assert str(path) == "test/raw"
    
    def test_path_normalization(self):
        """Test path normalization and cleaning."""
        config = Config()
        config.data_path.raw = "test//raw///"
        
        storage = StorageManager(config)
        
        path = storage.resolve_path('raw', 'file.csv')
        # Path should be normalized
        assert '///' not in str(path)
        assert str(path).endswith('test/raw/file.csv')
