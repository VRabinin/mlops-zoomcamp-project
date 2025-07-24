"""
Test configuration and data ingestion functionality.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.config.config import Config, DataPathConfig, get_config
from src.data.ingestion.crm_ingestion import CRMDataIngestion
from src.data.schemas.crm_schema import CRMDataSchema
from src.utils.storage import StorageManager


class TestConfig:
    """Test configuration management."""
    
    def test_data_path_config_defaults(self):
        """Test DataPathConfig default values."""
        config = DataPathConfig()
        assert config.raw == "data/raw"
        assert config.processed == "data/processed"
    
    def test_get_config_returns_config_instance(self):
        """Test that get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)
        assert hasattr(config, 'data_path')
        assert hasattr(config, 'mlflow')


class TestCRMDataSchema:
    """Test CRM data schema definition."""
    
    def test_schema_initialization(self):
        """Test schema initializes with expected attributes."""
        schema = CRMDataSchema()
        assert isinstance(schema.required_columns, list)
        assert isinstance(schema.column_types, dict)
        assert len(schema.required_columns) > 0
    
    def test_target_column(self):
        """Test target column identification."""
        schema = CRMDataSchema()
        target = schema.get_target_column()
        assert target == 'deal_stage'
    
    def test_feature_columns(self):
        """Test feature column extraction."""
        schema = CRMDataSchema()
        features = schema.get_feature_columns()
        target = schema.get_target_column()
        assert target not in features
        assert len(features) > 0


class TestStorageManager:
    """Test StorageManager functionality."""
    
    def test_local_storage_initialization(self):
        """Test StorageManager initializes correctly for local storage."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        config.data_path.features = "test/features"
        
        storage = StorageManager(config)
        
        assert not storage.use_s3  # Should detect local environment
        assert storage.config.data_path.raw == "test/raw"
        assert storage.config.data_path.processed == "test/processed"
        assert storage.config.data_path.features == "test/features"
    
    def test_path_resolution_local(self):
        """Test path resolution for local storage."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        storage = StorageManager(config)
        
        # Test path resolution without filename
        raw_path = storage.resolve_path('raw')
        assert str(raw_path) == "test/raw"
        
        # Test path resolution with filename
        file_path = storage.resolve_path('raw', 'data.csv')
        assert str(file_path).endswith('test/raw/data.csv')
    
    def test_get_full_path(self):
        """Test get_full_path method."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        storage = StorageManager(config)
        
        full_path = storage.get_full_path('raw', 'test.csv')
        assert full_path.endswith('test/raw/test.csv')
        assert os.path.isabs(full_path)  # Should be absolute path
    
    @patch.dict(os.environ, {'USE_S3_STORAGE': 'true'})
    def test_s3_storage_detection(self):
        """Test S3 storage detection from environment variable."""
        config = Config()
        config.storage.endpoint_url = "http://localhost:9000"
        config.storage.access_key = "test"
        config.storage.secret_key = "test"
        config.storage.buckets = {"data_lake": "test-bucket"}
        config.storage.s3_paths = {"raw": "raw", "processed": "processed"}
        
        with patch('src.utils.storage.boto3.client'):
            storage = StorageManager(config)
            assert storage.use_s3
    
    def test_working_directory(self):
        """Test get_working_directory method."""
        config = Config()
        config.data_path.raw = "test/raw"
        
        storage = StorageManager(config)
        
        # Should create directory and return path
        work_dir = storage.get_working_directory('raw')
        assert str(work_dir) == "test/raw"


class TestCRMDataIngestion:
    """Test CRM data ingestion functionality."""
    
    def test_initialization(self):
        """Test CRMDataIngestion initializes correctly."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        ingestion = CRMDataIngestion(config)
        
        # Test that ingestion has config and storage manager
        assert ingestion.config == config
        assert hasattr(ingestion, 'storage')
        assert isinstance(ingestion.storage, StorageManager)
        
        # Test that storage manager has correct paths configured
        assert ingestion.storage.config.data_path.raw == "test/raw"
        assert ingestion.storage.config.data_path.processed == "test/processed"
    
    def test_load_data_with_sample_dataframe(self):
        """Test loading data with a sample DataFrame."""
        # Create sample data
        sample_data = pd.DataFrame({
            'opportunity_id': ['OPP001', 'OPP002'],
            'sales_agent': ['Agent1', 'Agent2'],
            'close_value': [1000.0, 2000.0],
            'deal_stage': ['Prospecting', 'Closed Won']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            file_path = Path(f.name)
        
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        try:
            ingestion = CRMDataIngestion(config)
            
            # Mock the smart storage loading to return our test data
            with patch.object(ingestion.storage, 'load_dataframe', return_value=sample_data):
                df = ingestion.load_data(file_path)
                assert len(df) == 2
                assert 'opportunity_id' in df.columns
                assert df['close_value'].sum() == 3000.0
        finally:
            file_path.unlink()  # Clean up
    
    def test_save_processed_data(self):
        """Test saving processed data using smart storage."""
        sample_data = pd.DataFrame({
            'opportunity_id': ['OPP001', 'OPP002'],
            'sales_agent': ['Agent1', 'Agent2'],
            'close_value': [1000.0, 2000.0]
        })
        
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        ingestion = CRMDataIngestion(config)
        
        # Mock the smart storage saving
        expected_path = "/test/path/processed_data.csv"
        with patch.object(ingestion.storage, 'save_dataframe', return_value=expected_path) as mock_save:
            result_path = ingestion.save_processed_data(sample_data, "test_file.csv")
            
            # Verify save_dataframe was called with correct parameters
            mock_save.assert_called_once_with(sample_data, 'processed', 'test_file.csv')
            assert result_path == expected_path
    
    def test_find_csv_files(self):
        """Test finding CSV files using smart storage."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        ingestion = CRMDataIngestion(config)
        
        # Mock the smart storage file listing
        mock_files = ['sales_pipeline.csv', 'accounts.csv', 'data_dictionary.csv']
        with patch.object(ingestion.storage, 'list_files', return_value=mock_files):
            csv_files = ingestion.find_csv_files()
            
            # Should return Path objects
            assert len(csv_files) == 3
            assert all(isinstance(f, Path) for f in csv_files)
            assert csv_files[0].name == 'sales_pipeline.csv'
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create sample data with issues
        sample_data = pd.DataFrame({
            'Opportunity ID': ['OPP001', 'OPP002', 'OPP001'],  # Duplicate
            'Sales Agent ': ['Agent1', 'Agent2', 'Agent1'],  # Space in column name
            'Close Value': [1000.0, 2000.0, 1000.0]
        })
        
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        ingestion = CRMDataIngestion(config)
        
        cleaned_df = ingestion.clean_data(sample_data)
        
        # Check column names are cleaned
        assert 'opportunity_id' in cleaned_df.columns
        assert 'sales_agent' in cleaned_df.columns
        
        # Check duplicates are removed
        assert len(cleaned_df) == 2
    
    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        ingestion = CRMDataIngestion(config)
        
        empty_df = pd.DataFrame()
        is_valid, issues = ingestion.validate_data(empty_df)
        
        assert not is_valid
        assert len(issues) > 0
        assert any("empty" in issue.lower() for issue in issues)
    
    def test_calculate_quality_score(self):
        """Test data quality score calculation."""
        config = Config()
        config.data_path.raw = "test/raw"
        config.data_path.processed = "test/processed"
        
        ingestion = CRMDataIngestion(config)
        
        # Perfect data
        perfect_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        score = ingestion._calculate_quality_score(perfect_df, [])
        assert score == 1.0
        
        # Data with missing values
        missing_df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', None]
        })
        score_missing = ingestion._calculate_quality_score(missing_df, [])
        assert score_missing < 1.0
        
        # Data with issues
        score_issues = ingestion._calculate_quality_score(perfect_df, ['issue1', 'issue2'])
        assert score_issues < 1.0


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing."""
    return pd.DataFrame({
        'opportunity_id': ['OPP001', 'OPP002', 'OPP003'],
        'sales_agent': ['Agent1', 'Agent2', 'Agent1'],
        'client_id': ['CLIENT001', 'CLIENT002', 'CLIENT003'],
        'product': ['Product A', 'Product B', 'Product A'],
        'deal_stage': ['Prospecting', 'Closed Won', 'Negotiation/Review'],
        'close_value': [1000.0, 2000.0, 1500.0],
        'probability': [0.3, 1.0, 0.7]
    })


def test_sample_dataframe_fixture(sample_dataframe):
    """Test the sample DataFrame fixture."""
    assert len(sample_dataframe) == 3
    assert 'opportunity_id' in sample_dataframe.columns
    assert sample_dataframe['close_value'].sum() == 4500.0
