"""
Test configuration and data ingestion functionality.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from src.config.config import Config, DataConfig, get_config
from src.data.ingestion.crm_ingestion import CRMDataIngestion
from src.data.schemas.crm_schema import CRMDataSchema


class TestConfig:
    """Test configuration management."""
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.raw_data_path == "data/raw"
        assert config.processed_data_path == "data/processed"
        assert config.kaggle_dataset == "innocentmfa/crm-sales-opportunities"
    
    def test_get_config_returns_config_instance(self):
        """Test that get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)
        assert hasattr(config, 'data')
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


class TestCRMDataIngestion:
    """Test CRM data ingestion functionality."""
    
    def test_initialization(self):
        """Test CRMDataIngestion initializes correctly."""
        config = {"raw_data_path": "test/raw", "processed_data_path": "test/processed"}
        ingestion = CRMDataIngestion(config)
        assert ingestion.config == config
        assert str(ingestion.raw_data_path).endswith("test/raw")
    
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
        
        config = {"raw_data_path": "test/raw", "processed_data_path": "test/processed"}
        ingestion = CRMDataIngestion(config)
        
        try:
            df = ingestion.load_data(file_path)
            assert len(df) == 2
            assert 'opportunity_id' in df.columns
            assert df['close_value'].sum() == 3000.0
        finally:
            file_path.unlink()  # Clean up
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create sample data with issues
        sample_data = pd.DataFrame({
            'Opportunity ID': ['OPP001', 'OPP002', 'OPP001'],  # Duplicate
            'Sales Agent ': ['Agent1', 'Agent2', 'Agent1'],  # Space in column name
            'Close Value': [1000.0, 2000.0, 1000.0]
        })
        
        config = {"raw_data_path": "test/raw", "processed_data_path": "test/processed"}
        ingestion = CRMDataIngestion(config)
        
        cleaned_df = ingestion.clean_data(sample_data)
        
        # Check column names are cleaned
        assert 'opportunity_id' in cleaned_df.columns
        assert 'sales_agent' in cleaned_df.columns
        
        # Check duplicates are removed
        assert len(cleaned_df) == 2
    
    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        config = {"raw_data_path": "test/raw", "processed_data_path": "test/processed"}
        ingestion = CRMDataIngestion(config)
        
        empty_df = pd.DataFrame()
        is_valid, issues = ingestion.validate_data(empty_df)
        
        assert not is_valid
        assert len(issues) > 0
        assert any("empty" in issue.lower() for issue in issues)
    
    def test_calculate_quality_score(self):
        """Test data quality score calculation."""
        config = {"raw_data_path": "test/raw", "processed_data_path": "test/processed"}
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
