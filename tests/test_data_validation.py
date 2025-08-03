"""
Test data validation and feature engineering functionality.
"""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.config import Config
from src.data.preprocessing.feature_engineering import FeatureEngineer
from src.data.schemas.crm_schema import CRMDataSchema
from src.data.validation.data_validator import DataValidator


class TestDataValidator:
    """Test DataValidator functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.schema = CRMDataSchema()
        self.validator = DataValidator(self.schema)

    def test_initialization(self):
        """Test DataValidator initializes correctly."""
        assert self.validator.schema == self.schema
        assert hasattr(self.validator, "required_columns")
        assert hasattr(self.validator, "column_types")

    def test_validate_schema_success(self):
        """Test successful schema validation."""
        valid_df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002"],
                "sales_agent": ["Agent1", "Agent2"],
                "account": ["CLIENT001", "CLIENT002"],
                "product": ["Product A", "Product B"],
                "deal_stage": ["Prospecting", "Closed Won"],
                "engage_date": ["2024-01-01", "2024-01-02"],
                "close_date": ["2024-01-15", "2024-01-16"],
                "close_value": [1000.0, 2000.0],
            }
        )

        is_valid, issues = self.validator.validate_schema(valid_df)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        invalid_df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002"],
                "sales_agent": ["Agent1", "Agent2"]
                # Missing required columns
            }
        )

        is_valid, issues = self.validator.validate_schema(invalid_df)
        assert is_valid is False
        assert len(issues) > 0
        assert any("missing" in issue.lower() for issue in issues)

    def test_validate_schema_wrong_types(self):
        """Test schema validation with wrong data types."""
        invalid_df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002"],
                "sales_agent": ["Agent1", "Agent2"],
                "account": ["CLIENT001", "CLIENT002"],
                "product": ["Product A", "Product B"],
                "deal_stage": ["Prospecting", "Closed Won"],
                "engage_date": ["2024-01-01", "2024-01-02"],
                "close_date": ["2024-01-15", "2024-01-16"],
                "close_value": ["invalid", "data"],  # Should be numeric
            }
        )

        is_valid, issues = self.validator.validate_schema(invalid_df)
        assert is_valid is False
        assert len(issues) > 0

    def test_validate_data_quality_no_issues(self):
        """Test data quality validation with clean data."""
        clean_df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", "OPP003"],
                "sales_agent": ["Agent1", "Agent2", "Agent3"],
                "close_value": [1000.0, 2000.0, 1500.0],
                "deal_stage": ["Prospecting", "Closed Won", "Negotiation/Review"],
            }
        )

        quality_score, issues = self.validator.validate_data_quality(clean_df)
        assert quality_score == 1.0
        assert len(issues) == 0

    def test_validate_data_quality_with_missing_values(self):
        """Test data quality validation with missing values."""
        df_with_nulls = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", None],
                "sales_agent": ["Agent1", None, "Agent3"],
                "close_value": [1000.0, np.nan, 1500.0],
                "deal_stage": ["Prospecting", "Closed Won", None],
            }
        )

        quality_score, issues = self.validator.validate_data_quality(df_with_nulls)
        assert quality_score < 1.0
        assert len(issues) > 0
        assert any("missing" in issue.lower() for issue in issues)

    def test_validate_data_quality_with_duplicates(self):
        """Test data quality validation with duplicate records."""
        df_with_dupes = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP001", "OPP003"],  # Duplicate
                "sales_agent": ["Agent1", "Agent1", "Agent3"],
                "close_value": [1000.0, 1000.0, 1500.0],
                "deal_stage": ["Prospecting", "Prospecting", "Negotiation/Review"],
            }
        )

        quality_score, issues = self.validator.validate_data_quality(df_with_dupes)
        assert quality_score < 1.0
        assert len(issues) > 0
        assert any("duplicate" in issue.lower() for issue in issues)

    def test_validate_data_quality_with_outliers(self):
        """Test data quality validation with outliers."""
        df_with_outliers = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", "OPP003"],
                "sales_agent": ["Agent1", "Agent2", "Agent3"],
                "close_value": [1000.0, 2000.0, 1000000.0],  # Outlier
                "deal_stage": ["Prospecting", "Closed Won", "Negotiation/Review"],
            }
        )

        quality_score, issues = self.validator.validate_data_quality(df_with_outliers)
        assert quality_score < 1.0
        assert len(issues) > 0
        assert any("outlier" in issue.lower() for issue in issues)

    def test_check_missing_values(self):
        """Test missing values detection."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, None, 4],
                "col2": ["a", "b", "c", None],
                "col3": [1.0, 2.0, 3.0, 4.0],
            }
        )

        missing_info = self.validator._check_missing_values(df)
        assert "col1" in missing_info
        assert "col2" in missing_info
        assert "col3" not in missing_info
        assert missing_info["col1"] == 1
        assert missing_info["col2"] == 1

    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        df = pd.DataFrame(
            {
                "normal_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "with_outliers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # 100 is outlier
            }
        )

        outliers = self.validator._detect_outliers(
            df, ["normal_values", "with_outliers"]
        )
        assert len(outliers["normal_values"]) == 0
        assert len(outliers["with_outliers"]) > 0
        assert 100 in outliers["with_outliers"]

    def test_validate_date_columns(self):
        """Test date column validation."""
        df = pd.DataFrame(
            {
                "valid_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "invalid_date": ["2024-01-01", "invalid", "2024-01-03"],
                "numeric_col": [1, 2, 3],
            }
        )

        date_issues = self.validator._validate_date_columns(
            df, ["valid_date", "invalid_date"]
        )
        assert len(date_issues) > 0
        assert any("invalid_date" in issue for issue in date_issues)

    def test_validate_categorical_columns(self):
        """Test categorical column validation."""
        df = pd.DataFrame(
            {
                "deal_stage": [
                    "Prospecting",
                    "Closed Won",
                    "Closed Lost",
                    "Invalid Stage",
                ],
                "sales_agent": ["Agent1", "Agent2", "Agent3", "Agent4"],
            }
        )

        categorical_issues = self.validator._validate_categorical_columns(df)
        # Should detect unusual categorical values if configured
        assert isinstance(categorical_issues, list)

    def test_comprehensive_validation(self):
        """Test comprehensive validation combining all checks."""
        complex_df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", "OPP003", "OPP004"],
                "sales_agent": ["Agent1", "Agent2", None, "Agent4"],  # Missing value
                "account": ["CLIENT001", "CLIENT002", "CLIENT003", "CLIENT004"],
                "product": ["Product A", "Product B", "Product A", "Product C"],
                "deal_stage": [
                    "Prospecting",
                    "Closed Won",
                    "Closed Lost",
                    "Invalid",
                ],  # Invalid stage
                "engage_date": [
                    "2024-01-01",
                    "2024-01-02",
                    "invalid",
                    "2024-01-04",
                ],  # Invalid date
                "close_date": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
                "close_value": [1000.0, 2000.0, np.nan, 1000000.0],  # Missing + outlier
            }
        )

        is_valid, issues = self.validator.validate_comprehensive(complex_df)
        assert is_valid is False
        assert len(issues) > 0

        # Should detect multiple types of issues
        issue_text = " ".join(issues).lower()
        assert "missing" in issue_text or "null" in issue_text
        assert "outlier" in issue_text or "invalid" in issue_text


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.config = Config()
        self.engineer = FeatureEngineer(self.config)

    def test_initialization(self):
        """Test FeatureEngineer initializes correctly."""
        assert self.engineer.config == self.config
        assert hasattr(self.engineer, "feature_store_path")

    def test_create_datetime_features(self):
        """Test datetime feature creation."""
        df = pd.DataFrame(
            {
                "engage_date": pd.to_datetime(
                    ["2024-01-15", "2024-06-15", "2024-12-15"]
                ),
                "close_date": pd.to_datetime(
                    ["2024-01-30", "2024-06-30", "2024-12-30"]
                ),
            }
        )

        enhanced_df = self.engineer.create_datetime_features(df)

        # Check new datetime features
        assert "engage_month" in enhanced_df.columns
        assert "engage_quarter" in enhanced_df.columns
        assert "engage_day_of_week" in enhanced_df.columns
        assert "close_month" in enhanced_df.columns
        assert "sales_cycle_days" in enhanced_df.columns

        # Verify values
        assert enhanced_df["engage_month"].iloc[0] == 1
        assert enhanced_df["engage_quarter"].iloc[1] == 2
        assert enhanced_df["sales_cycle_days"].iloc[0] == 15

    def test_create_sales_features(self):
        """Test sales-specific feature creation."""
        df = pd.DataFrame(
            {
                "sales_agent": ["Agent1", "Agent2", "Agent1", "Agent3"],
                "account": ["CLIENT001", "CLIENT002", "CLIENT001", "CLIENT003"],
                "product": ["Product A", "Product B", "Product A", "Product C"],
                "close_value": [1000.0, 2000.0, 1500.0, 3000.0],
            }
        )

        enhanced_df = self.engineer.create_sales_features(df)

        # Check aggregated features
        assert "agent_avg_deal_size" in enhanced_df.columns
        assert "agent_total_deals" in enhanced_df.columns
        assert "account_deal_count" in enhanced_df.columns
        assert "product_avg_value" in enhanced_df.columns

        # Verify calculations
        agent1_avg = enhanced_df[enhanced_df["sales_agent"] == "Agent1"][
            "agent_avg_deal_size"
        ].iloc[0]
        assert agent1_avg == 1250.0  # (1000 + 1500) / 2

    def test_create_categorical_features(self):
        """Test categorical feature encoding."""
        df = pd.DataFrame(
            {
                "deal_stage": [
                    "Prospecting",
                    "Closed Won",
                    "Closed Lost",
                    "Negotiation/Review",
                ],
                "product": ["Product A", "Product B", "Product A", "Product C"],
                "sales_agent": ["Agent1", "Agent2", "Agent1", "Agent3"],
            }
        )

        enhanced_df = self.engineer.create_categorical_features(df)

        # Check encoded features
        encoded_cols = [col for col in enhanced_df.columns if "encoded" in col]
        assert len(encoded_cols) > 0

        # Check frequency encoding
        freq_cols = [col for col in enhanced_df.columns if "freq" in col]
        assert len(freq_cols) > 0

    def test_create_win_probability_features(self):
        """Test win probability feature creation."""
        df = pd.DataFrame(
            {
                "deal_stage": [
                    "Prospecting",
                    "Qualification",
                    "Proposal/Price Quote",
                    "Negotiation/Review",
                    "Closed Won",
                    "Closed Lost",
                ],
                "close_value": [1000.0, 2000.0, 1500.0, 3000.0, 2500.0, 1200.0],
                "sales_agent": [
                    "Agent1",
                    "Agent2",
                    "Agent1",
                    "Agent3",
                    "Agent1",
                    "Agent2",
                ],
            }
        )

        enhanced_df = self.engineer.create_win_probability_features(df)

        # Check stage-based features
        assert "stage_win_rate" in enhanced_df.columns
        assert "stage_progression_score" in enhanced_df.columns

        # Check agent performance features
        assert "agent_win_rate" in enhanced_df.columns
        assert "agent_avg_deal_size" in enhanced_df.columns

        # Verify win rate calculations
        won_stages = enhanced_df[enhanced_df["deal_stage"] == "Closed Won"]
        if len(won_stages) > 0:
            assert won_stages["stage_win_rate"].iloc[0] == 1.0

    def test_engineer_features_comprehensive(self):
        """Test comprehensive feature engineering pipeline."""
        df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", "OPP003", "OPP004"],
                "sales_agent": ["Agent1", "Agent2", "Agent1", "Agent3"],
                "account": ["CLIENT001", "CLIENT002", "CLIENT001", "CLIENT003"],
                "product": ["Product A", "Product B", "Product A", "Product C"],
                "deal_stage": [
                    "Prospecting",
                    "Closed Won",
                    "Negotiation/Review",
                    "Qualification",
                ],
                "engage_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "close_date": pd.to_datetime(
                    ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"]
                ),
                "close_value": [1000.0, 2000.0, 1500.0, 3000.0],
            }
        )

        features_df = self.engineer.engineer_features(df)

        # Should have more columns than original
        assert len(features_df.columns) > len(df.columns)

        # Should contain various feature types
        feature_cols = features_df.columns.tolist()

        # Datetime features
        datetime_features = [
            col
            for col in feature_cols
            if any(
                keyword in col
                for keyword in ["month", "quarter", "day_of_week", "cycle_days"]
            )
        ]
        assert len(datetime_features) > 0

        # Sales features
        sales_features = [
            col
            for col in feature_cols
            if any(keyword in col for keyword in ["agent_", "account_", "product_"])
        ]
        assert len(sales_features) > 0

        # Win probability features
        win_features = [
            col
            for col in feature_cols
            if any(keyword in col for keyword in ["win_rate", "progression_score"])
        ]
        assert len(win_features) > 0

    def test_save_and_load_features(self):
        """Test saving and loading feature data."""
        df = pd.DataFrame(
            {"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "target": [0, 1, 0]}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            self.config.data_path.features = temp_dir
            engineer = FeatureEngineer(self.config)

            # Save features
            saved_path = engineer.save_features(df, "test_features.csv")
            assert os.path.exists(saved_path)

            # Load features
            loaded_df = engineer.load_features("test_features.csv")
            pd.testing.assert_frame_equal(df, loaded_df)

    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Create features with varying correlation to target
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "highly_correlated": np.random.normal(0, 1, n_samples),
                "moderately_correlated": np.random.normal(0, 1, n_samples),
                "uncorrelated": np.random.normal(0, 1, n_samples),
                "constant_feature": [1] * n_samples,  # Should be removed
                "target": np.random.randint(0, 2, n_samples),
            }
        )

        # Make highly_correlated actually correlated with target
        df["highly_correlated"] = df["target"] + np.random.normal(0, 0.1, n_samples)

        selected_features = self.engineer.select_features(df, "target", max_features=3)

        # Should exclude constant features
        assert "constant_feature" not in selected_features

        # Should prioritize correlated features
        assert "highly_correlated" in selected_features

        # Should respect max_features limit
        assert len(selected_features) <= 3

    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        df = pd.DataFrame(
            {
                "large_scale": [1000, 2000, 3000],
                "small_scale": [0.1, 0.2, 0.3],
                "categorical": ["A", "B", "C"],
            }
        )

        scaled_df = self.engineer.scale_features(
            df, numeric_columns=["large_scale", "small_scale"]
        )

        # Numeric columns should be scaled
        assert scaled_df["large_scale"].std() == pytest.approx(1.0, abs=0.1)
        assert scaled_df["small_scale"].std() == pytest.approx(1.0, abs=0.1)

        # Categorical columns should remain unchanged
        assert scaled_df["categorical"].equals(df["categorical"])

    def test_missing_value_handling(self):
        """Test missing value handling in feature engineering."""
        df = pd.DataFrame(
            {
                "numeric_with_nulls": [1.0, 2.0, np.nan, 4.0],
                "categorical_with_nulls": ["A", "B", None, "D"],
                "date_with_nulls": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", None, "2024-01-04"]
                ),
            }
        )

        cleaned_df = self.engineer.handle_missing_values(df)

        # Should have no missing values
        assert cleaned_df.isnull().sum().sum() == 0

        # Numeric nulls should be filled appropriately
        assert not cleaned_df["numeric_with_nulls"].isnull().any()

        # Categorical nulls should be filled appropriately
        assert not cleaned_df["categorical_with_nulls"].isnull().any()


class TestFeatureEngineerIntegration:
    """Integration tests for feature engineering pipeline."""

    def test_end_to_end_feature_pipeline(self):
        """Test complete feature engineering pipeline."""
        # Create realistic CRM data
        np.random.seed(42)
        n_records = 50

        raw_data = pd.DataFrame(
            {
                "opportunity_id": [f"OPP{i:03d}" for i in range(1, n_records + 1)],
                "sales_agent": np.random.choice(
                    ["Agent1", "Agent2", "Agent3", "Agent4"], n_records
                ),
                "account": [
                    f"CLIENT{i:03d}" for i in np.random.randint(1, 20, n_records)
                ],
                "product": np.random.choice(
                    ["Product A", "Product B", "Product C"], n_records
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
                    n_records,
                ),
                "engage_date": pd.date_range("2024-01-01", periods=n_records, freq="D"),
                "close_date": pd.date_range("2024-01-15", periods=n_records, freq="D"),
                "close_value": np.random.uniform(500, 5000, n_records),
            }
        )

        config = Config()

        # 1. Validate data
        schema = CRMDataSchema()
        validator = DataValidator(schema)
        is_valid, issues = validator.validate_comprehensive(raw_data)

        # Should pass basic validation (our synthetic data is clean)
        assert is_valid or len(issues) == 0  # Allow for minor issues in synthetic data

        # 2. Engineer features
        engineer = FeatureEngineer(config)
        features_df = engineer.engineer_features(raw_data)

        # Should have significantly more features
        assert len(features_df.columns) > len(raw_data.columns)

        # 3. Validate feature quality
        feature_quality_score, feature_issues = validator.validate_data_quality(
            features_df
        )
        assert feature_quality_score > 0.8  # Should maintain high quality

        # 4. Verify feature types
        feature_cols = features_df.columns.tolist()

        # Should have datetime features
        datetime_features = [
            col
            for col in feature_cols
            if any(keyword in col for keyword in ["month", "quarter", "day", "cycle"])
        ]
        assert len(datetime_features) >= 5

        # Should have aggregated features
        agg_features = [
            col
            for col in feature_cols
            if any(keyword in col for keyword in ["avg", "count", "total", "rate"])
        ]
        assert len(agg_features) >= 3

        # Should maintain data integrity
        assert len(features_df) == len(raw_data)
        assert (
            features_df["opportunity_id"].nunique()
            == raw_data["opportunity_id"].nunique()
        )


@pytest.fixture
def sample_crm_data():
    """Fixture providing sample CRM data for testing."""
    return pd.DataFrame(
        {
            "opportunity_id": ["OPP001", "OPP002", "OPP003", "OPP004"],
            "sales_agent": ["Agent1", "Agent2", "Agent1", "Agent3"],
            "account": ["CLIENT001", "CLIENT002", "CLIENT001", "CLIENT003"],
            "product": ["Product A", "Product B", "Product A", "Product C"],
            "deal_stage": [
                "Prospecting",
                "Closed Won",
                "Negotiation/Review",
                "Qualification",
            ],
            "engage_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "close_date": pd.to_datetime(
                ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"]
            ),
            "close_value": [1000.0, 2000.0, 1500.0, 3000.0],
        }
    )


def test_sample_crm_data_fixture(sample_crm_data):
    """Test the sample CRM data fixture."""
    assert len(sample_crm_data) == 4
    assert "opportunity_id" in sample_crm_data.columns
    assert sample_crm_data["close_value"].sum() == 7500.0
    assert sample_crm_data["engage_date"].dtype == "datetime64[ns]"
