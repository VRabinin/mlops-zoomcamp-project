"""
Test model training and monitoring functionality.
"""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.models.training.monthly_win_probability import (
    ModelEvaluator,
    ModelTrainer,
    MonthlyWinProbabilityModel,
)
from src.monitoring.drift_monitor import ModelDriftMonitor
from src.monitoring.evidently_metrics_calculation import EvidentlyMetricsCalculator


class TestMonthlyWinProbabilityModel:
    """Test MonthlyWinProbabilityModel functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.config = Config()
        self.model = MonthlyWinProbabilityModel(self.config)

    def test_initialization(self):
        """Test model initializes correctly."""
        assert self.model.config == self.config
        assert hasattr(self.model, "model")
        assert hasattr(self.model, "feature_columns")
        assert hasattr(self.model, "target_column")

    def test_feature_preprocessing(self):
        """Test feature preprocessing pipeline."""
        df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", "OPP003"],
                "sales_agent": ["Agent1", "Agent2", "Agent1"],
                "close_value": [1000.0, 2000.0, 1500.0],
                "deal_stage": ["Prospecting", "Closed Won", "Negotiation/Review"],
                "engage_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
            }
        )

        processed_df = self.model.preprocess_features(df)

        # Should have more columns due to feature engineering
        assert len(processed_df.columns) >= len(df.columns)

        # Should maintain same number of rows
        assert len(processed_df) == len(df)

        # Should not have missing values in key columns
        assert not processed_df["close_value"].isnull().any()

    def test_target_encoding(self):
        """Test target variable encoding."""
        df = pd.DataFrame(
            {
                "deal_stage": [
                    "Prospecting",
                    "Closed Won",
                    "Closed Lost",
                    "Negotiation/Review",
                ]
            }
        )

        encoded_df = self.model.encode_target(df)

        # Should have target column
        assert "target" in encoded_df.columns

        # Should encode wins as 1, losses as 0
        win_mask = encoded_df["deal_stage"] == "Closed Won"
        loss_mask = encoded_df["deal_stage"] == "Closed Lost"

        assert encoded_df.loc[win_mask, "target"].iloc[0] == 1
        assert encoded_df.loc[loss_mask, "target"].iloc[0] == 0

    def test_model_training(self):
        """Test model training process."""
        # Create training data
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, n_samples),
                "feature_2": np.random.normal(0, 1, n_samples),
                "feature_3": np.random.normal(0, 1, n_samples),
            }
        )
        y = np.random.randint(0, 2, n_samples)

        # Train model
        trained_model = self.model.train(X, y)

        # Should return a trained model
        assert trained_model is not None
        assert hasattr(trained_model, "predict")
        assert hasattr(trained_model, "predict_proba")

        # Model should be stored internally
        assert self.model.model is not None

    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Create and train a simple model
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )
        y_train = np.random.randint(0, 2, 100)

        self.model.train(X_train, y_train)

        # Test prediction
        X_test = pd.DataFrame({"feature_1": [0.5, -0.5], "feature_2": [1.0, -1.0]})

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        # Should return correct shapes
        assert len(predictions) == 2
        assert probabilities.shape == (2, 2)  # 2 samples, 2 classes

        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Train a model with known features
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "important_feature": np.random.normal(0, 1, 100),
                "less_important": np.random.normal(0, 0.1, 100),
            }
        )
        # Make first feature actually important for target
        y_train = (X_train["important_feature"] > 0).astype(int)

        self.model.train(X_train, y_train)

        importance = self.model.get_feature_importance()

        # Should return feature importance
        assert isinstance(importance, (dict, pd.Series))
        assert len(importance) == 2

        # Important feature should have higher importance
        if isinstance(importance, dict):
            assert importance["important_feature"] > importance["less_important"]


class TestModelTrainer:
    """Test ModelTrainer functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.config = Config()
        self.trainer = ModelTrainer(self.config)

    def test_initialization(self):
        """Test trainer initializes correctly."""
        assert self.trainer.config == self.config
        assert hasattr(self.trainer, "mlflow_tracking_uri")
        assert hasattr(self.trainer, "experiment_name")

    def test_data_preparation(self):
        """Test data preparation for training."""
        df = pd.DataFrame(
            {
                "opportunity_id": ["OPP001", "OPP002", "OPP003", "OPP004"],
                "sales_agent": ["Agent1", "Agent2", "Agent1", "Agent3"],
                "close_value": [1000.0, 2000.0, 1500.0, 3000.0],
                "deal_stage": [
                    "Prospecting",
                    "Closed Won",
                    "Closed Lost",
                    "Negotiation/Review",
                ],
            }
        )

        X_train, X_test, y_train, y_test = self.trainer.prepare_training_data(
            df, test_size=0.5
        )

        # Should return correct shapes
        assert len(X_train) == 2
        assert len(X_test) == 2
        assert len(y_train) == 2
        assert len(y_test) == 2

        # Features should not include target or ID columns
        assert "opportunity_id" not in X_train.columns
        assert "deal_stage" not in X_train.columns
        assert "target" not in X_train.columns

    @patch("mlflow.start_run")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("mlflow.sklearn.log_model")
    def test_model_training_with_mlflow(
        self, mock_log_model, mock_log_param, mock_log_metric, mock_start_run
    ):
        """Test model training with MLflow integration."""
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=None)
        mock_start_run.return_value = mock_run

        # Create training data
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )
        y_train = np.random.randint(0, 2, 100)

        X_test = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 20),
                "feature_2": np.random.normal(0, 1, 20),
            }
        )
        y_test = np.random.randint(0, 2, 20)

        # Train model
        model, metrics = self.trainer.train_model(X_train, y_train, X_test, y_test)

        # Should return trained model and metrics
        assert model is not None
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # MLflow should be called
        mock_start_run.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metric.assert_called()
        mock_log_model.assert_called_once()

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        # Create synthetic data
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )
        y = np.random.randint(0, 2, 100)

        # Define parameter grid
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        best_params, best_score = self.trainer.optimize_hyperparameters(
            X, y, param_grid, cv=3
        )

        # Should return best parameters and score
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert 0.0 <= best_score <= 1.0


class TestModelEvaluator:
    """Test ModelEvaluator functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.config = Config()
        self.evaluator = ModelEvaluator(self.config)

    def test_initialization(self):
        """Test evaluator initializes correctly."""
        assert self.evaluator.config == self.config

    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation."""
        # Create test predictions
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]]
        )

        metrics = self.evaluator.calculate_metrics(y_true, y_pred, y_prob)

        # Should calculate all basic metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics

        # Verify accuracy calculation
        expected_accuracy = accuracy_score(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(expected_accuracy, abs=0.01)

    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        cm = self.evaluator.generate_confusion_matrix(y_true, y_pred)

        # Should return confusion matrix
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    def test_classification_report(self):
        """Test classification report generation."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        report = self.evaluator.generate_classification_report(y_true, y_pred)

        # Should return classification report
        assert isinstance(report, (str, dict))

    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        # Train a simple model
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "important_feature": np.random.normal(0, 1, 100),
                "noise_feature": np.random.normal(0, 0.1, 100),
            }
        )
        y = (X["important_feature"] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance_df = self.evaluator.analyze_feature_importance(model, X.columns)

        # Should return DataFrame with importance scores
        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 2

        # Should be sorted by importance
        assert importance_df["importance"].is_monotonic_decreasing


class TestModelDriftMonitor:
    """Test ModelDriftMonitor functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.config = Config()
        self.monitor = ModelDriftMonitor(self.config)

    def test_initialization(self):
        """Test drift monitor initializes correctly."""
        assert self.monitor.config == self.config
        assert hasattr(self.monitor, "drift_threshold")
        assert hasattr(self.monitor, "reference_data")

    def test_data_drift_detection(self):
        """Test data drift detection."""
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(5, 2, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        # Create current data with drift
        current_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(1, 1, 100),  # Mean shifted
                "feature_2": np.random.normal(5, 3, 100),  # Variance increased
                "target": np.random.randint(0, 2, 100),
            }
        )

        drift_report = self.monitor.detect_data_drift(reference_data, current_data)

        # Should return drift report
        assert isinstance(drift_report, dict)
        assert "drift_detected" in drift_report
        assert "feature_drift" in drift_report

        # Should detect drift in at least one feature
        feature_drift = drift_report["feature_drift"]
        assert any(feature_drift.values())

    def test_model_performance_drift(self):
        """Test model performance drift detection."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )
        y_true = np.random.randint(0, 2, 100)

        # Create predictions with different performance
        y_pred_good = np.where(
            np.random.random(100) < 0.9, y_true, 1 - y_true
        )  # 90% accuracy
        y_pred_poor = np.where(
            np.random.random(100) < 0.6, y_true, 1 - y_true
        )  # 60% accuracy

        # Calculate performance metrics
        perf_good = accuracy_score(y_true, y_pred_good)
        perf_poor = accuracy_score(y_true, y_pred_poor)

        drift_detected = self.monitor.detect_performance_drift(
            perf_good, perf_poor, threshold=0.1
        )

        # Should detect performance drift
        assert drift_detected is True

    def test_prediction_drift_detection(self):
        """Test prediction drift detection."""
        # Create reference predictions
        np.random.seed(42)
        reference_predictions = np.random.beta(2, 2, 1000)  # Beta distribution

        # Create current predictions with different distribution
        current_predictions = np.random.beta(5, 2, 1000)  # Different beta distribution

        drift_score = self.monitor.detect_prediction_drift(
            reference_predictions, current_predictions
        )

        # Should return drift score
        assert isinstance(drift_score, float)
        assert drift_score >= 0.0


class TestEvidentlyMetricsCalculator:
    """Test EvidentlyMetricsCalculator functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.config = Config()
        self.calculator = EvidentlyMetricsCalculator(self.config)

    def test_initialization(self):
        """Test calculator initializes correctly."""
        assert self.calculator.config == self.config

    @patch("evidently.metrics.DataDriftPreset")
    @patch("evidently.Report")
    def test_data_drift_report_generation(self, mock_report, mock_preset):
        """Test data drift report generation using Evidently."""
        # Mock Evidently components
        mock_report_instance = MagicMock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.run.return_value = None
        mock_report_instance.as_dict.return_value = {"drift_detected": True}

        # Create test data
        reference_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(1, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )

        report = self.calculator.generate_data_drift_report(
            reference_data, current_data
        )

        # Should generate report
        assert isinstance(report, dict)
        mock_report.assert_called_once()
        mock_report_instance.run.assert_called_once()

    @patch("evidently.metrics.ClassificationPreset")
    @patch("evidently.Report")
    def test_model_performance_report(self, mock_report, mock_preset):
        """Test model performance report generation."""
        # Mock Evidently components
        mock_report_instance = MagicMock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.run.return_value = None
        mock_report_instance.as_dict.return_value = {"accuracy": 0.85}

        # Create test data
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100),
                "prediction": np.random.randint(0, 2, 100),
            }
        )

        report = self.calculator.generate_performance_report(
            data, "target", "prediction"
        )

        # Should generate report
        assert isinstance(report, dict)
        mock_report.assert_called_once()
        mock_report_instance.run.assert_called_once()

    def test_metrics_extraction(self):
        """Test metrics extraction from Evidently reports."""
        # Mock Evidently report structure
        evidently_report = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {"dataset_drift": True, "drift_share": 0.3},
                },
                {
                    "metric": "ClassificationQualityMetric",
                    "result": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
                },
            ]
        }

        extracted_metrics = self.calculator.extract_key_metrics(evidently_report)

        # Should extract relevant metrics
        assert isinstance(extracted_metrics, dict)
        # Implementation depends on actual method structure


class TestIntegrationModelPipeline:
    """Integration tests for complete model pipeline."""

    def test_end_to_end_training_pipeline(self):
        """Test complete model training pipeline."""
        # Create realistic training data
        np.random.seed(42)
        n_samples = 200

        # Create features that correlate with win probability
        close_value = np.random.uniform(1000, 10000, n_samples)
        sales_cycle = np.random.uniform(10, 100, n_samples)
        agent_experience = np.random.choice([1, 2, 3, 4, 5], n_samples)

        # Create realistic win probability based on features
        win_prob = (
            0.3
            + 0.2 * (close_value / 10000)  # Base probability
            + 0.2 * (agent_experience / 5)  # Higher value = higher prob
            + 0.1 * (50 - sales_cycle) / 50  # More experience = higher prob
            + 0.2  # Shorter cycle = higher prob
            * np.random.random(n_samples)  # Random component
        )
        win_prob = np.clip(win_prob, 0, 1)

        training_data = pd.DataFrame(
            {
                "opportunity_id": [f"OPP{i:04d}" for i in range(n_samples)],
                "close_value": close_value,
                "sales_cycle_days": sales_cycle,
                "agent_experience_years": agent_experience,
                "feature_1": np.random.normal(0, 1, n_samples),
                "feature_2": np.random.normal(0, 1, n_samples),
                "deal_stage": [
                    "Closed Won" if p > 0.5 else "Closed Lost" for p in win_prob
                ],
            }
        )

        config = Config()

        # 1. Initialize components
        model = MonthlyWinProbabilityModel(config)
        trainer = ModelTrainer(config)
        evaluator = ModelEvaluator(config)

        # 2. Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            training_data, test_size=0.3
        )

        # 3. Train model
        trained_model = model.train(X_train, y_train)

        # 4. Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # 5. Evaluate performance
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_prob)

        # Verify pipeline success
        assert trained_model is not None
        assert len(y_pred) == len(y_test)
        assert y_prob.shape == (len(y_test), 2)
        assert metrics["accuracy"] > 0.5  # Should be better than random
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_model_persistence_and_loading(self):
        """Test model saving and loading."""
        # Create and train a simple model
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )
        y_train = np.random.randint(0, 2, 100)

        config = Config()
        model = MonthlyWinProbabilityModel(config)
        trained_model = model.train(X_train, y_train)

        # Test predictions before saving
        X_test = pd.DataFrame({"feature_1": [0.5, -0.5], "feature_2": [1.0, -1.0]})
        predictions_before = model.predict(X_test)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, "test_model.pkl")
            model.save_model(model_path)
            assert os.path.exists(model_path)

            # Load model into new instance
            new_model = MonthlyWinProbabilityModel(config)
            new_model.load_model(model_path)

            # Test predictions after loading
            predictions_after = new_model.predict(X_test)

            # Predictions should be identical
            np.testing.assert_array_equal(predictions_before, predictions_after)


@pytest.fixture
def sample_model_data():
    """Fixture providing sample data for model testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "opportunity_id": [f"OPP{i:03d}" for i in range(n_samples)],
            "close_value": np.random.uniform(1000, 10000, n_samples),
            "sales_cycle_days": np.random.uniform(10, 100, n_samples),
            "agent_experience": np.random.choice([1, 2, 3, 4, 5], n_samples),
            "deal_stage": np.random.choice(
                ["Prospecting", "Closed Won", "Closed Lost"], n_samples
            ),
        }
    )


def test_sample_model_data_fixture(sample_model_data):
    """Test the sample model data fixture."""
    assert len(sample_model_data) == 100
    assert "opportunity_id" in sample_model_data.columns
    assert "close_value" in sample_model_data.columns
    assert sample_model_data["close_value"].min() >= 1000
    assert sample_model_data["close_value"].max() <= 10000
