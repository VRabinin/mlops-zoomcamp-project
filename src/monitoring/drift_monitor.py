"""
Model Drift Monitoring using Evidently AI

This module provides comprehensive model drift monitoring capabilities
for the CRM monthly win probability prediction model.
"""

import datetime
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityMetric,
    ClassificationRocCurve,
    ColumnDriftMetric,
    ColumnQuantileMetric,
    ColumnValueRangeMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnsType,
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDriftedColumns,
    TestNumberOfDuplicatedColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfRowsWithMissingValues,
)

from src.config.config import Config, get_config
from src.utils.storage import StorageManager


class CRMDriftMonitor:
    """
    Model drift monitor for CRM win probability predictions.

    This class handles:
    - Reference data management
    - Current data collection with predictions
    - Drift detection using Evidently
    - Report generation and storage
    """

    def __init__(self, config: Config = None):
        """Initialize the drift monitor."""
        self.config = config or get_config()
        self.storage = StorageManager(self.config)
        self.logger = logging.getLogger(__name__)

        # Set up MLflow
        os.environ["AWS_ACCESS_KEY_ID"] = self.config.storage.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.config.storage.secret_key
        os.environ["AWS_DEFAULT_REGION"] = self.config.storage.region
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.storage.endpoint_url
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)

        # Feature columns for monitoring (matching the trained model)
        self.feature_columns = [
            "close_value_log",
            "close_value_category_encoded",
            "days_since_engage",
            "sales_cycle_days",
            "engage_month",
            "engage_quarter",
            "engage_day_of_week",
            "agent_win_rate",
            "agent_opportunity_count",
            "product_win_rate",
            "product_popularity",
            "revenue",
            "employees",
            "is_repeat_account",
            "account_frequency",
            "should_close_next_month",
            "is_overdue",
            "is_early_stage",
            "high_value_deal",
            "long_sales_cycle",
            "sales_velocity",
            "sales_agent_encoded",
            "product_encoded",
            "account_encoded",
            "manager_encoded",
            "regional_office_encoded",
            "sector_encoded",
        ]

        # Column mapping for Evidently (matching trained model features)
        self.column_mapping = ColumnMapping(
            target="is_won",
            prediction="prediction",
            numerical_features=[
                "close_value_log",
                "days_since_engage",
                "sales_cycle_days",
                "agent_win_rate",
                "agent_opportunity_count",
                "product_win_rate",
                "product_popularity",
                "revenue",
                "employees",
                "account_frequency",
                "sales_velocity",
            ],
            categorical_features=[
                "close_value_category_encoded",
                "engage_month",
                "engage_quarter",
                "engage_day_of_week",
                "is_repeat_account",
                "should_close_next_month",
                "is_overdue",
                "is_early_stage",
                "high_value_deal",
                "long_sales_cycle",
                "sales_agent_encoded",
                "product_encoded",
                "account_encoded",
                "manager_encoded",
                "regional_office_encoded",
                "sector_encoded",
            ],
        )

    def load_model(self) -> Any:
        """Load the latest model from MLflow registry."""
        try:
            model = mlflow.sklearn.load_model(
                "models:/monthly_win_probability_model/latest"
            )
            self.logger.info("✅ Model loaded successfully from MLflow registry")
            return model
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise

    def create_monthly_features(
        self, df: pd.DataFrame, current_date: datetime.datetime = None
    ) -> pd.DataFrame:
        """Create monthly prediction features for the input data."""
        if current_date is None:
            current_date = datetime.datetime.now()

        df = df.copy()

        # Ensure engage_date is datetime
        if "engage_date" in df.columns:
            df["engage_date"] = pd.to_datetime(df["engage_date"])

        # Calculate days since engagement
        df["days_since_engage"] = (current_date - df["engage_date"]).dt.days

        # Expected close timeframe
        avg_sales_cycle = df[df["is_closed"] == 1]["sales_cycle_days"].median()
        if pd.isna(avg_sales_cycle):
            avg_sales_cycle = 90  # Default to 90 days

        df["expected_close_date"] = df["engage_date"] + timedelta(days=avg_sales_cycle)
        df["days_to_expected_close"] = (
            df["expected_close_date"] - current_date
        ).dt.days

        # Monthly prediction flags
        df["should_close_next_month"] = (
            df["days_to_expected_close"].between(-15, 45).astype(int)
        )
        df["is_overdue"] = (df["days_to_expected_close"] < -30).astype(int)
        df["is_early_stage"] = (df["days_to_expected_close"] > 60).astype(int)

        # Sales velocity features
        df["sales_velocity"] = df["close_value"] / df["sales_cycle_days"]
        df["sales_velocity"] = df["sales_velocity"].fillna(0)

        # Risk factors
        df["high_value_deal"] = (
            df["close_value"] > df["close_value"].quantile(0.8)
        ).astype(int)
        df["long_sales_cycle"] = (
            df["sales_cycle_days"] > df["sales_cycle_days"].quantile(0.8)
        ).astype(int)

        return df

    def create_reference_data(
        self, snapshot_month: str, sample_size: int = 1000
    ) -> Tuple[bool, str]:
        """
        Create reference dataset with predictions for drift monitoring.

        Args:
            snapshot_month: Month snapshot to use as reference (e.g., "2017-05")
            sample_size: Number of samples to include in reference dataset

        Returns:
            Tuple of (success, message)
        """
        try:
            self.logger.info(f"Creating reference data for snapshot: {snapshot_month}")

            # Load feature data
            features_file = f"crm_features_{snapshot_month}.csv"
            df = self.storage.load_dataframe("features", features_file)

            # Sample data for reference
            if len(df) > sample_size:
                df_reference = df[df["is_won"] == 1].sample(
                    n=sample_size, random_state=42
                )
                self.logger.info(
                    f"Sampled {sample_size} records from {len(df)} total records"
                )
            else:
                df_reference = df.copy()
                self.logger.info(f"Using all {len(df)} records as reference")

            # Load model and make predictions
            model = self.load_model()

            # Apply feature engineering
            df_reference = self.create_monthly_features(df_reference)

            # Prepare features for prediction
            X = df_reference[self.feature_columns].fillna(0)

            # Generate predictions
            predictions = model.predict_proba(X)[:, 1]  # Probability of win
            prediction_binary = model.predict(X)  # Binary predictions

            # Add predictions to reference data
            df_reference = df_reference.copy()
            df_reference["prediction"] = predictions
            df_reference["prediction_binary"] = prediction_binary

            # Add timestamp
            df_reference["monitoring_timestamp"] = datetime.datetime.now()

            # Save reference data
            reference_file = f"reference_data_{snapshot_month}.csv"
            self.storage.save_dataframe(df_reference, "features", reference_file)

            self.logger.info(f"✅ Reference data created and saved: {reference_file}")
            self.logger.info(f"📊 Reference data shape: {df_reference.shape}")
            self.logger.info(f"🎯 Mean prediction probability: {predictions.mean():.3f}")
            self.logger.info(
                f"📈 Win rate in reference: {df_reference['is_won'].mean():.3f}"
            )

            return True, f"Reference data created successfully: {reference_file}"

        except Exception as e:
            error_msg = f"Error creating reference data: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def generate_current_predictions(self, current_month: str) -> Tuple[bool, str]:
        """
        Generate predictions for current month data.

        Args:
            current_month: Current month snapshot (e.g., "2017-06")

        Returns:
            Tuple of (success, message)
        """
        try:
            self.logger.info(
                f"Generating predictions for current month: {current_month}"
            )

            # Load current month features
            features_file = f"crm_features_{current_month}.csv"
            df_current = self.storage.load_dataframe("features", features_file)

            # Load model and make predictions
            model = self.load_model()

            # Apply feature engineering
            df_current = self.create_monthly_features(df_current)

            # Prepare features
            X = df_current[self.feature_columns].fillna(0)

            # Generate predictions
            predictions = model.predict_proba(X)[:, 1]
            prediction_binary = model.predict(X)

            # Add predictions to current data
            df_current = df_current.copy()
            df_current["prediction"] = predictions
            df_current["prediction_binary"] = prediction_binary
            df_current["monitoring_timestamp"] = datetime.datetime.now()

            # Save current data with predictions
            current_file = f"current_predictions_{current_month}.csv"
            self.storage.save_dataframe(df_current, "features", current_file)

            self.logger.info(
                f"✅ Current predictions generated and saved: {current_file}"
            )
            self.logger.info(f"📊 Current data shape: {df_current.shape}")
            self.logger.info(f"🎯 Mean prediction probability: {predictions.mean():.3f}")

            return True, f"Current predictions saved: {current_file}"

        except Exception as e:
            error_msg = f"Error generating current predictions: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def detect_drift(
        self, reference_month: str, current_month: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect model and data drift between reference and current data.

        Args:
            reference_month: Reference month snapshot
            current_month: Current month snapshot

        Returns:
            Tuple of (success, drift_results)
        """
        try:
            self.logger.info(f"Detecting drift: {reference_month} vs {current_month}")

            # Load reference data
            reference_file = f"reference_data_{reference_month}.csv"
            df_reference = self.storage.load_dataframe("features", reference_file)

            # Load current data
            current_file = f"current_predictions_{current_month}.csv"
            df_current = self.storage.load_dataframe("features", current_file)

            self.logger.info(f"📊 Reference data shape: {df_reference.shape}")
            self.logger.info(f"📊 Current data shape: {df_current.shape}")

            # Ensure we have the same columns for comparison
            common_columns = list(set(df_reference.columns) & set(df_current.columns))
            monitoring_features = [
                col for col in self.feature_columns if col in common_columns
            ]

            # Create Evidently reports
            drift_report = self._create_drift_report(
                df_reference, df_current, monitoring_features
            )
            model_performance_report = self._create_model_performance_report(
                df_reference, df_current
            )

            # Extract key metrics
            drift_results = self._extract_drift_metrics(
                drift_report, model_performance_report
            )

            # Save reports
            self._save_drift_reports(
                drift_report, model_performance_report, reference_month, current_month
            )

            self.logger.info("✅ Drift detection completed successfully")

            return True, drift_results

        except Exception as e:
            error_msg = f"Error detecting drift: {str(e)}"
            self.logger.error(error_msg)
            return False, {"error": error_msg}

    def _create_drift_report(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame, features: list
    ) -> Report:
        """Create Evidently drift detection report."""

        report = Report(
            metrics=[
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
                ColumnDriftMetric(column_name="prediction"),
                ColumnQuantileMetric(column_name="close_value_log", quantile=0.5),
                ColumnValueRangeMetric(column_name="agent_win_rate"),
                ColumnValueRangeMetric(column_name="product_win_rate"),
            ]
        )

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        return report

    def _create_model_performance_report(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> Report:
        """Create Evidently model performance report."""

        # Only create performance report if we have ground truth
        if "is_won" in current_data.columns and current_data["is_won"].notna().any():
            report = Report(
                metrics=[
                    ClassificationQualityMetric(),
                    ClassificationClassBalance(),
                    ClassificationConfusionMatrix(),
                    ClassificationRocCurve(),
                ]
            )

            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )
        else:
            # Create a minimal report without performance metrics
            report = Report(
                metrics=[
                    ColumnDriftMetric(column_name="prediction"),
                ]
            )

            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

        return report

    def _extract_drift_metrics(
        self, drift_report: Report, performance_report: Report
    ) -> Dict[str, Any]:
        """Extract key metrics from Evidently reports."""

        drift_dict = drift_report.as_dict()
        performance_dict = performance_report.as_dict()

        # Extract drift metrics
        dataset_drift = None
        prediction_drift = None
        num_drifted_columns = 0
        missing_values_share = 0

        for metric in drift_dict.get("metrics", []):
            if metric["metric"] == "DatasetDriftMetric":
                dataset_drift = metric["result"]["dataset_drift"]
                num_drifted_columns = metric["result"]["number_of_drifted_columns"]
            elif (
                metric["metric"] == "ColumnDriftMetric"
                and metric.get("result", {}).get("column_name") == "prediction"
            ):
                prediction_drift = metric["result"]["drift_score"]
            elif metric["metric"] == "DatasetMissingValuesMetric":
                missing_values_share = metric["result"]["current"][
                    "share_of_missing_values"
                ]

        # Extract performance metrics if available
        accuracy = None
        precision = None
        recall = None
        f1_score = None
        roc_auc = None

        for metric in performance_dict.get("metrics", []):
            if metric["metric"] == "ClassificationQualityMetric":
                result = metric["result"]["current"]
                accuracy = result.get("accuracy")
                precision = result.get("precision")
                recall = result.get("recall")
                f1_score = result.get("f1")
                roc_auc = result.get("roc_auc")

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_drift": dataset_drift,
            "prediction_drift": prediction_drift,
            "num_drifted_columns": num_drifted_columns,
            "missing_values_share": missing_values_share,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "roc_auc": roc_auc,
            "drift_detected": dataset_drift
            or (prediction_drift and prediction_drift > 0.1),
            "alert_level": self._determine_alert_level(
                dataset_drift, prediction_drift, num_drifted_columns
            ),
        }

    def _determine_alert_level(
        self, dataset_drift: bool, prediction_drift: float, num_drifted_columns: int
    ) -> str:
        """Determine alert level based on drift metrics."""

        if dataset_drift or (prediction_drift and prediction_drift > 0.2):
            return "HIGH"
        elif prediction_drift and prediction_drift > 0.1 or num_drifted_columns > 5:
            return "MEDIUM"
        elif num_drifted_columns > 2:
            return "LOW"
        else:
            return "NONE"

    def _save_drift_reports(
        self,
        drift_report: Report,
        performance_report: Report,
        reference_month: str,
        current_month: str,
    ):
        """Save Evidently reports to storage."""

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save HTML reports
        drift_html = f"drift_report_{reference_month}_{current_month}_{timestamp}.html"
        performance_html = (
            f"performance_report_{reference_month}_{current_month}_{timestamp}.html"
        )

        # Create reports directory path
        reports_path = "monitoring_reports"

        try:
            # Save drift report as HTML
            drift_report.save_html(f"/tmp/{drift_html}")
            with open(f"/tmp/{drift_html}", "rb") as f:
                self.storage.upload_file(f, reports_path, drift_html)

            # Save performance report as HTML
            performance_report.save_html(f"/tmp/{performance_html}")
            with open(f"/tmp/{performance_html}", "rb") as f:
                self.storage.upload_file(f, reports_path, performance_html)

            self.logger.info(f"📊 Reports saved: {drift_html}, {performance_html}")

            # Clean up temp files
            os.remove(f"/tmp/{drift_html}")
            os.remove(f"/tmp/{performance_html}")

        except Exception as e:
            self.logger.warning(f"Could not save HTML reports: {e}")

    def run_data_quality_tests(self, current_month: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Run data quality tests on current data.

        Args:
            current_month: Current month snapshot

        Returns:
            Tuple of (all_tests_passed, test_results)
        """
        try:
            self.logger.info(f"Running data quality tests for: {current_month}")

            # Load current data
            current_file = f"current_predictions_{current_month}.csv"
            df_current = self.storage.load_dataframe("features", current_file)

            # Create test suite
            tests = TestSuite(
                tests=[
                    TestNumberOfColumnsWithMissingValues(),
                    TestNumberOfRowsWithMissingValues(),
                    TestNumberOfConstantColumns(),
                    TestNumberOfDuplicatedRows(),
                    TestNumberOfDuplicatedColumns(),
                    TestColumnsType(),
                ]
            )

            # Run tests (using current data as both reference and current for structure tests)
            tests.run(reference_data=df_current, current_data=df_current)

            # Extract results
            test_results = tests.as_dict()

            # Check if all tests passed
            all_passed = all(
                test["status"] == "SUCCESS" for test in test_results["tests"]
            )

            # Create summary
            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_tests": len(test_results["tests"]),
                "passed_tests": sum(
                    1 for test in test_results["tests"] if test["status"] == "SUCCESS"
                ),
                "failed_tests": sum(
                    1 for test in test_results["tests"] if test["status"] == "FAIL"
                ),
                "all_tests_passed": all_passed,
                "test_details": test_results["tests"],
            }

            self.logger.info(
                f"✅ Data quality tests completed: {summary['passed_tests']}/{summary['total_tests']} passed"
            )

            return all_passed, summary

        except Exception as e:
            error_msg = f"Error running data quality tests: {str(e)}"
            self.logger.error(error_msg)
            return False, {"error": error_msg}
