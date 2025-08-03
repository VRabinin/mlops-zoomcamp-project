"""
Prefect Flow for Creating Reference Data for Drift Monitoring

This flow creates reference datasets with predictions that will be used
as baseline for drift detection in subsequent monitoring runs.
"""

import logging
from typing import Any, Dict, Tuple

from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import Config, get_config
from src.monitoring.drift_monitor import CRMDriftMonitor


@task(name="create_reference_data", retries=2)
def create_reference_data_task(
    config: Config, snapshot_month: str, sample_size: int
) -> Tuple[bool, str]:
    """
    Create reference dataset with predictions for drift monitoring.

    Args:
        config: Configuration object
        snapshot_month: Month snapshot to use as reference (e.g., "2017-05")
        sample_size: Number of samples to include in reference dataset

    Returns:
        Tuple of (success, message)
    """
    logger = get_run_logger()
    logger.info(f"🎯 Creating reference data for drift monitoring")
    logger.info(f"📅 Snapshot month: {snapshot_month}")
    logger.info(f"📊 Sample size: {sample_size}")

    try:
        # Initialize drift monitor
        drift_monitor = CRMDriftMonitor(config)

        # Create reference data
        success, message = drift_monitor.create_reference_data(
            snapshot_month, sample_size
        )

        if success:
            logger.info(f"✅ {message}")
        else:
            logger.error(f"❌ {message}")

        return success, message

    except Exception as e:
        error_msg = f"Failed to create reference data: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


@task(name="validate_reference_data", retries=1)
def validate_reference_data_task(
    config: Config, snapshot_month: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the created reference data.

    Args:
        config: Configuration object
        snapshot_month: Month snapshot to validate

    Returns:
        Tuple of (valid, validation_results)
    """
    logger = get_run_logger()
    logger.info(f"🔍 Validating reference data for {snapshot_month}")

    try:
        import pandas as pd

        from src.utils.storage import StorageManager

        storage = StorageManager(config)

        # Load reference data
        reference_file = f"reference_data_{snapshot_month}.csv"
        df = storage.load_dataframe("features", reference_file)

        # Validation checks
        validation_results = {
            "file_exists": True,
            "shape": df.shape,
            "has_predictions": "prediction" in df.columns,
            "has_target": "is_won" in df.columns,
            "prediction_range": None,
            "missing_predictions": 0,
            "valid": False,
        }

        if "prediction" in df.columns:
            validation_results["prediction_range"] = [
                float(df["prediction"].min()),
                float(df["prediction"].max()),
            ]
            validation_results["missing_predictions"] = int(
                df["prediction"].isna().sum()
            )

        # Check if validation passes
        validation_results["valid"] = (
            validation_results["has_predictions"]
            and validation_results["has_target"]
            and validation_results["missing_predictions"] == 0
            and df.shape[0] > 0
        )

        if validation_results["valid"]:
            logger.info(f"✅ Reference data validation passed")
            logger.info(f"📊 Shape: {validation_results['shape']}")
            logger.info(f"🎯 Prediction range: {validation_results['prediction_range']}")
        else:
            logger.warning(f"⚠️ Reference data validation failed")

        return validation_results["valid"], validation_results

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        return False, {"error": error_msg, "valid": False}


@flow(name="create_reference_data_flow", retries=1)
def create_reference_data_flow(snapshot_month: str, sample_size: int) -> Dict[str, Any]:
    """
    Main flow for creating reference data for drift monitoring.

    Args:
        snapshot_month: Month snapshot to use as reference (e.g., "2017-05")
        sample_size: Number of samples to include in reference dataset

    Returns:
        Dictionary with flow execution results
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Reference Data Creation Flow")

    # Get configuration
    config = get_config()

    # Use default snapshot month if not provided
    if snapshot_month == "":
        snapshot_month = config.first_snapshot_month
        logger.info(f"Using default snapshot month: {snapshot_month}")

    # Create reference data
    success, message = create_reference_data_task(config, snapshot_month, sample_size)

    if not success:
        logger.error(f"❌ Reference data creation failed: {message}")
        return {"status": "failed", "error": message, "snapshot_month": snapshot_month}

    # Validate reference data
    valid, validation_results = validate_reference_data_task(config, snapshot_month)

    if not valid:
        logger.error(f"❌ Reference data validation failed")
        return {
            "status": "failed",
            "error": "Reference data validation failed",
            "validation_results": validation_results,
            "snapshot_month": snapshot_month,
        }

    logger.info("✅ Reference Data Creation Flow completed successfully")

    return {
        "status": "success",
        "message": message,
        "snapshot_month": snapshot_month,
        "sample_size": sample_size,
        "validation_results": validation_results,
    }


if __name__ == "__main__":
    # For local testing
    import sys

    snapshot_month = sys.argv[1] if len(sys.argv) > 1 else ""
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    result = create_reference_data_flow(snapshot_month, sample_size)
    print(f"Flow result: {result}")
