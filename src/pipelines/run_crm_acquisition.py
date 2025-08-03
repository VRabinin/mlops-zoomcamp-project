"""CRM data ingestion flow using Prefect."""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import Config as Config
from src.config.config import get_config
from src.data.ingestion.crm_acquisition import CRMDataAcquisition
from src.data.preprocessing.feature_engineering import CRMFeatureEngineer
from src.data.validation.run_validation import DataValidationOrchestrator


@task(name="download_crm_data", retries=2)
def download_crm_data_task(config: Config, snapshot_month: str) -> Tuple[bool, str]:
    """Download CRM data from Kaggle.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (success, message).
    """
    logger = get_run_logger()
    logger.info("Starting CRM data download from Kaggle")

    acquisition = CRMDataAcquisition(config, snapshot_month)
    success, message = acquisition.download_dataset()

    if success:
        logger.info("✅ Data download completed successfully")
    else:
        logger.error(f"❌ Data download failed: {message}")

    return success, message


@task(name="load_and_enhance_data", retries=1)
def load_and_enhance_data_task(
    config: Config, snapshot_month: str
) -> Tuple[bool, list[pd.DataFrame], list[str], Dict[str, Any]]:
    """Load and enhance CRM data.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (success, sales_df, accounts_df, products_df, sales_teams_df, metadata).
    """
    logger = get_run_logger()
    logger.info("Loading and enhancing CRM data for simulation purposes")

    try:
        acquisition = CRMDataAcquisition(config, snapshot_month)

        # Load sales data
        df_sales = acquisition.load_data(Path("sales_pipeline.csv"))
        logger.info(f"📊 Loaded Sales data shape: {df_sales.shape}")

        # Enhance data
        dataframes, filenames = acquisition.enhance_data(df_sales)
        if not dataframes:
            raise ValueError("No dataframes returned from enhancement process")
        logger.info(f"🧹 Enhanced data shape: {dataframes[0].shape}")

        # Create metadata
        metadata = {
            "raw_shape": df_sales.shape,
            "enhanced_shape": dataframes[0].shape,
            "columns": list(dataframes[0].columns),
            "data_types": dataframes[0].dtypes.to_dict(),
        }

        logger.info("✅ Data loading and cleaning completed")
        return True, dataframes, filenames, metadata

    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        return False, [], [], {"error": str(e)}


@task(name="save_enhanced_data", retries=1)
def save_enhanced_data_task(
    dataframes: list[pd.DataFrame],
    filenames: list[str],
    config: Config,
    snapshot_month: str,
    suffix: str = "raw",
) -> Tuple[bool, list[str]]:
    """Save enhanced data to storage.

    Args:
        df: DataFrame to save.
        config: Configuration dictionary.
        suffix: File suffix.

    Returns:
        Tuple of (success, file_path).
    """
    logger = get_run_logger()
    logger.info(f"Saving {suffix} data")

    try:
        acquisition = CRMDataAcquisition(config, snapshot_month)
        acquisition.save_enhanced_data(dataframes, filenames, suffix)
        logger.info("✅ Data saving completed successfully")

        metadata = {
            "filenames": filenames,
            "data_shape": [df.shape for df in dataframes],
        }

        return True, metadata

    except Exception as e:
        logger.error(f"❌ Saving data failed: {str(e)}")
        return False, str(e)


@flow(name="crm_data_acquisition_flow", log_prints=True)
def crm_data_acquisition_flow(snapshot_month: str) -> Dict[str, Any]:
    """Main CRM data acquisition flow."""
    logger = get_run_logger()
    logger.info("🚀 Starting CRM data acquisition pipeline")

    # Get configuration
    config = get_config()

    # Task 1: Download data
    download_success, download_message = download_crm_data_task(config, snapshot_month)

    if not download_success:
        logger.error(f"Pipeline failed at download step: {download_message}")
        return {"status": "failed", "step": "download", "error": download_message}

    # Task 2: Load and clean data
    load_success, dataframes, filenames, load_metadata = load_and_enhance_data_task(
        config, snapshot_month
    )

    if not load_success:
        logger.error(
            f"Pipeline failed at enhance step: {load_metadata.get('error', 'Unknown error')}"
        )
        return {"status": "failed", "step": "load", "error": load_metadata.get("error")}

    # Task 3: Save cleaned data
    save_success, save_paths = save_enhanced_data_task(
        dataframes, filenames, config, snapshot_month, "raw"
    )

    if not save_success:
        logger.error(f"Pipeline failed at save step: {save_paths}")
        return {"status": "failed", "step": "save", "error": save_paths}

    # Create final summary
    summary = {
        "status": "success",
        "steps_completed": ["download", "load", "enhance", "save"],
        "data_shape": {
            "raw": load_metadata["raw_shape"],
            "enhanced": load_metadata["enhanced_shape"],
        },
        "files": {"processed_data": save_paths},
    }

    logger.info("🎉 CRM data acquisition pipeline completed successfully!")
    logger.info(f"📊 Processed data shape: {summary['data_shape']['enhanced']}")
    return summary


if __name__ == "__main__":
    # Setup logging for local execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = get_config()
    # Run the flow
    result = crm_data_acquisition_flow(config.first_snapshot_month)

    if result["status"] == "success":
        print("\n✅ Pipeline completed successfully!")
        print(f"💾 Files saved:")
        print(f"  - Processed data: {result['files']['processed_data']}")
    else:
        print(f"\n❌ Pipeline failed at {result['step']}: {result['error']}")
        exit(1)
