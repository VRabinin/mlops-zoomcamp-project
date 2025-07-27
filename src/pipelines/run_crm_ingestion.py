"""CRM data ingestion flow using Prefect."""

import logging
from logging import config
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import get_config
from src.data.ingestion.crm_ingestion import CRMDataIngestion
from src.data.validation.run_validation import DataValidationOrchestrator
from src.data.preprocessing.feature_engineering import CRMFeatureEngineer
from src.config.config import Config as Config

@task(name="download_crm_data", retries=2)
def download_crm_data_task(config: Config) -> Tuple[bool, str]:
    """Download CRM data from Kaggle.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (success, message).
    """
    logger = get_run_logger()
    logger.info("Starting CRM data download from Kaggle")
    
    ingestion = CRMDataIngestion(config)
    success, message = ingestion.download_dataset()
    
    if success:
        logger.info("✅ Data download completed successfully")
    else:
        logger.error(f"❌ Data download failed: {message}")
    
    return success, message


@task(name="load_and_clean_data", retries=1)
def load_and_clean_data_task(config: Config, snapshot_month: str) -> Tuple[bool, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and clean CRM data.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (success, sales_df, accounts_df, products_df, sales_teams_df, metadata).
    """
    logger = get_run_logger()
    logger.info("Loading and cleaning CRM data")
    
    try:
        ingestion = CRMDataIngestion(config, snapshot_month=snapshot_month)

        # Load sales data
        df_sales = ingestion.load_data(Path('sales_pipeline.csv'))
        logger.info(f"📊 Loaded Sales data shape: {df_sales.shape}")
        # Load accounts data
        df_accounts = ingestion.load_data(Path('accounts.csv'))
        logger.info(f"📊 Loaded Accounts data shape: {df_accounts.shape}")
        # Load products data
        df_products = ingestion.load_data(Path('products.csv'))
        logger.info(f"📊 Loaded Products data shape: {df_products.shape}")
        # Load sales teams data
        df_sales_teams = ingestion.load_data(Path('sales_teams.csv'))
        logger.info(f"📊 Loaded Sales Teams data shape: {df_sales_teams.shape}")

        # Clean data
        df_cleaned = ingestion.clean_data(df_sales)
        logger.info(f"🧹 Cleaned data shape: {df_cleaned.shape}")
        
        # Create metadata
        metadata = {
            'raw_shape': df_cleaned.shape,
            'cleaned_shape': df_cleaned.shape,
            'columns': list(df_cleaned.columns),
            'data_types': df_cleaned.dtypes.to_dict()
        }
        
        logger.info("✅ Data loading and cleaning completed")
        return True, df_cleaned, df_accounts, df_products, df_sales_teams, metadata
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        return False, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {'error': str(e)}


@task(name="validate_data_quality", retries=1)
def validate_data_quality_task(df: pd.DataFrame, config: Config) -> Tuple[bool, Dict[str, Any]]:
    """Validate data quality.
    
    Args:
        df: DataFrame to validate.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (validation_passed, validation_results).
    """
    logger = get_run_logger()
    logger.info("Running data quality validation")
    
    try:
        validator = DataValidationOrchestrator(config)
        results = validator.run_comprehensive_validation(df)
        
        # Log validation results
        logger.info(f"📋 Validation score: {results['overall_score']:.2f}")
        
        if results['overall_passed']:
            logger.info("✅ Data validation passed")
        else:
            logger.warning("⚠️ Data validation failed")
            
            # Log specific issues
            for category, validation in results['validations'].items():
                if isinstance(validation, dict) and not validation.get('passed', True):
                    logger.warning(f"  - {category}: Failed")
        
        return results['overall_passed'], results
        
    except Exception as e:
        logger.error(f"❌ Data validation failed: {str(e)}")
        return False, {'error': str(e)}


@task(name="engineer_features", retries=1)
def engineer_features_task(df_sales: pd.DataFrame, df_accounts: pd.DataFrame, df_products: pd.DataFrame, df_sales_teams: pd.DataFrame, config: Config) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """Engineer features from the data.
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (success, processed_dataframe, metadata).
    """
    logger = get_run_logger()
    logger.info("Starting feature engineering")
    try:
        feature_engineer = CRMFeatureEngineer(config)
        df_processed, feature_columns, metadata = feature_engineer.run_feature_engineering(df_sales, df_accounts, df_products, df_sales_teams)
        logger.info(f"🎯 Features created: {metadata['features_created']}")
        logger.info(f"📝 Feature columns: {len(feature_columns)}")
        logger.info("✅ Feature engineering completed")
        return True, df_processed, metadata
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {str(e)}")
        return False, pd.DataFrame(), {'error': str(e)}


@task(name="save_processed_data", retries=1)
def save_processed_data_task(df: pd.DataFrame, config: Config, suffix: str = "processed", snapshot_month: str = "XXXX-XX") -> Tuple[bool, str]:
    """Save processed data to storage.
    
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
        from src.utils.storage import StorageManager
        storage = StorageManager(config)
        
        # Use smart storage method for consistent path handling
        file_path = f"crm_{suffix}_{snapshot_month}.csv"

        saved_path = storage.save_dataframe(df, suffix, file_path)
        
        logger.info(f"💾 Data saved to: {saved_path}")
        return True, saved_path
        
    except Exception as e:
        logger.error(f"❌ Saving data failed: {str(e)}")
        return False, str(e)


@flow(name="crm_data_ingestion_flow", log_prints=True)
def crm_data_ingestion_flow(snapshot_month: str):
    """Main CRM data ingestion flow."""
    logger = get_run_logger()
    logger.info("🚀 Starting CRM data ingestion pipeline")
    
    # Get configuration
    config = get_config()
    
    # Task 1: Load and clean data
    load_success, df_cleaned, df_accounts, df_products, df_sales_teams, load_metadata = load_and_clean_data_task(config, snapshot_month)
    
    if not load_success:
        logger.error(f"Pipeline failed at load step: {load_metadata.get('error', 'Unknown error')}")
        return {"status": "failed", "step": "load", "error": load_metadata.get('error')}
    
    # Task 2: Save cleaned data
    save_success, save_path = save_processed_data_task(df_cleaned, config, "processed", snapshot_month)
    
    if not save_success:
        logger.error(f"Pipeline failed at save step: {save_path}")
        return {"status": "failed", "step": "save", "error": save_path}
    
    # Task 3: Validate data quality
    validation_passed, validation_results = validate_data_quality_task(df_cleaned, config)
    
    # Task 4: Engineer features
    features_success, df_features, features_metadata = engineer_features_task(df_cleaned, df_accounts, df_products, df_sales_teams, config)
    
    if not features_success:
        logger.error(f"Pipeline failed at feature engineering: {features_metadata.get('error', 'Unknown error')}")
        return {"status": "failed", "step": "features", "error": features_metadata.get('error')}
    
    # Task 5: Save features
    features_save_success, features_path = save_processed_data_task(df_features, config, "features", snapshot_month)
    
    if not features_save_success:
        logger.error(f"Pipeline failed at features save step: {features_path}")
        return {"status": "failed", "step": "features_save", "error": features_path}
    
    # Create final summary
    summary = {
        "status": "success",
        "steps_completed": ["download", "load", "clean", "save", "validate", "features", "features_save"],
        "data_shape": {
            "raw": load_metadata['raw_shape'],
            "cleaned": load_metadata['cleaned_shape'],
            "features": features_metadata['final_shape']
        },
        "validation": {
            "passed": validation_passed,
            "score": validation_results.get('overall_score', 0.0)
        },
        "features": {
            "created": features_metadata['features_created'],
            "total_columns": len(features_metadata['feature_columns'])
        },
        "files": {
            "processed_data": save_path,
            "features": features_path
        }
    }
    
    logger.info("🎉 CRM data ingestion pipeline completed successfully!")
    logger.info(f"📊 Final data shape: {summary['data_shape']['features']}")
    logger.info(f"🎯 Validation score: {summary['validation']['score']:.2f}")
    logger.info(f"✨ Features created: {summary['features']['created']}")
    
    return summary


if __name__ == "__main__":
    # Setup logging for local execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    config = get_config()
    # Run the flow
    result = crm_data_ingestion_flow(snapshot_month=config.start_snapshot_month)

    if result["status"] == "success":
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Data processed: {result['data_shape']['features']}")
        print(f"🎯 Validation score: {result['validation']['score']:.2f}")
        print(f"💾 Files saved:")
        print(f"  - Processed data: {result['files']['processed_data']}")
        print(f"  - Features: {result['files']['features']}")
    else:
        print(f"\n❌ Pipeline failed at {result['step']}: {result['error']}")
        exit(1)
