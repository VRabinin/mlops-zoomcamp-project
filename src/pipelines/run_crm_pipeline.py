"""CRM data ingestion flow using Prefect."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import get_config
from src.data.ingestion.crm_ingestion import CRMDataIngestion
from src.data.validation.run_validation import DataValidationOrchestrator
from src.data.preprocessing.feature_engineering import CRMFeatureEngineer


@task(name="download_crm_data", retries=2)
def download_crm_data_task(config: Dict[str, Any]) -> Tuple[bool, str]:
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
def load_and_clean_data_task(config: Dict[str, Any]) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """Load and clean CRM data.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (success, dataframe, metadata).
    """
    logger = get_run_logger()
    logger.info("Loading and cleaning CRM data")
    
    try:
        ingestion = CRMDataIngestion(config)
        
        # Load data
        df = ingestion.load_data()
        logger.info(f"📊 Loaded data shape: {df.shape}")
        
        # Clean data
        df_cleaned = ingestion.clean_data(df)
        logger.info(f"🧹 Cleaned data shape: {df_cleaned.shape}")
        
        # Create metadata
        metadata = {
            'raw_shape': df.shape,
            'cleaned_shape': df_cleaned.shape,
            'columns': list(df_cleaned.columns),
            'data_types': df_cleaned.dtypes.to_dict()
        }
        
        logger.info("✅ Data loading and cleaning completed")
        return True, df_cleaned, metadata
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        return False, pd.DataFrame(), {'error': str(e)}


@task(name="validate_data_quality", retries=1)
def validate_data_quality_task(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
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
def engineer_features_task(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
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
        feature_engineer = CRMFeatureEngineer({
            'target_column': config.get('target_column', 'deal_stage'),
            'test_size': config.get('test_size', 0.2),
            'random_state': config.get('random_state', 42)
        })
        
        df_processed, feature_columns, metadata = feature_engineer.run_feature_engineering(df)
        
        logger.info(f"🎯 Features created: {metadata['features_created']}")
        logger.info(f"📝 Feature columns: {len(feature_columns)}")
        
        logger.info("✅ Feature engineering completed")
        return True, df_processed, metadata
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {str(e)}")
        return False, pd.DataFrame(), {'error': str(e)}


@task(name="save_processed_data", retries=1)
def save_processed_data_task(df: pd.DataFrame, config: Dict[str, Any], suffix: str = "processed") -> Tuple[bool, str]:
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
        
        if suffix == "processed":
            if storage.use_s3:
                # Save to S3 with processed/ prefix
                file_path = f"processed/crm_data_processed.csv"
            else:
                # Save to local processed directory
                output_path = Path(config['processed_data_path']) / "crm_data_processed.csv"
                file_path = str(output_path)
        else:
            if storage.use_s3:
                # Save to S3 with features/ prefix
                file_path = f"features/crm_{suffix}.csv"
            else:
                # Save to local features directory
                output_path = Path(config.get('feature_store_path', 'data/features')) / f"crm_{suffix}.csv"
                file_path = str(output_path)
        
        saved_path = storage.save_dataframe(df, file_path)
        
        logger.info(f"💾 Data saved to: {saved_path}")
        return True, saved_path
        
    except Exception as e:
        logger.error(f"❌ Saving data failed: {str(e)}")
        return False, str(e)


@flow(name="crm_data_ingestion_flow", log_prints=True)
def crm_data_ingestion_flow():
    """Main CRM data ingestion flow."""
    logger = get_run_logger()
    logger.info("🚀 Starting CRM data ingestion pipeline")
    
    # Get configuration
    config = get_config()
    config_dict = {
        'raw_data_path': config.data.raw_data_path,
        'processed_data_path': config.data.processed_data_path,
        'feature_store_path': config.data.feature_store_path,
        'kaggle_dataset': config.data.kaggle_dataset,
        'target_column': config.model.target_column,
        'test_size': config.model.test_size,
        'random_state': config.model.random_state,
        'minio': {
            'endpoint_url': config.storage.endpoint_url,
            'access_key': config.storage.access_key,
            'secret_key': config.storage.secret_key,
            'region': config.storage.region,
            'buckets': config.storage.buckets
        }
    }
    
    # Task 1: Download data
    download_success, download_message = download_crm_data_task(config_dict)
    
    if not download_success:
        logger.error(f"Pipeline failed at download step: {download_message}")
        return {"status": "failed", "step": "download", "error": download_message}
    
    # Task 2: Load and clean data
    load_success, df_cleaned, load_metadata = load_and_clean_data_task(config_dict)
    
    if not load_success:
        logger.error(f"Pipeline failed at load step: {load_metadata.get('error', 'Unknown error')}")
        return {"status": "failed", "step": "load", "error": load_metadata.get('error')}
    
    # Task 3: Save cleaned data
    save_success, save_path = save_processed_data_task(df_cleaned, config_dict, "processed")
    
    if not save_success:
        logger.error(f"Pipeline failed at save step: {save_path}")
        return {"status": "failed", "step": "save", "error": save_path}
    
    # Task 4: Validate data quality
    validation_passed, validation_results = validate_data_quality_task(df_cleaned, config_dict)
    
    # Task 5: Engineer features
    features_success, df_features, features_metadata = engineer_features_task(df_cleaned, config_dict)
    
    if not features_success:
        logger.error(f"Pipeline failed at feature engineering: {features_metadata.get('error', 'Unknown error')}")
        return {"status": "failed", "step": "features", "error": features_metadata.get('error')}
    
    # Task 6: Save features
    features_save_success, features_path = save_processed_data_task(df_features, config_dict, "features")
    
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
    
    # Run the flow
    result = crm_data_ingestion_flow()
    
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
