"""Monthly Win Probability Training Flow using Prefect.

This flow implements the complete training pipeline for monthly win probability
prediction models, based on the analysis from the 02_monthly_win_probability_prediction notebook.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import get_config, Config
from src.models.training.monthly_win_probability import MonthlyWinProbabilityTrainer
from src.utils.storage import StorageManager


@task(name="load_features_data", retries=2)
def load_features_data_task(config: Config, snapshot_month: str) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """Load feature-engineered CRM data for training.
    
    Args:
        config: Configuration object.
        snapshot_month: Month snapshot to load (e.g., "2017-05").
        
    Returns:
        Tuple of (success, dataframe, metadata).
    """
    logger = get_run_logger()
    logger.info(f"Loading features data for snapshot: {snapshot_month}")
    
    try:
        storage = StorageManager(config)
        
        # Load features data
        file_path = f"crm_features_{snapshot_month}.csv"
        df = storage.load_dataframe("features", file_path)
        
        logger.info(f"✅ Features data loaded successfully")
        logger.info(f"📊 Data shape: {df.shape}")
        logger.info(f"📋 Columns: {list(df.columns)}")
        
        # Create metadata safely to avoid comparison errors
        metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
        }
        
        # Safe calculation of win rate
        if 'is_won' in df.columns:
            try:
                metadata["win_rate"] = float(df['is_won'].mean())
            except Exception as e:
                logger.warning(f"Could not calculate win rate: {e}")
                metadata["win_rate"] = None
        else:
            metadata["win_rate"] = None
            
        # Safe calculation of closed deals
        if 'is_closed' in df.columns:
            try:
                metadata["closed_deals"] = int(df['is_closed'].sum())
            except Exception as e:
                logger.warning(f"Could not calculate closed deals: {e}")
                metadata["closed_deals"] = None
        else:
            metadata["closed_deals"] = None
            
        # Safe date range calculation
        if 'engage_date' in df.columns:
            try:
                # Convert to datetime with error handling
                date_series = pd.to_datetime(df['engage_date'], errors='coerce')
                valid_dates = date_series.dropna()
                if len(valid_dates) > 0:
                    metadata["date_range"] = {
                        "min_engage": str(valid_dates.min()),
                        "max_engage": str(valid_dates.max())
                    }
                else:
                    metadata["date_range"] = {"min_engage": None, "max_engage": None}
            except Exception as e:
                logger.warning(f"Could not calculate date range: {e}")
                metadata["date_range"] = {"min_engage": None, "max_engage": None}
        else:
            metadata["date_range"] = {"min_engage": None, "max_engage": None}
        
        return True, df, metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to load features data: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, pd.DataFrame(), {"error": str(e)}


@task(name="validate_training_data", retries=1)
def validate_training_data_task(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """Validate that the data is suitable for monthly win probability training.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (validation_passed, validation_results).
    """
    logger = get_run_logger()
    logger.info("Validating training data for monthly win probability model")
    
    validation_results = {"issues": [], "warnings": []}
    
    try:
        # Check required columns
        required_columns = [
            'is_won', 'is_closed', 'sales_cycle_days', 'close_value'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results["issues"].append(f"Missing required columns: {missing_columns}")
        
        # Check data quality
        if len(df) < 1000:
            validation_results["warnings"].append(f"Small dataset size: {len(df)} samples")
        
        # Check target variable distribution
        if 'is_won' in df.columns:
            win_rate = df['is_won'].mean()
            if win_rate < 0.1 or win_rate > 0.9:
                validation_results["warnings"].append(
                    f"Imbalanced target variable. Win rate: {win_rate:.3f}"
                )
        
        # Check for null values in critical columns (warnings only for dates)
        critical_columns = ['is_won', 'close_value']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                validation_results["issues"].append(f"Null values found in {col}")
        
        # Check for null values in date columns (warnings only)
        date_columns = ['engage_date', 'close_date']
        for col in date_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                validation_results["warnings"].append(
                    f"Null values found in {col}: {null_count} out of {len(df)} records"
                )
        
        # Check date ranges (if we have valid dates)
        if 'engage_date' in df.columns:
            try:
                # Convert to datetime and check valid dates
                df_temp = df.copy()
                df_temp['engage_date'] = pd.to_datetime(df_temp['engage_date'], errors='coerce')
                valid_dates = df_temp['engage_date'].dropna()
                
                if len(valid_dates) > 0:
                    date_range = (valid_dates.max() - valid_dates.min()).days
                    if date_range < 30:
                        validation_results["warnings"].append(
                            f"Short date range: {date_range} days"
                        )
                else:
                    validation_results["warnings"].append("No valid engage_date values found")
            except Exception as e:
                validation_results["warnings"].append(f"Could not analyze engage_date: {str(e)}")
        
        validation_passed = len(validation_results["issues"]) == 0
        
        if validation_passed:
            logger.info("✅ Data validation passed")
        else:
            logger.warning("⚠️ Data validation found issues")
            
        for issue in validation_results["issues"]:
            logger.error(f"  - ISSUE: {issue}")
        for warning in validation_results["warnings"]:
            logger.warning(f"  - WARNING: {warning}")
        
        validation_results["passed"] = validation_passed
        validation_results["sample_count"] = len(df)
        validation_results["win_rate"] = df['is_won'].mean() if 'is_won' in df.columns else None
        
        return validation_passed, validation_results
        
    except Exception as e:
        logger.error(f"❌ Data validation failed: {str(e)}")
        return False, {"error": str(e), "passed": False}


@task(name="train_monthly_win_models", retries=1)
def train_monthly_win_models_task(df: pd.DataFrame, config: Config) -> Tuple[bool, Dict[str, Any]]:
    """Train monthly win probability models.
    
    Args:
        df: Training DataFrame.
        config: Configuration object.
        
    Returns:
        Tuple of (success, training_results).
    """
    logger = get_run_logger()
    logger.info("Starting monthly win probability model training")
    
    try:
        # Initialize trainer
        trainer = MonthlyWinProbabilityTrainer(config)
        
        # Run training pipeline
        results = trainer.run_training_pipeline(df)
        
        if results["status"] == "success":
            logger.info("✅ Model training completed successfully")
            logger.info(f"🏆 Best model: {results['best_model']['name']}")
            logger.info(f"📊 ROC AUC: {results['best_model']['roc_auc']:.4f}")
            logger.info(f"📝 Registered model: {results['model_info']['registered_model_name']}")
            return True, results
        else:
            logger.error(f"❌ Model training failed: {results.get('error', 'Unknown error')}")
            return False, results
            
    except Exception as e:
        logger.error(f"❌ Model training task failed: {str(e)}")
        return False, {"status": "failed", "error": str(e)}


@task(name="save_training_results", retries=1)
def save_training_results_task(results: Dict[str, Any], config: Config, 
                             snapshot_month: str) -> Tuple[bool, str]:
    """Save training results and model metadata.
    
    Args:
        results: Training results dictionary.
        config: Configuration object.
        snapshot_month: Month snapshot identifier.
        
    Returns:
        Tuple of (success, saved_path).
    """
    logger = get_run_logger()
    logger.info("Saving training results and metadata")
    
    try:
        import json
        from src.utils.storage import StorageManager
        
        storage = StorageManager(config)
        
        # Save training results as JSON
        file_path = f"monthly_win_training_results_{snapshot_month}.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            "status": results["status"],
            "experiment_name": results.get("experiment_name", ""),
            "training_timestamp": results.get("training_timestamp", ""),
            "best_model": results.get("best_model", {}),
            "dataset_info": results.get("dataset_info", {}),
            "model_performance_ranking": results.get("model_performance_ranking", []),
            "feature_count": len(results.get("feature_list", [])),
            "feature_list": results.get("feature_list", [])
        }
        
        if "model_info" in results:
            json_results["model_info"] = results["model_info"]
        
        # Save to storage
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_results, f, indent=2, default=str)
            temp_path = f.name
        
        saved_path = storage.save_file(temp_path, "experiments", file_path)
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        logger.info(f"💾 Training results saved to: {saved_path}")
        return True, saved_path
        
    except Exception as e:
        logger.error(f"❌ Failed to save training results: {str(e)}")
        return False, str(e)


@task(name="create_model_monitoring_setup", retries=1)
def create_model_monitoring_setup_task(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Create monitoring recommendations and setup for the trained model.
    
    Args:
        results: Training results dictionary.
        
    Returns:
        Tuple of (success, monitoring_setup).
    """
    logger = get_run_logger()
    logger.info("Creating model monitoring setup")
    
    try:
        best_model = results.get("best_model", {})
        
        monitoring_setup = {
            "model_name": "monthly_win_probability_model",
            "monitoring_metrics": {
                "performance_metrics": {
                    "roc_auc_threshold": max(0.75, best_model.get("roc_auc", 0.8) * 0.95),
                    "brier_score_threshold": min(0.25, best_model.get("brier_score", 0.2) * 1.1),
                    "accuracy_threshold": max(0.7, best_model.get("accuracy", 0.75) * 0.95)
                },
                "data_drift_monitoring": {
                    "feature_count": results.get("dataset_info", {}).get("features", 0),
                    "win_rate_baseline": results.get("dataset_info", {}).get("win_rate", 0.3),
                    "monitoring_frequency": "weekly"
                }
            },
            "alerts": {
                "performance_degradation": "ROC AUC drops below threshold",
                "data_drift": "Feature distributions change significantly",
                "prediction_drift": "Win rate predictions deviate from historical patterns"
            },
            "business_metrics": {
                "revenue_forecast_accuracy": "Track actual vs predicted revenue",
                "high_probability_deal_conversion": "Monitor deals with >60% win probability",
                "monthly_pipeline_performance": "Compare predictions to actual monthly results"
            },
            "recommendations": [
                "Update predictions weekly for active pipeline",
                "Focus on opportunities with >60% win probability",
                "Review low-probability deals for potential intervention",
                "Use expected revenue for sales forecasting",
                "Retrain model quarterly or when performance degrades"
            ]
        }
        
        logger.info("✅ Model monitoring setup created")
        logger.info(f"📊 ROC AUC threshold: {monitoring_setup['monitoring_metrics']['performance_metrics']['roc_auc_threshold']:.3f}")
        logger.info(f"🎯 Brier score threshold: {monitoring_setup['monitoring_metrics']['performance_metrics']['brier_score_threshold']:.3f}")
        
        return True, monitoring_setup
        
    except Exception as e:
        logger.error(f"❌ Failed to create monitoring setup: {str(e)}")
        return False, {"error": str(e)}


@flow(name="monthly_win_probability_training_flow", log_prints=True)
def monthly_win_probability_training_flow(snapshot_month: str):
    """Main flow for training monthly win probability prediction models.
    
    Args:
        snapshot_month: Month snapshot to use for training (e.g., "2017-05").
        
    Returns:
        Training pipeline results.
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Monthly Win Probability Training Pipeline")
    logger.info(f"📅 Training snapshot: {snapshot_month}")
    
    # Get configuration
    config = get_config()
    
    # Task 1: Load features data
    load_success, df, load_metadata = load_features_data_task(config, snapshot_month)
    
    if not load_success:
        logger.error(f"Pipeline failed at load step: {load_metadata.get('error', 'Unknown error')}")
        return {
            "status": "failed", 
            "step": "load", 
            "error": load_metadata.get('error'),
            "snapshot_month": snapshot_month
        }
    
    logger.info(f"📊 Loaded data: {load_metadata['shape']} samples")
    logger.info(f"🎯 Win rate: {load_metadata.get('win_rate', 'N/A')}")
    
    # Task 2: Validate training data
    validation_passed, validation_results = validate_training_data_task(df)
    
    if not validation_passed:
        logger.error(f"Pipeline failed at validation step")
        return {
            "status": "failed", 
            "step": "validation", 
            "error": validation_results.get('issues', []),
            "snapshot_month": snapshot_month
        }
    
    # Task 3: Train models
    training_success, training_results = train_monthly_win_models_task(df, config)
    
    if not training_success:
        logger.error(f"Pipeline failed at training step: {training_results.get('error', 'Unknown error')}")
        return {
            "status": "failed", 
            "step": "training", 
            "error": training_results.get('error'),
            "snapshot_month": snapshot_month
        }
    
#    # Task 4: Save training results
#    save_success, save_path = save_training_results_task(training_results, config, snapshot_month)
    
#    if not save_success:
#        logger.warning(f"Failed to save training results: {save_path}")
#        # Don't fail the pipeline for this

    # Task 4: Create monitoring setup
    monitoring_success, monitoring_setup = create_model_monitoring_setup_task(training_results)
    
    if not monitoring_success:
        logger.warning(f"Failed to create monitoring setup: {monitoring_setup.get('error', 'Unknown error')}")
        # Don't fail the pipeline for this
    
    # Create final summary
    summary = {
        "status": "success",
        "snapshot_month": snapshot_month,
        "steps_completed": ["load", "validate", "train", "save", "monitor_setup"],
        "data_info": {
            "samples": load_metadata["shape"][0],
            "features": load_metadata["shape"][1],
            "win_rate": load_metadata.get("win_rate"),
            "closed_deals": load_metadata.get("closed_deals")
        },
        "validation": {
            "passed": validation_passed,
            "warnings": validation_results.get("warnings", [])
        },
        "training": {
            "best_model": training_results.get("best_model", {}),
            "model_info": training_results.get("model_info", {}),
            "experiment_name": training_results.get("experiment_name", ""),
            "feature_count": len(training_results.get("feature_list", []))
        },
#        "files": {
#            "training_results": save_path if save_success else None
#        },
        "monitoring": monitoring_setup if monitoring_success else None
    }
    
    logger.info("🎉 Monthly Win Probability Training Pipeline completed successfully!")
    logger.info(f"🏆 Best model: {summary['training']['best_model'].get('name', 'Unknown')}")
    logger.info(f"📊 ROC AUC: {summary['training']['best_model'].get('roc_auc', 0):.4f}")
    logger.info(f"📝 Model registered: {summary['training']['model_info'].get('registered_model_name', 'Unknown')}")
    logger.info(f"🔍 Features used: {summary['training']['feature_count']}")
    logger.info(f"📅 Monitoring setup: {summary.get('monitoring', 'Unknown')}")
    
    return summary


if __name__ == "__main__":
    # Setup logging for local execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = get_config()
    snapshot_month = config.first_snapshot_month
    
    # Run the flow
    result = monthly_win_probability_training_flow(snapshot_month)
    
    if result["status"] == "success":
        print("\n✅ Monthly Win Probability Training Pipeline completed successfully!")
        print(f"📅 Snapshot: {result['snapshot_month']}")
        print(f"📊 Data: {result['data_info']['samples']:,} samples, {result['data_info']['features']} features")
        print(f"🏆 Best Model: {result['training']['best_model'].get('name', 'Unknown')}")
        print(f"🎯 ROC AUC: {result['training']['best_model'].get('roc_auc', 0):.4f}")
        print(f"📝 Model: {result['training']['model_info'].get('registered_model_name', 'Unknown')}")
        
        if result.get("files", {}).get("training_results"):
            print(f"💾 Results saved: {result['files']['training_results']}")
    else:
        print(f"\n❌ Pipeline failed at {result['step']}: {result['error']}")
        exit(1)
