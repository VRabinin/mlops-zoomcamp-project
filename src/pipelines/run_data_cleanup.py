"""
CRM Data Cleanup Flow using Prefect.

This pipeline cleans up processed and feature CSV files while preserving
the original Kaggle source data files (accounts.csv, products.csv, etc.).
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import get_config
from src.utils.storage import StorageManager


@task(name="identify_cleanup_files", retries=1)
def identify_cleanup_files_task(config_dict: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
    """Identify files to clean up, excluding Kaggle source files.
    
    Args:
        config_dict: Configuration dictionary.
        
    Returns:
        Tuple of (success, file_categories_dict).
    """
    logger = get_run_logger()
    logger.info("🔍 Identifying files for cleanup...")
    
    try:
        # Recreate config object from dict
        from src.config.config import Config
        config = Config.from_dict(config_dict)
        
        storage = StorageManager(config)
        
        # Files to preserve (Kaggle source files)
        kaggle_source_files = {
            'accounts.csv',
            'products.csv', 
            'sales_pipeline.csv',
            'sales_teams.csv',
            'data_dictionary.csv'
        }
        
        # Categories of files to clean
        cleanup_categories = {
            'processed_files': [],
            'feature_files': [], 
            'monitoring_files': [],
            'prediction_files': [],
            'temp_files': []
        }
        
        # Check raw directory for non-source files
        logger.info("📁 Scanning raw directory...")
        try:
            raw_files = storage.list_files('raw', '*.csv')
            for file_path in raw_files:
                filename = Path(file_path).name
                # If it's not a Kaggle source file and looks like a processed file
                if filename not in kaggle_source_files:
                    if 'enhanced' in filename or 'processed' in filename:
                        cleanup_categories['processed_files'].append(f"raw/{filename}")
                        logger.info(f"  📋 Found processed file in raw: {filename}")
        except Exception as e:
            logger.warning(f"Could not scan raw directory: {e}")
        
        # Check processed directory
        logger.info("📁 Scanning processed directory...")
        try:
            processed_files = storage.list_files('processed', '*.csv')
            for file_path in processed_files:
                filename = Path(file_path).name
                cleanup_categories['processed_files'].append(f"processed/{filename}")
                logger.info(f"  📋 Found processed file: {filename}")
        except Exception as e:
            logger.warning(f"Could not scan processed directory: {e}")
        
        # Check features directory
        logger.info("📁 Scanning features directory...")
        try:
            feature_files = storage.list_files('features', '*.csv')
            for file_path in feature_files:
                filename = Path(file_path).name
                if 'reference_data' in filename:
                    cleanup_categories['monitoring_files'].append(f"features/{filename}")
                    logger.info(f"  📋 Found monitoring file: {filename}")
                elif 'prediction' in filename or 'current_prediction' in filename:
                    cleanup_categories['prediction_files'].append(f"features/{filename}")
                    logger.info(f"  📋 Found prediction file: {filename}")
                else:
                    cleanup_categories['feature_files'].append(f"features/{filename}")
                    logger.info(f"  📋 Found feature file: {filename}")
        except Exception as e:
            logger.warning(f"Could not scan features directory: {e}")
        
        # Check monitoring directory
        logger.info("📁 Scanning monitoring directory...")
        try:
            monitoring_files = storage.list_files('monitoring', '*.csv')
            for file_path in monitoring_files:
                filename = Path(file_path).name
                cleanup_categories['monitoring_files'].append(f"monitoring/{filename}")
                logger.info(f"  📋 Found monitoring file: {filename}")
        except Exception as e:
            logger.warning(f"Could not scan monitoring directory: {e}")
        
        # Check for temp files in data_temp if it exists
        if hasattr(config.storage.paths, 'temp_data'):
            logger.info("📁 Scanning temp directory...")
            try:
                temp_files = storage.list_files('temp', '*.csv')
                for file_path in temp_files:
                    filename = Path(file_path).name
                    cleanup_categories['temp_files'].append(f"temp/{filename}")
                    logger.info(f"  📋 Found temp file: {filename}")
            except Exception as e:
                logger.warning(f"Could not scan temp directory: {e}")
        
        # Count totals
        total_files = sum(len(files) for files in cleanup_categories.values())
        logger.info(f"✅ File identification complete:")
        logger.info(f"  📊 Processed files: {len(cleanup_categories['processed_files'])}")
        logger.info(f"  🔧 Feature files: {len(cleanup_categories['feature_files'])}")
        logger.info(f"  📈 Monitoring files: {len(cleanup_categories['monitoring_files'])}")
        logger.info(f"  🎯 Prediction files: {len(cleanup_categories['prediction_files'])}")
        logger.info(f"  🗂️ Temp files: {len(cleanup_categories['temp_files'])}")
        logger.info(f"  🔢 Total files to clean: {total_files}")
        
        return True, cleanup_categories
        
    except Exception as e:
        error_msg = f"Failed to identify cleanup files: {str(e)}"
        logger.error(error_msg)
        return False, {}


@task(name="cleanup_file_category", retries=1)
def cleanup_file_category_task(
    config_dict: Dict[str, Any], 
    category_name: str, 
    file_list: List[str],
    dry_run: bool = False
) -> Tuple[bool, str]:
    """Clean up files in a specific category.
    
    Args:
        config_dict: Configuration dictionary.
        category_name: Name of the file category.
        file_list: List of file paths to clean.
        dry_run: If True, only log what would be deleted without actually deleting.
        
    Returns:
        Tuple of (success, message).
    """
    logger = get_run_logger()
    
    if not file_list:
        logger.info(f"📂 No files to clean in category: {category_name}")
        return True, f"No files in {category_name}"
    
    logger.info(f"🧹 Cleaning {category_name} ({len(file_list)} files)...")
    
    try:
        # Recreate config object from dict
        from src.config.config import Config
        config = Config.from_dict(config_dict)
        
        storage = StorageManager(config)
        
        cleaned_count = 0
        failed_count = 0
        
        for file_path in file_list:
            try:
                # Extract data type and filename from path
                parts = file_path.split('/', 1)
                if len(parts) == 2:
                    data_type, filename = parts
                else:
                    data_type = 'raw'
                    filename = file_path
                
                if dry_run:
                    logger.info(f"  🔍 Would delete: {file_path}")
                    cleaned_count += 1
                else:
                    # Delete the file
                    success = storage.delete_file(data_type, filename)
                    if success:
                        logger.info(f"  ✅ Deleted: {file_path}")
                        cleaned_count += 1
                    else:
                        logger.warning(f"  ❌ Failed to delete: {file_path}")
                        failed_count += 1
                        
            except Exception as e:
                logger.error(f"  ❌ Error deleting {file_path}: {e}")
                failed_count += 1
        
        action = "Would clean" if dry_run else "Cleaned"
        message = f"{action} {cleaned_count} files in {category_name}"
        if failed_count > 0:
            message += f" ({failed_count} failed)"
        
        logger.info(f"✅ {message}")
        return True, message
        
    except Exception as e:
        error_msg = f"Failed to clean {category_name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


@task(name="cleanup_summary", retries=0)
def cleanup_summary_task(cleanup_results: List[Tuple[str, bool, str]]) -> Dict[str, Any]:
    """Generate cleanup summary.
    
    Args:
        cleanup_results: List of (category, success, message) tuples.
        
    Returns:
        Summary dictionary.
    """
    logger = get_run_logger()
    logger.info("📋 Generating cleanup summary...")
    
    summary = {
        'total_categories': len(cleanup_results),
        'successful_categories': 0,
        'failed_categories': 0,
        'details': []
    }
    
    for category, success, message in cleanup_results:
        summary['details'].append({
            'category': category,
            'success': success,
            'message': message
        })
        
        if success:
            summary['successful_categories'] += 1
        else:
            summary['failed_categories'] += 1
    
    logger.info("✅ Cleanup Summary:")
    logger.info(f"  📊 Total categories processed: {summary['total_categories']}")
    logger.info(f"  ✅ Successful: {summary['successful_categories']}")
    logger.info(f"  ❌ Failed: {summary['failed_categories']}")
    
    for detail in summary['details']:
        status = "✅" if detail['success'] else "❌"
        logger.info(f"  {status} {detail['category']}: {detail['message']}")
    
    return summary


@flow(name="crm_data_cleanup_flow", retries=0)
def crm_data_cleanup_flow(
    dry_run: bool = True,
    include_processed: bool = True,
    include_features: bool = True,
    include_monitoring: bool = True,
    include_predictions: bool = True,
    include_temp: bool = True
) -> Dict[str, Any]:
    """
    CRM Data Cleanup Flow.
    
    Cleans up generated CSV files while preserving Kaggle source data.
    
    Args:
        dry_run: If True, only shows what would be deleted without actually deleting.
        include_processed: Whether to clean processed data files.
        include_features: Whether to clean feature files.
        include_monitoring: Whether to clean monitoring files.
        include_predictions: Whether to clean prediction files.
        include_temp: Whether to clean temporary files.
        
    Returns:
        Dictionary with cleanup results and summary.
    """
    logger = get_run_logger()
    logger.info("🚀 Starting CRM Data Cleanup Flow")
    logger.info(f"🔍 Mode: {'DRY RUN' if dry_run else 'ACTUAL CLEANUP'}")
    
    try:
        # Get configuration
        config = get_config()
        config_dict = config.to_dict()
        
        # Identify files to cleanup
        success, file_categories = identify_cleanup_files_task(config_dict)
        
        if not success:
            logger.error("❌ Failed to identify cleanup files")
            return {"success": False, "error": "File identification failed"}
        
        # Prepare cleanup tasks based on parameters
        cleanup_tasks = []
        
        if include_processed and file_categories.get('processed_files'):
            cleanup_tasks.append(('processed_files', file_categories['processed_files']))
        
        if include_features and file_categories.get('feature_files'):
            cleanup_tasks.append(('feature_files', file_categories['feature_files']))
        
        if include_monitoring and file_categories.get('monitoring_files'):
            cleanup_tasks.append(('monitoring_files', file_categories['monitoring_files']))
        
        if include_predictions and file_categories.get('prediction_files'):
            cleanup_tasks.append(('prediction_files', file_categories['prediction_files']))
        
        if include_temp and file_categories.get('temp_files'):
            cleanup_tasks.append(('temp_files', file_categories['temp_files']))
        
        if not cleanup_tasks:
            logger.info("ℹ️ No file categories selected for cleanup")
            return {
                "success": True,
                "message": "No file categories selected for cleanup",
                "file_categories": file_categories
            }
        
        # Execute cleanup tasks
        cleanup_results = []
        for category_name, file_list in cleanup_tasks:
            success, message = cleanup_file_category_task(config_dict, category_name, file_list, dry_run)
            cleanup_results.append((category_name, success, message))
        
        # Generate summary
        summary = cleanup_summary_task(cleanup_results)
        
        # Final result
        result = {
            "success": True,
            "dry_run": dry_run,
            "file_categories": file_categories,
            "cleanup_results": cleanup_results,
            "summary": summary
        }
        
        logger.info("✅ CRM Data Cleanup Flow completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"CRM Data Cleanup Flow failed: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


if __name__ == "__main__":
    # Run the flow locally for testing
    result = crm_data_cleanup_flow(dry_run=True)
    print("Flow result:", result)
