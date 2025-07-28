"""
Prefect Flow for Model Drift Monitoring

This flow performs comprehensive drift monitoring by comparing current
month predictions against reference data and generating alerts.
"""

import logging
from typing import Dict, Any, Tuple, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

from src.config.config import get_config, Config
from src.monitoring.drift_monitor import CRMDriftMonitor


@task(name="generate_current_predictions", retries=2)
def generate_current_predictions_task(config: Config, current_month: str) -> Tuple[bool, str]:
    """
    Generate predictions for current month data.
    
    Args:
        config: Configuration object
        current_month: Current month snapshot (e.g., "2017-06")
        
    Returns:
        Tuple of (success, message)
    """
    logger = get_run_logger()
    logger.info(f"🎯 Generating predictions for current month: {current_month}")
    
    try:
        # Initialize drift monitor
        drift_monitor = CRMDriftMonitor(config)
        
        # Generate current predictions
        success, message = drift_monitor.generate_current_predictions(current_month)
        
        if success:
            logger.info(f"✅ {message}")
        else:
            logger.error(f"❌ {message}")
            
        return success, message
        
    except Exception as e:
        error_msg = f"Failed to generate current predictions: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


@task(name="run_data_quality_tests", retries=1)
def run_data_quality_tests_task(config: Config, current_month: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Run data quality tests on current data.
    
    Args:
        config: Configuration object
        current_month: Current month snapshot
        
    Returns:
        Tuple of (all_tests_passed, test_results)
    """
    logger = get_run_logger()
    logger.info(f"🔍 Running data quality tests for: {current_month}")
    
    try:
        # Initialize drift monitor
        drift_monitor = CRMDriftMonitor(config)
        
        # Run data quality tests
        all_passed, test_results = drift_monitor.run_data_quality_tests(current_month)
        
        if all_passed:
            logger.info(f"✅ All data quality tests passed")
        else:
            logger.warning(f"⚠️ Some data quality tests failed")
            failed_tests = [test for test in test_results.get('test_details', []) 
                          if test.get('status') == 'FAIL']
            for test in failed_tests:
                logger.warning(f"Failed test: {test.get('name', 'Unknown')}")
        
        return all_passed, test_results
        
    except Exception as e:
        error_msg = f"Data quality tests failed: {str(e)}"
        logger.error(error_msg)
        return False, {"error": error_msg}


@task(name="detect_model_drift", retries=2)
def detect_model_drift_task(config: Config, reference_month: str, current_month: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect model and data drift between reference and current data.
    
    Args:
        config: Configuration object
        reference_month: Reference month snapshot
        current_month: Current month snapshot
        
    Returns:
        Tuple of (success, drift_results)
    """
    logger = get_run_logger()
    logger.info(f"🕵️ Detecting drift: {reference_month} vs {current_month}")
    
    try:
        # Initialize drift monitor
        drift_monitor = CRMDriftMonitor(config)
        
        # Detect drift
        success, drift_results = drift_monitor.detect_drift(reference_month, current_month)
        
        if success:
            alert_level = drift_results.get('alert_level', 'NONE')
            drift_detected = drift_results.get('drift_detected', False)
            
            if drift_detected:
                logger.warning(f"🚨 Drift detected! Alert level: {alert_level}")
                if drift_results.get('dataset_drift'):
                    logger.warning("📊 Dataset drift detected")
                if drift_results.get('prediction_drift'):
                    logger.warning(f"🎯 Prediction drift score: {drift_results['prediction_drift']:.3f}")
                logger.warning(f"📈 Number of drifted columns: {drift_results.get('num_drifted_columns', 0)}")
            else:
                logger.info("✅ No significant drift detected")
                
            logger.info(f"📊 Drift analysis completed successfully")
        else:
            logger.error(f"❌ Drift detection failed")
            
        return success, drift_results
        
    except Exception as e:
        error_msg = f"Drift detection failed: {str(e)}"
        logger.error(error_msg)
        return False, {"error": error_msg}


@task(name="save_monitoring_results", retries=1)
def save_monitoring_results_task(
    config: Config, 
    current_month: str,
    drift_results: Dict[str, Any],
    quality_results: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Save monitoring results to storage for historical tracking.
    
    Args:
        config: Configuration object
        current_month: Current month snapshot
        drift_results: Results from drift detection
        quality_results: Results from data quality tests
        
    Returns:
        Tuple of (success, message)
    """
    logger = get_run_logger()
    logger.info(f"💾 Saving monitoring results for: {current_month}")
    
    try:
        import json
        import datetime
        from src.utils.storage import StorageManager
        
        storage = StorageManager(config)
        
        # Combine results
        monitoring_summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "current_month": current_month,
            "drift_detection": drift_results,
            "data_quality": quality_results,
            "overall_status": "healthy" if not drift_results.get('drift_detected', False) and quality_results.get('all_tests_passed', False) else "attention_needed"
        }
        
        # Save as JSON
        results_file = f"monitoring_results_{current_month}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to JSON string and save
        results_json = json.dumps(monitoring_summary, indent=2, default=str)
        
        # Save to monitoring results path
        from io import StringIO
        results_buffer = StringIO(results_json)
        storage.upload_file(results_buffer, "monitoring_results", results_file)
        
        logger.info(f"✅ Monitoring results saved: {results_file}")
        
        return True, f"Monitoring results saved: {results_file}"
        
    except Exception as e:
        error_msg = f"Failed to save monitoring results: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


@task(name="send_drift_alerts", retries=1)
def send_drift_alerts_task(
    config: Config,
    current_month: str,
    drift_results: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Send alerts if significant drift is detected.
    
    Args:
        config: Configuration object
        current_month: Current month snapshot
        drift_results: Results from drift detection
        
    Returns:
        Tuple of (success, message)
    """
    logger = get_run_logger()
    
    alert_level = drift_results.get('alert_level', 'NONE')
    drift_detected = drift_results.get('drift_detected', False)
    
    if not drift_detected:
        logger.info("ℹ️ No alerts to send - no significant drift detected")
        return True, "No alerts needed"
    
    logger.warning(f"🚨 Sending drift alert for {current_month} - Level: {alert_level}")
    
    try:
        # Create alert message
        alert_message = f"""
        🚨 MODEL DRIFT ALERT - {alert_level} SEVERITY
        
        Month: {current_month}
        Timestamp: {drift_results.get('timestamp')}
        
        Drift Summary:
        - Dataset Drift: {drift_results.get('dataset_drift', 'Unknown')}
        - Prediction Drift Score: {drift_results.get('prediction_drift', 'N/A')}
        - Drifted Columns: {drift_results.get('num_drifted_columns', 0)}
        - Missing Values: {drift_results.get('missing_values_share', 0):.1%}
        
        Model Performance:
        - Accuracy: {drift_results.get('accuracy', 'N/A')}
        - F1 Score: {drift_results.get('f1_score', 'N/A')}
        - ROC AUC: {drift_results.get('roc_auc', 'N/A')}
        
        Recommended Actions:
        - Review model performance metrics
        - Consider model retraining
        - Investigate data pipeline changes
        - Check feature engineering logic
        """
        
        # For now, just log the alert (in production, you would send to Slack, email, etc.)
        logger.warning(alert_message)
        
        # TODO: Implement actual alert mechanisms
        # - Send to Slack webhook
        # - Send email notifications
        # - Create JIRA tickets
        # - Post to monitoring dashboard
        
        return True, f"Alert sent for {alert_level} level drift"
        
    except Exception as e:
        error_msg = f"Failed to send drift alerts: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


@flow(name="drift_monitoring_flow", retries=1)
def drift_monitoring_flow(
    current_month: str,
    reference_month: str = ''
) -> Dict[str, Any]:
    """
    Main flow for model drift monitoring.
    
    Args:
        current_month: Current month snapshot to monitor (e.g., "2017-06")
        reference_month: Reference month for comparison (defaults to first snapshot)
        
    Returns:
        Dictionary with flow execution results
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Model Drift Monitoring Flow")
    logger.info(f"📅 Current month: {current_month}")
    
    # Get configuration
    config = get_config()
    
    # Use default reference month if not provided
    if reference_month == '':
        #Take previous month as reference. For this, we assume current_month is in "YYYY-MM" format
        from datetime import datetime, timedelta
        current_date = datetime.strptime(current_month, "%Y-%m")
        reference_date = current_date - timedelta(days=10)  # Roughly one month back
        reference_month = reference_date.strftime("%Y-%m")
        logger.info(f"Using default reference month: {reference_month}")
    else:
        logger.info(f"Using provided reference month: {reference_month}")
    
    logger.info(f"📊 Comparing against reference: {reference_month}")
    
    # Step 1: Generate predictions for current month
    pred_success, pred_message = generate_current_predictions_task(config, current_month)
    
    if not pred_success:
        logger.error(f"❌ Failed to generate predictions: {pred_message}")
        return {
            "status": "failed",
            "error": pred_message,
            "current_month": current_month,
            "reference_month": reference_month
        }
    
    # Step 2: Run data quality tests
    quality_passed, quality_results = run_data_quality_tests_task(config, current_month)
    
    # Step 3: Detect drift
    drift_success, drift_results = detect_model_drift_task(config, reference_month, current_month)
    
    if not drift_success:
        logger.error(f"❌ Drift detection failed")
        return {
            "status": "failed",
            "error": "Drift detection failed",
            "drift_results": drift_results,
            "current_month": current_month,
            "reference_month": reference_month
        }
    
    # Step 4: Save monitoring results
    save_success, save_message = save_monitoring_results_task(
        config, current_month, drift_results, quality_results
    )
    
    # Step 5: Send alerts if needed
    alert_success, alert_message = send_drift_alerts_task(config, current_month, drift_results)
    
    # Determine overall status
    overall_success = pred_success and drift_success and save_success and alert_success
    drift_detected = drift_results.get('drift_detected', False)
    alert_level = drift_results.get('alert_level', 'NONE')
    
    if overall_success:
        if drift_detected:
            logger.warning(f"⚠️ Monitoring completed with drift detected (Level: {alert_level})")
        else:
            logger.info("✅ Monitoring completed successfully - No drift detected")
    else:
        logger.error("❌ Some monitoring steps failed")
    
    return {
        "status": "success" if overall_success else "partial_failure",
        "current_month": current_month,
        "reference_month": reference_month,
        "predictions_generated": pred_success,
        "drift_detection": {
            "success": drift_success,
            "drift_detected": drift_detected,
            "alert_level": alert_level,
            "results": drift_results
        },
        "data_quality": {
            "all_passed": quality_passed,
            "results": quality_results
        },
        "alerts_sent": alert_success,
        "results_saved": save_success,
        "messages": {
            "predictions": pred_message,
            "alerts": alert_message,
            "save": save_message
        }
    }


if __name__ == "__main__":
    # For local testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_drift_monitoring.py <current_month> [reference_month]")
        sys.exit(1)
    
    current_month = sys.argv[1]
    reference_month = sys.argv[2] if len(sys.argv) > 2 else ''
    
    result = drift_monitoring_flow(current_month, reference_month)
    print(f"Flow result: {result}")
