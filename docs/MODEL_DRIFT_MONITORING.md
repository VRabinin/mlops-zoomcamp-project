# Model Drift Monitoring with Evidently AI

This document describes the comprehensive model drift monitoring system implemented for the CRM Win Probability Prediction model using Evidently AI.

## Overview

The monitoring system provides:
- **Reference Data Management**: Baseline datasets for drift comparison
- **Automated Drift Detection**: Statistical tests for data and model drift
- **Real-time Monitoring**: Prefect pipelines for automated monitoring
- **Interactive Dashboard**: Streamlit interface for monitoring insights
- **Alert System**: Configurable alerts for drift detection

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw CRM Data  │ -> │  Feature Pipeline │ -> │ Current Features│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Reference Data  │ <- │ Reference Pipeline│ <- │   ML Model      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         v                                               v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Drift Detection │ -> │ Evidently Reports│ -> │ Streamlit UI    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. CRMDriftMonitor Class

Core monitoring functionality in `src/monitoring/drift_monitor.py`:

- **Reference Data Creation**: Generate baseline datasets with predictions
- **Current Predictions**: Score new data with the latest model
- **Drift Detection**: Statistical comparison using Evidently
- **Report Generation**: HTML and JSON reports for analysis

### 2. Prefect Pipelines

#### Reference Data Creation Pipeline
- **File**: `src/pipelines/run_reference_data_creation.py`
- **Purpose**: Create baseline datasets for drift comparison
- **Schedule**: Manual execution when needed
- **Parameters**: `snapshot_month`, `sample_size`

#### Drift Monitoring Pipeline
- **File**: `src/pipelines/run_drift_monitoring.py`
- **Purpose**: Detect drift between current and reference data
- **Schedule**: Weekly (configurable)
- **Parameters**: `current_month`, `reference_month`

### 3. Streamlit Dashboard

Interactive monitoring interface in `src/monitoring/streamlit_dashboard.py`:

- **Overview Tab**: Key metrics and drift status
- **Controls Tab**: Manual execution of monitoring tasks
- **Reports Tab**: Access to Evidently HTML reports

## Quick Start

### 1. Deploy Monitoring Pipelines

```bash
# Deploy reference data creation flow
make prefect-deploy-reference-creation

# Deploy drift monitoring flow
make prefect-deploy-drift-monitoring
```

### 2. Create Reference Data

```bash
# Create reference data for baseline period
make prefect-run-reference-creation
```

### 3. Run Drift Monitoring

```bash
# Run monitoring for current period
make prefect-run-drift-monitoring
```

### 4. View Results

```bash
# Start Streamlit app
make streamlit-app

# Navigate to "Model Monitoring" tab
```

## Monitoring Metrics

### Data Drift Metrics
- **Dataset Drift**: Overall distribution changes across features
- **Feature Drift**: Individual feature distribution changes
- **Missing Values**: Changes in data completeness
- **Data Quality**: Schema and type consistency

### Model Drift Metrics
- **Prediction Drift**: Changes in model output distribution
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Prediction Confidence**: Distribution of prediction probabilities

### Alert Levels
- **NONE**: No significant drift detected
- **LOW**: Minor drift, monitor trends
- **MEDIUM**: Notable drift, plan retraining
- **HIGH**: Critical drift, immediate action required

## Configuration

### Evidently Settings

The monitoring system is configured for the CRM prediction model:

```python
# Feature columns monitored
feature_columns = [
    'close_value_log', 'close_value_category_encoded',
    'days_since_engage', 'sales_cycle_days',
    'agent_win_rate', 'agent_opportunity_count',
    'product_win_rate', 'product_popularity',
    # ... additional features
]

# Column mapping for Evidently
column_mapping = ColumnMapping(
    target='is_won',
    prediction='prediction',
    numerical_features=[...],
    categorical_features=[...]
)
```

### Drift Thresholds

```python
# Alert level thresholds
def _determine_alert_level(dataset_drift, prediction_drift, num_drifted_columns):
    if dataset_drift or prediction_drift > 0.2:
        return "HIGH"
    elif prediction_drift > 0.1 or num_drifted_columns > 5:
        return "MEDIUM"
    elif num_drifted_columns > 2:
        return "LOW"
    else:
        return "NONE"
```

## Makefile Commands

### Core Monitoring
```bash
make prefect-deploy-reference-creation  # Deploy reference data flow
make prefect-deploy-drift-monitoring    # Deploy monitoring flow
make prefect-run-reference-creation     # Create reference data
make prefect-run-drift-monitoring       # Run drift monitoring
make monitor-demo                       # Complete monitoring demo
```

### Utilities
```bash
make prefect-deployments               # List all deployments
make prefect-flows                     # List flow runs
make prefect-ui                        # Open Prefect UI
```

## Output Files

### Reference Data
- **Location**: `features/reference_data_{month}.csv`
- **Content**: Sampled features with model predictions
- **Usage**: Baseline for drift comparison

### Current Predictions
- **Location**: `features/current_predictions_{month}.csv`
- **Content**: Current features with fresh predictions
- **Usage**: Comparison against reference

### Monitoring Results
- **Location**: `monitoring_results/monitoring_results_{month}_{timestamp}.json`
- **Content**: Drift metrics and test results
- **Usage**: Historical tracking and alerting

### Evidently Reports
- **Location**: `monitoring_reports/`
- **Files**: 
  - `drift_report_{ref}_{current}_{timestamp}.html`
  - `performance_report_{ref}_{current}_{timestamp}.html`
- **Usage**: Detailed analysis and debugging

## Integration with Streamlit

The monitoring dashboard is integrated into the main Streamlit application:

1. **Tab Integration**: Added "Model Monitoring" tab
2. **Real-time Updates**: Automatic refresh of monitoring status
3. **Interactive Controls**: Run monitoring tasks from UI
4. **Visual Reports**: Display key metrics and trends

## Best Practices

### 1. Reference Data Management
- Update reference data quarterly or when major model changes occur
- Maintain multiple reference periods for seasonal comparison
- Balance sample size (1000+ samples recommended)

### 2. Monitoring Frequency
- **Real-time**: For high-volume prediction systems
- **Daily**: For business-critical models
- **Weekly**: For standard monitoring (current default)
- **Monthly**: For stable models with low change rates

### 3. Alert Response
- **HIGH**: Stop predictions, investigate immediately, retrain model
- **MEDIUM**: Increase monitoring frequency, plan retraining
- **LOW**: Document trends, continue monitoring
- **NONE**: Standard monitoring schedule

### 4. Performance Optimization
- Use sampling for large datasets (> 10K records)
- Store processed reports for faster dashboard loading
- Implement incremental monitoring for continuous data streams

## Troubleshooting

### Common Issues

1. **Missing Evidently Dependencies**
   ```bash
   pip install evidently==0.7.11
   ```

2. **Storage Access Errors**
   - Check MinIO connectivity
   - Verify bucket permissions
   - Ensure storage configuration is correct

3. **Model Loading Failures**
   - Verify MLflow connectivity
   - Check model registry
   - Ensure model version compatibility

4. **Prefect Flow Errors**
   - Check Prefect server status
   - Verify work pool configuration
   - Review flow logs in Prefect UI

### Debug Commands

```bash
# Check system status
make prefect-deployments

# View flow runs
make prefect-flows

# Check MLflow models
curl http://localhost:5005/api/2.0/mlflow/registered-models/list

# Test MinIO connectivity
curl http://localhost:9000/minio/health/live
```

## Future Enhancements

### Planned Features
- **Automated Retraining**: Trigger model retraining on high drift
- **Slack/Email Alerts**: Integration with notification systems
- **Custom Metrics**: Business-specific drift indicators
- **A/B Testing**: Compare model versions automatically
- **Seasonal Adjustment**: Account for known seasonal patterns

### Advanced Monitoring
- **Concept Drift**: Detect changes in target variable relationship
- **Covariate Shift**: Monitor input feature distribution changes
- **Prior Probability Shift**: Track changes in class distribution
- **Performance Degradation**: Real-time accuracy monitoring

## Resources

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Prefect Documentation](https://docs.prefect.io/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Streamlit Components](https://docs.streamlit.io/)
