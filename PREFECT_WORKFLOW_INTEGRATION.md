# Prefect Workflow Integration - Refactoring Summary

## Overview

The Streamlit application has been successfully refactored to trigger Prefect flows on the Prefect server instead of running them locally. This provides better workflow orchestration, monitoring, and scalability.

## Key Changes

### 1. New Prefect Client Module (`src/utils/prefect_client.py`)

Created a comprehensive Prefect client interface that provides:

- **Server Health Checking**: Verify Prefect server connectivity
- **Deployment Management**: List and manage available deployments  
- **Flow Triggering**: Trigger flows via API or CLI with parameters
- **Flow Monitoring**: Get flow run status and recent runs
- **Flow Control**: Cancel running flows

Key Features:
- Async/sync compatibility for Streamlit integration
- Fallback to CLI when API has issues
- Proper error handling and logging
- Support for flow parameters and metadata

### 2. Enhanced Monitoring Dashboard (`src/monitoring/streamlit_dashboard.py`)

Updated the monitoring controls to use Prefect server:

- **Server-Based Flow Triggering**: All monitoring flows now run on Prefect server
- **Real-time Status**: Monitor flow runs with live status updates
- **Parameter Support**: Proper parameter handling for reference data and drift monitoring
- **Flow Management**: Cancel running flows directly from the UI

### 3. New Workflow Control Tab in Streamlit App

Added a dedicated "🚀 Workflow Control" tab with three sections:

#### 🚀 Run Pipelines
- **Data Acquisition**: Download CRM data with snapshot month parameter
- **Data Ingestion**: Process data with month parameter  
- **Model Training**: Train models with training month parameter
- **Reference Data Creation**: Create monitoring baselines with period and sample size
- **Drift Monitoring**: Run drift analysis with current and reference periods

#### 📊 Flow Status  
- **Recent Flow Runs**: Table view of recent executions
- **Running Flows**: Detailed view of currently executing flows
- **Flow Control**: Cancel running flows
- **Real-time Updates**: Refresh button for latest status

#### ⚙️ Deployments
- **Deployment Info**: View all available deployments
- **Quick Run**: Trigger any deployment with one click
- **Deployment Status**: See deployment health and configuration

### 4. Parameter Management

Each flow type now has proper parameter handling:

- **Data Acquisition**: `snapshot_month` (required)
- **Data Ingestion**: `snapshot_month` (required)  
- **Model Training**: `current_month` (required)
- **Reference Data**: `reference_period`, `sample_size`
- **Drift Monitoring**: `current_month`, `reference_period`

## Usage Examples

### Starting the Enhanced Streamlit App

```bash
# Ensure Prefect server is running
make application-start

# Deploy flows to Prefect server
make prefect-deploy

# Start Streamlit with workflow integration
make streamlit-dev
```

### Triggering Flows Programmatically

```python
from src.utils.prefect_client import PrefectFlowManager, CRMDeployments

# Initialize client
manager = PrefectFlowManager()

# Check server health
is_healthy, message = manager.check_server_health()

# List deployments
deployments = manager.get_deployments_sync()

# Trigger data acquisition
success, flow_run_id, message = manager.trigger_deployment_sync(
    CRMDeployments.DATA_ACQUISITION, 
    {"snapshot_month": "2017-06"}
)

# Monitor flow run
status_success, flow_info = manager.get_flow_run_status_sync(flow_run_id)
```

### Available Deployment Constants

```python
from src.utils.prefect_client import CRMDeployments

# Available deployment names
CRMDeployments.DATA_ACQUISITION        # "crm-data-acquisition"
CRMDeployments.DATA_INGESTION          # "crm-data-ingestion"  
CRMDeployments.MONTHLY_TRAINING        # "monthly-win-probability-training"
CRMDeployments.REFERENCE_DATA          # "reference-data-creation"
CRMDeployments.DRIFT_MONITORING        # "model-drift-monitoring"
```

## Technical Implementation Details

### Async/Sync Compatibility

The Prefect client handles Streamlit's synchronous nature while using Prefect's async API:

```python
def run_async(self, coro):
    """Helper to run async functions in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Handle nested event loops
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)
```

### CLI Fallback Mechanism

When the async API fails (common with some Prefect versions), the client falls back to CLI:

```python
def trigger_deployment_sync(self, deployment_name: str, parameters: Dict[str, Any] = None):
    try:
        # Try async API first
        return self.run_async(self.trigger_deployment(deployment_name, parameters))
    except Exception as e:
        logger.warning(f"Async trigger failed: {e}, trying CLI method")
        # Fallback to CLI
        return self.trigger_deployment_cli(deployment_name, parameters)
```

### Error Handling and Logging

Comprehensive error handling throughout:

- Server connectivity issues
- Deployment not found errors  
- Parameter validation errors
- Flow execution failures
- Timeout handling

## Benefits of the Refactoring

### 1. **Better Orchestration**
- Centralized workflow management through Prefect server
- Proper dependency handling and scheduling
- Resource management and scaling

### 2. **Enhanced Monitoring** 
- Real-time flow status in Prefect UI
- Historical run tracking
- Performance metrics and logs

### 3. **Improved Reliability**
- Retry mechanisms for failed flows
- Better error handling and recovery
- Resource isolation between flows

### 4. **Developer Experience**
- Clean separation between UI and workflow logic
- Easier testing and debugging
- Better code maintainability

### 5. **Production Readiness**
- Scalable to multiple workers
- Remote execution capabilities
- Proper secrets and configuration management

## Troubleshooting

### Common Issues

1. **"No deployments found"**
   ```bash
   make prefect-deploy  # Deploy flows first
   ```

2. **"Prefect server not accessible"**
   ```bash
   make application-start  # Start infrastructure
   ```

3. **"Parameter validation failed"**
   - Check flow requirements in deployment logs
   - Ensure all required parameters are provided

4. **"Flow runs stuck in PENDING"**
   ```bash
   docker logs mlops-prefect-setup  # Check worker status
   ```

### Verification Commands

```bash
# Check Prefect status
make prefect-status-all

# List deployments  
make prefect-deployments

# View recent flows
make prefect-flows

# Open Prefect UI
make prefect-ui
```

## Future Enhancements

### Planned Improvements

1. **Scheduled Workflows**: Add cron-based scheduling
2. **Flow Chaining**: Automatic dependency management between flows
3. **Advanced Parameters**: Dynamic parameter validation
4. **Notification Integration**: Email/Slack alerts for flow status
5. **Resource Management**: CPU/memory limits per flow type

### Integration Opportunities

- **CI/CD Integration**: Trigger flows from GitHub Actions
- **API Endpoints**: REST API for external workflow triggering
- **Event-Driven Flows**: React to data changes automatically
- **Multi-Environment**: Dev/staging/prod workflow environments

This refactoring establishes a solid foundation for enterprise-grade MLOps workflow management while maintaining the user-friendly Streamlit interface.
