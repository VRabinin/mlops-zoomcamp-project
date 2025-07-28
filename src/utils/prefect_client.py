"""
Prefect Client Utilities for triggering flows on Prefect Server.

This module provides utilities for interacting with the Prefect API
to trigger deployments, monitor flow runs, and manage workflow execution
from the Streamlit application.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import json

from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.filters import FlowRunFilter, DeploymentFilter
from prefect.client.schemas.sorting import FlowRunSort
from prefect.exceptions import PrefectHTTPStatusError

logger = logging.getLogger(__name__)


class PrefectFlowManager:
    """Manager for interacting with Prefect server to trigger and monitor flows."""
    
    def __init__(self, api_url: str = None):
        """Initialize the Prefect client manager.
        
        Args:
            api_url: Prefect API URL. Defaults to environment variable or localhost.
        """
        self.api_url = api_url or os.getenv('PREFECT_API_URL', 'http://localhost:4200/api')
        #self.api_url = api_url or os.getenv('PREFECT_API_URL', 'http://mlops-prefect-server:4200/api')
        self.base_url = self.api_url.replace('/api', '')
        
        # Set environment variable for the Prefect client
        os.environ['PREFECT_API_URL'] = self.api_url
        
        logger.info(f"Initialized Prefect client with API URL: {self.api_url}")
    
    def check_server_health(self) -> Tuple[bool, str]:
        """Check if Prefect server is accessible.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True, "✅ Prefect server is healthy"
            else:
                return False, f"❌ Prefect server returned status {response.status_code}"
        except requests.RequestException as e:
            return False, f"❌ Cannot connect to Prefect server: {str(e)}"
    
    async def get_deployments(self) -> List[Dict[str, Any]]:
        """Get list of available deployments.
        
        Returns:
            List of deployment information dictionaries.
        """
        try:
            async with PrefectClient(api=self.api_url) as client:
                deployments = await client.read_deployments()
                result = []
                for deployment in deployments:
                    # Get flow information
                    flow = await client.read_flow(deployment.flow_id)
                    
                    result.append({
                        'id': str(deployment.id),
                        'name': deployment.name,
                        'flow_name': flow.name,
                        'flow_id': str(deployment.flow_id),
                        'description': deployment.description,
                        'tags': deployment.tags,
                        'is_schedule_active': len(deployment.schedules) > 0 and not deployment.paused,
                        'paused': deployment.paused,
                        'status': deployment.status.value if hasattr(deployment.status, 'value') else str(deployment.status),
                        'created': deployment.created,
                        'updated': deployment.updated,
                        'work_pool_name': deployment.work_pool_name
                    })
                return result
        except Exception as e:
            logger.error(f"Error fetching deployments: {e}")
            return []
    
    async def trigger_deployment(self, deployment_name: str, parameters: Dict[str, Any] = None) -> Tuple[bool, str, str]:
        """Trigger a deployment run.
        
        Args:
            deployment_name: Name of the deployment to run
            parameters: Optional parameters to pass to the flow
            
        Returns:
            Tuple of (success, flow_run_id, message)
        """
        #try:
        while True:
            async with PrefectClient(api=self.api_url) as client:
                # Find deployment by name
                deployments = await client.read_deployments(
                    deployment_filter=DeploymentFilter(name={"any_": [deployment_name]})
                )
                
                if not deployments:
                    return False, "", f"❌ Deployment '{deployment_name}' not found"
                
                deployment = deployments[0]
                
                # Create flow run using the newer API
                from prefect.client.schemas.objects import FlowRun
                flow_run_create = {
                    "flow_id": deployment.flow_id,
                    "deployment_id": deployment.id,
                    "parameters": parameters or {},
                    "context": {},
                    "parent_task_run_id": None,
                    "state_type": "SCHEDULED",
                    "state_name": "Scheduled"
                }
                
                flow_run = await client.create_flow_run_from_deployment(
                    deployment_id=deployment.id,
                    parameters=parameters or {}
                )
                
                flow_run_id = str(flow_run.id)
                message = f"✅ Flow run created successfully: {flow_run_id}"
                
                logger.info(f"Triggered deployment '{deployment_name}' with run ID: {flow_run_id}")
                
                return True, flow_run_id, message
                
        #except Exception as e:
        #    error_msg = f"❌ Error triggering deployment '{deployment_name}': {str(e)}"
        #    logger.error(error_msg)
        #    return False, "", error_msg
    
    def trigger_deployment_cli(self, deployment_name: str, parameters: Dict[str, Any] = None) -> Tuple[bool, str, str]:
        """Trigger a deployment using CLI (more reliable for some Prefect versions).
        
        Args:
            deployment_name: Name of the deployment to run
            parameters: Optional parameters to pass to the flow
            
        Returns:
            Tuple of (success, flow_run_id, message)
        """
        try:
            import subprocess
            import json
            
            # Build the command - need to find the full deployment name
            # First get deployment info to find the flow name
            deployments = self.get_deployments_sync()
            full_deployment_name = None
            
            for dep in deployments:
                if dep['name'] == deployment_name:
                    full_deployment_name = f"{dep['flow_name']}/{dep['name']}"
                    break
            
            if not full_deployment_name:
                return False, "", f"❌ Deployment '{deployment_name}' not found"
            
            # Build the command
            cmd = ['prefect', 'deployment', 'run', full_deployment_name]
            
            # Add parameters
            if parameters:
                for key, value in parameters.items():
                    cmd.extend(['--param', f'{key}={json.dumps(value)}'])
            
            # Set environment
            env = os.environ.copy()
            env['PREFECT_API_URL'] = self.api_url
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # Extract flow run ID from output
                output_lines = result.stdout.strip().split('\n')
                flow_run_id = None
                
                for line in output_lines:
                    if 'Created flow run' in line:
                        # Parse line like: "Created flow run 'happy-goose' with id: 12345..."
                        import re
                        match = re.search(r'([a-f0-9-]{36})', line)
                        if match:
                            flow_run_id = match.group(1)
                            break
                
                if not flow_run_id:
                    # Try to get from all output
                    full_output = result.stdout
                    import re
                    matches = re.findall(r'([a-f0-9-]{36})', full_output)
                    if matches:
                        flow_run_id = matches[-1]  # Use the last UUID found
                
                flow_run_id = flow_run_id or "unknown"
                message = f"✅ Flow run created successfully: {flow_run_id}"
                return True, flow_run_id, message
            else:
                error_msg = f"❌ CLI error: {result.stderr}"
                logger.error(error_msg)
                return False, "", error_msg
                
        except Exception as e:
            error_msg = f"❌ Error triggering deployment '{deployment_name}' via CLI: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    async def get_flow_run_status(self, flow_run_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Get status of a specific flow run.
        
        Args:
            flow_run_id: UUID of the flow run
            
        Returns:
            Tuple of (success, flow_run_info)
        """
        try:
            async with PrefectClient(api=self.api_url) as client:
                flow_run = await client.read_flow_run(flow_run_id)
                
                return True, {
                    'id': str(flow_run.id),
                    'name': flow_run.name,
                    'flow_name': flow_run.flow_name,
                    'state_name': flow_run.state_name,
                    'state_type': flow_run.state_type.value if flow_run.state_type else None,
                    'start_time': flow_run.start_time,
                    'end_time': flow_run.end_time,
                    'total_run_time': flow_run.total_run_time,
                    'created': flow_run.created,
                    'parameters': flow_run.parameters,
                    'tags': flow_run.tags
                }
                
        except Exception as e:
            logger.error(f"Error fetching flow run {flow_run_id}: {e}")
            return False, {}
    
    async def get_recent_flow_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent flow runs.
        
        Args:
            limit: Maximum number of flow runs to return
            
        Returns:
            List of flow run information dictionaries.
        """
        try:
            async with PrefectClient(api=self.api_url) as client:
                flow_runs = await client.read_flow_runs(
                    limit=limit,
                    sort=FlowRunSort.CREATED_DESC
                )
                
                return [
                    {
                        'id': str(run.id),
                        'name': run.name,
                        'flow_name': run.flow_name,
                        'state_name': run.state_name,
                        'state_type': run.state_type.value if run.state_type else None,
                        'start_time': run.start_time,
                        'end_time': run.end_time,
                        'total_run_time': run.total_run_time,
                        'created': run.created,
                        'parameters': run.parameters,
                        'tags': run.tags
                    }
                    for run in flow_runs
                ]
                
        except Exception as e:
            logger.error(f"Error fetching recent flow runs: {e}")
            return []
    
    async def cancel_flow_run(self, flow_run_id: str) -> Tuple[bool, str]:
        """Cancel a running flow.
        
        Args:
            flow_run_id: UUID of the flow run to cancel
            
        Returns:
            Tuple of (success, message)
        """
        try:
            async with PrefectClient(api=self.api_url) as client:
                await client.set_flow_run_state(
                    flow_run_id=flow_run_id,
                    state={"type": "CANCELLED", "message": "Cancelled by user"}
                )
                
                return True, f"✅ Flow run {flow_run_id} cancelled successfully"
                
        except Exception as e:
            error_msg = f"❌ Error cancelling flow run {flow_run_id}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def run_async(self, coro):
        """Helper to run async functions in sync context."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                # If no loop is running, use asyncio.run
                return asyncio.run(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)
    
    # Synchronous wrapper methods for Streamlit
    def get_deployments_sync(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_deployments."""
        return self.run_async(self.get_deployments())
    
    def trigger_deployment_sync(self, deployment_name: str, parameters: Dict[str, Any] = None) -> Tuple[bool, str, str]:
        """Synchronous wrapper for trigger_deployment."""
        try:
            # First try the async API
            return self.run_async(self.trigger_deployment(deployment_name, parameters))
        except Exception as e:
            logger.warning(f"Async trigger failed: {e}, trying CLI method")
            # Fallback to CLI method
            return self.trigger_deployment_cli(deployment_name, parameters)
    
    def get_flow_run_status_sync(self, flow_run_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Synchronous wrapper for get_flow_run_status."""
        return self.run_async(self.get_flow_run_status(flow_run_id))
    
    def get_recent_flow_runs_sync(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_recent_flow_runs."""
        return self.run_async(self.get_recent_flow_runs(limit))
    
    def cancel_flow_run_sync(self, flow_run_id: str) -> Tuple[bool, str]:
        """Synchronous wrapper for cancel_flow_run."""
        return self.run_async(self.cancel_flow_run(flow_run_id))


# Common deployment names for the CRM MLOps project
class CRMDeployments:
    """Constants for CRM deployment names."""
    DATA_ACQUISITION = "crm-data-acquisition"
    DATA_INGESTION = "crm-data-ingestion"
    MONTHLY_TRAINING = "monthly-win-probability-training"
    REFERENCE_DATA = "reference-data-creation"
    DRIFT_MONITORING = "model-drift-monitoring"
    DATA_CLEANUP = "crm-data-cleanup"


def get_prefect_manager() -> PrefectFlowManager:
    """Get a configured Prefect flow manager instance."""
    return PrefectFlowManager()


def format_flow_run_state(state_type: str, state_name: str) -> str:
    """Format flow run state with appropriate emoji and styling.
    
    Args:
        state_type: The state type (COMPLETED, FAILED, RUNNING, etc.)
        state_name: The state name
        
    Returns:
        Formatted state string with emoji
    """
    state_emoji = {
        "COMPLETED": "✅",
        "FAILED": "❌", 
        "RUNNING": "🏃",
        "PENDING": "⏳",
        "SCHEDULED": "📅",
        "CANCELLED": "🚫",
        "CRASHED": "💥",
        "CANCELLING": "🛑"
    }
    
    emoji = state_emoji.get(state_type, "⚪")
    return f"{emoji} {state_name}"


def format_duration(total_run_time: timedelta) -> str:
    """Format duration in a human-readable way.
    
    Args:
        total_run_time: Duration timedelta
        
    Returns:
        Formatted duration string
    """
    if not total_run_time:
        return "N/A"
    
    total_seconds = int(total_run_time.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"
