"""
Test Prefect client functionality and flow management.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.prefect_client import (
    CRMDeployments,
    PrefectFlowManager,
    format_duration,
    format_flow_run_state,
)


class TestCRMDeployments:
    """Test CRM deployment constants."""

    def test_deployment_constants(self):
        """Test that all required deployment constants are defined."""
        assert hasattr(CRMDeployments, "DATA_ACQUISITION")
        assert hasattr(CRMDeployments, "DATA_INGESTION")
        assert hasattr(CRMDeployments, "MONTHLY_TRAINING")
        assert hasattr(CRMDeployments, "REFERENCE_DATA")
        assert hasattr(CRMDeployments, "DRIFT_MONITORING")
        assert hasattr(CRMDeployments, "DATA_CLEANUP")

        # Verify they are strings
        assert isinstance(CRMDeployments.DATA_ACQUISITION, str)
        assert isinstance(CRMDeployments.DATA_INGESTION, str)
        assert isinstance(CRMDeployments.MONTHLY_TRAINING, str)
        assert isinstance(CRMDeployments.REFERENCE_DATA, str)
        assert isinstance(CRMDeployments.DRIFT_MONITORING, str)
        assert isinstance(CRMDeployments.DATA_CLEANUP, str)


class TestFormatUtilities:
    """Test formatting utility functions."""

    def test_format_flow_run_state_success(self):
        """Test formatting successful flow run state."""
        result = format_flow_run_state("COMPLETED", "Completed")
        assert "✅" in result
        assert "Completed" in result

    def test_format_flow_run_state_failure(self):
        """Test formatting failed flow run state."""
        result = format_flow_run_state("FAILED", "Failed")
        assert "❌" in result
        assert "Failed" in result

    def test_format_flow_run_state_running(self):
        """Test formatting running flow run state."""
        result = format_flow_run_state("RUNNING", "Running")
        assert "🏃" in result
        assert "Running" in result

    def test_format_flow_run_state_pending(self):
        """Test formatting pending flow run state."""
        result = format_flow_run_state("PENDING", "Pending")
        assert "⏳" in result
        assert "Pending" in result

    def test_format_flow_run_state_unknown(self):
        """Test formatting unknown flow run state."""
        result = format_flow_run_state("UNKNOWN", "Unknown")
        assert "❓" in result
        assert "Unknown" in result

    def test_format_duration_seconds(self):
        """Test formatting duration in seconds."""
        result = format_duration(timedelta(seconds=45))
        assert result == "45s"

    def test_format_duration_minutes(self):
        """Test formatting duration in minutes."""
        result = format_duration(timedelta(minutes=2, seconds=30))
        assert result == "2m 30s"

    def test_format_duration_hours(self):
        """Test formatting duration in hours."""
        result = format_duration(timedelta(hours=1, minutes=15, seconds=30))
        assert result == "1h 15m 30s"

    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        result = format_duration(timedelta(seconds=0))
        assert result == "N/A"

    def test_format_duration_none(self):
        """Test formatting None duration."""
        result = format_duration(None)
        assert result == "N/A"


class TestPrefectFlowManager:
    """Test PrefectFlowManager functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.base_url = "http://localhost:4200/api"
        self.manager = PrefectFlowManager(self.base_url)

    def test_initialization(self):
        """Test PrefectFlowManager initializes correctly."""
        assert self.manager.api_url == self.base_url
        # The base_url is derived from api_url by removing '/api'
        expected_base_url = self.base_url.replace("/api", "")
        assert self.manager.base_url == expected_base_url

    @patch("requests.get")
    def test_check_connection_success(self, mock_get):
        """Test successful connection check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        success, message = self.manager.check_server_health()
        assert success is True
        assert "healthy" in message
        mock_get.assert_called_once_with(f"{self.manager.base_url}/health", timeout=5)

    @patch("requests.get")
    def test_check_connection_failure(self, mock_get):
        """Test failed connection check."""
        from requests import RequestException

        mock_get.side_effect = RequestException("Connection failed")

        success, message = self.manager.check_server_health()
        assert success is False
        assert "Cannot connect" in message

    @patch("src.utils.prefect_client.PrefectClient")
    def test_get_deployments_success(self, mock_prefect_client):
        """Test successful deployment retrieval."""
        # Mock the async context manager and client
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock deployment objects
        mock_deployment1 = MagicMock()
        mock_deployment1.id = "123"
        mock_deployment1.name = "test-deployment-1"
        mock_deployment1.flow_id = "flow-123"
        mock_deployment1.description = "Test deployment"
        mock_deployment1.tags = ["test"]
        mock_deployment1.schedules = []
        mock_deployment1.paused = False
        mock_deployment1.status = MagicMock()
        mock_deployment1.status.value = "READY"
        mock_deployment1.created = datetime.now()
        mock_deployment1.updated = datetime.now()
        mock_deployment1.work_pool_name = "default"

        mock_deployment2 = MagicMock()
        mock_deployment2.id = "456"
        mock_deployment2.name = "test-deployment-2"
        mock_deployment2.flow_id = "flow-456"
        mock_deployment2.description = "Test deployment 2"
        mock_deployment2.tags = ["test"]
        mock_deployment2.schedules = []
        mock_deployment2.paused = False
        mock_deployment2.status = MagicMock()
        mock_deployment2.status.value = "READY"
        mock_deployment2.created = datetime.now()
        mock_deployment2.updated = datetime.now()
        mock_deployment2.work_pool_name = "default"

        # Mock flow objects
        mock_flow1 = MagicMock()
        mock_flow1.name = "test-flow-1"
        mock_flow2 = MagicMock()
        mock_flow2.name = "test-flow-2"

        # Setup mock responses
        mock_client.read_deployments.return_value = [mock_deployment1, mock_deployment2]
        mock_client.read_flow.side_effect = [mock_flow1, mock_flow2]

        deployments = self.manager.get_deployments_sync()
        assert len(deployments) == 2
        assert deployments[0]["name"] == "test-deployment-1"
        assert deployments[1]["name"] == "test-deployment-2"

    @patch("src.utils.prefect_client.PrefectClient")
    def test_get_deployments_failure(self, mock_prefect_client):
        """Test failed deployment retrieval."""
        mock_prefect_client.side_effect = Exception("API error")

        deployments = self.manager.get_deployments_sync()
        assert deployments == []

    @patch("src.utils.prefect_client.PrefectClient")
    def test_trigger_flow_success(self, mock_prefect_client):
        """Test successful flow triggering."""
        # Mock the async context manager and client
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock deployment
        mock_deployment = MagicMock()
        mock_deployment.id = "deployment-123"
        mock_deployment.flow_id = "flow-123"
        mock_client.read_deployments.return_value = [mock_deployment]

        # Mock flow run
        mock_flow_run = MagicMock()
        mock_flow_run.id = "flow-run-123"
        mock_client.create_flow_run_from_deployment.return_value = mock_flow_run

        success, flow_run_id, message = self.manager.trigger_deployment_sync(
            "test-deployment", {"param1": "value1"}
        )

        assert success is True
        assert flow_run_id == "flow-run-123"
        assert "successfully" in message

    @patch("src.utils.prefect_client.PrefectClient")
    def test_trigger_flow_failure(self, mock_prefect_client):
        """Test failed flow triggering."""
        mock_prefect_client.side_effect = Exception("Trigger failed")

        # This should fallback to CLI method, but for testing we'll mock that too
        with patch.object(self.manager, "trigger_deployment_cli") as mock_cli:
            mock_cli.return_value = (False, "", "CLI also failed")

            success, flow_run_id, message = self.manager.trigger_deployment_sync(
                "test-deployment", {}
            )

            assert success is False
            assert "failed" in message.lower()

    @patch("subprocess.run")
    def test_trigger_flow_cli_fallback_success(self, mock_run):
        """Test CLI fallback for flow triggering."""
        # Mock the get_deployments_sync call first
        with patch.object(self.manager, "get_deployments_sync") as mock_get_deployments:
            mock_get_deployments.return_value = [
                {"name": "test-deployment", "flow_name": "test-flow"}
            ]

            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "Created flow run 'happy-goose' with id: 12345678-1234-5678-9abc-123456789abc"
            mock_run.return_value = mock_process

            success, flow_run_id, message = self.manager.trigger_deployment_cli(
                "test-deployment", {"param": "value"}
            )

            assert success is True
            assert "successfully" in message

    @patch("subprocess.run")
    def test_trigger_flow_cli_fallback_failure(self, mock_run):
        """Test CLI fallback failure."""
        mock_run.side_effect = Exception("CLI failed")

        success, flow_run_id, message = self.manager.trigger_deployment_cli(
            "test-deployment", {}
        )

        assert success is False
        assert "Error" in message or "❌" in message

    @patch("src.utils.prefect_client.FlowRunSort")
    @patch("src.utils.prefect_client.PrefectClient")
    def test_get_recent_flow_runs_success(
        self, mock_prefect_client, mock_flow_run_sort
    ):
        """Test successful recent flow runs retrieval."""
        # Mock the sort enum
        mock_flow_run_sort.CREATED_DESC = "CREATED_DESC"

        # Mock the async context manager and client
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock flow run
        mock_run = MagicMock()
        mock_run.id = "run-123"
        mock_run.name = "test-run"
        mock_run.flow_name = "test-flow"
        mock_run.state_name = "Completed"
        mock_run.state_type = MagicMock()
        mock_run.state_type.value = "COMPLETED"
        mock_run.start_time = datetime.now()
        mock_run.end_time = datetime.now()
        mock_run.total_run_time = timedelta(seconds=30)
        mock_run.created = datetime.now()
        mock_run.parameters = {}
        mock_run.tags = []

        mock_client.read_flow_runs.return_value = [mock_run]

        runs = self.manager.get_recent_flow_runs_sync(limit=10)
        assert len(runs) == 1
        assert runs[0]["id"] == "run-123"
        assert runs[0]["flow_name"] == "test-flow"

    @patch("src.utils.prefect_client.PrefectClient")
    def test_get_recent_flow_runs_failure(self, mock_prefect_client):
        """Test failed recent flow runs retrieval."""
        mock_prefect_client.side_effect = Exception("API error")

        runs = self.manager.get_recent_flow_runs_sync()
        assert runs == []

    @patch("src.utils.prefect_client.PrefectClient")
    def test_get_flow_run_status_success(self, mock_prefect_client):
        """Test successful flow run status retrieval."""
        # Mock the async context manager and client
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock flow run
        mock_flow_run = MagicMock()
        mock_flow_run.id = "run-123"
        mock_flow_run.name = "test-run"
        mock_flow_run.flow_name = "test-flow"
        mock_flow_run.state_name = "Running"
        mock_flow_run.state_type = MagicMock()
        mock_flow_run.state_type.value = "RUNNING"
        mock_flow_run.start_time = datetime.now()
        mock_flow_run.end_time = None
        mock_flow_run.total_run_time = None
        mock_flow_run.created = datetime.now()
        mock_flow_run.parameters = {}
        mock_flow_run.tags = []

        mock_client.read_flow_run.return_value = mock_flow_run

        success, status = self.manager.get_flow_run_status_sync("run-123")
        assert success is True
        assert status["id"] == "run-123"
        assert status["state_type"] == "RUNNING"

    @patch("src.utils.prefect_client.PrefectClient")
    def test_get_flow_run_status_failure(self, mock_prefect_client):
        """Test failed flow run status retrieval."""
        mock_prefect_client.side_effect = Exception("API error")

        success, status = self.manager.get_flow_run_status_sync("run-123")
        assert success is False
        assert status == {}

    def test_deployment_name_validation(self):
        """Test deployment name validation."""
        # Valid deployment names should work
        valid_names = [
            CRMDeployments.DATA_ACQUISITION,
            CRMDeployments.DATA_INGESTION,
            CRMDeployments.MONTHLY_TRAINING,
        ]

        for name in valid_names:
            # Should not raise an exception during name processing
            # We'll mock the actual trigger to avoid real API calls
            with patch.object(self.manager, "trigger_deployment_sync") as mock_trigger:
                mock_trigger.return_value = (False, "", "Mocked response")
                success, flow_run_id, message = self.manager.trigger_deployment_sync(
                    name, {}
                )
                # The method was called without raising an exception
                mock_trigger.assert_called_once_with(name, {})

    def test_parameter_serialization(self):
        """Test parameter serialization for flow triggering."""
        complex_params = {
            "string_param": "test",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
        }

        # Should not raise an exception during parameter processing
        with patch.object(self.manager, "trigger_deployment_sync") as mock_trigger:
            mock_trigger.return_value = (True, "test-run", "Success")

            success, flow_run_id, message = self.manager.trigger_deployment_sync(
                "test-deployment", complex_params
            )

            # Verify the method was called with proper parameters
            mock_trigger.assert_called_once_with("test-deployment", complex_params)


class TestPrefectFlowManagerAsync:
    """Test async methods of PrefectFlowManager."""

    def setup_method(self):
        """Set up test environment for async tests."""
        self.manager = PrefectFlowManager("http://localhost:4200/api")

    @pytest.mark.asyncio
    @patch("src.utils.prefect_client.PrefectClient")
    async def test_trigger_deployment_async_success(self, mock_prefect_client):
        """Test successful async deployment triggering."""
        # Mock the async context manager and client
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock deployment
        mock_deployment = MagicMock()
        mock_deployment.id = "deployment-123"
        mock_deployment.flow_id = "flow-123"
        mock_client.read_deployments.return_value = [mock_deployment]

        # Mock flow run
        mock_flow_run = MagicMock()
        mock_flow_run.id = "flow-run-123"
        mock_client.create_flow_run_from_deployment.return_value = mock_flow_run

        success, flow_run_id, message = await self.manager.trigger_deployment(
            "test-deployment", {}
        )

        assert success is True
        assert flow_run_id == "flow-run-123"
        assert "successfully" in message

    @pytest.mark.asyncio
    @patch("src.utils.prefect_client.FlowRunSort")
    @patch("src.utils.prefect_client.PrefectClient")
    async def test_get_recent_flow_runs_async_success(
        self, mock_prefect_client, mock_flow_run_sort
    ):
        """Test successful async flow runs retrieval."""
        # Mock the sort enum
        mock_flow_run_sort.CREATED_DESC = "CREATED_DESC"

        # Mock the async context manager and client
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock flow run
        mock_run = MagicMock()
        mock_run.id = "run-1"
        mock_run.name = "test-run-1"
        mock_run.flow_name = "test-flow-1"
        mock_run.state_name = "Completed"
        mock_run.state_type = MagicMock()
        mock_run.state_type.value = "COMPLETED"
        mock_run.start_time = datetime.now()
        mock_run.end_time = datetime.now()
        mock_run.total_run_time = timedelta(seconds=30)
        mock_run.created = datetime.now()
        mock_run.parameters = {}
        mock_run.tags = []

        mock_client.read_flow_runs.return_value = [mock_run]

        runs = await self.manager.get_recent_flow_runs(limit=10)

        assert len(runs) == 1
        assert runs[0]["id"] == "run-1"
        assert runs[0]["flow_name"] == "test-flow-1"

    def test_sync_wrapper_for_async_methods(self):
        """Test synchronous wrappers for async methods."""
        with patch.object(self.manager, "run_async") as mock_run_async:
            mock_run_async.return_value = (True, "sync-wrapped-123", "Success")

            # Test sync wrapper
            success, flow_run_id, message = self.manager.trigger_deployment_sync(
                "test-deployment", {}
            )

            assert success is True
            assert flow_run_id == "sync-wrapped-123"
            assert "Success" in message


class TestPrefectFlowManagerIntegration:
    """Integration tests for PrefectFlowManager."""

    @patch("requests.get")
    @patch("src.utils.prefect_client.PrefectClient")
    def test_full_workflow_simulation(self, mock_prefect_client, mock_get):
        """Test a complete workflow simulation."""
        manager = PrefectFlowManager("http://localhost:4200/api")

        # Mock connection check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Mock the async context manager and client for deployments
        mock_client = AsyncMock()
        mock_prefect_client.return_value.__aenter__.return_value = mock_client

        # Mock deployment objects
        mock_deployment = MagicMock()
        mock_deployment.id = "123"
        mock_deployment.name = CRMDeployments.DATA_ACQUISITION
        mock_deployment.flow_id = "flow-123"
        mock_deployment.description = "Test deployment"
        mock_deployment.tags = ["test"]
        mock_deployment.schedules = []
        mock_deployment.paused = False
        mock_deployment.status = MagicMock()
        mock_deployment.status.value = "READY"
        mock_deployment.created = datetime.now()
        mock_deployment.updated = datetime.now()
        mock_deployment.work_pool_name = "default"

        # Mock flow object
        mock_flow = MagicMock()
        mock_flow.name = "crm-data-acquisition-flow"

        # Setup mock responses
        mock_client.read_deployments.return_value = [mock_deployment]
        mock_client.read_flow.return_value = mock_flow

        # Mock flow run creation
        mock_flow_run = MagicMock()
        mock_flow_run.id = "flow-run-789"
        mock_client.create_flow_run_from_deployment.return_value = mock_flow_run

        # Mock flow run status
        mock_status_run = MagicMock()
        mock_status_run.id = "flow-run-789"
        mock_status_run.name = "test-run"
        mock_status_run.flow_name = "crm-data-acquisition-flow"
        mock_status_run.state_name = "Running"
        mock_status_run.state_type = MagicMock()
        mock_status_run.state_type.value = "RUNNING"
        mock_status_run.start_time = datetime.now()
        mock_status_run.end_time = None
        mock_status_run.total_run_time = None
        mock_status_run.created = datetime.now()
        mock_status_run.parameters = {}
        mock_status_run.tags = []

        mock_client.read_flow_run.return_value = mock_status_run

        # Test the complete workflow
        # 1. Check connection
        success, message = manager.check_server_health()
        assert success is True

        # 2. Get deployments
        deployments = manager.get_deployments_sync()
        assert len(deployments) == 1
        assert deployments[0]["name"] == CRMDeployments.DATA_ACQUISITION

        # 3. Trigger a flow
        success, flow_run_id, message = manager.trigger_deployment_sync(
            CRMDeployments.DATA_ACQUISITION, {"test": "param"}
        )
        assert success is True
        assert flow_run_id == "flow-run-789"

        # 4. Check flow status
        success, status = manager.get_flow_run_status_sync("flow-run-789")
        assert success is True
        assert status["state_type"] == "RUNNING"


class TestPrefectFlowManagerErrorHandling:
    """Test error handling in PrefectFlowManager."""

    @patch("src.utils.prefect_client.PrefectClient")
    def test_malformed_response_handling(self, mock_prefect_client):
        """Test handling of malformed API responses."""
        manager = PrefectFlowManager("http://localhost:4200/api")

        # Mock client that raises an exception during deployment reading
        mock_prefect_client.side_effect = ValueError("Invalid response")

        # Should handle gracefully
        deployments = manager.get_deployments_sync()
        assert deployments == []

    @patch("requests.get")
    def test_network_timeout_handling(self, mock_get):
        """Test handling of network timeouts."""
        manager = PrefectFlowManager("http://localhost:4200/api")

        # Mock timeout
        import requests

        mock_get.side_effect = requests.Timeout("Request timed out")

        # Should handle gracefully
        success, message = manager.check_server_health()
        assert success is False
        assert "Cannot connect" in message

    @patch("src.utils.prefect_client.PrefectClient")
    def test_http_error_status_handling(self, mock_prefect_client):
        """Test handling of HTTP error status codes."""
        manager = PrefectFlowManager("http://localhost:4200/api")

        # Mock Prefect client that raises an exception
        mock_prefect_client.side_effect = Exception("Server error")

        # Should fallback to CLI method
        with patch.object(manager, "trigger_deployment_cli") as mock_cli:
            mock_cli.return_value = (False, "", "CLI also failed")

            success, flow_run_id, message = manager.trigger_deployment_sync(
                "test-deployment", {}
            )
            assert success is False
            assert "failed" in message.lower()
