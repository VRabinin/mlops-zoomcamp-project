"""
Streamlit component for displaying Evidently monitoring dashboards.

This module provides functionality to embed Evidently drift monitoring
reports and dashboards into the Streamlit application.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config.config import get_config
from src.monitoring.drift_monitor import CRMDriftMonitor
from src.utils.prefect_client import (
    CRMDeployments,
    PrefectFlowManager,
    format_duration,
    format_flow_run_state,
)
from src.utils.storage import StorageManager


class EvidentiallyDashboard:
    """Streamlit component for Evidently monitoring dashboards."""

    def __init__(self):
        self.config = get_config()
        self.storage = StorageManager(self.config)
        self.drift_monitor = CRMDriftMonitor(self.config)
        self.prefect_manager = PrefectFlowManager()

    def show_monitoring_overview(self):
        """Display monitoring overview with key metrics."""
        st.subheader("🔍 Model Drift Monitoring Overview")

        # Get latest monitoring results
        monitoring_results = self.get_latest_monitoring_results()

        if not monitoring_results:
            st.info("No monitoring results found. Run drift monitoring first.")
            # self.show_monitoring_controls()
            return

        # Display key metrics
        self.display_drift_metrics(monitoring_results)

        # Display alerts and recommendations
        self.display_alerts_and_recommendations(monitoring_results)

        # Display historical trends
        self.display_historical_trends()

    def get_latest_monitoring_results(self) -> Optional[Dict[str, Any]]:
        """Get the latest monitoring results from storage."""
        # try:
        while True:
            # List monitoring result files
            files = self.storage.list_files(
                "monitoring_results", "monitoring_results_*.json"
            )
            files = self.storage._list_files_s3(
                self.storage.config.storage.buckets.get("data_lake"),
                self.storage.resolve_path("monitoring_results"),
            )
            if not files:
                return None
            # Sort by filename to get the latest
            latest_file = sorted(files)[-1]
            # Load the latest results
            content = self.storage.load_text_file(
                "monitoring_results", str(latest_file).split("/")[-1]
            )
            return json.loads(content)

        # except Exception as e:
        #    st.error(f"Error loading monitoring results: {e}")
        #    return None

    def display_drift_metrics(self, monitoring_results: Dict[str, Any]):
        """Display drift detection metrics."""
        drift_data = monitoring_results.get("drift_detection", {})
        quality_data = monitoring_results.get("data_quality", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            drift_detected = drift_data.get("drift_detected", False)
            status_color = "🔴" if drift_detected else "🟢"
            st.metric(
                label="Drift Status",
                value=f"{status_color} {'Drift Detected' if drift_detected else 'No Drift'}",
                help="Overall drift detection status",
            )

        with col2:
            alert_level = drift_data.get("alert_level", "NONE")
            level_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟠", "NONE": "🟢"}.get(
                alert_level, "⚪"
            )
            st.metric(
                label="Alert Level",
                value=f"{level_color} {alert_level}",
                help="Drift alert severity level",
            )

        with col3:
            num_drifted = drift_data.get("results", {}).get("num_drifted_columns", 0)
            st.metric(
                label="Drifted Columns",
                value=num_drifted,
                help="Number of features showing drift",
            )

        with col4:
            quality_passed = quality_data.get("all_passed", True)
            quality_status = "🟢 Passed" if quality_passed else "🔴 Failed"
            st.metric(
                label="Data Quality",
                value=quality_status,
                help="Data quality test results",
            )

        # Detailed drift metrics
        st.subheader("📊 Detailed Drift Metrics")

        results = drift_data.get("results", {})

        col1, col2 = st.columns(2)

        with col1:
            prediction_drift = results.get("prediction_drift")
            if prediction_drift is not None:
                st.metric(
                    label="Prediction Drift Score",
                    value=f"{prediction_drift:.3f}",
                    help="Drift score for model predictions (0-1, higher = more drift)",
                )

            missing_values = results.get("missing_values_share", 0)
            st.metric(
                label="Missing Values",
                value=f"{missing_values:.1%}",
                help="Share of missing values in current data",
            )

        with col2:
            dataset_drift = results.get("dataset_drift", False)
            dataset_status = "🔴 Yes" if dataset_drift else "🟢 No"
            st.metric(
                label="Dataset Drift",
                value=dataset_status,
                help="Whether overall dataset distribution has drifted",
            )

            # Model performance metrics if available
            accuracy = results.get("accuracy")
            if accuracy is not None:
                st.metric(
                    label="Model Accuracy",
                    value=f"{accuracy:.3f}",
                    help="Current model accuracy",
                )

    def display_alerts_and_recommendations(self, monitoring_results: Dict[str, Any]):
        """Display alerts and recommendations based on monitoring results."""
        drift_data = monitoring_results.get("drift_detection", {})
        alert_level = drift_data.get("alert_level", "NONE")
        drift_detected = drift_data.get("drift_detected", False)

        if not drift_detected:
            st.success("✅ **No Drift Detected** - Model is performing as expected")
            return

        st.subheader("🚨 Alerts & Recommendations")

        if alert_level == "HIGH":
            st.error("🔴 **HIGH PRIORITY ALERT**")
            st.markdown(
                """
            **Immediate Actions Required:**
            - 🔄 **Retrain the model** with recent data
            - 🕵️ **Investigate data pipeline** for changes
            - 📊 **Review feature engineering** logic
            - 🎯 **Update reference data** baseline
            """
            )

        elif alert_level == "MEDIUM":
            st.warning("🟡 **MEDIUM PRIORITY ALERT**")
            st.markdown(
                """
            **Recommended Actions:**
            - 📈 **Monitor closely** over next few periods
            - 🔍 **Analyze drifted features** in detail
            - 📋 **Plan model retraining** schedule
            - 📊 **Review data sources** for changes
            """
            )

        elif alert_level == "LOW":
            st.info("🟠 **LOW PRIORITY ALERT**")
            st.markdown(
                """
            **Monitoring Actions:**
            - 👁️ **Continue monitoring** trends
            - 📝 **Document observations** for analysis
            - 🔄 **Consider gradual** model updates
            """
            )

        # Additional context
        results = drift_data.get("results", {})
        if results.get("num_drifted_columns", 0) > 0:
            st.write(
                f"**Drifted Features:** {results['num_drifted_columns']} features showing drift"
            )

        if results.get("prediction_drift"):
            st.write(f"**Prediction Drift Score:** {results['prediction_drift']:.3f}")

    def display_historical_trends(self):
        """Display historical drift monitoring trends."""
        st.subheader("📈 Historical Monitoring Trends")

        try:
            # Get all monitoring result files
            files = self.storage.list_files(
                "monitoring_results", "monitoring_results_*.json"
            )

            if len(files) < 2:
                st.info(
                    "Insufficient historical data for trend analysis. Run monitoring for multiple periods."
                )
                return

            # Load historical data
            historical_data = []
            for file in sorted(files):
                try:
                    content = self.storage.load_text_file("monitoring_results", file)
                    result = json.loads(content)

                    # Extract key metrics
                    drift_results = result.get("drift_detection", {}).get("results", {})
                    historical_data.append(
                        {
                            "timestamp": result.get("timestamp"),
                            "current_month": result.get("current_month"),
                            "drift_detected": drift_results.get(
                                "drift_detected", False
                            ),
                            "prediction_drift": drift_results.get(
                                "prediction_drift", 0
                            ),
                            "num_drifted_columns": drift_results.get(
                                "num_drifted_columns", 0
                            ),
                            "alert_level": drift_results.get("alert_level", "NONE"),
                            "accuracy": drift_results.get("accuracy"),
                            "missing_values_share": drift_results.get(
                                "missing_values_share", 0
                            ),
                        }
                    )
                except Exception as e:
                    st.warning(f"Could not parse {file}: {e}")
                    continue

            if not historical_data:
                st.warning("No valid historical monitoring data found.")
                return

            df = pd.DataFrame(historical_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Create trend charts
            col1, col2 = st.columns(2)

            with col1:
                # Drift score trend
                fig_drift = px.line(
                    df,
                    x="current_month",
                    y="prediction_drift",
                    title="Prediction Drift Score Over Time",
                    markers=True,
                )
                fig_drift.add_hline(
                    y=0.1,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Warning Threshold",
                )
                fig_drift.add_hline(
                    y=0.2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Critical Threshold",
                )
                st.plotly_chart(fig_drift, use_container_width=True)

            with col2:
                # Number of drifted columns
                fig_columns = px.bar(
                    df,
                    x="current_month",
                    y="num_drifted_columns",
                    title="Number of Drifted Columns",
                    color="alert_level",
                    color_discrete_map={
                        "NONE": "green",
                        "LOW": "orange",
                        "MEDIUM": "yellow",
                        "HIGH": "red",
                    },
                )
                st.plotly_chart(fig_columns, use_container_width=True)

            # Model performance trend (if available)
            if df["accuracy"].notna().any():
                fig_perf = px.line(
                    df[df["accuracy"].notna()],
                    x="current_month",
                    y="accuracy",
                    title="Model Accuracy Over Time",
                    markers=True,
                )
                st.plotly_chart(fig_perf, use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying historical trends: {e}")

    def show_monitoring_controls(self):
        """Display controls for running monitoring via Prefect server."""
        st.subheader("🎛️ Monitoring Controls")

        # Check Prefect server status first
        is_healthy, health_msg = self.prefect_manager.check_server_health()
        if not is_healthy:
            st.error(health_msg)
            st.info(
                "💡 **Troubleshooting:** Ensure Prefect server is running with `make application-start`"
            )
            return

        st.success(health_msg)

        # Get available deployments
        deployments = self.prefect_manager.get_deployments_sync()
        deployment_names = [d["name"] for d in deployments]

        if not deployments:
            st.warning("⚠️ No Prefect deployments found")
            st.info(
                "💡 **Setup required:** Deploy flows first with `make prefect-deploy`"
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Create Reference Data**")
            st.write("Trigger reference data creation flow on Prefect server")

            # Get available periods
            try:
                feature_files = self.storage.list_files(
                    "features", "crm_features_*.csv"
                )
                periods = [f.split("_")[-1].replace(".csv", "") for f in feature_files]
                periods = sorted(list(set(periods)))
            except:
                periods = ["2017-05", "2017-06", "2017-07"]

            ref_period = st.selectbox("Reference Period", periods, key="ref_period")
            sample_size = st.number_input(
                "Sample Size",
                min_value=100,
                max_value=5000,
                value=1000,
                key="ref_sample",
            )

            # Check if reference data deployment is available
            ref_deployment_available = CRMDeployments.REFERENCE_DATA in deployment_names

            if ref_deployment_available:
                if st.button(
                    "🚀 Create Reference Data (Prefect)", key="create_ref_prefect"
                ):
                    with st.spinner("Triggering reference data creation flow..."):
                        try:
                            parameters = {
                                "reference_period": ref_period,
                                "sample_size": sample_size,
                            }

                            (
                                success,
                                flow_run_id,
                                message,
                            ) = self.prefect_manager.trigger_deployment_sync(
                                CRMDeployments.REFERENCE_DATA, parameters
                            )

                            if success:
                                st.success(message)
                                st.info(
                                    f"🔍 **Track progress:** Flow run ID `{flow_run_id}`"
                                )
                                st.info(
                                    f"🌐 **Monitor in UI:** [Prefect Dashboard](http://localhost:4200/flow-runs)"
                                )

                                # Store flow run ID in session state for tracking
                                if "flow_runs" not in st.session_state:
                                    st.session_state.flow_runs = []
                                st.session_state.flow_runs.append(
                                    {
                                        "id": flow_run_id,
                                        "name": "Reference Data Creation",
                                        "triggered_at": datetime.now(),
                                    }
                                )
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.warning(f"⚠️ Deployment '{CRMDeployments.REFERENCE_DATA}' not found")
                st.info("💡 Deploy flows first: `make prefect-deploy`")

        with col2:
            st.write("**Run Drift Monitoring**")
            st.write("Trigger drift monitoring flow on Prefect server")

            current_period = st.selectbox(
                "Current Period",
                periods,
                index=1 if len(periods) > 1 else 0,
                key="current_period",
            )
            ref_period_monitor = st.selectbox(
                "Reference Period", periods, key="ref_period_monitor"
            )

            # Check if drift monitoring deployment is available
            drift_deployment_available = (
                CRMDeployments.DRIFT_MONITORING in deployment_names
            )

            if drift_deployment_available:
                if st.button(
                    "🚀 Run Monitoring (Prefect)", key="run_monitoring_prefect"
                ):
                    with st.spinner("Triggering drift monitoring flow..."):
                        try:
                            parameters = {
                                "current_month": current_period,
                                "reference_period": ref_period_monitor,
                            }

                            (
                                success,
                                flow_run_id,
                                message,
                            ) = self.prefect_manager.trigger_deployment_sync(
                                CRMDeployments.DRIFT_MONITORING, parameters
                            )

                            if success:
                                st.success(message)
                                st.info(
                                    f"🔍 **Track progress:** Flow run ID `{flow_run_id}`"
                                )
                                st.info(
                                    f"🌐 **Monitor in UI:** [Prefect Dashboard](http://localhost:4200/flow-runs)"
                                )

                                # Store flow run ID in session state for tracking
                                if "flow_runs" not in st.session_state:
                                    st.session_state.flow_runs = []
                                st.session_state.flow_runs.append(
                                    {
                                        "id": flow_run_id,
                                        "name": "Drift Monitoring",
                                        "triggered_at": datetime.now(),
                                    }
                                )

                                # Auto-refresh monitoring results after a delay
                                st.info(
                                    "ℹ️ Monitoring results will be available after flow completion"
                                )
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.warning(
                    f"⚠️ Deployment '{CRMDeployments.DRIFT_MONITORING}' not found"
                )
                st.info("💡 Deploy flows first: `make prefect-deploy`")

        # Show recent flow runs
        self.show_recent_flow_runs()

        # Show deployment status
        self.show_deployment_status(deployments)

    def show_recent_flow_runs(self):
        """Display recent flow runs and their status."""
        st.subheader("🏃 Recent Flow Runs")

        try:
            flow_runs = self.prefect_manager.get_recent_flow_runs_sync(limit=10)

            if not flow_runs:
                st.info("No recent flow runs found")
                return

            # Create a nice table of flow runs
            for run in flow_runs:
                with st.expander(
                    f"{run['flow_name']} - {format_flow_run_state(run['state_type'], run['state_name'])}"
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Flow:** {run['flow_name']}")
                        st.write(f"**Run ID:** `{run['id'][:8]}...`")
                        st.write(
                            f"**Created:** {run['created'].strftime('%Y-%m-%d %H:%M:%S') if run['created'] else 'N/A'}"
                        )

                    with col2:
                        st.write(
                            f"**Status:** {format_flow_run_state(run['state_type'], run['state_name'])}"
                        )
                        if run["start_time"]:
                            st.write(
                                f"**Started:** {run['start_time'].strftime('%H:%M:%S')}"
                            )
                        if run["end_time"]:
                            st.write(
                                f"**Ended:** {run['end_time'].strftime('%H:%M:%S')}"
                            )

                    with col3:
                        if run["total_run_time"]:
                            st.write(
                                f"**Duration:** {format_duration(run['total_run_time'])}"
                            )

                        # Show parameters if any
                        if run["parameters"]:
                            st.write("**Parameters:**")
                            for key, value in run["parameters"].items():
                                st.write(f"  • {key}: {value}")

                        # Cancel button for running flows
                        if run["state_type"] == "RUNNING":
                            if st.button(f"🛑 Cancel", key=f"cancel_{run['id'][:8]}"):
                                (
                                    success,
                                    message,
                                ) = self.prefect_manager.cancel_flow_run_sync(run["id"])
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)

        except Exception as e:
            st.error(f"Error fetching flow runs: {e}")

    def show_deployment_status(self, deployments: List[Dict[str, Any]]):
        """Display deployment status and information."""
        st.subheader("📋 Available Deployments")

        if not deployments:
            st.warning("No deployments found")
            return

        # Group deployments by type
        crm_deployments = [
            d
            for d in deployments
            if any(
                crm_name in d["name"]
                for crm_name in [
                    "crm",
                    "monthly",
                    "reference",
                    "drift",
                    "acquisition",
                    "ingestion",
                ]
            )
        ]

        if crm_deployments:
            st.write("**CRM MLOps Deployments:**")
            for deployment in crm_deployments:
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**{deployment['name']}**")
                    if deployment["description"]:
                        st.caption(deployment["description"])

                with col2:
                    schedule_status = (
                        "📅 Scheduled"
                        if deployment["is_schedule_active"]
                        else "🚫 Manual"
                    )
                    st.write(schedule_status)

                with col3:
                    if st.button(f"▶️ Run", key=f"run_{deployment['name']}"):
                        # Trigger deployment without parameters
                        (
                            success,
                            flow_run_id,
                            message,
                        ) = self.prefect_manager.trigger_deployment_sync(
                            deployment["name"]
                        )
                        if success:
                            st.success(f"✅ Triggered: {deployment['name']}")
                            st.info(f"Run ID: {flow_run_id}")
                        else:
                            st.error(message)

        # Show other deployments if any
        other_deployments = [d for d in deployments if d not in crm_deployments]
        if other_deployments:
            with st.expander("Other Deployments"):
                for deployment in other_deployments:
                    st.write(
                        f"• **{deployment['name']}** - {deployment.get('description', 'No description')}"
                    )

    def show_evidently_reports(self):
        """Display available Evidently HTML reports."""
        st.subheader("📄 Evidently Reports")

        try:
            # List available HTML reports
            report_files = self.storage.list_files("monitoring_reports", "*.html")

            if not report_files:
                st.info(
                    "No Evidently reports found. Run drift monitoring to generate reports."
                )
                return

            # Group reports by type
            drift_reports = [
                f
                for f in report_files
                if str(f).split("/")[-1].startswith("drift_report")
            ]
            performance_reports = [
                f
                for f in report_files
                if str(f).split("/")[-1].startswith("performance_report")
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Drift Reports**")
                for report in sorted(drift_reports, reverse=True)[:5]:  # Show latest 5
                    report_name = str(report).split("/")[-1]
                    if st.button(f"📊 {report_name}", key=f"drift_{report_name}"):
                        self.display_html_report("monitoring_reports", report_name)

            with col2:
                st.write("**Performance Reports**")
                for report in sorted(performance_reports, reverse=True)[
                    :5
                ]:  # Show latest 5
                    report_name = str(report).split("/")[-1]
                    if st.button(f"🎯 {report_name}", key=f"perf_{report_name}"):
                        self.display_html_report("monitoring_reports", report_name)

        except Exception as e:
            st.error(f"Error listing reports: {e}")

    def display_html_report(self, folder: str, filename: str):
        """Display an HTML report in Streamlit."""
        try:
            file_content = self.storage.load_text_file(folder, filename)
            components.html(file_content, height=600, scrolling=True)

        except Exception as e:
            st.error(f"Error displaying report: {e}")


def show_monitoring_dashboard():
    """Main function to display the monitoring dashboard."""
    dashboard = EvidentiallyDashboard()

    # Create tabs for different monitoring views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎛️ Controls", "📄 Reports"])

    with tab1:
        dashboard.show_monitoring_overview()

    with tab2:
        dashboard.show_monitoring_controls()

    with tab3:
        dashboard.show_evidently_reports()
