"""
Streamlit component for displaying Evidently monitoring dashboards.

This module provides functionality to embed Evidently drift monitoring
reports and dashboards into the Streamlit application.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.config.config import get_config
from src.utils.storage import StorageManager
from src.monitoring.drift_monitor import CRMDriftMonitor
import streamlit.components.v1 as components

class EvidentiallyDashboard:
    """Streamlit component for Evidently monitoring dashboards."""
    
    def __init__(self):
        self.config = get_config()
        self.storage = StorageManager(self.config)
        self.drift_monitor = CRMDriftMonitor(self.config)
    
    def show_monitoring_overview(self):
        """Display monitoring overview with key metrics."""
        st.subheader("🔍 Model Drift Monitoring Overview")
        
        # Get latest monitoring results
        monitoring_results = self.get_latest_monitoring_results()
        
        if not monitoring_results:
            st.info("No monitoring results found. Run drift monitoring first.")
            #self.show_monitoring_controls()
            return
        
        # Display key metrics
        self.display_drift_metrics(monitoring_results)
        
        # Display alerts and recommendations
        self.display_alerts_and_recommendations(monitoring_results)
        
        # Display historical trends
        self.display_historical_trends()
    
    def get_latest_monitoring_results(self) -> Optional[Dict[str, Any]]:
        """Get the latest monitoring results from storage."""
        #try:
        while True:
            # List monitoring result files
            files = self.storage.list_files("monitoring_results", "monitoring_results_*.json")
            st.info(self.storage.list_files("monitoring_results", "monitoring_results_*.json"))
            st.info(self.storage.resolve_path("monitoring_results"))
            files = self.storage._list_files_s3(self.storage.config.storage.buckets.get('data_lake'), self.storage.resolve_path("monitoring_results"))
            if not files:
                return None
            
            # Sort by filename to get the latest
            latest_file = sorted(files)[-1]
            
            # Load the latest results
            st.info(f"Loading latest monitoring results from: {latest_file}")
            st.info(self.storage.get_bucket_for_data_type("monitoring_results"))
            st.info(self.storage.get_s3_path("monitoring_results", latest_file))
            content = self.storage.load_text_file("monitoring_results", str(latest_file).split('/')[-1])
            return json.loads(content)

        #except Exception as e:
        #    st.error(f"Error loading monitoring results: {e}")
        #    return None
    
    def display_drift_metrics(self, monitoring_results: Dict[str, Any]):
        """Display drift detection metrics."""
        drift_data = monitoring_results.get('drift_detection', {})
        quality_data = monitoring_results.get('data_quality', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            drift_detected = drift_data.get('drift_detected', False)
            status_color = "🔴" if drift_detected else "🟢"
            st.metric(
                label="Drift Status",
                value=f"{status_color} {'Drift Detected' if drift_detected else 'No Drift'}",
                help="Overall drift detection status"
            )
        
        with col2:
            alert_level = drift_data.get('alert_level', 'NONE')
            level_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟠", "NONE": "🟢"}.get(alert_level, "⚪")
            st.metric(
                label="Alert Level",
                value=f"{level_color} {alert_level}",
                help="Drift alert severity level"
            )
        
        with col3:
            num_drifted = drift_data.get('results', {}).get('num_drifted_columns', 0)
            st.metric(
                label="Drifted Columns",
                value=num_drifted,
                help="Number of features showing drift"
            )
        
        with col4:
            quality_passed = quality_data.get('all_passed', True)
            quality_status = "🟢 Passed" if quality_passed else "🔴 Failed"
            st.metric(
                label="Data Quality",
                value=quality_status,
                help="Data quality test results"
            )
        
        # Detailed drift metrics
        st.subheader("📊 Detailed Drift Metrics")
        
        results = drift_data.get('results', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_drift = results.get('prediction_drift')
            if prediction_drift is not None:
                st.metric(
                    label="Prediction Drift Score",
                    value=f"{prediction_drift:.3f}",
                    help="Drift score for model predictions (0-1, higher = more drift)"
                )
            
            missing_values = results.get('missing_values_share', 0)
            st.metric(
                label="Missing Values",
                value=f"{missing_values:.1%}",
                help="Share of missing values in current data"
            )
        
        with col2:
            dataset_drift = results.get('dataset_drift', False)
            dataset_status = "🔴 Yes" if dataset_drift else "🟢 No"
            st.metric(
                label="Dataset Drift",
                value=dataset_status,
                help="Whether overall dataset distribution has drifted"
            )
            
            # Model performance metrics if available
            accuracy = results.get('accuracy')
            if accuracy is not None:
                st.metric(
                    label="Model Accuracy",
                    value=f"{accuracy:.3f}",
                    help="Current model accuracy"
                )
    
    def display_alerts_and_recommendations(self, monitoring_results: Dict[str, Any]):
        """Display alerts and recommendations based on monitoring results."""
        drift_data = monitoring_results.get('drift_detection', {})
        alert_level = drift_data.get('alert_level', 'NONE')
        drift_detected = drift_data.get('drift_detected', False)
        
        if not drift_detected:
            st.success("✅ **No Drift Detected** - Model is performing as expected")
            return
        
        st.subheader("🚨 Alerts & Recommendations")
        
        if alert_level == "HIGH":
            st.error("🔴 **HIGH PRIORITY ALERT**")
            st.markdown("""
            **Immediate Actions Required:**
            - 🔄 **Retrain the model** with recent data
            - 🕵️ **Investigate data pipeline** for changes
            - 📊 **Review feature engineering** logic
            - 🎯 **Update reference data** baseline
            """)
        
        elif alert_level == "MEDIUM":
            st.warning("🟡 **MEDIUM PRIORITY ALERT**")
            st.markdown("""
            **Recommended Actions:**
            - 📈 **Monitor closely** over next few periods
            - 🔍 **Analyze drifted features** in detail
            - 📋 **Plan model retraining** schedule
            - 📊 **Review data sources** for changes
            """)
        
        elif alert_level == "LOW":
            st.info("🟠 **LOW PRIORITY ALERT**")
            st.markdown("""
            **Monitoring Actions:**
            - 👁️ **Continue monitoring** trends
            - 📝 **Document observations** for analysis
            - 🔄 **Consider gradual** model updates
            """)
        
        # Additional context
        results = drift_data.get('results', {})
        if results.get('num_drifted_columns', 0) > 0:
            st.write(f"**Drifted Features:** {results['num_drifted_columns']} features showing drift")
        
        if results.get('prediction_drift'):
            st.write(f"**Prediction Drift Score:** {results['prediction_drift']:.3f}")
    
    def display_historical_trends(self):
        """Display historical drift monitoring trends."""
        st.subheader("📈 Historical Monitoring Trends")
        
        try:
            # Get all monitoring result files
            files = self.storage.list_files("monitoring_results", "monitoring_results_*.json")
            
            if len(files) < 2:
                st.info("Insufficient historical data for trend analysis. Run monitoring for multiple periods.")
                return
            
            # Load historical data
            historical_data = []
            for file in sorted(files):
                try:
                    content = self.storage.load_text_file("monitoring_results", file)
                    result = json.loads(content)
                    
                    # Extract key metrics
                    drift_results = result.get('drift_detection', {}).get('results', {})
                    historical_data.append({
                        'timestamp': result.get('timestamp'),
                        'current_month': result.get('current_month'),
                        'drift_detected': drift_results.get('drift_detected', False),
                        'prediction_drift': drift_results.get('prediction_drift', 0),
                        'num_drifted_columns': drift_results.get('num_drifted_columns', 0),
                        'alert_level': drift_results.get('alert_level', 'NONE'),
                        'accuracy': drift_results.get('accuracy'),
                        'missing_values_share': drift_results.get('missing_values_share', 0)
                    })
                except Exception as e:
                    st.warning(f"Could not parse {file}: {e}")
                    continue
            
            if not historical_data:
                st.warning("No valid historical monitoring data found.")
                return
            
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create trend charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Drift score trend
                fig_drift = px.line(
                    df, 
                    x='current_month', 
                    y='prediction_drift',
                    title='Prediction Drift Score Over Time',
                    markers=True
                )
                fig_drift.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                                   annotation_text="Warning Threshold")
                fig_drift.add_hline(y=0.2, line_dash="dash", line_color="red", 
                                   annotation_text="Critical Threshold")
                st.plotly_chart(fig_drift, use_container_width=True)
            
            with col2:
                # Number of drifted columns
                fig_columns = px.bar(
                    df,
                    x='current_month',
                    y='num_drifted_columns',
                    title='Number of Drifted Columns',
                    color='alert_level',
                    color_discrete_map={
                        'NONE': 'green',
                        'LOW': 'orange',
                        'MEDIUM': 'yellow',
                        'HIGH': 'red'
                    }
                )
                st.plotly_chart(fig_columns, use_container_width=True)
            
            # Model performance trend (if available)
            if df['accuracy'].notna().any():
                fig_perf = px.line(
                    df[df['accuracy'].notna()],
                    x='current_month',
                    y='accuracy',
                    title='Model Accuracy Over Time',
                    markers=True
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying historical trends: {e}")
    
    def show_monitoring_controls(self):
        """Display controls for running monitoring."""
        st.subheader("🎛️ Monitoring Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Create Reference Data**")
            st.write("Create baseline dataset for drift comparison")
            
            # Get available periods
            try:
                feature_files = self.storage.list_files("features", "crm_features_*.csv")
                periods = [f.split('_')[-1].replace('.csv', '') for f in feature_files]
                periods = sorted(list(set(periods)))
            except:
                periods = ["2017-05", "2017-06", "2017-07"]
            
            ref_period = st.selectbox("Reference Period", periods, key="ref_period")
            sample_size = st.number_input("Sample Size", min_value=100, max_value=5000, value=1000, key="ref_sample")
            
            if st.button("Create Reference Data", key="create_ref"):
                with st.spinner("Creating reference data..."):
                    try:
                        success, message = self.drift_monitor.create_reference_data(ref_period, sample_size)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            st.write("**Run Drift Monitoring**")
            st.write("Compare current data against reference")
            
            current_period = st.selectbox("Current Period", periods, index=1 if len(periods) > 1 else 0, key="current_period")
            ref_period_monitor_var = st.selectbox("Reference Period", periods, key="ref_period_monitor")

            if st.button("Run Monitoring", key="run_monitoring"):
                with st.spinner("Running drift monitoring..."):
                    try:
                        # Generate current predictions
                        pred_success, pred_msg = self.drift_monitor.generate_current_predictions(current_period)
                        if not pred_success:
                            st.error(f"Failed to generate predictions: {pred_msg}")
                            return
                        
                        # Detect drift
                        drift_success, drift_results = self.drift_monitor.detect_drift(ref_period_monitor_var, current_period)
                        if drift_success:
                            st.success("✅ Monitoring completed successfully!")
                            
                            # Show quick results
                            if drift_results.get('drift_detected'):
                                st.warning(f"🚨 Drift detected! Alert level: {drift_results.get('alert_level')}")
                            else:
                                st.info("ℹ️ No significant drift detected")

                            st.rerun()  # Refresh to show new results
                        else:
                            st.error(f"❌ Monitoring failed: {drift_results}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    def show_evidently_reports(self):
        """Display available Evidently HTML reports."""
        st.subheader("📄 Evidently Reports")
        
        try:
            # List available HTML reports
            report_files = self.storage.list_files("monitoring_reports", "*.html")
            
            if not report_files:
                st.info("No Evidently reports found. Run drift monitoring to generate reports.")
                return
            
            # Group reports by type
            drift_reports = [f for f in report_files if str(f).split('/')[-1].startswith("drift_report")]
            performance_reports = [f for f in report_files if str(f).split('/')[-1].startswith("performance_report")]

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Drift Reports**")
                for report in sorted(drift_reports, reverse=True)[:5]:  # Show latest 5
                    report_name = str(report).split('/')[-1]
                    if st.button(f"📊 {report_name}", key=f"drift_{report_name}"):
                        self.display_html_report("monitoring_reports", report_name)
            
            with col2:
                st.write("**Performance Reports**")
                for report in sorted(performance_reports, reverse=True)[:5]:  # Show latest 5
                    report_name = str(report).split('/')[-1]
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
