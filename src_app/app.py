"""
Monthly Win Probability Prediction - Streamlit Application

This application provides an interactive interface for predicting the probability
of winning sales opportunities in the next month using the trained MLflow model.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import storage manager and config
try:
    from src.config.config import get_config
    from src.utils.prefect_client import (
        CRMDeployments,
        PrefectFlowManager,
        format_duration,
        format_flow_run_state,
    )
    from src.utils.storage import StorageManager

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    st.warning("Config module not available - using fallback configuration")

warnings.filterwarnings("ignore")


def safe_convert_to_string(value):
    """Convert any UUID objects to strings to avoid PyArrow conversion errors."""
    import uuid

    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def safe_dataframe_display(data_dict):
    """Safely convert dictionary data for DataFrame display, handling UUID objects."""
    safe_data = {}
    for key, value in data_dict.items():
        if isinstance(value, list):
            safe_data[key] = [safe_convert_to_string(item) for item in value]
        else:
            safe_data[key] = safe_convert_to_string(value)
    return safe_data


# Configuration - supports both local and Docker environments
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5005")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MODEL_NAME = "monthly_win_probability_model"

# Set up environment for MLflow artifact access
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
os.environ["MINIO_ENDPOINT"] = MINIO_ENDPOINT

# Configure page
st.set_page_config(
    page_title="CRM Win Probability Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }

    .prediction-high {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }

    .prediction-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }

    .prediction-low {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }

    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data()
def get_current_period():
    try:
        # while True:
        if CONFIG_AVAILABLE:
            # Use storage manager for intelligent data loading
            config = get_config()
            storage_manager = StorageManager(config)

            # Try to load the latest features file
            try:
                # while True:
                # Look for the most recent CRM features file
                feature_files = storage_manager.list_files(
                    "features", "crm_features_*.csv"
                )
                if feature_files:
                    # Sort by name to get the most recent (assuming date in filename)
                    latest_file = sorted(feature_files)[-1]
                    filename = (
                        Path(latest_file).name
                        if storage_manager.use_s3
                        else latest_file
                    )
                    current_period = filename.split("_")[-1].replace(".csv", "")
                    # st.success(f"✅ Data loaded from {'MinIO' if storage_manager.use_s3 else 'local'}: {filename}")
                    return current_period
                else:
                    return None
            except Exception as storage_error:
                st.error(
                    f"Storage manager failed to load current period: {storage_error}"
                )
                return None
        else:
            # Fallback to local file check when config not available
            data_path = Path(__file__).parent.parent / "data" / "features"
            if data_path.exists():
                feature_files = list(data_path.glob("crm_features_*.csv"))
                if feature_files:
                    latest_file = sorted(feature_files)[-1]
                    current_period = latest_file.name.split("_")[-1].replace(".csv", "")
                    return current_period
            return None
    except Exception as storage_error:
        st.error(f"Storage manager failed to load current period: {storage_error}")
        return None


def get_next_period(current_period=None):
    """
    Calculate the next period based on the current period.
    Expected format: YYYY-MM (e.g., '2017-05')
    Returns the next month in the same format.
    """
    if current_period is None:
        current_period = get_current_period()

    if not current_period:
        return None

    try:
        # Parse the period string (expected format: YYYY-MM)
        year, month = map(int, current_period.split("-"))

        # Validate month range
        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}")

        # Calculate next month
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1

        # Format as YYYY-MM
        return f"{next_year:04d}-{next_month:02d}"

    except (ValueError, AttributeError) as e:
        st.error(f"Error parsing period '{current_period}': {e}")
        return None


def validate_period_format(period):
    """
    Validate that a period string is in the expected YYYY-MM format.
    Returns (is_valid, error_message)
    """
    if not period:
        return False, "Period is empty or None"

    try:
        parts = period.split("-")
        if len(parts) != 2:
            return False, f"Expected format YYYY-MM, got '{period}'"

        year, month = map(int, parts)

        if year < 1900 or year > 2100:
            return False, f"Year {year} seems invalid"

        if not (1 <= month <= 12):
            return False, f"Month {month} must be between 1-12"

        return True, "Valid format"

    except (ValueError, AttributeError) as e:
        return False, f"Invalid format '{period}': {e}"


def get_period_range(start_period, num_periods=12):
    """
    Generate a range of periods starting from start_period.
    Useful for creating simulation sequences.
    """
    if not start_period:
        return []

    is_valid, error_msg = validate_period_format(start_period)
    if not is_valid:
        st.error(f"Invalid start period: {error_msg}")
        return []

    periods = [start_period]
    current = start_period

    for _ in range(num_periods - 1):
        next_period = get_next_period(current)
        if not next_period:
            break
        periods.append(next_period)
        current = next_period

    return periods


@st.cache_data
def load_periods():
    """Load unique periods from the data"""
    try:
        # while True:
        if CONFIG_AVAILABLE:
            # Use storage manager for intelligent data loading
            config = get_config()
            storage_manager = StorageManager(config)
            df = storage_manager.load_dataframe("raw", "sales_pipeline.csv")
            df["creation_year_month"] = pd.to_datetime(df["engage_date"]).dt.to_period(
                "M"
            )
            periods = (
                df[df["creation_year_month"].notnull()]["creation_year_month"]
                .unique()
                .astype(str)
                .tolist()
            )
    except Exception as storage_error:
        st.error(f"Storage manager failed to load periods: {storage_error}")
    return periods


@st.cache_data()
def load_data():
    """Load the CRM features dataset from MinIO or local storage"""
    try:
        config = get_config()
        storage_manager = StorageManager(config)
        current_period = get_current_period()
        df = storage_manager.load_dataframe_by_type(
            "features", f"crm_features_{current_period}.csv"
        )
        df["engage_date"] = pd.to_datetime(df["engage_date"])
        df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("**Troubleshooting steps:**")
        st.info("1. Ensure MinIO is running: `make application-start`")
        st.info("2. Check if data files exist in MinIO bucket")
        st.info("3. Verify data pipeline has run: `make data-pipeline`")
        return None
        # Convert date columns
    return df

    # try:
    while True:
        if CONFIG_AVAILABLE:
            # Use storage manager for intelligent data loading
            config = get_config()
            storage_manager = StorageManager(config)

            # Try to load the latest features file
            # try:
            while True:
                # Look for the most recent CRM features file
                feature_files = storage_manager.list_files(
                    "features", "crm_features_*.csv"
                )
                if feature_files:
                    # Sort by name to get the most recent (assuming date in filename)
                    latest_file = sorted(feature_files)[-1]
                    filename = (
                        Path(latest_file).name
                        if storage_manager.use_s3
                        else latest_file
                    )
                    current_period = get_current_period()
                    df = storage_manager.load_dataframe_by_type(
                        "features", f"crm_features_{current_period}.csv"
                    )
                    # df = storage_manager.load_dataframe_by_type('features', filename)
                    # st.success(f"✅ Data loaded from {'MinIO' if storage_manager.use_s3 else 'local'}: {filename}")
                break
            # except Exception as storage_error:
            #    st.warning(f"Storage manager failed: {storage_error}")
            #    # Fallback to local file
            #    data_path = project_root / "data" / "features" / "crm_features.csv"
            #    df = pd.read_csv(data_path)
            #    st.info("📁 Loaded data from local fallback")
        else:
            # Fallback to local file when config not available
            data_path = project_root / "data" / "features" / "crm_features.csv"
            df = pd.read_csv(data_path)
            # st.info("📁 Loaded data from local file (config unavailable)")

        # Convert date columns
        df["engage_date"] = pd.to_datetime(df["engage_date"])
        df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce")

        return df


#    except Exception as e:
#        st.error(f"❌ Error loading data: {e}")
#        st.info("**Troubleshooting steps:**")
#        st.info("1. Ensure MinIO is running: `make application-start`")
#        st.info("2. Check if data files exist in MinIO bucket")
#        st.info("3. Verify data pipeline has run: `make data-pipeline`")
#        return None


def check_system_status():
    """Check the status of required services"""
    status = {}

    # Check MLflow
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
        status["MLflow"] = "✅ Running" if response.status_code == 200 else "❌ Error"
    except:
        status["MLflow"] = "❌ Not accessible"

    # Check MinIO
    try:
        response = requests.get(f"{MINIO_ENDPOINT}/minio/health/live", timeout=5)
        status["MinIO"] = "✅ Running" if response.status_code == 200 else "❌ Error"
    except:
        status["MinIO"] = "❌ Not accessible"

    # Check data availability using storage manager
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            storage_manager = StorageManager(config)

            try:
                # Look for the most recent CRM features file
                feature_files = storage_manager.list_files(
                    "features", "crm_features*.csv"
                )

                if feature_files:
                    # Sort by name to get the most recent (assuming date in filename)
                    latest_file = sorted(feature_files)[-1]
                    filename = (
                        Path(latest_file).name
                        if storage_manager.use_s3
                        else latest_file
                    )
                    status["Data (Features)"] = f"✅ Available in the file: {filename}"
                else:
                    status["Data (Features)"] = "❌ Missing"

            except Exception as storage_error:
                st.warning(f"Storage manager failed: {storage_error}")
                status["Data (Features)"] = "❌ Missing"

            # Count available feature files
            try:
                feature_files = storage_manager.list_files(
                    "features", "crm_features*.csv"
                )
                file_count = len(feature_files)
                status["Feature Files"] = (
                    f"✅ {file_count} files" if file_count > 0 else "❌ No files"
                )
            except:
                status["Feature Files"] = "❌ Cannot list"

        except Exception as e:
            status["Data (Features)"] = f"❌ Error: {str(e)[:30]}..."
            status["Feature Files"] = "❌ Storage error"
    else:
        # Fallback to local file check
        data_path = project_root / "data" / "features" / "crm_features.csv"
        status["Data File"] = "✅ Available" if data_path.exists() else "❌ Missing"

    # Check model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        models = client.search_registered_models()
        model_exists = any(m.name == MODEL_NAME for m in models)
        status["Model"] = "✅ Registered" if model_exists else "❌ Not found"
    except:
        status["Model"] = "❌ Error checking"

    return status


@st.cache_resource
def load_model():
    """Load the trained MLflow model"""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Load the latest model from registry
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("**Troubleshooting steps:**")
        st.info("1. Ensure MLflow server is running: `make dev-start`")
        st.info("2. Ensure MinIO is running: `make application-start`")
        st.info(
            "3. Verify model is registered: Check MLflow UI at http://localhost:5005"
        )
        st.info("4. Train model if needed: Run the training notebook")
        return None


def get_feature_columns():
    """Get the feature columns used by the model"""
    return [
        "close_value_log",
        "close_value_category_encoded",
        "days_since_engage",
        "sales_cycle_days",
        "engage_month",
        "engage_quarter",
        "engage_day_of_week",
        "agent_win_rate",
        "agent_opportunity_count",
        "product_win_rate",
        "product_popularity",
        "revenue",
        "employees",
        "is_repeat_account",
        "account_frequency",
        "should_close_next_month",
        "is_overdue",
        "is_early_stage",
        "high_value_deal",
        "long_sales_cycle",
        "sales_velocity",
        "sales_agent_encoded",
        "product_encoded",
        "account_encoded",
        "manager_encoded",
        "regional_office_encoded",
        "sector_encoded",
    ]


def create_monthly_features(df, current_date=None):
    """Create monthly prediction features for the input data"""
    if current_date is None:
        current_date = datetime.now()

    df = df.copy()

    # Calculate days since engagement
    df["days_since_engage"] = (current_date - df["engage_date"]).dt.days

    # Expected close timeframe
    avg_sales_cycle = df[df["is_closed"] == 1]["sales_cycle_days"].median()
    if pd.isna(avg_sales_cycle):
        avg_sales_cycle = 90  # Default to 90 days

    df["expected_close_date"] = df["engage_date"] + timedelta(days=avg_sales_cycle)
    df["days_to_expected_close"] = (df["expected_close_date"] - current_date).dt.days

    # Monthly prediction flags
    df["should_close_next_month"] = (
        df["days_to_expected_close"].between(-15, 45).astype(int)
    )
    df["is_overdue"] = (df["days_to_expected_close"] < -30).astype(int)
    df["is_early_stage"] = (df["days_to_expected_close"] > 60).astype(int)

    # Sales velocity features
    df["sales_velocity"] = df["close_value"] / df["sales_cycle_days"]
    df["sales_velocity"] = df["sales_velocity"].fillna(0)

    # Risk factors
    df["high_value_deal"] = (
        df["close_value"] > df["close_value"].quantile(0.8)
    ).astype(int)
    df["long_sales_cycle"] = (
        df["sales_cycle_days"] > df["sales_cycle_days"].quantile(0.8)
    ).astype(int)

    return df


def create_single_prediction_input():
    """Create input form for single opportunity prediction"""
    st.subheader("🎯 Single Opportunity Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Opportunity Details**")
            opportunity_id = st.text_input(
                "Opportunity ID",
                value=f"NEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            close_value = st.number_input(
                "Deal Value ($)", min_value=1.0, value=1000.0, step=100.0
            )
            engage_date = st.date_input(
                "Engagement Date", value=datetime.now() - timedelta(days=30)
            )

            # Sales agent selection
            df = load_data()
            if df is not None:
                agents = df["sales_agent"].unique()
                selected_agent = st.selectbox("Sales Agent", agents)

                # Product selection
                products = df["product"].unique()
                selected_product = st.selectbox("Product", products)

                # Account selection
                accounts = df["account"].unique()
                selected_account = st.selectbox("Account", accounts)

        with col2:
            st.write("**Account & Context**")
            revenue = st.number_input(
                "Account Revenue ($)", min_value=0.0, value=1000000.0, step=10000.0
            )
            employees = st.number_input(
                "Account Employees", min_value=1, value=100, step=1
            )

            regional_office = st.selectbox(
                "Regional Office", ["Central", "East", "West"]
            )
            sector = st.selectbox(
                "Sector",
                [
                    "technolgy",
                    "retail",
                    "finance",
                    "medical",
                    "services",
                    "marketing",
                    "employment",
                    "government",
                ],
            )

            is_repeat_customer = st.checkbox("Repeat Customer")

            st.write("**Sales Cycle Context**")
            estimated_sales_cycle = st.number_input(
                "Estimated Sales Cycle (days)", min_value=1, value=90, step=1
            )

        submitted = st.form_submit_button("🚀 Predict Win Probability")

        if submitted:
            # Create prediction data
            prediction_data = create_prediction_data(
                opportunity_id,
                close_value,
                engage_date,
                selected_agent,
                selected_product,
                selected_account,
                revenue,
                employees,
                regional_office,
                sector,
                is_repeat_customer,
                estimated_sales_cycle,
            )

            return prediction_data

    return None


def create_prediction_data(
    opportunity_id,
    close_value,
    engage_date,
    sales_agent,
    product,
    account,
    revenue,
    employees,
    regional_office,
    sector,
    is_repeat_customer,
    sales_cycle_days,
):
    """Create a prediction-ready dataset from form inputs"""

    # Load historical data for context
    df = load_data()
    if df is None:
        return None

    # Create new opportunity record
    new_opportunity = {
        "opportunity_id": opportunity_id,
        "sales_agent": sales_agent,
        "product": product,
        "account": account,
        "deal_stage": "Engaging",  # Assume engaging stage for prediction
        "engage_date": pd.to_datetime(engage_date),
        "close_date": pd.NaT,
        "close_value": close_value,
        "close_value_log": np.log(close_value) if close_value > 0 else 0,
        "revenue": revenue,
        "employees": employees,
        "regional_office": regional_office,
        "sector": sector,
        "is_repeat_account": 1 if is_repeat_customer else 0,
        "sales_cycle_days": sales_cycle_days,
        "is_closed": 0,
        "is_won": 0,
        "is_lost": 0,
        "is_open": 1,
    }

    # Get agent performance metrics from historical data
    agent_stats = (
        df[df["sales_agent"] == sales_agent].agg(
            {"opportunity_id": "count", "is_won": "mean"}
        )
        if sales_agent in df["sales_agent"].values
        else pd.Series({"opportunity_id": 1, "is_won": 0.5})
    )

    new_opportunity["agent_opportunity_count"] = agent_stats["opportunity_id"]
    new_opportunity["agent_win_rate"] = agent_stats["is_won"]

    # Get product performance metrics
    product_stats = (
        df[df["product"] == product].agg({"opportunity_id": "count", "is_won": "mean"})
        if product in df["product"].values
        else pd.Series({"opportunity_id": 1, "is_won": 0.5})
    )

    new_opportunity["product_popularity"] = product_stats["opportunity_id"]
    new_opportunity["product_win_rate"] = product_stats["is_won"]

    # Account frequency
    new_opportunity["account_frequency"] = (
        df[df["account"] == account]["opportunity_id"].count()
        if account in df["account"].values
        else 1
    )

    # Date features
    engage_dt = pd.to_datetime(engage_date)
    new_opportunity["engage_month"] = engage_dt.month
    new_opportunity["engage_quarter"] = engage_dt.quarter
    new_opportunity["engage_day_of_week"] = engage_dt.dayofweek

    # Deal value category (simplified)
    value_quantiles = df["close_value"].quantile([0.25, 0.5, 0.75])
    if close_value <= value_quantiles[0.25]:
        new_opportunity["close_value_category_encoded"] = 0  # Small
    elif close_value <= value_quantiles[0.5]:
        new_opportunity["close_value_category_encoded"] = 1  # Medium
    elif close_value <= value_quantiles[0.75]:
        new_opportunity["close_value_category_encoded"] = 2  # Large
    else:
        new_opportunity["close_value_category_encoded"] = 3  # Very Large

    # Create encodings for categorical variables
    new_opportunity["sales_agent_encoded"] = (
        df[df["sales_agent"] == sales_agent]["sales_agent_encoded"].iloc[0]
        if sales_agent in df["sales_agent"].values
        else 0
    )
    new_opportunity["product_encoded"] = (
        df[df["product"] == product]["product_encoded"].iloc[0]
        if product in df["product"].values
        else 0
    )
    new_opportunity["account_encoded"] = (
        df[df["account"] == account]["account_encoded"].iloc[0]
        if account in df["account"].values
        else 0
    )
    new_opportunity["manager_encoded"] = (
        df[df["sales_agent"] == sales_agent]["manager_encoded"].iloc[0]
        if sales_agent in df["sales_agent"].values
        else 0
    )
    new_opportunity["regional_office_encoded"] = (
        df[df["regional_office"] == regional_office]["regional_office_encoded"].iloc[0]
        if regional_office in df["regional_office"].values
        else 0
    )
    new_opportunity["sector_encoded"] = (
        df[df["sector"] == sector]["sector_encoded"].iloc[0]
        if sector in df["sector"].values
        else 0
    )

    # Convert to DataFrame
    new_df = pd.DataFrame([new_opportunity])

    # Create monthly features
    new_df = create_monthly_features(new_df)

    return new_df


def display_prediction_results(prediction_data, model):
    """Display prediction results with visualizations"""
    if prediction_data is None or model is None:
        return

    # Get feature columns and prepare data
    feature_cols = get_feature_columns()

    # Handle missing features
    for col in feature_cols:
        if col not in prediction_data.columns:
            prediction_data[col] = 0  # Default value for missing features

    X = prediction_data[feature_cols].fillna(0)

    try:
        # Make prediction
        win_probability = model.predict_proba(X)[0, 1]
        prediction_class = model.predict(X)[0]

        # Calculate expected revenue
        expected_revenue = prediction_data["close_value"].iloc[0] * win_probability

        # Display results
        st.success("✅ Prediction completed successfully!")

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Win Probability",
                value=f"{win_probability:.1%}",
                help="Probability of winning this opportunity",
            )

        with col2:
            st.metric(
                label="Deal Value",
                value=f"${prediction_data['close_value'].iloc[0]:,.0f}",
                help="Total value of the opportunity",
            )

        with col3:
            st.metric(
                label="Expected Revenue",
                value=f"${expected_revenue:,.0f}",
                help="Probability-weighted expected revenue",
            )

        with col4:
            risk_level = (
                "🟢 Low"
                if win_probability > 0.7
                else "🟡 Medium"
                if win_probability > 0.4
                else "🔴 High"
            )
            st.metric(
                label="Risk Level",
                value=risk_level,
                help="Risk of not closing this deal",
            )

        # Probability gauge
        st.subheader("📊 Win Probability Gauge")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=win_probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Win Probability (%)"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgray"},
                        {"range": [30, 60], "color": "yellow"},
                        {"range": [60, 80], "color": "orange"},
                        {"range": [80, 100], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("💡 Recommendations")

        if win_probability > 0.7:
            st.success("🎯 **High Priority Opportunity**")
            st.write("- Maintain regular contact with the client")
            st.write("- Prepare detailed proposal and timeline")
            st.write("- Schedule final decision meeting")
        elif win_probability > 0.4:
            st.warning("⚠️ **Medium Priority - Needs Attention**")
            st.write("- Identify and address potential objections")
            st.write("- Strengthen value proposition")
            st.write("- Consider involving senior sales support")
        else:
            st.error("🚨 **Low Priority - High Risk**")
            st.write("- Re-evaluate opportunity qualification")
            st.write("- Consider alternative approaches or timeline")
            st.write("- Focus resources on higher probability deals")

        # Feature importance (simplified)
        st.subheader("🔍 Key Factors")

        key_factors = {
            "Agent Win Rate": prediction_data["agent_win_rate"].iloc[0],
            "Product Win Rate": prediction_data["product_win_rate"].iloc[0],
            "Deal Size": "Large"
            if prediction_data["close_value"].iloc[0] > 5000
            else "Medium"
            if prediction_data["close_value"].iloc[0] > 1000
            else "Small",
            "Sales Cycle": f"{prediction_data['sales_cycle_days'].iloc[0]} days",
            "Account Status": "Repeat Customer"
            if prediction_data["is_repeat_account"].iloc[0]
            else "New Customer",
        }

        for factor, value in key_factors.items():
            if isinstance(value, float):
                st.write(f"**{factor}**: {value:.1%}")
            else:
                st.write(f"**{factor}**: {value}")

    except Exception as e:
        st.error(f"Error making prediction: {e}")


def show_simulation_controls():
    st.header("🔧 Simulation Control")
    st.write("Control the simulation environment for testing and development purposes.")
    if not CONFIG_AVAILABLE:
        st.error("❌ Configuration not available - cannot connect to Prefect server")
        return
    # Initialize Prefect manager
    try:
        prefect_manager = PrefectFlowManager()
    except Exception as e:
        st.error(f"❌ Error initializing Prefect client: {e}")
        return
    # Check Prefect server health
    is_healthy, health_msg = prefect_manager.check_server_health()
    st.write("**Prefect Server Status:**", health_msg)
    if not is_healthy:
        st.info(
            "💡 **Troubleshooting:** Ensure Prefect server is running with `make application-start`"
        )
        return

    # Get deployments
    try:
        deployments = prefect_manager.get_deployments_sync()
        deployment_names = [d["name"] for d in deployments]
        if not deployments:
            st.warning("⚠️ No Prefect deployments found")
            st.info(
                "💡 **Setup required:** Deploy flows first with `make prefect-deploy`"
            )
            return
    except Exception as e:
        st.error(f"❌ Error fetching deployments: {e}")
        return

    col1, col2 = st.columns([0.3, 0.7])
    # Create workflow control sections
    with col1:
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🚀 Simulation Control",
                "📊 Flow Status",
                "⚙️ Deployments",
                "⚠️ Advanced Controls",
            ]
        )

        with tab1:
            st.subheader("🔄 Reset Simulation")
            period = st.selectbox(
                "Select Start Period",
                options=load_periods(),
                index=0,
                key="start_period",
            )
            if st.button("Reset to Start Period"):
                if CRMDeployments.DATA_CLEANUP in deployment_names:
                    parameters = {"dry_run": True, "file_categories": ["all"]}
                    trigger_flow(
                        prefect_manager,
                        CRMDeployments.DATA_CLEANUP,
                        f"Data Cleanup (Execute)",
                        parameters,
                    )
                else:
                    st.warning("❌ Cleanup Pipeline is not deployed")
            st.subheader("📊 Move to the next period")
            if st.button("Simulate Period Change"):
                next_period = get_next_period(get_current_period())
                if next_period:
                    parameters = {"snapshot_month": next_period}
                    trigger_flow(
                        prefect_manager,
                        CRMDeployments.DATA_INGESTION,
                        "Period Change Simulation",
                        parameters,
                    )
                    # Refresh data cache
                    st.cache_data.clear()
                else:
                    st.error("❌ Could not determine next period")
        with tab2:
            show_flow_status(prefect_manager)
        with tab3:
            show_deployment_info(prefect_manager, deployments)
        with tab4:
            show_advanced_pipeline_controls(prefect_manager, deployments)
    with col2:
        pass


def show_advanced_pipeline_controls(
    prefect_manager: "PrefectFlowManager", deployments: List[Dict[str, Any]]
):
    """Show controls for running different MLOps pipelines."""
    st.subheader("⚠️ Pipeline Execution")

    deployment_names = [d["name"] for d in deployments]

    # Data Acquisition Pipeline
    st.write("### 📥 Data Acquisition")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.write("Download and acquire raw CRM sales data from Kaggle.")
        # Parameters for data acquisition
        snapshot_month = st.selectbox(
            "Snapshot Month", load_periods(), key="acquisition_snapshot_month"
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if CRMDeployments.DATA_ACQUISITION in deployment_names:
            if st.button("🚀 Run Data Acquisition", key="run_acquisition"):
                parameters = {"snapshot_month": snapshot_month}
                trigger_flow(
                    prefect_manager,
                    CRMDeployments.DATA_ACQUISITION,
                    "Data Acquisition",
                    parameters,
                )
        else:
            st.warning("❌ Not deployed")

    # Data Ingestion Pipeline
    st.write("### 🔄 Data Ingestion")
    col1, col2 = st.columns([2, 2])
    with col1:
        if CONFIG_AVAILABLE:
            try:
                config = get_config()
                storage_manager = StorageManager(config)
                feature_files = storage_manager.list_files(
                    "raw", "sales_pipeline_*.csv"
                )
                periods = [f.split("_")[-1].replace(".csv", "") for f in feature_files]
                periods = sorted(list(set(periods)))
                # Add common periods if none found
                if not periods:
                    st.warning("❌ Not raw data available - cannot run ingestion")
            except:
                st.warning(
                    "❌ Getting periods from raw data failed - cannot run ingestion"
                )
        st.write("Process and validate acquired data, create features.")
        ingestion_month = st.selectbox(
            "Processing Month", periods, key="ingestion_snapshot_month"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if CRMDeployments.DATA_INGESTION in deployment_names:
            if st.button("🚀 Run Data Ingestion", key="run_ingestion"):
                parameters = {"snapshot_month": ingestion_month}
                trigger_flow(
                    prefect_manager,
                    CRMDeployments.DATA_INGESTION,
                    "Data Ingestion",
                    parameters,
                )
        else:
            st.warning("❌ Not deployed")

    # Model Training Pipeline
    st.write("### 🎯 Model Training")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.write("Train monthly win probability prediction model.")
        training_month = st.selectbox(
            "Training Month", periods, key="training_snapshot_month"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if CRMDeployments.MONTHLY_TRAINING in deployment_names:
            if st.button("🚀 Run Model Training", key="run_training"):
                parameters = {"current_month": training_month}
                trigger_flow(
                    prefect_manager,
                    CRMDeployments.MONTHLY_TRAINING,
                    "Model Training",
                    parameters,
                )
        else:
            st.warning("❌ Not deployed")

    # Reference Data Creation
    st.write("### 📊 Reference Data Creation")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.write("Create reference dataset for drift monitoring.")

        # Get available periods for parameters
        if CONFIG_AVAILABLE:
            try:
                config = get_config()
                storage_manager = StorageManager(config)
                feature_files = storage_manager.list_files(
                    "features", "crm_features_*.csv"
                )
                periods = [f.split("_")[-1].replace(".csv", "") for f in feature_files]
                periods = sorted(list(set(periods)))
                if not periods:
                    st.warning("❌ Not raw data available - cannot run ingestion")
            except:
                st.warning(
                    "❌ Getting periods from raw data failed - cannot run ingestion"
                )

        ref_period = st.selectbox(
            "Reference Period", periods, key="ref_period_workflow"
        )
        sample_size = st.number_input(
            "Sample Size",
            min_value=100,
            max_value=5000,
            value=1000,
            key="ref_sample_workflow",
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if CRMDeployments.REFERENCE_DATA in deployment_names:
            if st.button("🚀 Create Reference Data", key="run_reference"):
                parameters = {
                    "reference_period": ref_period,
                    "sample_size": sample_size,
                }
                trigger_flow(
                    prefect_manager,
                    CRMDeployments.REFERENCE_DATA,
                    "Reference Data Creation",
                    parameters,
                )
        else:
            st.warning("❌ Not deployed")

    # Drift Monitoring
    st.write("### 🔍 Drift Monitoring")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.write("Analyze model drift and data quality.")
        current_period = st.selectbox(
            "Current Period",
            periods,
            index=1 if len(periods) > 1 else 0,
            key="current_period_workflow",
        )
        ref_period_monitor = st.selectbox(
            "Reference Period", periods, key="ref_period_monitor_workflow"
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if CRMDeployments.DRIFT_MONITORING in deployment_names:
            if st.button("🚀 Run Drift Monitoring", key="run_drift"):
                parameters = {
                    "current_month": current_period,
                    "reference_period": ref_period_monitor,
                }
                trigger_flow(
                    prefect_manager,
                    CRMDeployments.DRIFT_MONITORING,
                    "Drift Monitoring",
                    parameters,
                )
        else:
            st.warning("❌ Not deployed")

    # Data Cleanup
    st.write("### 🧹 Data Cleanup")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.write("Clean up processed CSV files (preserves Kaggle source data).")
        dry_run_cleanup = st.checkbox(
            "Dry Run (preview only)", value=True, key="dry_run_cleanup"
        )

        # File category selection
        st.write("**Select cleanup categories:**")
        cleanup_all = st.checkbox("All processed files", key="cleanup_all")
        cleanup_features = st.checkbox(
            "Feature files", key="cleanup_features", disabled=cleanup_all
        )
        cleanup_processed = st.checkbox(
            "Processed files", key="cleanup_processed", disabled=cleanup_all
        )
        cleanup_temp = st.checkbox(
            "Temporary files", key="cleanup_temp", disabled=cleanup_all
        )

        # Build file categories list
        file_categories = []
        if cleanup_all:
            file_categories = ["all"]
        else:
            if cleanup_features:
                file_categories.append("features")
            if cleanup_processed:
                file_categories.append("processed")
            if cleanup_temp:
                file_categories.append("temp")

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if CRMDeployments.DATA_CLEANUP in deployment_names:
            # Only enable button if at least one category is selected
            button_disabled = len(file_categories) == 0
            if button_disabled:
                st.warning("⚠️ Select at least one cleanup category")

            if st.button(
                "🧹 Run Data Cleanup", key="run_cleanup", disabled=button_disabled
            ):
                parameters = {
                    "dry_run": dry_run_cleanup,
                    "file_categories": file_categories,
                }
                action_text = "Preview" if dry_run_cleanup else "Execute"
                trigger_flow(
                    prefect_manager,
                    CRMDeployments.DATA_CLEANUP,
                    f"Data Cleanup ({action_text})",
                    parameters,
                )
        else:
            st.warning("❌ Not deployed")


def show_flow_status(prefect_manager: "PrefectFlowManager"):
    """Show recent flow runs and their status."""
    st.subheader("📊 Recent Flow Runs")

    try:
        flow_runs = prefect_manager.get_recent_flow_runs_sync(limit=15)

        if not flow_runs:
            st.info("No recent flow runs found")
            return

        # Create a dataframe for better display
        runs_data = []
        for run in flow_runs:
            run_data = {
                "Flow": safe_convert_to_string(run["flow_name"]),
                "Status": format_flow_run_state(run["state_type"], run["state_name"]),
                "Started": run["start_time"].strftime("%m-%d %H:%M")
                if run["start_time"]
                else "N/A",
                "Duration": format_duration(run["total_run_time"])
                if run["total_run_time"]
                else "N/A",
                "Run ID": str(run["id"])[:8] + "...",
            }
            runs_data.append(run_data)

        runs_df = pd.DataFrame(runs_data)
        st.dataframe(runs_df, use_container_width=True)

        # Show detailed view for running flows
        running_flows = [run for run in flow_runs if run["state_type"] == "RUNNING"]
        if running_flows:
            st.write("### 🏃 Currently Running Flows")
            for run in running_flows:
                with st.expander(f"{run['flow_name']} - Running"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(
                            f"**Started:** {run['start_time'].strftime('%Y-%m-%d %H:%M:%S') if run['start_time'] else 'N/A'}"
                        )
                        st.write(f"**Run ID:** `{run['id']}`")

                    with col2:
                        if run["parameters"]:
                            st.write("**Parameters:**")
                            for key, value in run["parameters"].items():
                                st.write(f"  • {key}: {value}")

                    with col3:
                        if st.button("🛑 Cancel Flow", key=f"cancel_{run['id'][:8]}"):
                            success, message = prefect_manager.cancel_flow_run_sync(
                                run["id"]
                            )
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

        # Refresh button
        if st.button("🔄 Refresh Status"):
            st.rerun()

    except Exception as e:
        st.error(f"Error fetching flow runs: {e}")


def show_deployment_info(
    prefect_manager: "PrefectFlowManager", deployments: List[Dict[str, Any]]
):
    """Show deployment information and management."""
    st.subheader("⚙️ Deployment Information")

    if not deployments:
        st.info("No deployments found")
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
            with st.expander(f"📋 {deployment['name']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Flow Name:** {deployment['flow_name']}")
                    st.write(
                        f"**Description:** {deployment.get('description', 'No description')}"
                    )
                    st.write(
                        f"**Created:** {deployment['created'].strftime('%Y-%m-%d %H:%M') if deployment['created'] else 'N/A'}"
                    )
                    st.write(
                        f"**Updated:** {deployment['updated'].strftime('%Y-%m-%d %H:%M') if deployment['updated'] else 'N/A'}"
                    )

                with col2:
                    schedule_status = (
                        "📅 Scheduled"
                        if deployment["is_schedule_active"]
                        else "🚫 Manual Only"
                    )
                    st.write(f"**Schedule:** {schedule_status}")

                    if deployment["tags"]:
                        st.write("**Tags:**")
                        for tag in deployment["tags"]:
                            st.write(f"  • {tag}")

                    # Quick run button
                    if st.button("▶️ Quick Run", key=f"quick_run_{deployment['name']}"):
                        trigger_flow(
                            prefect_manager, deployment["name"], deployment["flow_name"]
                        )


def trigger_flow(
    prefect_manager: "PrefectFlowManager",
    deployment_name: str,
    display_name: str,
    parameters: Dict[str, Any] = None,
):
    """Trigger a flow and show the result."""
    with st.spinner(f"Triggering {display_name} flow..."):
        try:
            success, flow_run_id, message = prefect_manager.trigger_deployment_sync(
                deployment_name, parameters
            )

            if success:
                st.success(message)
                st.info(f"🔍 **Track progress:** Flow run ID `{flow_run_id}`")
                st.info(
                    "🌐 **Monitor in UI:** [Prefect Dashboard](http://localhost:4200/flow-runs)"
                )

                # Store in session state for quick access
                if "recent_flow_runs" not in st.session_state:
                    st.session_state.recent_flow_runs = []
                st.session_state.recent_flow_runs.append(
                    {
                        "id": flow_run_id,
                        "name": display_name,
                        "deployment": deployment_name,
                        "triggered_at": datetime.now(),
                        "parameters": parameters,
                    }
                )

                # Keep only last 10 runs
                if len(st.session_state.recent_flow_runs) > 10:
                    st.session_state.recent_flow_runs = (
                        st.session_state.recent_flow_runs[-10:]
                    )
            else:
                st.error(message)
        except Exception as e:
            st.error(f"Error triggering {display_name}: {e}")


def show_pipeline_overview():
    """Show overview of current sales pipeline"""
    st.subheader("📈 Sales Pipeline Overview")

    df = load_data()
    if df is None:
        return

    # Filter to open opportunities
    pipeline = df[df["is_closed"] == 0].copy()

    if len(pipeline) == 0:
        st.info("No open opportunities found in the pipeline.")
        return

    # Create monthly features for pipeline
    pipeline = create_monthly_features(pipeline)

    # Load model and make predictions
    model = load_model()
    if model is None:
        return

    feature_cols = get_feature_columns()

    # Handle missing features
    for col in feature_cols:
        if col not in pipeline.columns:
            pipeline[col] = 0

    X = pipeline[feature_cols].fillna(0)

    try:
        # Predict for all opportunities
        pipeline["win_probability"] = model.predict_proba(X)[:, 1]
        pipeline["expected_revenue"] = (
            pipeline["close_value"] * pipeline["win_probability"]
        )

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Opportunities", len(pipeline))

        with col2:
            st.metric("Total Pipeline Value", f"${pipeline['close_value'].sum():,.0f}")

        with col3:
            st.metric("Expected Revenue", f"${pipeline['expected_revenue'].sum():,.0f}")

        with col4:
            avg_win_prob = pipeline["win_probability"].mean()
            st.metric("Avg Win Probability", f"{avg_win_prob:.1%}")

        # Pipeline by probability categories
        pipeline["probability_category"] = pd.cut(
            pipeline["win_probability"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=[
                "Low (0-30%)",
                "Medium (30-60%)",
                "High (60-80%)",
                "Very High (80%+)",
            ],
        )

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            prob_dist = pipeline["probability_category"].value_counts()
            fig = px.pie(
                values=prob_dist.values,
                names=prob_dist.index,
                title="Opportunities by Win Probability",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            revenue_by_prob = pipeline.groupby("probability_category")[
                "expected_revenue"
            ].sum()
            fig = px.bar(
                x=revenue_by_prob.index,
                y=revenue_by_prob.values,
                title="Expected Revenue by Probability Category",
            )
            fig.update_layout(yaxis_title="Expected Revenue ($)")
            st.plotly_chart(fig, use_container_width=True)

        # Top opportunities table
        st.subheader("🏆 Top 10 Opportunities by Win Probability")
        top_opps = pipeline.nlargest(10, "win_probability")[
            [
                "opportunity_id",
                "sales_agent",
                "account",
                "product",
                "close_value",
                "win_probability",
                "expected_revenue",
            ]
        ].round(3)

        # Format the dataframe for display
        top_opps["close_value"] = top_opps["close_value"].apply(lambda x: f"${x:,.0f}")
        top_opps["win_probability"] = top_opps["win_probability"].apply(
            lambda x: f"{x:.1%}"
        )
        top_opps["expected_revenue"] = top_opps["expected_revenue"].apply(
            lambda x: f"${x:,.0f}"
        )

        st.dataframe(top_opps, use_container_width=True)

    except Exception as e:
        st.error(f"Error analyzing pipeline: {e}")


def main():
    """Main application"""

    # Header
    st.title("🎯 CRM Monthly Win Probability Predictor")

    # Display current period
    current_period = get_current_period()
    if current_period:
        next_period = get_next_period(current_period)
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📅 **Current Period:** {current_period}")
        with col2:
            if next_period:
                st.info(f"➡️ **forecasted Period:** {next_period}")
            else:
                st.warning("⚠️ **Next Period:** Unable to calculate")
    else:
        st.warning("⚠️ **Current Period:** Not available - please run data pipeline")

    st.markdown("---")

    # Create tabs for navigation
    tab2, tab1, tab5, tab4, tab3 = st.tabs(
        [
            "📊 Pipeline Overview",
            "🔮 Single Prediction",
            "🔍 Model Monitoring",
            "🔧 Simulation Control",
            "🤖 Model Information",
        ]
    )

    with tab1:
        st.header("Individual Opportunity Prediction")
        st.write(
            "Enter opportunity details to predict win probability for the next month."
        )

        prediction_data = create_single_prediction_input()

        if prediction_data is not None:
            model = load_model()
            display_prediction_results(prediction_data, model)

    with tab2:
        st.header("Sales Pipeline Analysis")
        st.write("Overview of all open opportunities with win probability predictions.")

        show_pipeline_overview()

    with tab5:
        st.header("Model Drift Monitoring")
        st.write("Monitor model performance and detect data drift using Evidently AI.")

        # try:
        from src.monitoring.streamlit_dashboard import show_monitoring_dashboard

        show_monitoring_dashboard()
        # except ImportError:
        #    st.error("Monitoring dashboard not available - missing dependencies")
        #    st.info("Install monitoring dependencies: pip install evidently")
        # except Exception as e:
        #    st.error(f"Error loading monitoring dashboard: {e}")

    with tab4:
        show_simulation_controls()

    with tab3:
        st.header("Model Information")

        # System Status Check
        st.subheader("🔧 System Status")
        with st.spinner("Checking system status..."):
            status = check_system_status()

        col1, col2 = st.columns(2)
        with col1:
            for service, status_text in list(status.items())[: len(status) // 2 + 1]:
                st.write(f"**{service}**: {status_text}")
        with col2:
            for service, status_text in list(status.items())[len(status) // 2 + 1 :]:
                st.write(f"**{service}**: {status_text}")

        # Storage Information
        st.subheader("💾 Storage Configuration")
        if CONFIG_AVAILABLE:
            try:
                config = get_config()
                storage_manager = StorageManager(config)
                storage_info = storage_manager.get_storage_info()

                st.write(f"**Storage Type**: {storage_info['storage_type']}")
                if storage_info["use_s3"]:
                    st.write(f"**Data Bucket**: {storage_info.get('bucket', 'N/A')}")
                    st.write(
                        f"**MinIO Endpoint**: {storage_info.get('endpoint', 'N/A')}"
                    )
                else:
                    st.write("**Local Path**: data/features/")

            except Exception as e:
                st.error(f"Error getting storage info: {e}")
        else:
            st.info("Storage manager not available - using local files")

        # Connection URLs
        st.write("**Service URLs:**")
        st.write(f"- MLflow UI: [{MLFLOW_TRACKING_URI}]({MLFLOW_TRACKING_URI})")
        st.write("- MinIO UI: [http://localhost:9001](http://localhost:9001)")

        if not all("✅" in status_text for status_text in status.values()):
            st.warning(
                "⚠️ Some services are not available. Please check the troubleshooting guide below."
            )

        st.subheader("About the Model")
        st.write(
            """
        This application uses a machine learning model trained to predict the probability
        of winning sales opportunities within the next month. The model considers various
        factors including:

        - **Historical Performance**: Agent and product win rates
        - **Deal Characteristics**: Value, sales cycle, engagement timeline
        - **Account Context**: Company size, sector, relationship history
        - **Timing Factors**: Seasonality, sales cycle stage, urgency
        """
        )

        st.subheader("Model Performance")
        st.info(
            """
        The model has been trained and validated on historical CRM data with:
        - High accuracy in probability calibration
        - Strong discrimination between wins and losses
        - Regular retraining to maintain performance
        """
        )

        st.subheader("Usage Guidelines")
        st.write(
            """
        **Best Practices:**
        - Use predictions as guidance, not absolute truth
        - Combine with sales judgment and market knowledge
        - Focus on high-probability opportunities for resource allocation
        - Review and act on low-probability deals to improve outcomes

        **Limitations:**
        - Based on historical patterns, may not capture new market dynamics
        - Requires accurate and complete input data
        - Should be updated regularly with new data
        """
        )

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and MLflow • MLOps Zoomcamp Project*")


if __name__ == "__main__":
    main()
