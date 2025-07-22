"""Configuration management for MLOps platform."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    feature_store_path: str = "data/features"
    kaggle_dataset: str = "innocentmfa/crm-sales-opportunities"
    train_test_split: float = 0.8
    validation_split: float = 0.2


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "crm-sales-prediction"
    artifact_location: str = "mlruns"
    model_registry_name: str = "crm-sales-model"


@dataclass
class PrefectConfig:
    """Prefect configuration."""
    api_url: str = "http://localhost:4200"
    work_pool: str = "default-agent-pool"
    deployment_name: str = "crm-training-pipeline"


@dataclass
class ModelConfig:
    """Model training configuration."""
    target_column: str = "deal_stage"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    models_to_train: list = field(default_factory=lambda: [
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
        "xgboost"
    ])


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    drift_threshold: float = 0.1
    performance_threshold: float = 0.8
    monitoring_frequency: str = "daily"
    alert_recipients: list = field(default_factory=lambda: ["admin@mlops-platform.com"])


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    prefect: PrefectConfig = field(default_factory=PrefectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    database_url: str = "postgresql://mlops_user:mlops_password@localhost:5432/mlops"
    redis_url: str = "redis://localhost:6379"


def load_config_from_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default path.
        
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        # Look for config in standard locations
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "development.yaml"
    
    if not Path(config_path).exists():
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration with environment variable overrides.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Config instance with all settings.
    """
    # Load from YAML
    yaml_config = load_config_from_yaml(config_path)
    
    # Create base config
    config = Config()
    
    # Update with YAML values
    if yaml_config:
        # Data config
        if 'data' in yaml_config:
            for key, value in yaml_config['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # MLflow config
        if 'mlflow' in yaml_config:
            for key, value in yaml_config['mlflow'].items():
                if hasattr(config.mlflow, key):
                    setattr(config.mlflow, key, value)
        
        # Prefect config
        if 'prefect' in yaml_config:
            for key, value in yaml_config['prefect'].items():
                if hasattr(config.prefect, key):
                    setattr(config.prefect, key, value)
        
        # Model config
        if 'model' in yaml_config:
            for key, value in yaml_config['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Monitoring config
        if 'monitoring' in yaml_config:
            for key, value in yaml_config['monitoring'].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
        
        # Top-level config
        for key in ['environment', 'debug', 'log_level', 'database_url', 'redis_url']:
            if key in yaml_config:
                setattr(config, key, yaml_config[key])
    
    # Override with environment variables
    config.data.raw_data_path = os.getenv('RAW_DATA_PATH', config.data.raw_data_path)
    config.data.processed_data_path = os.getenv('PROCESSED_DATA_PATH', config.data.processed_data_path)
    config.data.kaggle_dataset = os.getenv('KAGGLE_DATASET', config.data.kaggle_dataset)
    
    config.mlflow.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', config.mlflow.tracking_uri)
    config.mlflow.experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', config.mlflow.experiment_name)
    
    config.prefect.api_url = os.getenv('PREFECT_API_URL', config.prefect.api_url)
    
    config.environment = os.getenv('ENVIRONMENT', config.environment)
    config.debug = os.getenv('DEBUG', str(config.debug)).lower() == 'true'
    config.log_level = os.getenv('LOG_LEVEL', config.log_level)
    config.database_url = os.getenv('DATABASE_URL', config.database_url)
    config.redis_url = os.getenv('REDIS_URL', config.redis_url)
    
    return config
