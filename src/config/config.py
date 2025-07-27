"""Configuration management for MLOps platform."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DataPathConfig:
    """Data pipeline configuration."""
    # Local filesystem paths (for direct execution)
    raw: str = "data/raw"
    processed: str = "data/processed"
    features: str = "data/features"


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "http://localhost:5000"
    artifact_location: str = "mlruns"


@dataclass
class PrefectConfig:
    """Prefect configuration."""
    api_url: str = "http://localhost:4200"
    work_pool: str = "default-agent-pool"


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    monitoring_frequency: str = "daily"
    alert_recipients: list = field(default_factory=lambda: ["admin@mlops-platform.com"])


@dataclass
class StorageConfig:
    """Storage configuration for S3/MinIO."""
    endpoint_url: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    region: str = "us-east-1"
    buckets: Dict[str, str] = field(default_factory=lambda: {
        "mlflow_artifacts": "mlflow-artifacts",
        "data_lake": "data-lake",
        "model_artifacts": "model-artifacts",
        "configurations": "configurations"
    })
    # S3/MinIO paths for different data types (within buckets)
    s3_paths: Dict[str, str] = field(default_factory=lambda: {
        "raw": "raw",
        "processed": "processed", 
        "features": "features",
        "models": "models",
        "experiments": "experiments",
        "prefect_flows": "prefect-flows"
    })
    # Legacy data_paths for backward compatibility
    data_paths: Dict[str, str] = field(default_factory=lambda: {
        "raw": "raw",
        "processed": "processed", 
        "features": "features",
        "models": "models",
        "experiments": "experiments",
        "prefect_flows": "prefect-flows"
    })


@dataclass
class Config:
    """Main configuration class."""
    first_snapshot_month: str = "XXXX-XX" # Placeholder for actual month
    data_path: DataPathConfig = field(default_factory=DataPathConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    prefect: PrefectConfig = field(default_factory=PrefectConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
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
    
    For staging and production environments, configuration is loaded entirely
    from environment variables. For development, YAML config is used as base
    with environment variable overrides.
    
    Args:
        config_path: Path to YAML config file (ignored for staging/production).
        
    Returns:
        Config instance with all settings.
    """
    # Determine environment first
    environment = os.getenv('ENVIRONMENT', 'development')
    
    # Create base config
    config = Config()
    
    # For staging and production, skip YAML loading - use env vars only
    yaml_config = {}
    if environment == 'development':
        yaml_config = load_config_from_yaml(config_path)
    
    # Update with YAML values
    if yaml_config:
        # Data config
        if 'data_path' in yaml_config:
            for key, value in yaml_config['data_path'].items():
                if key == 'local_paths' and isinstance(value, dict):
                    # Handle nested local_paths structure
                    for local_key, local_value in value.items():
                        if hasattr(config.data_path, local_key):
                            setattr(config.data_path, local_key, local_value)
                elif hasattr(config.data_path, key):
                    setattr(config.data_path, key, value)

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
        
        # Storage config (MinIO/S3)
        if 'storage' in yaml_config:
            for key, value in yaml_config['storage'].items():
                if key == 's3_paths' and isinstance(value, dict):
                    # Handle nested s3_paths structure
                    config.storage.s3_paths.update(value)
                    # Keep data_paths in sync for backward compatibility
                    config.storage.data_paths.update(value)
                elif hasattr(config.storage, key):
                    setattr(config.storage, key, value)
        
        # Top-level config
        for key in ['first_snapshot_month', 'environment', 'debug', 'log_level', 'database_url', 'redis_url']:
            if key in yaml_config:
                setattr(config, key, yaml_config[key])
    
    # Override with environment variables (comprehensive for staging/production)
    
    # Data configuration overrides
    config.data_path.raw = os.getenv('RAW_DATA_PATH', config.data_path.raw)
    config.data_path.processed = os.getenv('PROCESSED_DATA_PATH', config.data_path.processed)
    config.data_path.features = os.getenv('FEATURE_STORE_PATH', config.data_path.features)

    # MLflow configuration overrides
    config.mlflow.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', config.mlflow.tracking_uri)
    config.mlflow.artifact_location = os.getenv('MLFLOW_ARTIFACT_LOCATION', config.mlflow.artifact_location)
    
    # Prefect configuration overrides
    config.prefect.api_url = os.getenv('PREFECT_API_URL', config.prefect.api_url)
    config.prefect.work_pool = os.getenv('PREFECT_WORK_POOL', config.prefect.work_pool)
    
    # Storage configuration overrides
    config.storage.endpoint_url = os.getenv('MINIO_ENDPOINT', config.storage.endpoint_url)
    config.storage.access_key = os.getenv('MINIO_ROOT_USER', config.storage.access_key)
    config.storage.secret_key = os.getenv('MINIO_ROOT_PASSWORD', config.storage.secret_key)
    config.storage.region = os.getenv('AWS_DEFAULT_REGION', config.storage.region)
    
    # Override bucket names from environment
    config.storage.buckets['mlflow_artifacts'] = os.getenv('MLFLOW_ARTIFACTS_BUCKET', config.storage.buckets['mlflow_artifacts'])
    config.storage.buckets['data_lake'] = os.getenv('DATA_LAKE_BUCKET', config.storage.buckets['data_lake'])
    config.storage.buckets['model_artifacts'] = os.getenv('MODEL_ARTIFACTS_BUCKET', config.storage.buckets['model_artifacts'])
    
    # Override S3 data paths from environment (update both s3_paths and data_paths for compatibility)
    s3_raw_path = os.getenv('S3_RAW_DATA_PATH', config.storage.s3_paths['raw'])
    s3_processed_path = os.getenv('S3_PROCESSED_DATA_PATH', config.storage.s3_paths['processed'])
    s3_features_path = os.getenv('S3_FEATURES_DATA_PATH', config.storage.s3_paths['features'])
    s3_models_path = os.getenv('S3_MODELS_PATH', config.storage.s3_paths['models'])
    s3_experiments_path = os.getenv('S3_EXPERIMENTS_PATH', config.storage.s3_paths['experiments'])
    s3_prefect_path = os.getenv('S3_PREFECT_FLOWS_PATH', config.storage.s3_paths['prefect_flows'])
    
    # Update both s3_paths and data_paths
    config.storage.s3_paths.update({
        'raw': s3_raw_path,
        'processed': s3_processed_path,
        'features': s3_features_path,
        'models': s3_models_path,
        'experiments': s3_experiments_path,
        'prefect_flows': s3_prefect_path
    })
    
    # Keep data_paths in sync for backward compatibility
    config.storage.data_paths.update(config.storage.s3_paths)
    
    # Top-level configuration overrides
    config.environment = os.getenv('ENVIRONMENT', config.environment)
    config.debug = os.getenv('DEBUG', str(config.debug)).lower() == 'true'
    config.log_level = os.getenv('LOG_LEVEL', config.log_level)
    config.database_url = os.getenv('DATABASE_URL', config.database_url)
    config.redis_url = os.getenv('REDIS_URL', config.redis_url)
    
    return config
