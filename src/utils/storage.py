"""Storage utilities for file and S3 operations."""

import os
import io
import logging
import boto3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO
from botocore.exceptions import ClientError, NoCredentialsError


class StorageManager:
    """Unified storage manager for local files and S3/MinIO."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage manager.
        
        Args:
            config: Configuration dictionary with storage settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Determine if we're running in Docker/containerized environment
        self.use_s3 = self._should_use_s3()
        
        if self.use_s3:
            self._setup_s3_client()
        
        self.logger.info(f"Storage mode: {'S3/MinIO' if self.use_s3 else 'Local filesystem'}")
    
    def _should_use_s3(self) -> bool:
        """Determine if S3 storage should be used.
        
        Returns:
            True if S3 should be used, False for local storage.
        """
        # Check for Docker environment indicator
        if os.path.exists('/.dockerenv'):
            self.logger.info("Docker environment detected - using S3 storage")
            return True
        
        # Check for explicit environment variable
        if os.getenv('USE_S3_STORAGE', '').lower() in ['true', '1', 'yes']:
            self.logger.info("S3 storage forced via environment variable")
            return True
        
        # Check if we're running in Prefect
        if os.getenv('PREFECT_API_URL'):
            self.logger.info("Prefect environment detected - using S3 storage")
            return True
        
        # Default to local storage
        self.logger.info("Local environment detected - using local filesystem")
        return False
    
    def _setup_s3_client(self):
        """Setup S3/MinIO client."""
        try:
            minio_config = self.config.get('minio', {})
            
            # Get S3 configuration
            endpoint_url = minio_config.get('endpoint_url', 'http://localhost:9000')
            access_key = minio_config.get('access_key', 'minioadmin')
            secret_key = minio_config.get('secret_key', 'minioadmin')
            region = minio_config.get('region', 'us-east-1')
            
            # Override with environment variables if available
            endpoint_url = os.getenv('MINIO_ENDPOINT', endpoint_url)
            access_key = os.getenv('MINIO_ROOT_USER', access_key)
            secret_key = os.getenv('MINIO_ROOT_PASSWORD', secret_key)
            
            # For Docker environment, use service name
            if os.path.exists('/.dockerenv'):
                endpoint_url = 'http://minio:9000'
            
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            
            # Get bucket configuration
            buckets_config = minio_config.get('buckets', {})
            self.data_bucket = buckets_config.get('data_lake', 'data-lake')
            
            self.logger.info(f"S3 client configured - endpoint: {endpoint_url}, bucket: {self.data_bucket}")
            
            # Ensure bucket exists
            self._ensure_bucket_exists(self.data_bucket)
            
        except Exception as e:
            self.logger.error(f"Failed to setup S3 client: {str(e)}")
            raise
    
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure S3 bucket exists.
        
        Args:
            bucket_name: Name of the bucket to check/create.
        """
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"Bucket '{bucket_name}' exists")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    self.s3_client.create_bucket(Bucket=bucket_name)
                    self.logger.info(f"Created bucket '{bucket_name}'")
                except ClientError as create_error:
                    self.logger.error(f"Failed to create bucket '{bucket_name}': {create_error}")
                    raise
            else:
                self.logger.error(f"Error checking bucket '{bucket_name}': {e}")
                raise
    
    def save_dataframe(self, df: pd.DataFrame, path: str, **kwargs) -> str:
        """Save DataFrame to storage.
        
        Args:
            df: DataFrame to save.
            path: File path (local or S3 key).
            **kwargs: Additional arguments for to_csv().
            
        Returns:
            Full path/URL where file was saved.
        """
        if self.use_s3:
            return self._save_dataframe_s3(df, path, **kwargs)
        else:
            return self._save_dataframe_local(df, path, **kwargs)
    
    def _save_dataframe_local(self, df: pd.DataFrame, path: str, **kwargs) -> str:
        """Save DataFrame to local filesystem.
        
        Args:
            df: DataFrame to save.
            path: Local file path.
            **kwargs: Additional arguments for to_csv().
            
        Returns:
            Full local path where file was saved.
        """
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set default CSV parameters
        csv_kwargs = {'index': False}
        csv_kwargs.update(kwargs)
        
        df.to_csv(local_path, **csv_kwargs)
        self.logger.info(f"DataFrame saved to local file: {local_path}")
        return str(local_path.absolute())
    
    def _save_dataframe_s3(self, df: pd.DataFrame, path: str, **kwargs) -> str:
        """Save DataFrame to S3/MinIO.
        
        Args:
            df: DataFrame to save.
            path: S3 key path.
            **kwargs: Additional arguments for to_csv().
            
        Returns:
            S3 URI where file was saved.
        """
        # Convert DataFrame to CSV string
        csv_kwargs = {'index': False}
        csv_kwargs.update(kwargs)
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, **csv_kwargs)
        csv_content = csv_buffer.getvalue()
        
        # Clean path for S3 key
        s3_key = path.lstrip('/')
        
        try:
            self.s3_client.put_object(
                Bucket=self.data_bucket,
                Key=s3_key,
                Body=csv_content,
                ContentType='text/csv'
            )
            
            s3_uri = f"s3://{self.data_bucket}/{s3_key}"
            self.logger.info(f"DataFrame saved to S3: {s3_uri}")
            return s3_uri
            
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to S3: {str(e)}")
            raise
    
    def load_dataframe(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from storage.
        
        Args:
            path: File path (local or S3 key).
            **kwargs: Additional arguments for read_csv().
            
        Returns:
            Loaded DataFrame.
        """
        if self.use_s3:
            return self._load_dataframe_s3(path, **kwargs)
        else:
            return self._load_dataframe_local(path, **kwargs)
    
    def _load_dataframe_local(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from local filesystem.
        
        Args:
            path: Local file path.
            **kwargs: Additional arguments for read_csv().
            
        Returns:
            Loaded DataFrame.
        """
        df = pd.read_csv(path, **kwargs)
        self.logger.info(f"DataFrame loaded from local file: {path}")
        return df
    
    def _load_dataframe_s3(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from S3/MinIO.
        
        Args:
            path: S3 key path.
            **kwargs: Additional arguments for read_csv().
            
        Returns:
            Loaded DataFrame.
        """
        # Clean path for S3 key
        s3_key = path.lstrip('/')
        
        try:
            response = self.s3_client.get_object(Bucket=self.data_bucket, Key=s3_key)
            csv_content = response['Body'].read().decode('utf-8')
            
            df = pd.read_csv(io.StringIO(csv_content), **kwargs)
            s3_uri = f"s3://{self.data_bucket}/{s3_key}"
            self.logger.info(f"DataFrame loaded from S3: {s3_uri}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load DataFrame from S3: {str(e)}")
            raise
    
    def exists(self, path: str) -> bool:
        """Check if file exists in storage.
        
        Args:
            path: File path (local or S3 key).
            
        Returns:
            True if file exists, False otherwise.
        """
        if self.use_s3:
            return self._exists_s3(path)
        else:
            return self._exists_local(path)
    
    def _exists_local(self, path: str) -> bool:
        """Check if local file exists.
        
        Args:
            path: Local file path.
            
        Returns:
            True if file exists, False otherwise.
        """
        return Path(path).exists()
    
    def _exists_s3(self, path: str) -> bool:
        """Check if S3 object exists.
        
        Args:
            path: S3 key path.
            
        Returns:
            True if object exists, False otherwise.
        """
        s3_key = path.lstrip('/')
        
        try:
            self.s3_client.head_object(Bucket=self.data_bucket, Key=s3_key)
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                return False
            else:
                self.logger.error(f"Error checking S3 object: {e}")
                raise
    
    def list_files(self, prefix: str = "") -> list:
        """List files in storage with given prefix.
        
        Args:
            prefix: Path prefix to filter files.
            
        Returns:
            List of file paths.
        """
        if self.use_s3:
            return self._list_files_s3(prefix)
        else:
            return self._list_files_local(prefix)
    
    def _list_files_local(self, prefix: str = "") -> list:
        """List local files with given prefix.
        
        Args:
            prefix: Path prefix to filter files.
            
        Returns:
            List of local file paths.
        """
        if not prefix:
            prefix = "."
        
        prefix_path = Path(prefix)
        if prefix_path.is_file():
            return [str(prefix_path)]
        elif prefix_path.is_dir():
            return [str(f) for f in prefix_path.rglob("*") if f.is_file()]
        else:
            # Try as glob pattern
            parent = prefix_path.parent
            pattern = prefix_path.name
            return [str(f) for f in parent.glob(pattern) if f.is_file()]
    
    def _list_files_s3(self, prefix: str = "") -> list:
        """List S3 objects with given prefix.
        
        Args:
            prefix: S3 key prefix to filter objects.
            
        Returns:
            List of S3 keys.
        """
        s3_prefix = prefix.lstrip('/')
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.data_bucket,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'] for obj in response['Contents']]
            
        except Exception as e:
            self.logger.error(f"Failed to list S3 objects: {str(e)}")
            raise
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage configuration information.
        
        Returns:
            Dictionary with storage configuration details.
        """
        info = {
            'storage_type': 'S3/MinIO' if self.use_s3 else 'Local',
            'use_s3': self.use_s3
        }
        
        if self.use_s3:
            info.update({
                'bucket': self.data_bucket,
                'endpoint': getattr(self, 's3_client', {}).meta.endpoint_url if hasattr(self, 's3_client') else None
            })
        
        return info


def create_storage_manager(config: Dict[str, Any]) -> StorageManager:
    """Factory function to create storage manager.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Configured StorageManager instance.
    """
    return StorageManager(config)
