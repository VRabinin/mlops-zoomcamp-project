"""Storage utilities for file and S3 operations."""

from fnmatch import fnmatch
import os
import io
import logging
import boto3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO
from botocore.exceptions import ClientError, NoCredentialsError
from src.config.config import Config as Config


class StorageManager:
    """Unified storage manager for local files and S3/MinIO."""
    
    def __init__(self, config: Config):
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
        
        # Check if we're running in a Prefect worker (not just client)
        if os.getenv('PREFECT_WORKER_TYPE') or os.getenv('PREFECT_API_URL', '').startswith('https://'):
            self.logger.info("Prefect worker environment detected - using S3 storage")
            return True
        
        # Default to local storage
        self.logger.info("Local environment detected - using local filesystem")
        return False
    
    def _setup_s3_client(self):
        """Setup S3/MinIO client."""
        try:
            # Try new storage config first, fallback to legacy minio config
            storage_config = self.config.storage
            
            # For Docker environment, use service name
            #if os.path.exists('/.dockerenv'):
            #    endpoint_url = 'http://minio:9000'
            
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.storage.endpoint_url if hasattr(self.config.storage, 'endpoint_url') else None,
                aws_access_key_id=storage_config.access_key,
                aws_secret_access_key=storage_config.secret_key,
                region_name=storage_config.region
            )
            
            # Ensure buckets exist
            for bucket in list(self.config.storage.buckets.values()):
                self._ensure_bucket_exists(bucket)
            
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
    
    def _save_dataframe_s3(self, df: pd.DataFrame, bucket: str, path: str, **kwargs) -> str:
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
                Bucket=bucket,
                Key=s3_key,
                Body=csv_content,
                ContentType='text/csv'
            )
            
            s3_uri = f"s3://{bucket}/{s3_key}"
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
            bucket = self.config.storage.buckets.get('data_lake')
            return self._load_dataframe_s3(bucket, path, **kwargs)
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
    
    def _load_dataframe_s3(self, bucket: str, path: str, **kwargs) -> pd.DataFrame:
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
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            csv_content = response['Body'].read().decode('utf-8')
            
            df = pd.read_csv(io.StringIO(csv_content), **kwargs)
            s3_uri = f"s3://{bucket}/{s3_key}"
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
            # For the exists method, we need to determine the bucket and key from the path
            # This is a generic method, so we'll need to parse the path
            if '/' in path:
                # Assume path format is bucket/key or data_type/filename
                parts = path.split('/', 1)
                if len(parts) == 2:
                    # Try to determine if first part is a bucket or data type
                    first_part = parts[0]
                    if first_part in ['raw', 'processed', 'features', 'monitoring']:
                        # It's a data type, convert to bucket/key
                        bucket = self.get_bucket_for_data_type(first_part) 
                        s3_key = self.get_s3_path(first_part, parts[1])
                    else:
                        # Assume it's bucket/key format
                        bucket = first_part
                        s3_key = parts[1]
                else:
                    # Single path, use default bucket
                    bucket = self.config.storage.buckets.get('data_lake', 'data-lake')
                    s3_key = path
            else:
                # Single filename, use default bucket
                bucket = self.config.storage.buckets.get('data_lake', 'data-lake') 
                s3_key = path
            
            return self._exists_s3(bucket, s3_key)
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
    
    def _exists_s3(self, bucket: str, path: str) -> bool:
        """Check if S3 object exists.
        
        Args:
            bucket: S3 bucket name.
            path: S3 key path.
            
        Returns:
            True if object exists, False otherwise.
        """
        s3_key = path.lstrip('/')
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=s3_key)
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
            return self._list_files_s3(self.config.storage.buckets.get('data_lake'), prefix)
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

    def _list_files_s3(self, bucket: str, prefix: str = "") -> list:
        """List S3 objects with given prefix.
        
        Args:
            bucket: S3 bucket name.
            prefix: S3 key prefix to filter objects.
            
        Returns:
            List of S3 keys.
        """
        s3_prefix = prefix.lstrip('/')
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
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
                'bucket': self.config.storage.buckets.get('data_lake'),
                'endpoint': getattr(self, 's3_client', {}).meta.endpoint_url if hasattr(self, 's3_client') else None
            })
        
        return info


    def get_s3_path(self, data_type: str, filename: str) -> str:
        """Get S3 path for a given data type and filename.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'features', 'models', etc.)
            filename: Name of the file
            
        Returns:
            Full S3 path
        """
        if data_type not in self.config.storage.s3_paths:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(self.config.storage.s3_paths.keys())}")
        
        base_path = self.config.storage.s3_paths[data_type]
        return f"{base_path}/{filename}".lstrip('/')
    
    def get_bucket_for_data_type(self, data_type: str) -> str:
        """Get appropriate bucket for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Bucket name
        """
        # Map data types to buckets
        data_type_to_bucket = {
            'raw': self.config.storage.buckets['data_lake'],
            'processed': self.config.storage.buckets['data_lake'],
            'features': self.config.storage.buckets['data_lake'],
            'experiments': self.config.storage.buckets['mlflow_artifacts'],
            'models': self.config.storage.buckets['model_artifacts'],
            'prefect_flows': self.config.storage.buckets['data_lake'],
            'monitoring_results': self.config.storage.buckets['data_lake'],
            'monitoring_reports': self.config.storage.buckets['data_lake']
        }
        
        return data_type_to_bucket.get(data_type)
    
    def save_dataframe(self, df: pd.DataFrame, data_type: str, filename: str, **kwargs) -> str:
        """Save DataFrame with automatic bucket and path selection.
        
        Args:
            df: DataFrame to save
            data_type: Type of data ('raw', 'processed', 'features', etc.)
            filename: Name of the file
            **kwargs: Additional arguments for to_csv()
            
        Returns:
            Full path/URL where file was saved
        """
        if self.use_s3:
            s3_path = self.get_s3_path(data_type, filename)
            bucket = self.get_bucket_for_data_type(data_type)
            return self._save_dataframe_s3(df, bucket, s3_path, **kwargs)
        else:
            # For local storage, use the resolved path from local_paths
            local_path = self.resolve_path(data_type, filename)
            return self._save_dataframe_local(df, str(local_path), **kwargs)
    
    def load_dataframe_by_type(self, data_type: str, filename: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame with automatic bucket and path selection.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'features', etc.)
            filename: Name of the file
            **kwargs: Additional arguments for read_csv()
            
        Returns:
            Loaded DataFrame
        """
        if self.use_s3:
            s3_path = self.get_s3_path(data_type, filename)
            bucket = self.get_bucket_for_data_type(data_type)
            return self._load_dataframe_s3(bucket, s3_path, **kwargs)
        else:
            # For local storage, use the resolved path from local_paths
            local_path = self.resolve_path(data_type, filename)
            return self._load_dataframe_local(str(local_path), **kwargs)


    def resolve_path(self, data_type: str, filename: str = None) -> Union[str, Path]:
        """Resolve the appropriate path for a given data type and filename.
        
        This is the central method that encapsulates all storage location logic.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'features', 'models', etc.)
            filename: Optional filename to append to the path
            
        Returns:
            Complete path (S3 key or local Path) ready for use
        """
        if self.use_s3:
            # For S3, return the S3 key
            if filename:
                return self.get_s3_path(data_type, filename)
            else:
                # Return just the base path
                if data_type not in self.config.storage.s3_paths:
                    raise ValueError(f"Unknown data type: {data_type}. Available: {list(self.config.storage.s3_paths.keys())}")
                return self.config.storage.s3_paths[data_type]
        else:
            # For local filesystem, return Path object
            if data_type not in self.config.data_path.__dict__:
                raise ValueError(f"Unknown data type: {data_type}. Available: {list(self.config.data_path.__dict__.keys())}")
            base_path = Path(self.config.data_path.__dict__[data_type])
            if filename:
                return base_path / filename
            else:
                return base_path
    
    def ensure_path_exists(self, data_type: str) -> Union[str, Path]:
        """Ensure that the directory/bucket for a data type exists.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'features', etc.)
            
        Returns:
            The path that was ensured to exist
        """
        path = self.resolve_path(data_type)
        
        if self.use_s3:
            # For S3, ensure the bucket exists (already done in setup)
            bucket = self.get_bucket_for_data_type(data_type)
            self._ensure_bucket_exists(bucket)
            return path
        else:
            # For local filesystem, create directory if it doesn't exist
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured local directory exists: {path}")
            return path
    
    def get_full_path(self, data_type: str, filename: str) -> str:
        """Get the full path/URI for a file, ensuring the parent directory exists.
        
        Args:
            data_type: Type of data
            filename: Name of the file
            
        Returns:
            Full path/URI ready for file operations
        """
        # Ensure the base path exists
        self.ensure_path_exists(data_type)
        
        # Get the complete path including filename
        full_path = self.resolve_path(data_type, filename)
        
        if self.use_s3:
            # Return S3 URI
            bucket = self.get_bucket_for_data_type(data_type)
            return f"s3://{bucket}/{full_path}"
        else:
            # Return absolute local path
            return str(Path(full_path).absolute())
    
    
    def load_dataframe(self, data_type: str, filename: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame using intelligent path resolution.
        
        This replaces the need for components to handle storage logic.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'features', etc.)
            filename: Name of the file
            **kwargs: Additional arguments for read_csv()
            
        Returns:
            Loaded DataFrame
        """
        # This method combines the functionality of load_dataframe_by_type
        # with the new path resolution logic
        return self.load_dataframe_by_type(data_type, filename, **kwargs)
    
    def file_exists(self, data_type: str, filename: str) -> bool:
        """Check if a file exists using intelligent path resolution.
        
        Args:
            data_type: Type of data
            filename: Name of the file
            
        Returns:
            True if file exists, False otherwise
        """
        if self.use_s3:
            bucket = self.get_bucket_for_data_type(data_type)
            s3_path = self.get_s3_path(data_type, filename)
            return self._exists_s3(bucket, s3_path)
        else:
            local_path = self.resolve_path(data_type, filename)
            return self._exists_local(str(local_path))
    
    def list_files(self, data_type: str, pattern: str = "*") -> list:
        """List files of a specific data type with optional pattern matching.
        
        Args:
            data_type: Type of data
            pattern: Glob pattern for filtering files (default: all files)
            
        Returns:
            List of file paths/keys
        """
        if self.use_s3:
            prefix = self.resolve_path(data_type)
            files = self._list_files_s3(self.config.storage.buckets.get('data_lake'), prefix)
            
            # Apply pattern filtering for S3
            if pattern != "*":
                import fnmatch
                files = [f for f in files if fnmatch.fnmatch(Path(f).name, pattern)]
            
            return files
        else:
            base_path = self.resolve_path(data_type)
            if pattern == "*":
                pattern = "**/*"
            
            return [str(f) for f in Path(base_path).glob(pattern) if f.is_file()]
    
    def get_working_directory(self, data_type: str) -> Union[str, Path]:
        """Get the working directory for a specific data type.
        
        This is useful for components that need to know where to operate.
        
        Args:
            data_type: Type of data
            
        Returns:
            Working directory path
        """
        return self.ensure_path_exists(data_type)

    def cleanup(self, data_type: str, filename_mask: str):
        """Cleanup files matching a specific pattern in the storage.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'features', etc.)
            filename_mask: Optional list of filename patterns to match for deletion
        """
        if self.use_s3:
            #TODO: Fix non-working cleanup
            bucket = self.get_bucket_for_data_type(data_type)
            prefix = self.resolve_path(data_type).lstrip('/')
            self._cleanup_s3(bucket, prefix, filename_mask)
        else:
            local_path = self.resolve_path(data_type)
            self._cleanup_local(local_path, filename_mask)

    def _cleanup_local(self, path: Union[str, Path], filename_mask: str = None):
        """Cleanup local files matching a specific pattern.
        
        Args:
            path: Path to the directory containing files to delete
            filename_mask: Optional list of filename patterns to match for deletion
        """
        if filename_mask is None:
            filename_mask = ["*"]

        for file in Path(path).glob(filename_mask):
            try:
                file.unlink(missing_ok=True)
                self.logger.info(f"Deleted local file: {file}")
            except Exception as e:
                self.logger.error(f"Failed to delete local file {file}: {str(e)}")

    def _cleanup_s3(self, bucket: str, prefix: str, filename_mask: str):
        """Cleanup S3 files matching a specific pattern.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix to filter objects
            filename_mask: Optional list of filename patterns to match for deletion
        """
        if filename_mask is None:
            filename_mask = ["*"]

        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in response:
                return
            
            for obj in response['Contents']:
                key = obj['Key']
                if fnmatch.fnmatch(Path(key).name, filename_mask):
                    self.s3_client.delete_object(Bucket=bucket, Key=key)
                    self.logger.info(f"Deleted S3 object: {key}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup S3 files: {str(e)}")
    
    def load_text_file(self, data_type: str, filename: str) -> str:
        """Load text content from a file.
        
        Args:
            data_type: Type of data (features, models, etc.)
            filename: Name of the file to load
            
        Returns:
            Text content of the file
        """
        if self.use_s3:
            bucket = self.get_bucket_for_data_type(data_type)
            s3_path = self.get_s3_path(data_type, filename)
            return self._load_text_file_s3(bucket, s3_path)
        else:
            local_path = self.resolve_path(data_type, filename)
            return self._load_text_file_local(str(local_path))
    
    def _load_text_file_local(self, path: str) -> str:
        """Load text file from local filesystem."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load text file {path}: {e}")
            raise
    
    def _load_text_file_s3(self, bucket: str, key: str) -> str:
        """Load text file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to load text file from S3 {bucket}/{key}: {e}")
            raise
    
    def upload_file(self, file_obj: Union[BinaryIO, io.StringIO], data_type: str, filename: str):
        """Upload a file object to storage.
        
        Args:
            file_obj: File-like object to upload
            data_type: Type of data (features, models, etc.)
            filename: Name of the file
        """
        if self.use_s3:
            bucket = self.get_bucket_for_data_type(data_type)
            s3_path = self.get_s3_path(data_type, filename)
            self._upload_file_s3(file_obj, bucket, s3_path)
        else:
            local_path = self.resolve_path(data_type, filename)
            self._upload_file_local(file_obj, str(local_path))
    
    def _upload_file_local(self, file_obj: Union[BinaryIO, io.StringIO], path: str):
        """Upload file to local filesystem."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Handle both binary and text file objects
            if isinstance(file_obj, io.StringIO):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(file_obj.getvalue())
            else:
                with open(path, 'wb') as f:
                    file_obj.seek(0)
                    f.write(file_obj.read())
        except Exception as e:
            self.logger.error(f"Failed to upload file to {path}: {e}")
            raise
    
    def _upload_file_s3(self, file_obj: Union[BinaryIO, io.StringIO], bucket: str, key: str):
        """Upload file to S3."""
        try:
            self._ensure_bucket_exists(bucket)
            
            # Handle both binary and text file objects
            if isinstance(file_obj, io.StringIO):
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=file_obj.getvalue().encode('utf-8')
                )
            else:
                file_obj.seek(0)
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=file_obj.read()
                )
        except Exception as e:
            self.logger.error(f"Failed to upload file to S3 {bucket}/{key}: {e}")
            raise
    
    def delete_file(self, data_type: str, filename: str) -> bool:
        """Delete a file from storage.
        
        Args:
            data_type: Type of data (raw, processed, features, etc.).
            filename: Name of the file to delete.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            if self.use_s3:
                bucket = self.get_bucket_for_data_type(data_type)
                key = self.get_s3_path(data_type, filename)
                return self._delete_file_s3(bucket, key)
            else:
                path = self.resolve_path(data_type, filename)
                return self._delete_file_local(str(path))
        except Exception as e:
            self.logger.error(f"Failed to delete file {data_type}/{filename}: {e}")
            return False
    
    def _delete_file_local(self, path: str) -> bool:
        """Delete file from local filesystem.
        
        Args:
            path: Full path to the file.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            file_path = Path(path)
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted local file: {path}")
                return True
            else:
                self.logger.warning(f"Local file does not exist: {path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete local file {path}: {e}")
            return False
    
    def _delete_file_s3(self, bucket: str, key: str) -> bool:
        """Delete file from S3.
        
        Args:
            bucket: S3 bucket name.
            key: S3 object key.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            # Check if object exists first
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.logger.warning(f"S3 object does not exist: {bucket}/{key}")
                    return False
                else:
                    raise
            
            # Delete the object
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            self.logger.info(f"Deleted S3 object: {bucket}/{key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete S3 object {bucket}/{key}: {e}")
            return False

def create_storage_manager(config: Config) -> StorageManager:
    """Factory function to create storage manager.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Configured StorageManager instance.
    """
    return StorageManager(config)
