"""CRM data ingestion from Kaggle."""

import os
import logging
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import kaggle

from src.data.schemas.crm_schema import CRMDataSchema
from src.utils.storage import StorageManager


class CRMDataIngestion:
    """Handle CRM data ingestion from Kaggle dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CRM data ingestion.
        
        Args:
            config: Configuration dictionary containing data paths and settings.
        """
        self.config = config
        
        # For backward compatibility, still support direct path config
        self.raw_data_path = Path(config.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(config.get('processed_data_path', 'data/processed'))
        self.kaggle_dataset = 'innocentmfa/crm-sales-opportunities'
        
        # Initialize storage manager
        self.storage = StorageManager(config)
        
        # Create directories if using local storage
        if not self.storage.use_s3:
            self.raw_data_path.mkdir(parents=True, exist_ok=True)
            self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self.schema = CRMDataSchema()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"CRM Ingestion initialized - Storage: {self.storage.get_storage_info()}")
    
    def download_dataset(self) -> Tuple[bool, str]:
        """Download CRM dataset from Kaggle.
        
        Returns:
            Tuple of (success, message/error).
        """
        try:
            self.logger.info(f"Downloading dataset: {self.kaggle_dataset}")
            
            # Authenticate with Kaggle API
            kaggle.api.authenticate()
            
            if self.storage.use_s3:
                # For S3 storage, download to a temporary local directory first
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download dataset to temporary directory
                    kaggle.api.dataset_download_files(
                        self.kaggle_dataset,
                        path=temp_dir,
                        unzip=True
                    )
                    
                    # Find and upload CSV files to S3
                    temp_path = Path(temp_dir)
                    csv_files = list(temp_path.glob("*.csv"))
                    
                    if not csv_files:
                        return False, "No CSV files found in downloaded dataset"
                    
                    for csv_file in csv_files:
                        # Read and upload to S3 using typed storage
                        df = pd.read_csv(csv_file)
                        self.storage.save_dataframe_by_type(df, 'raw', csv_file.name)
                        self.logger.info(f"Uploaded {csv_file.name} to S3 raw data storage")
                    
                    self.logger.info(f"Dataset uploaded to S3 storage - {len(csv_files)} files")
            else:
                # For local storage, download directly to raw data path
                kaggle.api.dataset_download_files(
                    self.kaggle_dataset,
                    path=str(self.raw_data_path),
                    unzip=True
                )
                self.logger.info("Dataset downloaded to local storage")
            
            return True, "Dataset downloaded successfully"
            
        except Exception as e:
            error_msg = f"Failed to download dataset: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def find_csv_files(self) -> List[Path]:
        """Find CSV files in the raw data directory.
        
        Returns:
            List of CSV file paths.
        """
        if self.storage.use_s3:
            # List CSV files from S3
            files = self.storage.list_files("raw/")
            csv_files = [Path(f) for f in files if f.endswith('.csv')]
        else:
            # List CSV files from local directory
            csv_files = list(self.raw_data_path.glob("*.csv"))
        
        self.logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        return csv_files
    
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load CRM data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, auto-detect main CSV file.
            
        Returns:
            Loaded DataFrame.
        """
        if file_path is None:
            csv_files = self.find_csv_files()
            if not csv_files:
                raise FileNotFoundError("No CSV files found in raw data directory")
            
            # Look for the main sales pipeline data file
            file_path = None
            for csv_file in csv_files:
                file_name = csv_file.name.lower()
                # Prioritize sales_pipeline.csv or similar main data files
                if 'sales_pipeline' in file_name or 'pipeline' in file_name:
                    file_path = csv_file
                    break
                elif any(keyword in file_name for keyword in ['crm', 'sales', 'opportunities', 'data']):
                    # Exclude small files that are likely metadata
                    if self.storage.use_s3:
                        # For S3, we can't easily check file size, so use name patterns
                        if 'dictionary' not in file_name and 'readme' not in file_name:
                            file_path = csv_file
                    else:
                        # For local files, check file size
                        if csv_file.stat().st_size > 100000:  # > 100KB
                            file_path = csv_file
            
            # If no specific file found, use the largest file
            if file_path is None:
                if self.storage.use_s3:
                    # For S3, just use the first non-dictionary file
                    for csv_file in csv_files:
                        if 'dictionary' not in csv_file.name.lower():
                            file_path = csv_file
                            break
                else:
                    # For local files, find the largest file
                    file_path = max(csv_files, key=lambda f: f.stat().st_size if f.exists() else 0)
            
            if file_path is None:
                file_path = csv_files[0]  # Fallback to first file
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            if self.storage.use_s3:
                # For S3, load using typed storage method
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = self.storage.load_dataframe_by_type('raw', file_path.name, encoding=encoding)
                        self.logger.info(f"Data loaded successfully from S3 with encoding: {encoding}")
                        self.logger.info(f"Data shape: {df.shape}")
                        return df
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, try without specifying encoding
                df = self.storage.load_dataframe_by_type('raw', file_path.name)
                self.logger.info("Data loaded successfully from S3 with default encoding")
                self.logger.info(f"Data shape: {df.shape}")
                return df
            else:
                # For local files
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        self.logger.info(f"Data loaded successfully with encoding: {encoding}")
                        self.logger.info(f"Data shape: {df.shape}")
                        return df
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, try without specifying encoding
                df = pd.read_csv(file_path)
                self.logger.info("Data loaded successfully with default encoding")
                self.logger.info(f"Data shape: {df.shape}")
                return df
            
        except Exception as e:
            error_msg = f"Failed to load data from {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data.
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        self.logger.info("Cleaning data...")
        
        # Clean column names
        df_cleaned = self.schema.clean_column_names(df)
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Fill missing categorical values with 'Unknown'
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna('Unknown')
        
        # Fill missing numerical values with median
        numerical_cols = df_cleaned.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                self.logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        missing_after = df_cleaned.isnull().sum().sum()
        self.logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return df_cleaned
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data against schema.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        self.logger.info("Validating data schema...")
        
        is_valid, issues = self.schema.validate_schema(df)
        
        if issues:
            self.logger.warning(f"Validation issues found: {issues}")
        else:
            self.logger.info("Data validation passed")
        
        return is_valid, issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[str]) -> float:
        """Calculate data quality score.
        
        Args:
            df: DataFrame to score.
            issues: List of validation issues.
            
        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 1.0
        
        # Penalize for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 0.3
        
        # Penalize for duplicates
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 0.2
        
        # Penalize for validation issues
        if issues:
            # Don't penalize for extra columns (they might be useful)
            serious_issues = [issue for issue in issues if "Extra columns" not in issue]
            score -= len(serious_issues) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "crm_data_processed.csv") -> str:
        """Save processed data to storage.
        
        Args:
            df: Processed DataFrame.
            filename: Output filename.
            
        Returns:
            Path/URI where file was saved.
        """
        if self.storage.use_s3:
            # Save to S3 using typed storage method
            saved_path = self.storage.save_dataframe_by_type(df, 'processed', filename)
        else:
            # Save to local processed directory using typed storage method
            saved_path = self.storage.save_dataframe_by_type(df, 'processed', filename)
        
        self.logger.info(f"Processed data saved to: {saved_path}")
        return saved_path
    
    def run_ingestion(self) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """Run complete data ingestion pipeline.
        
        Returns:
            Tuple of (success, dataframe, metadata).
        """
        metadata = {
            'dataset': self.kaggle_dataset,
            'timestamp': pd.Timestamp.now(),
            'steps_completed': []
        }
        
        try:
            # Step 1: Download dataset
            success, message = self.download_dataset()
            if not success:
                return False, pd.DataFrame(), {**metadata, 'error': message}
            metadata['steps_completed'].append('download')
            
            # Step 2: Load data
            df = self.load_data()
            metadata['raw_shape'] = df.shape
            metadata['raw_columns'] = list(df.columns)
            metadata['steps_completed'].append('load')
            
            # Step 3: Clean data
            df_cleaned = self.clean_data(df)
            metadata['cleaned_shape'] = df_cleaned.shape
            metadata['cleaned_columns'] = list(df_cleaned.columns)
            metadata['steps_completed'].append('clean')
            
            # Step 4: Validate data
            is_valid, issues = self.validate_data(df_cleaned)
            metadata['validation'] = {
                'is_valid': is_valid,
                'issues': issues,
                'quality_score': self._calculate_quality_score(df_cleaned, issues)
            }
            metadata['steps_completed'].append('validate')
            
            # Step 5: Save processed data
            saved_path = self.save_processed_data(df_cleaned)
            metadata['output_path'] = saved_path
            metadata['steps_completed'].append('save')
            
            self.logger.info("Data ingestion pipeline completed successfully")
            self.logger.info(f"Quality score: {metadata['validation']['quality_score']:.2f}")
            
            return True, df_cleaned, metadata
            
        except Exception as e:
            error_msg = f"Data ingestion pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            return False, pd.DataFrame(), {**metadata, 'error': error_msg}


def main():
    """Main function for running CRM data ingestion."""
    from src.config.config import get_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = get_config()
    
    # Create ingestion instance
    config_dict = {
        'raw_data_path': config.data.raw_data_path,
        'processed_data_path': config.data.processed_data_path,
        'kaggle_dataset': 'innocentmfa/crm-sales-opportunities',
        'storage': {
            'endpoint_url': config.storage.endpoint_url,
            'access_key': config.storage.access_key,
            'secret_key': config.storage.secret_key,
            'region': config.storage.region,
            'buckets': config.storage.buckets,
            's3_paths': config.storage.s3_paths,
            'data_paths': config.storage.data_paths
        },
        # Legacy minio config for backward compatibility
        'minio': {
            'endpoint_url': config.storage.endpoint_url,
            'access_key': config.storage.access_key,
            'secret_key': config.storage.secret_key,
            'region': config.storage.region,
            'buckets': config.storage.buckets
        }
    }
    
    ingestion = CRMDataIngestion(config_dict)
    
    # Run ingestion
    success, df, metadata = ingestion.run_ingestion()
    
    if success:
        print(f"✅ Data ingestion completed successfully!")
        print(f"📊 Data shape: {df.shape}")
        print(f"🎯 Quality score: {metadata['validation']['quality_score']:.2f}")
        if metadata['validation']['issues']:
            print(f"⚠️  Issues: {metadata['validation']['issues']}")
    else:
        print(f"❌ Data ingestion failed: {metadata.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
