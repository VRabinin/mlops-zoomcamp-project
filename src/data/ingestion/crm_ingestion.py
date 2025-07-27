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
from src.config.config import Config

class CRMDataIngestion:
    """Handle CRM data ingestion from Kaggle dataset."""
    
    def __init__(self, config: Config, snapshot_month: str = "XXXX-XX"):
        """Initialize CRM data ingestion.
        
        Args:
            config: Configuration dictionary containing data paths and settings.
        """
        self.config = config
        self.kaggle_dataset = 'innocentmfa/crm-sales-opportunities'
        self.snapshot_month = snapshot_month
        os.environ['AWS_ACCESS_KEY_ID'] = config.storage.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.storage.secret_key
        # Initialize storage manager - this now handles all path logic
        self.storage = StorageManager(config)
        
        # Remove manual path handling - let StorageManager handle this
        # The storage manager will automatically create directories if needed
        
        # Initialize schema
        self.schema = CRMDataSchema()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"CRM Ingestion initialized - Storage: {self.storage.get_storage_info()}")
        
    def find_csv_files(self) -> List[Path]:
        """Find CSV files in the raw data directory.
        
        Returns:
            List of CSV file paths.
        """
        # Use smart storage method to list CSV files
        csv_files = self.storage.list_files('raw', '*.csv')
        csv_paths = [Path(f) for f in csv_files]
        
        self.logger.info(f"Found {len(csv_paths)} CSV files: {[f.name for f in csv_paths]}")
        return csv_paths
    
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
                if f'sales_pipeline_enhanced_{self.snapshot_month}' in file_name:
                    file_path = csv_file
                    break
            
            if file_path is None:
                raise FileNotFoundError(
                    f"Snapshot file sales_pipeline_enhanced_{self.snapshot_month}.csv not found. "
                    "Please ensure the file is named correctly or specify the path."
                )
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            # Try different encodings for data loading
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = self.storage.load_dataframe('raw', file_path.name, encoding=encoding)
                    self.logger.info(f"Data loaded successfully with encoding: {encoding}")
                    self.logger.info(f"Data shape: {df.shape}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try without specifying encoding
            df = self.storage.load_dataframe('raw', file_path.name)
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
        
        # Remove duplicates
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Fill missing categorical values with 'Unknown'
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna('Unknown')
        
        # Fill missing numerical values with median in Won or Lost deals
        numerical_cols = df_cleaned.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().any():
                median_val = df_cleaned[df_cleaned['deal_stage'].isin(['Won', 'Lost'])][col].median()
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
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "crm_data_processed_XXXX-XX.csv") -> str:
        """Save processed data to storage.
        
        Args:
            df: Processed DataFrame.
            filename: Output filename.
            
        Returns:
            Path/URI where file was saved.
        """
        # Use smart storage to handle both S3 and local paths
        saved_path = self.storage.save_dataframe(df, 'processed', filename)
        
        self.logger.info(f"Processed data saved to: {saved_path}")
        return saved_path
    
    def run_ingestion(self) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """Run complete data ingestion pipeline.
        
        Returns:
            Tuple of (success, dataframe, metadata).
        """
        metadata = {
            'timestamp': pd.Timestamp.now(),
            'steps_completed': []
        }
        
        try:           
            # Step 1: Load data
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
            saved_path = self.save_processed_data(df_cleaned, filename=f"crm_data_processed_{self.snapshot_month}.csv")
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
    ingestion = CRMDataIngestion(config, config.first_snapshot_month)
    
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
