"""CRM data ingestion from Kaggle."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import kaggle
import numpy as np
import pandas as pd

from src.config.config import Config
from src.data.schemas.crm_schema import CRMDataSchema
from src.utils.storage import StorageManager


class CRMDataAcquisition:
    """Handle CRM data ingestion from Kaggle dataset."""

    def __init__(self, config: Config, snapshot_month: str = "XXXX-XX"):
        """Initialize CRM data ingestion.

        Args:
            config: Configuration dictionary containing data paths and settings.
        """
        self.config = config
        self.kaggle_dataset = "innocentmfa/crm-sales-opportunities"
        self.snapshot_month = snapshot_month

        # Initialize storage manager - this now handles all path logic
        self.storage = StorageManager(config)

        # Remove manual path handling - let StorageManager handle this
        # The storage manager will automatically create directories if needed

        # Initialize schema
        self.schema = CRMDataSchema()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"CRM Ingestion initialized - Storage: {self.storage.get_storage_info()}"
        )

    def download_dataset(self) -> Tuple[bool, str]:
        """Download CRM dataset from Kaggle.

        Returns:
            Tuple of (success, message/error).
        """
        try:
            self.logger.info(f"Downloading dataset: {self.kaggle_dataset}")

            # Authenticate with Kaggle API
            kaggle.api.authenticate()
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download dataset to temporary directory
                kaggle.api.dataset_download_files(
                    self.kaggle_dataset, path=temp_dir, unzip=True
                )
                # Find and upload CSV files to S3
                temp_path = Path(temp_dir)
                csv_files = list(temp_path.glob("*.csv"))

                if not csv_files:
                    return False, "No CSV files found in downloaded dataset"

                for csv_file in csv_files:
                    # Read and upload to S3 using smart storage
                    df = pd.read_csv(csv_file)
                    self.storage.save_dataframe(df, "raw", csv_file.name)

            return True, "Dataset downloaded successfully"

        except Exception as e:
            error_msg = f"Failed to download dataset: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def find_csv_files(self, prefix: str = "raw") -> List[Path]:
        """Find CSV files in the raw data directory.

        Returns:
            List of CSV file paths.
        """
        # Use smart storage method to list CSV files
        csv_files = self.storage.list_files(prefix, "*.csv")
        csv_paths = [Path(f) for f in csv_files]

        self.logger.info(
            f"Found {len(csv_paths)} CSV files: {[f.name for f in csv_paths]}"
        )
        return csv_paths

    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load CRM data from CSV file.

        Args:
            file_path: Path to CSV file. If None, auto-detect main CSV file.

        Returns:
            Loaded DataFrame.
        """
        if file_path is None:
            csv_files = self.find_csv_files("raw")
            if not csv_files:
                raise FileNotFoundError("No CSV files found in raw data directory")

            # Look for the main sales pipeline data file
            file_path = None
            for csv_file in csv_files:
                file_name = csv_file.name.lower()
                # Prioritize sales_pipeline.csv or similar main data files
                if "sales_pipeline." in file_name:
                    file_path = csv_file
                    break

            if file_path is None:
                raise FileNotFoundError(
                    "No main sales_pipeline.csv file found in raw data directory"
                )

        self.logger.info(f"Loading data from: {file_path}")

        try:
            # Try different encodings for data loading
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = self.storage.load_dataframe(
                        "raw", file_path.name, encoding=encoding
                    )
                    self.logger.info(
                        f"Data loaded successfully with encoding: {encoding}"
                    )
                    self.logger.info(f"Data shape: {df.shape}")
                    return df
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, try without specifying encoding
            df = self.storage.load_dataframe("raw", file_path.name)
            self.logger.info("Data loaded successfully with default encoding")
            self.logger.info(f"Data shape: {df.shape}")
            return df

        except Exception as e:
            error_msg = f"Failed to load data from {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def _adjust_dates_and_stage(self, period, group):
        """
        Adjusts the dates and stages in the sales opportunities data.
        """
        # Calculate last date of the period
        last_date = (
            pd.to_datetime(period, errors="coerce").to_period("M").end_time.date()
        )
        # Add last date to the group in date format
        group["snapshot_date"] = last_date
        group.loc[
            pd.to_datetime(group["close_date"], errors="coerce")
            > group["snapshot_date"],
            "deal_stage",
        ] = "Engaging"
        group.loc[
            pd.to_datetime(group["close_date"], errors="coerce")
            > group["snapshot_date"],
            "close_date",
        ] = pd.NaT
        group.loc[
            pd.to_datetime(group["engage_date"], errors="coerce")
            > group["snapshot_date"],
            "deal_stage",
        ] = "Prospecting"
        group.loc[
            pd.to_datetime(group["engage_date"], errors="coerce")
            > group["snapshot_date"],
            "engage_date",
        ] = pd.NaT
        return group

    def enhance_data(
        self, df_sales: pd.DataFrame
    ) -> Tuple[list[pd.DataFrame], list[str]]:
        """Enhance CRM data for simulation.

        Args:
            df_sales: Raw sales opportunities data DataFrame.

        Returns:
            Tuple of (list of enhanced DataFrames, list of filenames).
        """
        dataframes = []
        filenames = []
        # Add opportunity creation date as offset of engagement date
        df_sales["random_number"] = np.random.randint(10, 30, df_sales.shape[0])
        # Create creation date by subtracting a random number of days from the engagement date
        df_sales["creation_date"] = pd.to_datetime(
            df_sales["engage_date"], errors="coerce"
        )
        # If engage_date is NaN, set creation_date to max of the engagement date
        m = pd.to_datetime(df_sales["engage_date"], errors="coerce").max()
        df_sales["creation_date"] = df_sales["creation_date"].fillna(m)
        df_sales["creation_date"] = df_sales["creation_date"] - pd.to_timedelta(
            df_sales["random_number"], unit="D"
        )
        df_sales = df_sales.drop(columns=["random_number"])

        # Add creation year-month
        df_sales["creation_year_month"] = df_sales["creation_date"].dt.to_period("M")
        # Split dataset by creation month and save to CSV
        # First months_to_aggregate months will be saved into a single fine, the rest will be saved separately
        df_sales["creation_year_month"] = df_sales["creation_year_month"].astype(str)
        # Save each period's data to a separate CSV file
        df_first_group = pd.DataFrame(columns=df_sales.columns)
        for period, group in df_sales.groupby("creation_year_month"):
            if period < self.snapshot_month:
                df_first_group = pd.concat([df_first_group, group])
            elif period == self.snapshot_month:
                df_first_group = pd.concat([df_first_group, group])
                df_first_group = self._adjust_dates_and_stage(period, df_first_group)
                # Save the aggregated data to a CSV file
                dataframes.append(df_first_group)
                filenames.append(f"sales_pipeline_enhanced_{period}.csv")
                self.logger.info(
                    f"Saved {len(df_first_group)} records for period {period} to CSV."
                )
            else:
                group = self._adjust_dates_and_stage(period, group)
                # Save each group to a separate CSV file
                dataframes.append(group)
                filenames.append(f"sales_pipeline_enhanced_{period}.csv")
                self.logger.info(
                    f"Saved {len(group)} records for period {period} to CSV."
                )
        if len(dataframes) == 0:
            self.logger.error(
                "No dataframes created during enhancement process. Check the configuration parameter 'first_snapshot_month'."
            )
            return [], []
        return dataframes, filenames

    def save_enhanced_data(
        self,
        dataframe_list: list[pd.DataFrame],
        filename_list: list[str],
        suffix: str = "raw",
    ) -> str:
        """Save enhanced data to storage.

        Args:
            df: Enhanced DataFrame.
            filename: Output filename.

        Returns:
            Path/URI where file was saved.
        """
        # Cleanup location before saving
        self.logger.info("Cleaning up previous enhanced data files...")
        self.storage.cleanup("raw", filename_mask="sales_pipeline_enhanced_*.csv")

        # Use smart storage to handle both S3 and local paths
        saved_paths = []
        for df, filename in zip(dataframe_list, filename_list):
            saved_path = self.storage.save_dataframe(df, suffix, filename)
            saved_paths.append(saved_path)
            self.logger.info(f"Enhanced data saved to: {saved_path}")
        return saved_paths

    def run_acquisition(self) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """Run complete data ingestion pipeline.

        Returns:
            Tuple of (success, dataframe, metadata).
        """
        metadata = {
            "dataset": self.kaggle_dataset,
            "timestamp": pd.Timestamp.now(),
            "steps_completed": [],
        }

        try:
            # Step 1: Download dataset
            success, message = self.download_dataset()
            if not success:
                return False, pd.DataFrame(), {**metadata, "error": message}
            metadata["steps_completed"].append("download")

            # Step 2: Load data
            df = self.load_data()
            metadata["raw_shape"] = df.shape
            metadata["raw_columns"] = list(df.columns)
            metadata["steps_completed"].append("load")

            # Step 3: Enhance data for simulation
            dataframes, filenames = self.enhance_data(df)
            metadata["dataframe_count"] = len(dataframes)
            metadata["raw_columns"] = list(dataframes[0].columns)
            metadata["steps_completed"].append("load")

            # Step 4: Save Enhanced Data
            self.save_enhanced_data(dataframes, filenames, "raw")
            self.logger.info("Data acquisition pipeline completed successfully")
            return True, df, metadata

        except Exception as e:
            error_msg = f"Data ingestion pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            return False, pd.DataFrame(), {**metadata, "error": error_msg}


def main():
    """Main function for running CRM data ingestion."""
    from src.config.config import get_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Get configuration
    config = get_config()
    ingestion = CRMDataAcquisition(config, config.first_snapshot_month)

    # Run ingestion
    success, df, metadata = ingestion.run_acquisition()

    if success:
        print("✅ Data acquisition completed successfully!")
        print(f"📊 Source Data shape: {df.shape}")
    else:
        print(f"❌ Data acquisition failed: {metadata.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
