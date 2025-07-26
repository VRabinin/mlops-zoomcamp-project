"""Feature engineering for CRM sales data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.config.config import Config
import logging
from pathlib import Path

from src.utils.storage import StorageManager


class CRMFeatureEngineer:
    """Feature engineering for CRM sales opportunities data."""
    
    def __init__(self, config: Config):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.target_column = 'deal_stage'
        self.test_size = 0.2
        self.random_state = 42
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize storage manager for consistent storage handling
        self.storage = StorageManager(config)
        
        self.logger = logging.getLogger(__name__)

    def create_features(self, df_sales: pd.DataFrame, df_accounts: pd.DataFrame, df_products: pd.DataFrame, df_sales_teams: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data.
        
        Args:
            df_sales: Input Sales DataFrame.
            df_sales_teams: Input Sales Teams DataFrame.
            df_accounts: Input Accounts DataFrame.
            df_products: Input Products DataFrame.

        Returns:
            DataFrame with new features.
        """
        df_features = df_sales.copy()

        self.logger.info("Creating new features...")
        
        # Close value features
        # Log transformation for close value (add 1 to handle 0 values)
        df_features['close_value_log'] = np.log1p(df_features['close_value'])
            
        # Close value categories
        df_features['close_value_category'] = pd.cut(
            df_features['close_value'],
            bins=[0, 1000, 5000, 10000, 50000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large', 'Enterprise'],
            include_lowest=True
        )
        
        # Deal stage features
        # Create deal stage progression indicator for actual stages
        stage_order = {
            'Prospecting': 1,
            'Engaging': 2,
            'Won': 4,
            'Lost': 0  # Lost deals get 0
        }
        df_features['deal_stage_order'] = df_features['deal_stage'].map(stage_order)
            
        # Create binary features for deal status
        df_features['is_closed'] = df_features['deal_stage'].isin(['Won', 'Lost']).astype(int)
        df_features['is_won'] = (df_features['deal_stage'] == 'Won').astype(int)
        df_features['is_lost'] = (df_features['deal_stage'] == 'Lost').astype(int)
        df_features['is_open'] = (1 - df_features['is_closed']).astype(int)

        #Join with Sales Agent
        df_features = df_features.merge(
            df_sales_teams,
            on='sales_agent',
            how='left'
        )

        # Check missing values after join
        missing_after_join = df_features['regional_office'].isnull().sum().sum()
        if missing_after_join > 0:
            self.logger.warning(f"Missing values after join with sales teams: {missing_after_join}")

        # Sales agent features
        # Count of opportunities per agent
        agent_counts = df_features['sales_agent'].value_counts()
        df_features['agent_opportunity_count'] = df_features['sales_agent'].map(agent_counts)
            
        # Agent performance metrics (if we have closed deals)
        agent_performance = df_features[df_features['is_closed']==1].groupby('sales_agent').agg({
            'is_won': ['mean', 'sum'],
            'close_value': ['mean', 'sum'],
            'opportunity_id': 'count'
        }).round(3)
        agent_performance.columns = ['win_rate', 'total_wins', 'avg_deal_value', 'total_revenue', 'closed_deals']
        agent_win_rates = agent_performance['win_rate']
        df_features['agent_win_rate'] = df_features['sales_agent'].map(agent_win_rates)
        
        # Product features
        # Product popularity
        product_counts = df_features['product'].value_counts()
        df_features['product_popularity'] = df_features['product'].map(product_counts)

        # Product performance
        product_performance = df_features[df_features['is_closed']==1].groupby('product').agg({
            'is_won': 'mean',
            'close_value': 'mean',
            'opportunity_id': 'count'
        }).round(3)

        product_performance.columns = ['product_win_rate', 'product_avg_value', 'product_deals_count']
        df_features['product_win_rate'] = df_features['product'].map(product_performance['product_win_rate'])

        # Date features
        # Convert to datetime, handling errors
        df_features['engage_date'] = pd.to_datetime(df_features['engage_date'], errors='coerce')
            
        # Extract date components
        df_features['engage_year'] = df_features['engage_date'].dt.year
        df_features['engage_month'] = df_features['engage_date'].dt.month
        df_features['engage_quarter'] = df_features['engage_date'].dt.quarter
        df_features['engage_day_of_week'] = df_features['engage_date'].dt.dayofweek
        df_features['engage_day_of_year'] = df_features['engage_date'].dt.dayofyear
        
        # Convert to datetime, handling errors
        df_features['close_date'] = pd.to_datetime(df_features['close_date'], errors='coerce')
            
        # Extract date components for closed deals
        mask = df_features['close_date'].notna()
        df_features.loc[mask, 'close_year'] = df_features.loc[mask, 'close_date'].dt.year
        df_features.loc[mask, 'close_month'] = df_features.loc[mask, 'close_date'].dt.month
        df_features.loc[mask, 'close_quarter'] = df_features.loc[mask, 'close_date'].dt.quarter
        df_features.loc[mask, 'close_day_of_week'] = df_features.loc[mask, 'close_date'].dt.dayofweek
        
        # Sales cycle duration (for closed deals)
        mask = df_features['close_date'].notna() & df_features['engage_date'].notna()
        df_features.loc[mask, 'sales_cycle_days'] = (
            df_features.loc[mask, 'close_date'] - df_features.loc[mask, 'engage_date']
        ).dt.days
            
        # Sales cycle categories
        df_features.loc[mask, 'sales_cycle_category'] = pd.cut(
            df_features.loc[mask, 'sales_cycle_days'],
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=['Quick', 'Short', 'Medium', 'Long', 'Very Long'],
            include_lowest=True
        )

        df_features = df_features.merge(
            df_accounts,
            on='account',
            how='left'
        )
        # Check missing values after join
        missing_after_join = df_features['account'].isnull().sum().sum()
        if missing_after_join > 0:
            self.logger.warning(f"Missing values after join with accounts: {missing_after_join}") 

        # Account features (instead of client_id)
        # Account frequency (repeat customers)
        account_counts = df_features['account'].value_counts()
        df_features['account_frequency'] = df_features['account'].map(account_counts)
        df_features['is_repeat_account'] = (df_features['account_frequency'] > 1).astype(int)
        
        # Remove expected value calculation since probability doesn't exist
        # Expected value calculation
        # if 'close_value' in df.columns and 'probability' in df.columns:
        #     df_features['expected_value'] = df_features['close_value'] * df_features['probability']
        
        self.logger.info(f"Created {len(df_features.columns) - len(df_features.columns)} new features")
        
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input DataFrame.
            fit: Whether to fit encoders (True for training, False for inference).
            
        Returns:
            DataFrame with encoded features.
        """
        df_encoded = df.copy()
        
        # Get categorical columns (excluding target if present)
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in categorical_columns:
            categorical_columns.remove(self.target_column)
        
        self.logger.info(f"Encoding categorical features: {categorical_columns}")
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    # Fit and transform
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    # Handle unseen categories by adding them
                    unique_values = df_encoded[col].unique()
                    self.label_encoders[col].fit(unique_values)
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
                else:
                    # Transform only (for inference)
                    if col in self.label_encoders:
                        # Handle unseen categories by mapping to a default value
                        known_categories = set(self.label_encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                        )
                        df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
                    else:
                        self.logger.warning(f"No encoder found for column {col}")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input DataFrame.
            fit: Whether to fit scaler (True for training, False for inference).
            
        Returns:
            DataFrame with scaled features.
        """
        df_scaled = df.copy()
        
        # Get numerical columns (excluding encoded categoricals and target)
        numerical_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it's numerical
        if self.target_column in numerical_columns:
            numerical_columns.remove(self.target_column)
        
        # Remove ID columns and other non-feature columns
        id_columns = [col for col in numerical_columns if 'id' in col.lower()]
        for col in id_columns:
            if col in numerical_columns:
                numerical_columns.remove(col)
        
        if numerical_columns:
            self.logger.info(f"Scaling numerical features: {numerical_columns}")
            
            if fit:
                scaled_values = self.scaler.fit_transform(df_scaled[numerical_columns])
            else:
                scaled_values = self.scaler.transform(df_scaled[numerical_columns])
            
            # Replace original columns with scaled versions
            for i, col in enumerate(numerical_columns):
                df_scaled[f'{col}_scaled'] = scaled_values[:, i]
        
        return df_scaled
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for model training.
        
        Args:
            df: DataFrame with features.
            
        Returns:
            List of feature column names.
        """
        # Start with all columns except target and IDs
        feature_columns = []
        
        for col in df.columns:
            # Skip target column
            if col == self.target_column:
                continue
            
            # Skip ID columns
            if any(id_word in col.lower() for id_word in ['id', 'opportunity']):
                continue
            
            # Skip original categorical columns if encoded versions exist
            if col + '_encoded' in df.columns:
                continue
            
            # Skip original numerical columns if scaled versions exist
            if col + '_scaled' in df.columns:
                continue
            
            # Include the column
            feature_columns.append(col)
        
        # Prioritize encoded and scaled versions
        for col in df.columns:
            if col.endswith('_encoded') or col.endswith('_scaled'):
                if col not in feature_columns:
                    feature_columns.append(col)
        
        self.feature_names = feature_columns
        self.logger.info(f"Selected {len(feature_columns)} feature columns")
        
        return feature_columns
    
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variable for training.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with prepared target.
        """
        df_target = df.copy()
        
        if self.target_column in df.columns:
            # For multi-class classification, we can use the deal stages directly
            # or create binary classification (won vs not won)
            
            # Create binary target (won vs not won)
            df_target['target_binary'] = (df_target[self.target_column] == 'Won').astype(int)
            
            # Create multi-class target (encode deal stages)
            if 'target_encoder' not in self.label_encoders:
                self.label_encoders['target_encoder'] = LabelEncoder()
            
            df_target['target_multiclass'] = self.label_encoders['target_encoder'].fit_transform(
                df_target[self.target_column]
            )
        
        return df_target
    
#    def split_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#        """Split data into train and test sets.
#        
#        Args:
#            df: Input DataFrame.
#            feature_columns: List of feature column names.
#            
#        Returns:
#            Tuple of (X_train, X_test, y_train, y_test).
#        """
#        # Prepare features and target
#        X = df[feature_columns]
#        y = df['target_binary'] if 'target_binary' in df.columns else df[self.target_column]
#        
#        # Split the data
#        X_train, X_test, y_train, y_test = train_test_split(
#            X, y,
#            test_size=self.test_size,
#            random_state=self.random_state,
#            stratify=y if y.dtype != 'float64' else None
#        )
#        
#        self.logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
#        
#        return X_train, X_test, y_train, y_test

    def run_feature_engineering(self, df_sales: pd.DataFrame, df_accounts: pd.DataFrame, df_products: pd.DataFrame, df_sales_teams: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """Run complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Tuple of (processed_dataframe, feature_columns, metadata).
        """
        metadata = {
            'original_shape': df_sales.shape,
            'original_columns': list(df_sales.columns),
            'steps_completed': []
        }
        
        # Step 1: Create new features
        df_features = self.create_features(df_sales, df_accounts, df_products, df_sales_teams)
        metadata['steps_completed'].append('create_features')
        metadata['features_created'] = len(df_features.columns) - len(df_sales.columns)

        # Step 2: Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, fit=True)
        metadata['steps_completed'].append('encode_categorical')
        
        # Step 3: Scale numerical features
        df_scaled = self.scale_numerical_features(df_encoded, fit=True)
        metadata['steps_completed'].append('scale_numerical')
        
        # Step 4: Prepare target
        df_target = self.prepare_target(df_scaled)
        metadata['steps_completed'].append('prepare_target')
        
        # Step 5: Get feature columns
        feature_columns = self.get_feature_columns(df_target)
        metadata['feature_columns'] = feature_columns
        metadata['final_shape'] = df_target.shape
        
        self.logger.info("Feature engineering completed successfully")
        
        return df_target, feature_columns, metadata


def main():
    """Main function for running feature engineering."""
    from src.config.config import get_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = get_config()
    
    # Create feature engineer
    feature_engineer = CRMFeatureEngineer(config)
    
    # Load processed data using smart storage
    try:
        df_sales = feature_engineer.storage.load_dataframe('processed', 'crm_data_processed.csv')
        df_sales_teams = feature_engineer.storage.load_dataframe('raw', 'sales_teams.csv')
        df_accounts = feature_engineer.storage.load_dataframe('raw', 'accounts.csv')
        df_products = feature_engineer.storage.load_dataframe('raw', 'products.csv')
        print(f"📊 Loaded sales: {df_sales.shape}")
        print(f"📊 Loaded sales teams: {df_sales_teams.shape}")
        print(f"📊 Loaded accounts: {df_accounts.shape}")
        print(f"📊 Loaded products: {df_products.shape}")
    except Exception as e:
        print(f"❌ Failed to load processed data: {str(e)}")
        print("Please run data ingestion first: make data-pipeline-flow")
        return 1
    
    # Run feature engineering
    df_processed, feature_columns, metadata = feature_engineer.run_feature_engineering(df_sales, df_accounts, df_products, df_sales_teams)
    
    # Save processed data with features using smart storage
    saved_path = feature_engineer.storage.save_dataframe(df_processed, 'features', 'crm_features.csv')
    
    print(f"✅ Feature engineering completed!")
    print(f"📊 Final shape: {df_processed.shape}")
    print(f"🎯 Features created: {metadata['features_created']}")
    print(f"📝 Feature columns: {len(feature_columns)}")
    print(f"💾 Saved to: {saved_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
