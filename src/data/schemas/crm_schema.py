"""CRM data schema definition and validation."""

from typing import List, Dict, Any, Tuple
import pandas as pd


class CRMDataSchema:
    """Schema definition for CRM sales opportunities dataset."""
    
    def __init__(self):
        """Initialize CRM data schema."""
        # Define required columns based on actual CRM dataset structure
        self.required_columns = [
            'opportunity_id',
            'sales_agent',
            'product',
            'account',
            'deal_stage',
            'engage_date',
            'close_date',
            'close_value'
        ]
        
        # Define expected data types
        self.column_types = {
            'opportunity_id': 'object',
            'sales_agent': 'object',
            'product': 'object',
            'account': 'object',
            'deal_stage': 'object',
            'engage_date': 'object',
            'close_date': 'object',
            'close_value': 'float64'
        }
        
        # Define target column
        self.target_column = 'deal_stage'
        
        # Define expected deal stages (based on actual data)
        self.valid_deal_stages = [
            'Won',
            'Lost',
            'Engaging',
            'Prospecting'
        ]
        
        # Define value ranges
        self.value_ranges = {
            'close_value': (0, float('inf'))
        }
    
    def get_target_column(self) -> str:
        """Get the target column name.
        
        Returns:
            Target column name.
        """
        return self.target_column
    
    def get_feature_columns(self) -> List[str]:
        """Get feature columns (all except target).
        
        Returns:
            List of feature column names.
        """
        return [col for col in self.required_columns if col != self.target_column]
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check extra columns (warn but don't fail)
        extra_columns = set(df.columns) - set(self.required_columns)
        if extra_columns:
            issues.append(f"Extra columns found: {extra_columns}")
        
        # Check data types for existing columns
        for col in self.required_columns:
            if col in df.columns:
                expected_type = self.column_types[col]
                actual_type = str(df[col].dtype)
                
                # Allow some flexibility in numeric types
                if expected_type == 'float64' and actual_type in ['int64', 'float64']:
                    continue
                elif expected_type == 'object' and actual_type == 'object':
                    continue
                else:
                    issues.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        # Check value ranges
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns:
                if df[col].min() < min_val:
                    issues.append(f"Column '{col}' has values below minimum {min_val}")
                if df[col].max() > max_val:
                    issues.append(f"Column '{col}' has values above maximum {max_val}")
        
        # Check deal stages
        if self.target_column in df.columns:
            invalid_stages = set(df[self.target_column].unique()) - set(self.valid_deal_stages)
            if invalid_stages:
                issues.append(f"Invalid deal stages found: {invalid_stages}")
        
        is_valid = len(issues) == 0 or all("Extra columns" in issue for issue in issues)
        return is_valid, issues
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names.
        
        Args:
            df: DataFrame with potentially messy column names.
            
        Returns:
            DataFrame with cleaned column names.
        """
        df_cleaned = df.copy()
        
        # Create mapping from messy names to clean names
        column_mapping = {}
        
        for col in df.columns:
            # Clean column name: lowercase, replace spaces/special chars with underscore
            clean_name = col.lower().strip()
            clean_name = clean_name.replace(' ', '_')
            clean_name = clean_name.replace('-', '_')
            clean_name = clean_name.replace('.', '_')
            clean_name = clean_name.replace('(', '').replace(')', '')
            
        # Handle common variations
        if 'opportunity' in clean_name and 'id' in clean_name:
            clean_name = 'opportunity_id'
        elif 'sales' in clean_name and 'agent' in clean_name:
            clean_name = 'sales_agent'
        elif 'account' in clean_name:
            clean_name = 'account'
        elif 'deal' in clean_name and 'stage' in clean_name:
            clean_name = 'deal_stage'
        elif 'close' in clean_name and 'value' in clean_name:
            clean_name = 'close_value'
        elif 'close' in clean_name and 'date' in clean_name:
            clean_name = 'close_date'
        elif 'engage' in clean_name and 'date' in clean_name:
            clean_name = 'engage_date'
        elif 'product' in clean_name:
            clean_name = 'product'
        
        column_mapping[col] = clean_name
        
        df_cleaned = df_cleaned.rename(columns=column_mapping)
        return df_cleaned
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information.
        
        Returns:
            Dictionary with schema details.
        """
        return {
            'required_columns': self.required_columns,
            'column_types': self.column_types,
            'target_column': self.target_column,
            'feature_columns': self.get_feature_columns(),
            'valid_deal_stages': self.valid_deal_stages,
            'value_ranges': self.value_ranges
        }
