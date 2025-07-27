"""Data validation orchestration."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.data.schemas.crm_schema import CRMDataSchema
from src.utils.storage import StorageManager
from src.config.config import Config

class DataValidationOrchestrator:
    """Orchestrate data validation processes."""
    
    def __init__(self, config: Config):
        """Initialize data validation orchestrator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.schema = CRMDataSchema()
        
        # Initialize storage manager for consistent storage handling
        self.storage = StorageManager(config)
        
        self.logger = logging.getLogger(__name__)
    
    def load_processed_data(self, filename: str = "crm_data_processed.csv") -> pd.DataFrame:
        """Load processed data for validation.
        
        Args:
            filename: Name of the processed data file.
            
        Returns:
            Loaded DataFrame.
        """
        # Use smart storage to load data regardless of storage type
        try:
            df = self.storage.load_dataframe('processed', filename)
            self.logger.info(f"Loaded processed data: {df.shape}")
            return df
        except Exception as e:
            raise FileNotFoundError(f"Processed data file not found: {filename}. Error: {str(e)}")
    
    def run_comprehensive_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive data validation.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Validation results dictionary.
        """
        results = {
            'timestamp': pd.Timestamp.now(),
            'data_shape': df.shape,
            'validations': {}
        }
        
        # Schema validation
        is_valid, issues = self.schema.validate_schema(df)
        results['validations']['schema'] = {
            'passed': is_valid,
            'issues': issues
        }
        
        # Data quality checks
        quality_results = self._run_quality_checks(df)
        results['validations']['quality'] = quality_results
        
        # Business rule validation
        business_results = self._run_business_rules(df)
        results['validations']['business_rules'] = business_results
        
        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)
        results['overall_passed'] = results['overall_score'] >= 0.7
        
        return results
    
    def _run_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run data quality checks.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Quality check results.
        """
        results = {}
        
        # Missing values check
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100
        results['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentage.to_dict(),
            'passed': missing_percentage.max() < 10  # Fail if any column has >10% missing
        }
        
        # Duplicate rows check
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        results['duplicates'] = {
            'count': int(duplicate_count),
            'percentage': float(duplicate_percentage),
            'passed': duplicate_percentage < 5  # Fail if >5% duplicates
        }
        
        # Data type consistency
        type_issues = []
        for col in self.schema.required_columns:
            if col in df.columns:
                expected_type = self.schema.column_types[col]
                actual_type = str(df[col].dtype)
                
                if expected_type == 'float64' and actual_type not in ['int64', 'float64']:
                    type_issues.append(f"{col}: expected numeric, got {actual_type}")
                elif expected_type == 'object' and actual_type != 'object':
                    type_issues.append(f"{col}: expected text, got {actual_type}")
        
        results['data_types'] = {
            'issues': type_issues,
            'passed': len(type_issues) == 0
        }
        
        return results
    
    def _run_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run business rule validation.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Business rule validation results.
        """
        results = {}
        
        # Deal stage validation
        if 'deal_stage' in df.columns:
            invalid_stages = set(df['deal_stage'].unique()) - set(self.schema.valid_deal_stages)
            results['deal_stages'] = {
                'invalid_stages': list(invalid_stages),
                'passed': len(invalid_stages) == 0
            }
        
        # Close value validation
        if 'close_value' in df.columns:
            negative_values = (df['close_value'] < 0).sum()
            zero_values = (df['close_value'] == 0).sum()
            results['close_values'] = {
                'negative_count': int(negative_values),
                'zero_count': int(zero_values),
                'passed': negative_values == 0  # No negative values allowed
            }
        
        # Date validation
        if 'engage_date' in df.columns and 'close_date' in df.columns:
            # Convert to datetime for comparison
            try:
                engage_dates = pd.to_datetime(df['engage_date'])
                close_dates = pd.to_datetime(df['close_date'])
                
                # Check if close date is after engage date
                invalid_date_order = (close_dates < engage_dates).sum()
                results['date_consistency'] = {
                    'invalid_date_order_count': int(invalid_date_order),
                    'passed': invalid_date_order == 0
                }
            except Exception:
                results['date_consistency'] = {
                    'error': 'Could not parse dates',
                    'passed': False
                }
        
        # Closed deals validation
        if 'deal_stage' in df.columns and 'close_date' in df.columns:
            closed_deals = df['deal_stage'].isin(['Won', 'Lost'])
            
            # Closed deals should have close dates
            closed_without_date = (closed_deals & df['close_date'].isna()).sum()
            
            results['closed_deals_consistency'] = {
                'closed_without_date_count': int(closed_without_date),
                'passed': closed_without_date == 0
            }
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall validation score.
        
        Args:
            results: Validation results.
            
        Returns:
            Overall score between 0.0 and 1.0.
        """
        score = 1.0
        validations = results['validations']
        
        # Schema validation (30% weight)
        if not validations['schema']['passed']:
            score -= 0.3
        
        # Quality checks (40% weight)
        quality = validations['quality']
        if not quality['missing_values']['passed']:
            score -= 0.15
        if not quality['duplicates']['passed']:
            score -= 0.15
        if not quality['data_types']['passed']:
            score -= 0.10
        
        # Business rules (30% weight)
        business = validations['business_rules']
        passed_rules = sum(1 for rule in business.values() if rule.get('passed', False))
        total_rules = len(business)
        if total_rules > 0:
            business_score = passed_rules / total_rules
            score -= (1 - business_score) * 0.3
        
        return max(0.0, score)
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report.
        
        Args:
            results: Validation results.
            
        Returns:
            Formatted validation report.
        """
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Data Shape: {results['data_shape']}")
        report.append(f"Overall Score: {results['overall_score']:.2f}")
        report.append(f"Overall Status: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
        report.append("")
        
        # Schema validation
        schema = results['validations']['schema']
        report.append("SCHEMA VALIDATION")
        report.append("-" * 20)
        report.append(f"Status: {'✅ PASSED' if schema['passed'] else '❌ FAILED'}")
        if schema['issues']:
            report.append("Issues:")
            for issue in schema['issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        # Quality checks
        quality = results['validations']['quality']
        report.append("QUALITY CHECKS")
        report.append("-" * 20)
        
        # Missing values
        missing = quality['missing_values']
        report.append(f"Missing Values: {'✅ PASSED' if missing['passed'] else '❌ FAILED'}")
        if not missing['passed']:
            for col, pct in missing['percentages'].items():
                if pct > 0:
                    report.append(f"  - {col}: {pct:.1f}% missing")
        
        # Duplicates
        duplicates = quality['duplicates']
        report.append(f"Duplicates: {'✅ PASSED' if duplicates['passed'] else '❌ FAILED'}")
        if not duplicates['passed']:
            report.append(f"  - {duplicates['count']} duplicate rows ({duplicates['percentage']:.1f}%)")
        
        # Data types
        types = quality['data_types']
        report.append(f"Data Types: {'✅ PASSED' if types['passed'] else '❌ FAILED'}")
        if types['issues']:
            for issue in types['issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        # Business rules
        business = results['validations']['business_rules']
        report.append("BUSINESS RULES")
        report.append("-" * 20)
        
        for rule_name, rule_result in business.items():
            status = '✅ PASSED' if rule_result['passed'] else '❌ FAILED'
            report.append(f"{rule_name.replace('_', ' ').title()}: {status}")
            
            if not rule_result['passed']:
                # Add specific details based on rule type
                if 'invalid_stages' in rule_result:
                    report.append(f"  - Invalid stages: {rule_result['invalid_stages']}")
                elif 'negative_count' in rule_result:
                    report.append(f"  - Negative values: {rule_result['negative_count']}")
                elif 'out_of_range_count' in rule_result:
                    report.append(f"  - Out of range values: {rule_result['out_of_range_count']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function for running data validation."""
    from src.config.config import get_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = get_config()
    
    
    # Create validation orchestrator
    validator = DataValidationOrchestrator(config)
    
    try:
        # Load processed data
        df = validator.load_processed_data(f'crm_data_processed_{config.first_snapshot_month}.csv')
        
        # Run validation
        results = validator.run_comprehensive_validation(df)
        
        # Generate and print report
        report = validator.generate_validation_report(results)
        print(report)
        
        # Return appropriate exit code
        return 0 if results['overall_passed'] else 1
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
