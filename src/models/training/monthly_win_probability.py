"""Monthly Win Probability Model Training Module.

This module implements the specialized model for predicting the probability 
of winning opportunities in the next month based on the analysis from the 
02_monthly_win_probability_prediction notebook.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import warnings

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, 
    brier_score_loss
)
import xgboost as xgb
from mlflow.models.signature import infer_signature

from src.config.config import Config

warnings.filterwarnings('ignore')


class MonthlyWinProbabilityTrainer:
    """Trainer for monthly win probability prediction models."""
    
    def __init__(self, config: Config):
        """Initialize the trainer with configuration.
        
        Args:
            config: Configuration object with MLflow and storage settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        
        # Set up MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        
        # Set up S3/MinIO credentials for MLflow artifacts
        import os
        os.environ['AWS_ACCESS_KEY_ID'] = config.storage.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.storage.secret_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = config.storage.endpoint_url
        
        self.experiment_name = "monthly_win_probability"
        
        # Ensure experiment exists
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            self.logger.warning(f"Could not create/get MLflow experiment: {e}")
    
    def create_monthly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for monthly win probability prediction.
        
        Args:
            df: Input DataFrame with CRM features.
            
        Returns:
            DataFrame with additional monthly prediction features.
        """
        df_monthly = df.copy()
        
        # Ensure date columns are datetime
        date_columns = ['engage_date', 'close_date', 'snapshot_date', 'creation_date']
        for col in date_columns:
            if col in df_monthly.columns:
                df_monthly[col] = pd.to_datetime(df_monthly[col], errors='coerce')
        
        # Current date simulation (use snapshot date in dataset + 1 month for realistic simulation)
        current_date = df_monthly['snapshot_date'].max() + timedelta(days=30)
        self.logger.info(f"Simulating predictions for: {current_date.strftime('%Y-%m-%d')}")
        
        # Calculate days since engagement
        df_monthly['days_since_engage'] = (current_date - df_monthly['engage_date']).dt.days
        
        # Expected close timeframe based on historical patterns
        # Only calculate for closed deals to avoid NaN in sales_cycle_days
        closed_deals = df_monthly[df_monthly['is_closed'] == 1]
        if len(closed_deals) > 0:
            avg_sales_cycle = closed_deals['sales_cycle_days'].median()
        else:
            avg_sales_cycle = 90  # Default assumption
            
        df_monthly['expected_close_date'] = df_monthly['engage_date'] + timedelta(days=avg_sales_cycle)
        df_monthly['days_to_expected_close'] = (df_monthly['expected_close_date'] - current_date).dt.days
        
        # Monthly prediction flags
        df_monthly['should_close_next_month'] = df_monthly['days_to_expected_close'].between(-15, 45)
        df_monthly['is_overdue'] = df_monthly['days_to_expected_close'] < -30
        df_monthly['is_early_stage'] = df_monthly['days_to_expected_close'] > 60
        
        # Sales velocity features - handle division by zero and NaN values
        # Only calculate for deals with valid sales cycle days
        valid_sales_cycle_mask = (df_monthly['sales_cycle_days'].notna()) & (df_monthly['sales_cycle_days'] > 0)
        df_monthly['sales_velocity'] = 0.0  # Initialize with zeros
        df_monthly.loc[valid_sales_cycle_mask, 'sales_velocity'] = (
            df_monthly.loc[valid_sales_cycle_mask, 'close_value'] / 
            df_monthly.loc[valid_sales_cycle_mask, 'sales_cycle_days']
        )
        
        # Risk factors - handle potential NaN values in close_value and sales_cycle_days
        df_monthly['high_value_deal'] = df_monthly['close_value'] > df_monthly['close_value'].quantile(0.8)
        df_monthly['long_sales_cycle'] = (
            df_monthly['sales_cycle_days'] > df_monthly['sales_cycle_days'].quantile(0.8)
        ).fillna(False)
        
        # Convert boolean columns to int to avoid issues downstream
        bool_columns = ['should_close_next_month', 'is_overdue', 'is_early_stage', 'high_value_deal', 'long_sales_cycle']
        for col in bool_columns:
            df_monthly[col] = df_monthly[col].astype(int)
        
        self.logger.info(f"Monthly prediction features created:")
        self.logger.info(f"- Opportunities that should close next month: {df_monthly['should_close_next_month'].sum():,}")
        self.logger.info(f"- Overdue opportunities: {df_monthly['is_overdue'].sum():,}")
        self.logger.info(f"- Early stage opportunities: {df_monthly['is_early_stage'].sum():,}")
        
        return df_monthly
    
    def prepare_monthly_prediction_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data specifically for monthly win probability prediction.
        
        Args:
            df: Input DataFrame with monthly features.
            
        Returns:
            Tuple of (features_df, target_series, feature_list).
        """
        # Filter to opportunities that are in a predictable timeframe
        training_data = df[
            (df['is_closed'] == 1) |  # Include all closed deals for training
            (df['should_close_next_month'] == True)  # Include current pipeline
        ].copy()
        
        self.logger.info(f"Training data size: {len(training_data):,} opportunities")
        self.logger.info(f"Closed deals: {training_data['is_closed'].sum():,}")
        self.logger.info(f"Open deals for prediction: {(training_data['is_closed'] == 0).sum():,}")
        
        # Define features for monthly prediction
        monthly_features = [
            # Deal characteristics
            'close_value_log', 'close_value_category_encoded',
            
            # Time-based features
            'days_since_engage', 'sales_cycle_days', 
            'engage_month', 'engage_quarter', 'engage_day_of_week',
            
            # Agent and product performance
            'agent_win_rate', 'agent_opportunity_count',
            'product_win_rate', 'product_popularity',
            
            # Account features
            'revenue', 'employees', 'is_repeat_account', 'account_frequency',
            
            # Temporal risk factors
            'should_close_next_month', 'is_overdue', 'is_early_stage',
            'high_value_deal', 'long_sales_cycle',
            
            # Velocity and momentum
            'sales_velocity',
            
            # Categorical encodings
            'sales_agent_encoded', 'product_encoded', 'account_encoded',
            'manager_encoded', 'regional_office_encoded', 'sector_encoded'
        ]
        
        # Handle missing values in computed features
        for col in ['sales_velocity', 'days_since_engage']:
            if col in training_data.columns:
                training_data[col] = training_data[col].fillna(training_data[col].median())
        
        # Select features that exist in the dataset
        available_features = [f for f in monthly_features if f in training_data.columns]
        missing_features = [f for f in monthly_features if f not in training_data.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        self.logger.info(f"Using {len(available_features)} features for monthly prediction")
        
        X = training_data[available_features].copy()
        y = training_data['is_won'].copy()  # Predict win probability
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        self.logger.info(f"Monthly prediction dataset:")
        self.logger.info(f"Features: {X.shape[1]}")
        self.logger.info(f"Samples: {X.shape[0]:,}")
        self.logger.info(f"Win rate: {y.mean():.3f}")

        return X, y, available_features, training_data

    def create_temporal_split(self, X: pd.DataFrame, y: pd.Series, df_training: pd.DataFrame, 
                            test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create a temporal split that respects time ordering.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            df_training: Original training DataFrame.
            test_size: Proportion of data for test set.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        # Reset indices to ensure alignment
        df_training = df_training.reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # For closed deals, use temporal split
        closed_mask = df_training['is_closed'] == 1
        open_mask = df_training['is_closed'] == 0
        
        # Get closed deals and sort by close_date
        closed_indices = df_training[closed_mask].index.tolist()
        
        if len(closed_indices) > 0:
            # Sort closed deals by close_date for temporal split
            self.logger.info(f"Found {len(closed_indices):,} closed deals for temporal split")
            closed_data_with_index = df_training.loc[closed_indices].copy()

            # Handle missing close_dates
            if 'close_date' in closed_data_with_index.columns:
                # Fill missing close_dates with engage_date + average sales cycle
                avg_cycle = closed_data_with_index['sales_cycle_days'].median()
                if pd.isna(avg_cycle):
                    avg_cycle = 90  # Default
                
                missing_close_date = closed_data_with_index['close_date'].isna()
                closed_data_with_index.loc[missing_close_date, 'close_date'] = (
                    pd.to_datetime(closed_data_with_index.loc[missing_close_date, 'engage_date']) + 
                    timedelta(days=avg_cycle)
                )
                
                # Sort by close_date
                closed_data_with_index['close_date'] = pd.to_datetime(closed_data_with_index['close_date'])
                closed_data_sorted = closed_data_with_index.sort_values('close_date')
                sorted_indices = closed_data_sorted.index.tolist()
            else:
                # If no close_date, just use the original order
                sorted_indices = closed_indices
            
            # Split closed deals temporally
            split_idx = int(len(sorted_indices) * (1 - test_size))
            self.logger.info(f"Temporal split: {len(sorted_indices)} closed deals, {split_idx} train, {len(sorted_indices) - split_idx} test")
            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]
        else:
            train_indices = []
            test_indices = []

        #TODO: Review handling of open deals
        ## Add open deals (split randomly but ensure indices exist)
        #open_indices = df_training[open_mask].index.tolist()
        #if len(open_indices) > 0:
        #    # Add 80% of open deals to training, 20% to test
        #    open_split = int(len(open_indices) * 0.8)
        #    train_indices.extend(open_indices[:open_split])
        #    test_indices.extend(open_indices[open_split:])
        
        # Create train/test sets using valid indices
        try:
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
        except Exception as e:
            self.logger.error(f"Error creating train/test split: {e}")
            # Fallback to random split
            self.logger.warning("Falling back to random split due to indexing issues")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        self.logger.info("Temporal Split Results:")
        self.logger.info(f"Training set: {len(X_train):,} samples, win rate: {y_train.mean():.3f}")
        self.logger.info(f"Test set: {len(X_test):,} samples, win rate: {y_test.mean():.3f}")
        self.logger.info(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_monthly_win_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Train and calibrate models specifically for monthly win probability prediction.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            
        Returns:
            Dictionary with model results.
        """
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        
        self.logger.info("Training Monthly Win Probability Models:")
        self.logger.info("=" * 60)
        
        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            with mlflow.start_run(run_name=f"monthly_win_prob_{name}"):
                # Train base model
                model.fit(X_train, y_train)
                
                # Calibrate probabilities using isotonic regression
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated_model.fit(X_train, y_train)
                
                # Predictions
                y_pred = calibrated_model.predict(X_test)
                y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                brier_score = brier_score_loss(y_test, y_pred_proba)
                
                # Log metrics
                mlflow.log_param("model_type", name)
                mlflow.log_param("calibration", "isotonic")
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("brier_score", brier_score)
                
                # Log model
                mlflow.sklearn.log_model(
                    calibrated_model, 
                    artifact_path=name,
                    input_example=X_test.iloc[:5], 
                    signature=infer_signature(X_train, y_train)
                )
                
                results[name] = {
                    'model': calibrated_model,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'brier_score': brier_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                self.logger.info(f"  Accuracy: {accuracy:.4f}")
                self.logger.info(f"  ROC AUC: {roc_auc:.4f}")
                self.logger.info(f"  Brier Score: {brier_score:.4f} (lower is better)")
        
        return results
    
    def register_best_model(self, results: Dict[str, Dict[str, Any]], X_test: pd.DataFrame, 
                          y_test: pd.Series) -> mlflow.models.model.ModelInfo:
        """Register the best model for monthly predictions.
        
        Args:
            results: Training results from all models.
            X_test: Test features for final evaluation.
            y_test: Test target for final evaluation.
            
        Returns:
            MLflow model info for the registered model.
        """
        # Find best model based on ROC AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_model = results[best_model_name]['model']
        
        self.logger.info(f"Best Model: {best_model_name.upper()}")
        self.logger.info(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
        self.logger.info(f"Brier Score: {results[best_model_name]['brier_score']:.4f}")
        
        with mlflow.start_run(run_name=f"monthly_win_model_final_{best_model_name}"):
            # Log final model metrics
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            final_auc = roc_auc_score(y_test, y_pred_proba)
            final_brier = brier_score_loss(y_test, y_pred_proba)
            
            mlflow.log_param("model_type", best_model_name)
            mlflow.log_param("purpose", "monthly_win_probability")
            mlflow.log_param("calibration", "isotonic")
            mlflow.log_param("features_count", X_test.shape[1])
            
            mlflow.log_metric("final_roc_auc", final_auc)
            mlflow.log_metric("final_brier_score", final_brier)
            mlflow.log_metric("training_samples", len(X_test))
            
            # Log model signature
            signature = infer_signature(X_test, y_pred_proba)
            
            # Register model
            model_info = mlflow.sklearn.log_model(
                best_model, 
                artifact_path="model", 
                signature=signature,
                registered_model_name="monthly_win_probability_model"
            )
            
            self.logger.info(f"✅ Model registered successfully!")
            self.logger.info(f"   Model Name: monthly_win_probability_model")
            self.logger.info(f"   Version: {model_info.registered_model_version}")
            self.logger.info(f"   ROC AUC: {final_auc:.4f}")
            self.logger.info(f"   Brier Score: {final_brier:.4f}")
            
            return model_info
    
    def generate_training_summary(self, results: Dict[str, Dict[str, Any]], 
                                X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Generate a comprehensive training summary.
        
        Args:
            results: Training results from all models.
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Training summary dictionary.
        """
        # Create results DataFrame
        results_df = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'ROC AUC': results['roc_auc'],
                'Brier Score': results['brier_score']
            }
            for name, results in results.items()
        }).T
        
        best_model_name = results_df['ROC AUC'].idxmax()
        
        summary = {
            "status": "success",
            "experiment_name": self.experiment_name,
            "training_timestamp": datetime.now().isoformat(),
            "best_model": {
                "name": best_model_name,
                "roc_auc": results_df.loc[best_model_name, 'ROC AUC'],
                "brier_score": results_df.loc[best_model_name, 'Brier Score'],
                "accuracy": results_df.loc[best_model_name, 'Accuracy']
            },
            "dataset_info": {
                "train_samples": len(X_train),
                "features": X_train.shape[1],
                "win_rate": y_train.mean(),
                "feature_count": X_train.shape[1]
            },
            "model_results": results_df.round(4).to_dict(),
            "model_performance_ranking": results_df.sort_values('ROC AUC', ascending=False).index.tolist()
        }
        
        return summary
    
    def run_training_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete monthly win probability training pipeline.
        
        Args:
            df: Input DataFrame with CRM features.
            
        Returns:
            Training pipeline results.
        """
        try:
            self.logger.info("🚀 Starting Monthly Win Probability Training Pipeline")
            
            # Step 1: Create monthly features
            df_monthly = self.create_monthly_features(df)
            
            # Step 2: Prepare training data
            X, y, feature_list, training_data = self.prepare_monthly_prediction_data(df_monthly)
            
            # Step 3: Create temporal split
            X_train, X_test, y_train, y_test = self.create_temporal_split(X, y, training_data)
            
            # Step 4: Train models
            results = self.train_monthly_win_models(X_train, y_train, X_test, y_test)
            
            # Step 5: Register best model
            model_info = self.register_best_model(results, X_test, y_test)
            
            # Step 6: Generate summary
            summary = self.generate_training_summary(results, X_train, y_train)
            
            # Handle model_info attributes safely
            model_info_dict = {}
            if hasattr(model_info, 'name'):
                model_info_dict["registered_model_name"] = model_info.name
            elif hasattr(model_info, 'registered_model_name'):
                model_info_dict["registered_model_name"] = model_info.registered_model_name
            else:
                model_info_dict["registered_model_name"] = "monthly_win_probability_model"
                
            if hasattr(model_info, 'version'):
                model_info_dict["model_version"] = model_info.version
            elif hasattr(model_info, 'registered_model_version'):
                model_info_dict["model_version"] = model_info.registered_model_version
            else:
                model_info_dict["model_version"] = "unknown"
                
            if hasattr(model_info, 'source'):
                model_info_dict["model_uri"] = model_info.source
            elif hasattr(model_info, 'model_uri'):
                model_info_dict["model_uri"] = model_info.model_uri
            else:
                model_info_dict["model_uri"] = "unknown"
                
            summary["model_info"] = model_info_dict
            summary["feature_list"] = feature_list
            
            self.logger.info("🎉 Monthly Win Probability Training Pipeline Completed!")
            self.logger.info(f"📊 Best Model: {summary['best_model']['name']}")
            self.logger.info(f"🎯 ROC AUC: {summary['best_model']['roc_auc']:.4f}")
            self.logger.info(f"✨ Model registered: {model_info_dict['registered_model_name']} v{model_info_dict['model_version']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
