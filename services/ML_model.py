#!/usr/bin/env python3
"""
Universal ML Model
Single XGBoost model that handles all car makes/models using ordinal encodings.
Replaces individual make/model-specific models for better efficiency and accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.encodings import encode_listings, get_encoding_stats

logger = logging.getLogger(__name__)


class UniversalMLModel:
    """
    Universal XGBoost model for car deal prediction across all makes/models.
    
    Features:
    - Single model handles all vehicle types using ordinal encodings
    - Accumulates training data from multiple make/model groups
    - Better accuracy through larger, diverse training dataset
    - Consistent prediction logic across all vehicles
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize universal ML model.
        
        Args:
            model_path: Directory to save/load models (defaults to project archive/ml_models)
        """
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "archive" / "ml_models"
        
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.model: Optional[xgb.Booster] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.training_stats: Dict = {}
        
        # File paths
        self.model_file = self.model_path / "universal_model.json"
        self.scaler_file = self.model_path / "universal_scaler.pkl"
        self.stats_file = self.model_path / "universal_training_stats.json"
    
    def prepare_universal_features(self, listings: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare features for universal model training/prediction.
        
        Args:
            listings: List of car listings (dealer or private)
            
        Returns:
            pd.DataFrame: Prepared features with universal encodings
        """
        if not listings:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(listings)
        
        # Add universal encodings (make_encoded, model_encoded)
        encoded_listings = encode_listings(listings)
        encoded_df = pd.DataFrame(encoded_listings)
        
        # Ensure required columns exist with defaults
        required_columns = {
            'asking_price': 0,
            'mileage': 50000,
            'year': 2015,
            'make_encoded': 0,
            'model_encoded': 0
        }
        
        for col, default_val in required_columns.items():
            if col not in encoded_df.columns:
                encoded_df[col] = default_val
                logger.warning(f"Column '{col}' missing, filled with default: {default_val}")
        
        # Calculate age from year
        current_year = datetime.now().year
        encoded_df['age'] = current_year - encoded_df['year']
        
        # Calculate market value from price markers (for dealer data only)
        if 'price_vs_market' in encoded_df.columns:
            # Market value = asking price + price vs market difference
            encoded_df['market_value'] = encoded_df['asking_price'] + encoded_df['price_vs_market']
        else:
            # For private listings, we'll predict market_value, so set to asking_price as placeholder
            encoded_df['market_value'] = encoded_df['asking_price']
        
        # Extract and process spec information (existing logic from ML_analyser.py)
        self._process_spec_features(encoded_df)
        
        # Extract fuel type and transmission (existing logic)
        self._process_fuel_transmission_features(encoded_df)
        
        # Define feature columns for universal model
        self.feature_columns = [
            'asking_price', 
            'mileage', 
            'age', 
            'make_encoded',          # NEW: Universal make encoding
            'model_encoded',         # NEW: Universal model encoding  
            'fuel_type_numeric', 
            'transmission_numeric',
            'engine_size',
            'spec_numeric'
        ]
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
                logger.warning(f"Feature column '{col}' missing, filled with 0")
        
        # Clean and validate feature data
        for col in self.feature_columns:
            # Convert to numeric, replacing invalid values with 0
            encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce').fillna(0)
        
        # For training data, we need market_value as target
        # For prediction data, market_value will be predicted
        
        logger.debug(f"Prepared {len(encoded_df)} listings with {len(self.feature_columns)} features")
        return encoded_df[self.feature_columns + (['market_value'] if 'market_value' in encoded_df.columns else [])]
    
    def _process_spec_features(self, df: pd.DataFrame):
        """Process vehicle specification features (existing logic from ML_analyser.py)."""
        # Extract engine size from spec if available
        df['engine_size'] = 1.6  # Default
        
        if 'spec' in df.columns:
            import re
            engine_size_pattern = r'(\d+\.\d+)(?:\s|L|$|T)'
            
            for idx, row in df.iterrows():
                if pd.notna(row.get('spec')):
                    spec_text = str(row['spec'])
                    match = re.search(engine_size_pattern, spec_text)
                    if match:
                        try:
                            df.at[idx, 'engine_size'] = float(match.group(1))
                        except (ValueError, TypeError):
                            pass
        
        # Create unique spec encoding - each spec gets its own ID
        if 'spec' in df.columns:
            # Normalize spec strings (lowercase, strip whitespace)
            df['spec_normalized'] = df['spec'].fillna('').astype(str).str.lower().str.strip()

            # Assign unique numeric IDs to each spec using factorize
            # factorize returns (codes, uniques) where codes are 0-based, so add 1 to start from 1
            spec_codes, spec_uniques = pd.factorize(df['spec_normalized'])
            df['spec_numeric'] = spec_codes + 1

            logger.debug(f"Universal model: Found {len(spec_uniques)} unique specs in dataset")
        else:
            # No spec column available, use default
            df['spec_numeric'] = 1
    
    def _process_fuel_transmission_features(self, df: pd.DataFrame):
        """Process fuel type and transmission features (existing logic)."""
        # Fuel type processing
        fuel_map = {
            'Unknown': 0, 'Petrol': 1, 'Diesel': 2, 
            'Hybrid': 3, 'Electric': 4, 'Plugin Hybrid': 5
        }
        
        df['fuel_type_numeric'] = 0
        if 'fuel_type' in df.columns:
            df['fuel_type_numeric'] = df['fuel_type'].map(fuel_map).fillna(0)
        
        # Transmission processing
        trans_map = {
            'Unknown': 0, 'Manual': 1, 'Automatic': 2, 
            'Semi-Auto': 3, 'CVT': 4, 'DSG': 5
        }
        
        df['transmission_numeric'] = 0
        if 'transmission' in df.columns:
            df['transmission_numeric'] = df['transmission'].map(trans_map).fillna(0)
    
    def train_universal_model(self, dealer_listings: List[Dict[str, Any]]) -> bool:
        """
        Train universal XGBoost model on dealer listings with retail price markers.
        
        Args:
            dealer_listings: List of dealer car listings with price_vs_market data
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Prepare features
            features_df = self.prepare_universal_features(dealer_listings)
            
            if len(features_df) < 20:
                logger.error(f"Insufficient training data: {len(features_df)} samples (minimum 20)")
                return False
            
            # Separate features and target
            X = features_df[self.feature_columns]
            y = features_df['market_value']
            
            # Split for training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            
            # XGBoost parameters optimized for universal model
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,          # Slightly deeper for universal model
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,   # Higher for stability with diverse data
                'seed': 42
            }
            
            # Train the model
            logger.info(f"Training universal XGBoost model on {len(X_train)} samples...")
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=150,     # More rounds for universal model
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=15,
                verbose_eval=False
            )
            
            # Evaluate model performance
            val_preds = self.model.predict(dval)
            rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            r2 = r2_score(y_val, val_preds)
            
            # Store training statistics
            self.training_stats = {
                'training_date': datetime.now().isoformat(),
                'total_samples': len(features_df),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'feature_columns': self.feature_columns,
                'encoding_stats': get_encoding_stats()
            }
            
            logger.info(f"Universal model trained successfully:")
            logger.info(f"  - RMSE: £{rmse:,.0f}")
            logger.info(f"  - R² Score: {r2:.4f}")
            logger.info(f"  - Training samples: {len(X_train):,}")
            
            # Save model components
            success = self.save_model()
            if success:
                logger.info("Universal model saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training universal model: {e}")
            return False
    
    def predict_market_values(self, private_listings: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Predict market values for private seller listings using universal model.
        
        Args:
            private_listings: List of private seller car listings
            
        Returns:
            pd.DataFrame: Original listings with predictions added
        """
        if not self.model or not self.scaler:
            raise ValueError("Universal model not loaded. Call load_model() first.")
        
        if not private_listings:
            return pd.DataFrame()
        
        try:
            # Prepare features (without market_value target)
            features_df = self.prepare_universal_features(private_listings)
            
            if len(features_df) == 0:
                logger.warning("No valid features prepared for prediction")
                return pd.DataFrame()
            
            # Get features for prediction
            X = features_df[self.feature_columns]
            
            # Scale features using trained scaler
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            dtest = xgb.DMatrix(X_scaled)
            predicted_market_values = self.model.predict(dtest)
            
            # Create results DataFrame
            results_df = pd.DataFrame(private_listings)
            
            # Add predictions
            results_df['predicted_market_value'] = predicted_market_values
            results_df['predicted_profit_margin'] = (
                (predicted_market_values - results_df['asking_price']) / results_df['asking_price']
            )
            results_df['potential_profit'] = predicted_market_values - results_df['asking_price']
            
            logger.info(f"Predicted market values for {len(results_df)} private listings")
            logger.info(f"Average predicted market value: £{predicted_market_values.mean():,.0f}")
            logger.info(f"Average profit margin: {results_df['predicted_profit_margin'].mean():.2%}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error predicting market values: {e}")
            return pd.DataFrame()
    
    def save_model(self) -> bool:
        """Save universal model components to disk."""
        try:
            # Save XGBoost model
            if self.model:
                self.model.save_model(str(self.model_file))
            
            # Save scaler
            if self.scaler:
                with open(self.scaler_file, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # Save training stats
            if self.training_stats:
                import json
                with open(self.stats_file, 'w') as f:
                    json.dump(self.training_stats, f, indent=2)
            
            logger.info(f"Universal model saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving universal model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load universal model components from disk."""
        try:
            # Check if model files exist
            if not self.model_file.exists():
                logger.warning(f"Universal model file not found: {self.model_file}")
                return False
            
            # Load XGBoost model
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_file))
            
            # Load scaler
            if self.scaler_file.exists():
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                logger.error(f"Scaler file not found: {self.scaler_file}")
                return False
            
            # Load training stats
            if self.stats_file.exists():
                import json
                with open(self.stats_file, 'r') as f:
                    self.training_stats = json.load(f)
                    self.feature_columns = self.training_stats.get('feature_columns', [])
            
            logger.info(f"Universal model loaded from {self.model_path}")
            if self.training_stats:
                logger.info(f"  - Trained on: {self.training_stats.get('training_date', 'Unknown')}")
                logger.info(f"  - R² Score: {self.training_stats.get('r2_score', 0):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading universal model: {e}")
            return False
    
    def is_model_available(self) -> bool:
        """Check if a trained universal model is available."""
        return (self.model_file.exists() and 
                self.scaler_file.exists() and 
                self.stats_file.exists())
    
    def get_model_info(self) -> Dict:
        """Get information about the current universal model."""
        info = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'model_path': str(self.model_path),
            'model_files_exist': self.is_model_available(),
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats
        }
        
        return info


# Global universal model instance
universal_model = UniversalMLModel()

# Convenience functions
def train_universal_model(dealer_listings: List[Dict[str, Any]]) -> bool:
    """Train the universal model on dealer listings."""
    return universal_model.train_universal_model(dealer_listings)

def predict_with_universal_model(private_listings: List[Dict[str, Any]]) -> pd.DataFrame:
    """Predict market values using the universal model."""
    return universal_model.predict_market_values(private_listings)

def load_universal_model() -> bool:
    """Load the universal model from disk."""
    return universal_model.load_model()

def is_universal_model_available() -> bool:
    """Check if a trained universal model is available."""
    return universal_model.is_model_available()

def get_universal_model_info() -> Dict:
    """Get information about the universal model."""
    return universal_model.get_model_info()


if __name__ == "__main__":
    # Test the universal model
    print("Testing Universal ML Model...")
    
    # Check if model exists
    model_available = is_universal_model_available()
    print(f"Universal model available: {model_available}")
    
    if model_available:
        # Load and show model info
        success = load_universal_model()
        if success:
            info = get_universal_model_info()
            print(f"\nModel Info:")
            for key, value in info.items():
                if key != 'training_stats':
                    print(f"  {key}: {value}")
            
            if info['training_stats']:
                stats = info['training_stats']
                print(f"\nTraining Stats:")
                print(f"  Date: {stats.get('training_date', 'Unknown')}")
                print(f"  Samples: {stats.get('total_samples', 0)}")
                print(f"  RMSE: £{stats.get('rmse', 0):,.0f}")
                print(f"  R² Score: {stats.get('r2_score', 0):.4f}")
    else:
        print("No universal model found. Run weekly training to create one.")