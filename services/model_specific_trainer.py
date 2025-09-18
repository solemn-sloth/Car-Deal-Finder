#!/usr/bin/env python3
"""
Model-Specific Training Functions
Handles training individual car models without one-hot encoding.
Each model trains only on its own data for specialized learning.
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

from config.config import get_xgboost_params

logger = logging.getLogger(__name__)


def prepare_model_specific_features(listings: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare features for model-specific training (no one-hot encoding needed).

    Args:
        listings: List of car listings for a specific make/model

    Returns:
        DataFrame with prepared features
    """
    # Convert to DataFrame
    df = pd.DataFrame(listings)

    if df.empty:
        return df

    # Ensure required columns exist
    required_columns = ['asking_price', 'mileage', 'year']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' missing from listings data")
            df[col] = np.nan

    # Handle legacy price_numeric column
    if 'price_numeric' in df.columns and 'asking_price' not in df.columns:
        df['asking_price'] = df['price_numeric']

    # Compute car age from year
    current_year = datetime.now().year
    df['age'] = current_year - df['year']

    # Calculate market value from price markers
    if 'price_vs_market' in df.columns:
        # If price is above market, market value is lower than asking price
        # If price is below market, market value is higher than asking price
        df['market_value'] = df['asking_price'] + df['price_vs_market']
    else:
        logger.warning("'price_vs_market' not in data, setting market_value to asking_price")
        df['market_value'] = df['asking_price']

    # Extract engine size from spec/subtitle
    df['engine_size'] = np.nan

    # Extract engine size from spec column if available
    if 'spec' in df.columns and not df['spec'].isna().all():
        # Extract engine size (like "2.0", "1.6L", "3.0 TDI")
        import re
        for idx, spec in df['spec'].items():
            if pd.notna(spec):
                # Look for patterns like "1.6", "2.0L", "3.0 TDI"
                match = re.search(r'(\d+\.?\d*)[lL]?(?:\s*(?:tdi|tsi|tfsi|diesel|petrol))?', str(spec).lower())
                if match:
                    try:
                        engine_size = float(match.group(1))
                        if 0.5 <= engine_size <= 8.0:  # Reasonable range
                            df.at[idx, 'engine_size'] = engine_size
                    except ValueError:
                        continue

    # Fill missing engine sizes with median or default
    if not df['engine_size'].isna().all():
        median_engine = df['engine_size'].median()
        df['engine_size'] = df['engine_size'].fillna(median_engine)
    else:
        df['engine_size'] = 1.6  # Default engine size

    logger.info(f"Extracted engine sizes: min={df['engine_size'].min():.1f}, max={df['engine_size'].max():.1f}, mean={df['engine_size'].mean():.2f}")

    # Extract and encode fuel type (simplified)
    df['fuel_type_numeric'] = 1  # Default to petrol
    if 'spec' in df.columns:
        # Simple fuel type detection
        fuel_map = {'diesel': 2, 'hybrid': 3, 'electric': 4, 'plugin': 3}
        for idx, spec in df['spec'].items():
            if pd.notna(spec):
                spec_lower = str(spec).lower()
                for fuel_type, code in fuel_map.items():
                    if fuel_type in spec_lower:
                        df.at[idx, 'fuel_type_numeric'] = code
                        break

    # Extract and encode transmission (simplified)
    df['transmission_numeric'] = 1  # Default to manual
    if 'spec' in df.columns:
        # Simple transmission detection
        trans_map = {'automatic': 2, 'auto': 2, 'cvt': 3, 'dsg': 4}
        for idx, spec in df['spec'].items():
            if pd.notna(spec):
                spec_lower = str(spec).lower()
                for trans_type, code in trans_map.items():
                    if trans_type in spec_lower:
                        df.at[idx, 'transmission_numeric'] = code
                        break

    # Create categorical spec encoding (simplified version of previous logic)
    df['spec_numeric'] = 1  # Default to base
    if 'spec' in df.columns:
        spec_categories = {
            'luxury': ['luxury', 'premium', 'executive', 'lounge', 'se l'],
            'sport': ['sport', 'r-line', 'gti', 'gtd', 'r', 'gte', 'm sport', 'amg'],
            'awd': ['4wd', 'awd', '4x4', 'quattro', '4motion'],
            'tech': ['panoramic', 'leather', 'navigation', 'tech pack', 'dsg']
        }

        for idx, spec in df['spec'].items():
            if pd.notna(spec):
                spec_lower = str(spec).lower()

                # Check categories in order of priority
                if any(trim in spec_lower for trim in spec_categories['luxury']):
                    df.at[idx, 'spec_numeric'] = 3  # Luxury
                elif any(trim in spec_lower for trim in spec_categories['sport']):
                    df.at[idx, 'spec_numeric'] = 2  # Sport
                elif any(trim in spec_lower for trim in spec_categories['awd']):
                    df.at[idx, 'spec_numeric'] = 4  # AWD
                elif any(trim in spec_lower for trim in spec_categories['tech']):
                    df.at[idx, 'spec_numeric'] = 5  # Tech

    logger.info(f"spec_numeric stats: min={df['spec_numeric'].min():.2f}, max={df['spec_numeric'].max():.2f}, mean={df['spec_numeric'].mean():.2f}")

    # Select features for model training (no make/model encoding needed!)
    feature_columns = [
        'asking_price',
        'mileage',
        'age',
        'market_value',
        'fuel_type_numeric',
        'transmission_numeric',
        'engine_size',
        'spec_numeric'
    ]

    # Ensure all feature columns exist
    existing_features = [col for col in feature_columns if col in df.columns]

    # Drop rows with missing values in core features
    core_features = ['asking_price', 'mileage', 'age', 'market_value']
    df_features = df[existing_features].dropna(subset=[col for col in core_features if col in df.columns])

    logger.info(f"Prepared {len(df_features)} listings with complete feature data")
    logger.info(f"Features used: {existing_features}")

    return df_features


def train_model_specific(make: str, model: str, dealer_listings: List[Dict[str, Any]]) -> bool:
    """
    Train XGBoost model for a specific car make/model.

    Args:
        make: Car make (e.g., "BMW")
        model: Car model (e.g., "3 Series")
        dealer_listings: List of dealer listings with price markers

    Returns:
        bool: True if training succeeded, False otherwise
    """
    try:
        logger.info(f"Training model-specific XGBoost for {make} {model}")

        # Prepare features
        df = prepare_model_specific_features(dealer_listings)

        if df.empty:
            logger.error(f"No valid features prepared for {make} {model}")
            return False

        if len(df) < 10:
            logger.error(f"Insufficient data for {make} {model}: {len(df)} samples (need at least 10)")
            return False

        # Prepare training data
        feature_columns = [col for col in df.columns if col != 'market_value']
        X = df[feature_columns].values
        y = df['market_value'].values

        logger.info(f"Training with {len(X)} samples and {len(feature_columns)} features")

        # Split data
        test_size = min(0.2, max(0.1, 20/len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)

        # Get XGBoost parameters
        xgb_params = get_xgboost_params()

        # Train model
        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=False,
            early_stopping_rounds=10
        )

        # Evaluate model
        y_pred = model_xgb.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model performance for {make} {model}: MSE={mse:.2f}, R²={r2:.3f}")

        # Save model and scaler
        success = save_model_specific(make, model, model_xgb, scaler)

        if success:
            logger.info(f"✅ Successfully trained and saved {make} {model}")
        else:
            logger.error(f"❌ Failed to save {make} {model}")

        return success

    except Exception as e:
        logger.error(f"Error training {make} {model}: {e}")
        return False


def save_model_specific(make: str, model: str, xgb_model: xgb.Booster, scaler: StandardScaler) -> bool:
    """
    Save model and scaler to organized sub-folder structure.

    Args:
        make: Car make
        model: Car model
        xgb_model: Trained XGBoost model
        scaler: Feature scaler

    Returns:
        bool: True if save succeeded, False otherwise
    """
    try:
        # Create clean folder names
        make_clean = make.lower().replace(' ', '_').replace('-', '_')
        model_clean = model.lower().replace(' ', '_').replace('-', '_')

        # Create directory structure
        project_root = Path(__file__).parent.parent
        model_dir = project_root / "archive" / "ml_models" / make_clean / model_clean
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = model_dir / "model.xgb"
        xgb_model.save_model(str(model_file))

        # Save scaler
        scaler_file = model_dir / "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

        logger.info(f"Saved model to {model_dir}")
        return True

    except Exception as e:
        logger.error(f"Error saving model for {make} {model}: {e}")
        return False


def load_model_specific(make: str, model: str) -> Tuple[Optional[xgb.Booster], Optional[StandardScaler]]:
    """
    Load model-specific XGBoost model and scaler from sub-folder structure.

    Args:
        make: Car make
        model: Car model

    Returns:
        Tuple of (XGBoost model, scaler) or (None, None) if not found
    """
    try:
        # Create clean folder names
        make_clean = make.lower().replace(' ', '_').replace('-', '_')
        model_clean = model.lower().replace(' ', '_').replace('-', '_')

        # Build paths
        project_root = Path(__file__).parent.parent
        model_dir = project_root / "archive" / "ml_models" / make_clean / model_clean
        model_file = model_dir / "model.xgb"
        scaler_file = model_dir / "scaler.pkl"

        # Check if files exist
        if not model_file.exists() or not scaler_file.exists():
            logger.debug(f"Model files not found for {make} {model}")
            return None, None

        # Load model
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(model_file))

        # Load scaler
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        logger.debug(f"Loaded model for {make} {model}")
        return xgb_model, scaler

    except Exception as e:
        logger.error(f"Error loading model for {make} {model}: {e}")
        return None, None


if __name__ == "__main__":
    # Test with some sample data
    logging.basicConfig(level=logging.INFO)

    sample_listings = [
        {
            'make': 'BMW',
            'model': '3 Series',
            'asking_price': 25000,
            'mileage': 50000,
            'year': 2020,
            'spec': '2.0 TDI Sport',
            'price_vs_market': -1000
        }
    ]

    success = train_model_specific('BMW', '3 Series', sample_listings)
    print(f"Training test: {'Success' if success else 'Failed'}")