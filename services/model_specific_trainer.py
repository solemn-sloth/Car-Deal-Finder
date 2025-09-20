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
import warnings
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Suppress XGBoost warnings about file format
warnings.filterwarnings("ignore", message=".*UBJSON format.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import get_xgboost_params
from src.output_manager import get_output_manager

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

    # Ensure asking_price and mileage are numeric
    df['asking_price'] = pd.to_numeric(df['asking_price'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    # Fill missing values with reasonable defaults
    df['asking_price'] = df['asking_price'].fillna(0)
    df['mileage'] = df['mileage'].fillna(50000)

    # Compute car age from year (ensure year is numeric)
    current_year = datetime.now().year

    # Convert year to numeric, handling any string values
    try:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Fill any invalid years with a reasonable default (e.g., 2015)
        df['year'] = df['year'].fillna(2015)

        df['age'] = current_year - df['year']
    except Exception as e:
        logger.error(f"Error processing year column: {e}")
        # Set default age if calculation fails
        df['age'] = 9  # Default age for cars

    # Check if this is training data (has market_value) or prediction data (no market_value)
    is_training_data = 'market_value' in df.columns

    if is_training_data:
        # Filter out rows without real market values for training
        valid_market_mask = (df['market_value'].notna()) & (df['market_value'] > 0)

        if valid_market_mask.any():
            # Keep only listings with actual market values
            df = df[valid_market_mask].copy()
            logger.info(f"Using {len(df)} listings with actual market_value from retail price scraping")

        else:
            # No valid market values found in training data
            logger.error("No valid market_value data found - cannot train without real market prices")
            return pd.DataFrame()  # Return empty DataFrame
    else:
        # For prediction data, we don't need market_value
        logger.info(f"Processing {len(df)} listings for prediction (no market_value needed)")

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

    # Extract and encode fuel type with unique IDs
    if 'spec' in df.columns:
        # Extract fuel type from spec text first
        df['fuel_type_extracted'] = 'petrol'  # Default

        # Fuel type detection patterns (expanded for better detection)
        fuel_patterns = {'diesel': ['diesel', 'tdi', 'hdi', 'dci', 'd4d', 'bluetec',
                                   # BMW diesel codes
                                   '318d', '320d', '325d', '330d', '335d', '340d',
                                   # Audi diesel codes
                                   'a3 tdi', 'a4 tdi', 'a6 tdi',
                                   # Mercedes diesel codes
                                   'cdi', 'bluetec'],
                        'plugin': ['plugin', 'phev', 'plug-in',
                                  # BMW plugin hybrid codes
                                  '225xe', '330e', '530e', '740e', 'x5 xdrive40e'],
                        'hybrid': ['hybrid', 'hev'],
                        'electric': ['electric', 'ev', 'bev',
                                    # Specific electric models
                                    'i3', 'leaf', 'model 3', 'e-tron']}

        for idx, spec in df['spec'].items():
            if pd.notna(spec):
                spec_lower = str(spec).lower()
                for fuel_type, patterns in fuel_patterns.items():
                    if any(pattern in spec_lower for pattern in patterns):
                        df.at[idx, 'fuel_type_extracted'] = fuel_type
                        break

        # Now create unique numeric IDs for each fuel type found in the dataset
        fuel_codes, fuel_uniques = pd.factorize(df['fuel_type_extracted'])
        df['fuel_type_numeric'] = fuel_codes + 1

        logger.info(f"Found {len(fuel_uniques)} unique fuel types: {list(fuel_uniques)}")
    else:
        # No spec column available, use default
        df['fuel_type_numeric'] = 1

    # Extract and encode transmission with unique IDs
    if 'spec' in df.columns:
        # Extract transmission type from spec text first
        df['transmission_extracted'] = 'manual'  # Default

        # Transmission detection patterns (keeping existing logic but expanded)
        trans_patterns = {'automatic': ['automatic', 'auto'],
                         'cvt': ['cvt'],
                         'dsg': ['dsg'],
                         'semi-auto': ['semi-auto', 'semi auto']}

        for idx, spec in df['spec'].items():
            if pd.notna(spec):
                spec_lower = str(spec).lower()
                for trans_type, patterns in trans_patterns.items():
                    if any(pattern in spec_lower for pattern in patterns):
                        df.at[idx, 'transmission_extracted'] = trans_type
                        break

        # Now create unique numeric IDs for each transmission type found in the dataset
        trans_codes, trans_uniques = pd.factorize(df['transmission_extracted'])
        df['transmission_numeric'] = trans_codes + 1

        logger.info(f"Found {len(trans_uniques)} unique transmission types: {list(trans_uniques)}")
    else:
        # No spec column available, use default
        df['transmission_numeric'] = 1

    # Create unique spec encoding - each spec gets its own ID
    if 'spec' in df.columns:
        # Normalize spec strings (lowercase, strip whitespace)
        df['spec_normalized'] = df['spec'].fillna('').astype(str).str.lower().str.strip()

        # Assign unique numeric IDs to each spec using factorize
        # factorize returns (codes, uniques) where codes are 0-based, so add 1 to start from 1
        spec_codes, spec_uniques = pd.factorize(df['spec_normalized'])
        df['spec_numeric'] = spec_codes + 1

        logger.info(f"Found {len(spec_uniques)} unique specs in dataset")
        if len(spec_uniques) <= 10:  # Log spec mappings for small datasets
            for i, spec in enumerate(spec_uniques):
                logger.info(f"  Spec ID {i+1}: '{spec}'")
    else:
        # No spec column available, use default
        df['spec_numeric'] = 1

    logger.info(f"spec_numeric stats: min={df['spec_numeric'].min():.2f}, max={df['spec_numeric'].max():.2f}, mean={df['spec_numeric'].mean():.2f}")

    # Select features for model training (no make/model encoding needed!)
    feature_columns = [
        'asking_price',
        'mileage',
        'age',
        'fuel_type_numeric',
        'transmission_numeric',
        'engine_size',
        'spec_numeric'
    ]

    # Ensure all feature columns exist
    existing_features = [col for col in feature_columns if col in df.columns]


    # Drop rows with missing values in core features
    if is_training_data:
        # For training: require market_value for target variable
        core_features = ['asking_price', 'mileage', 'age', 'market_value']
        # CRITICAL: Need to include market_value in the returned DataFrame for training!
        features_with_target = existing_features + ['market_value']

        df_features = df[features_with_target].dropna(subset=[col for col in core_features if col in df.columns])
    else:
        # For prediction: don't require market_value (we're trying to predict it)
        core_features = ['asking_price', 'mileage', 'age']

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
        logger.debug(f"Training model-specific XGBoost for {make} {model}")


        # Prepare features
        df = prepare_model_specific_features(dealer_listings)

        if df.empty:
            # Check if the issue is specifically missing market_value data
            # Create test data without market_value column to see if data exists
            test_listings = []
            for listing in dealer_listings:
                test_listing = listing.copy()
                if 'market_value' in test_listing:
                    del test_listing['market_value']
                test_listings.append(test_listing)

            test_df = prepare_model_specific_features(test_listings)
            if not test_df.empty:
                # Data exists but no market_value - this triggers retail scraping
                error_msg = f"No valid market_value data found - cannot train without real market prices for {make} {model}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # No data at all
                error_msg = f"No valid features prepared for {make} {model}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        if len(df) < 10:
            logger.error(f"Insufficient data for {make} {model}: {len(df)} samples (need at least 10)")
            return False


        # Prepare training data with detailed error handling
        try:
            # Extract features and target
            feature_columns = [col for col in df.columns if col != 'market_value']
            X = df[feature_columns].values
            y = df['market_value'].values

        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise

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
    logging.basicConfig(level=logging.ERROR)

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