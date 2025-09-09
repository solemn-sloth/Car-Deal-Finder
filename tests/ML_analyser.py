#!/usr/bin/env python
"""
Car Deal ML Analyzer - Test Script

Uses XGBoost to identify profitable car deals by:
1. Training on dealer listings (with price markers)
2. Predicting on private listings
3. Filtering for deals with high predicted profit margin
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Keep GradientBoostingRegressor import in case we need it as a fallback
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from playwright.sync_api import sync_playwright

# Set up proper import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the retail price scraper for price marker data
from services.retail_price_scraper import scrape_price_marker, batch_scrape_price_markers

# Import main scraping components from your core codebase
from services.network_requests import AutoTraderAPIClient
from services.data_adapter import NetworkDataAdapter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("car_deal_ml.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("car_deal_ml")

# Constants
MIN_PROFIT_MARGIN = 0.15  # 15% minimum profit margin
DATABASE_PATH = "car_deals_database.pkl"
MODELS_PATH = "ml_models"

# Create models directory if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)


# Data Collection Functions
# Not needed - using AutoTraderAPIClient instead
# def build_search_url(make: str, model: str = None, postcode: str = "SW1A1AA", radius: int = 50) -> str:
#     """Build search URL for AutoTrader based on make and model"""
#     base_url = "https://www.autotrader.co.uk/car-search"
#     params = [
#         f"make={make.lower()}",
#         f"postcode={postcode}",
#         f"radius={radius}",
#     ]
#     
#     if model:
#         params.append(f"model={model.lower()}")
#     
#     return f"{base_url}?{'&'.join(params)}"


def scrape_listings(make: str, model: str = None, max_pages: int = None) -> List[Dict[str, Any]]:
    """Scrape car listings for a given make and model
    
    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (e.g., '3-series')
        max_pages: Maximum number of pages to scrape (None for all available pages)
    
    Returns:
        List of dictionaries containing car listing details
    """
    logger.info(f"Scraping listings for {make} {model or ''}")
    
    # Initialize the AutoTrader API client from your main code
    api_client = AutoTraderAPIClient()
    
    try:
        # Use the existing get_all_cars method to scrape listings with no page limit
        cars = api_client.get_all_cars(
            make=make,
            model=model,
            max_pages=max_pages  # Passing None to get all pages
        )
        
        # Convert the listings to a consistent format
        listings = []
        for car in cars:
            # Extract the processed data from the API response
            listing = {
                # Core vehicle data
                'make': car.get('make', ''),
                'model': car.get('model', ''),
                'year': car.get('year', 0),
                'price_numeric': car.get('price', 0),
                'mileage': car.get('mileage', 0),
                
                # Additional data
                'seller_type': 'Dealer' if car.get('seller_type', '').upper() == 'TRADE' else 'Private',  # Match the format expected by the filter function
                'url': car.get('url', ''),
                'image_url': car.get('image_url', ''),
                'image_url_2': car.get('image_url_2', ''),
                'location': car.get('location', ''),
                
                # Add spec data for feature extraction
                'spec': f"{car.get('engine_size', '')} {car.get('fuel_type', '')} {car.get('transmission', '')}"
            }
            
            listings.append(listing)
        
        logger.info(f"Scraped {len(listings)} listings for {make} {model or ''}")
        return listings
        
    except Exception as e:
        logger.error(f"Error scraping listings: {e}")
        return []


def enrich_with_price_markers(listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich dealer listings with price marker data using parallel processing
    
    Args:
        listings: List of car listings
        
    Returns:
        Enriched listings with price_vs_market field added
    """
    # Filter out listings without URLs
    valid_listings = []
    valid_urls = []
    
    for listing in listings:
        if 'url' in listing and listing['url']:
            valid_listings.append(listing)
            valid_urls.append(listing['url'])
        else:
            logger.warning(f"Skipping listing without URL: {listing.get('make', '')} {listing.get('model', '')}")
    
    total_urls = len(valid_urls)
    logger.info(f"Starting to process {total_urls} URLs in parallel for price markers")
    print(f"\n{'='*60}")
    print(f"PRICE MARKER SCRAPING: {total_urls} URLs")
    print(f"{'='*60}")
    
    if not valid_urls:
        return []
    
    # Create a progress display function
    start_time = datetime.now()
    
    def progress_callback(completed, total):
        # Calculate time elapsed and estimated time remaining
        elapsed = datetime.now() - start_time
        if completed > 0:
            avg_time_per_item = elapsed.total_seconds() / completed
            remaining_items = total - completed
            est_time_remaining = remaining_items * avg_time_per_item
            
            # Format as minutes and seconds
            minutes, seconds = divmod(est_time_remaining, 60)
            time_str = f"{int(minutes)}m {int(seconds)}s"
            
            # Calculate speed (items per minute)
            if elapsed.total_seconds() > 0:
                speed = completed * 60 / elapsed.total_seconds()
                
                # Add speed and time info to progress report
                logger.info(f"Progress: {completed}/{total} ({completed/total:.1%}) - Speed: {speed:.1f} URLs/min - Est. remaining: {time_str}")
    
    try:
        # Use batch processing with multiprocessing to get all price markers in parallel
        # Pass in our progress callback for better monitoring
        results = batch_scrape_price_markers(
            valid_urls, 
            progress_callback=progress_callback
        )
        
        print(f"\n{'='*60}")
        print(f"PROCESSING RESULTS")
        print(f"{'='*60}")
        
        # Merge price marker data with original listings
        enriched_listings = []
        success_count = 0
        error_count = 0
        
        for listing in valid_listings:
            url = listing['url']
            
            if url in results:
                price_marker = results[url]
                
                # Check if this is an error result
                if 'Error' in price_marker.get('marker_text', ''):
                    error_count += 1
                    logger.warning(f"Error for {listing.get('make', '')} {listing.get('model', '')}: {price_marker.get('marker_text', 'Unknown error')}")
                    listing['price_vs_market'] = 0.0  # Default to market value
                    enriched_listings.append(listing)
                else:
                    # Successful result
                    success_count += 1
                    # Add price marker data to the listing
                    enriched_listing = {**listing, 'price_vs_market': price_marker['market_difference']}
                    enriched_listings.append(enriched_listing)
                    
                    # Log but don't spam console
                    if success_count % 10 == 0:  # Log every 10th success
                        logger.info(f"Enriched {success_count} listings with price markers")
            else:
                # URL was processed but no result
                error_count += 1
                logger.error(f"No price marker result for URL: {url}")
                listing['price_vs_market'] = 0.0  # Default to market value
                enriched_listings.append(listing)
        
        # Final summary
        total_time = datetime.now() - start_time
        minutes, seconds = divmod(total_time.total_seconds(), 60)
        
        print(f"\n{'='*60}")
        print(f"PRICE MARKER SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {int(minutes)}m {int(seconds)}s")
        print(f"Successful: {success_count}/{total_urls} ({success_count/total_urls:.1%})")
        print(f"Failed: {error_count}/{total_urls} ({error_count/total_urls:.1%})")
        print(f"Average speed: {total_urls*60/total_time.total_seconds():.1f} URLs/minute")
        print(f"{'='*60}\n")
                
    except Exception as e:
        logger.error(f"Error in batch price marker processing: {e}")
        # Fall back to sequential processing as a backup
        logger.warning("Falling back to sequential processing")
        
        enriched_listings = []
        total = len(valid_listings)
        
        print("\nFalling back to sequential processing...")
        
        for i, listing in enumerate(valid_listings):
            try:
                # Show progress for sequential processing too
                print(f"\rProgress: {i+1}/{total} ({(i+1)/total:.1%})", end="", flush=True)
                
                # Scrape price marker for this listing
                price_marker = scrape_price_marker(listing['url'])
                
                # Add price marker data to the listing
                enriched_listing = {**listing, 'price_vs_market': price_marker['market_difference']}
                enriched_listings.append(enriched_listing)
                
            except Exception as e:
                logger.error(f"Error enriching listing with price marker: {e}")
                # Add without price marker
                listing['price_vs_market'] = 0.0  # Default to market value
                enriched_listings.append(listing)
        
        print("\nSequential processing complete")
    
    return enriched_listings


def filter_listings_by_seller_type(listings: List[Dict[str, Any]], seller_type: str) -> List[Dict[str, Any]]:
    """Filter listings by seller type (dealer or private)
    
    Args:
        listings: List of car listings
        seller_type: 'Dealer' or 'Private'
        
    Returns:
        Filtered list of listings
    """
    # Debug seller types
    seller_types = set(listing.get('seller_type', 'Unknown') for listing in listings)
    logger.info(f"Found seller types in data: {seller_types}")
    
    filtered = [listing for listing in listings if listing.get('seller_type') == seller_type]
    logger.info(f"Filtered {len(filtered)} {seller_type.lower()} listings from {len(listings)} total")
    return filtered


# Feature Engineering Functions
def prepare_features(listings: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract and prepare features for model training or prediction
    
    Args:
        listings: List of car listings
        
    Returns:
        DataFrame with prepared features
    """
    # Convert listings to DataFrame
    df = pd.DataFrame(listings)
    
    # Ensure required columns exist
    required_columns = ['price_numeric', 'mileage', 'year']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' missing from listings data")
            df[col] = np.nan
    
    # Rename price_numeric to asking_price for clarity
    if 'price_numeric' in df.columns:
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
    df['engine_size'] = np.nan  # Initialize with NaN
    
    # First check if we have a spec column
    spec_column = 'spec'
    if 'spec' not in df.columns and 'subtitle' in df.columns:
        # Use subtitle as spec if spec doesn't exist
        spec_column = 'subtitle'
    
    if spec_column in df.columns:
        # Extract engine size using regex pattern
        import re
        engine_size_pattern = r'(\d+\.\d+)(?:\s|L|$|T)'  # Matches patterns like "1.4", "2.0L", "1.6T"
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(spec_column)):
                spec_text = str(row[spec_column])
                match = re.search(engine_size_pattern, spec_text)
                if match:
                    try:
                        # Convert to float and store
                        df.at[idx, 'engine_size'] = float(match.group(1))
                    except (ValueError, TypeError):
                        # Keep as NaN if conversion fails
                        pass
        
        # Fill missing engine sizes with median (more robust than mean)
        if not df['engine_size'].isna().all():
            median_engine = df['engine_size'].median()
            df['engine_size'] = df['engine_size'].fillna(median_engine)
        else:
            # If all are NaN, use a reasonable default
            df['engine_size'] = 1.6
            
        logger.info(f"Extracted engine sizes: min={df['engine_size'].min():.1f}, max={df['engine_size'].max():.1f}, mean={df['engine_size'].mean():.2f}")
    else:
        # No spec or subtitle column available
        df['engine_size'] = 1.6  # Default value
        
    # Process spec text and create spec_numeric for ML model
    if spec_column in df.columns:
        # Make a copy of the spec column to preserve original
        df['spec_text'] = df[spec_column].fillna('')
        
        # Clean up spec text a bit
        df['spec_text'] = df['spec_text'].str.lower()
        
        # Define trim levels and their numeric values (higher = more premium/expensive)
        # The base value is 1.0, with premium trims getting higher values
        trim_levels = {
            # VW Golf variants
            'r-line': 2.5,
            'gtd': 2.2,
            'gti': 2.5,
            'r': 3.0,
            'gte': 2.3,
            'match': 1.5,
            'life': 1.2,
            'style': 1.7,
            
            # General trim levels across brands
            'sport': 1.8,
            'se': 1.3,
            's': 1.0,
            'se l': 1.6,
            'luxury': 2.0,
            'premium': 2.0,
            'executive': 2.2,
            'amg': 3.0,
            'lounge': 1.7,
            'm sport': 2.3,
            'edition': 1.5,
            
            # Drivetrain specs
            '4wd': 1.5,
            'awd': 1.5,
            '4x4': 1.5,
            'quattro': 1.5,
            '4motion': 1.5,
            
            # Common value-adding features
            'panoramic': 1.3,
            'leather': 1.4,
            'navigation': 1.2,
            'tech pack': 1.3,
            'dsg': 1.2,
        }
        
        # Initialize spec_numeric with base value of 1.0
        df['spec_numeric'] = 1.0
        
        # Apply multipliers for each trim level found in spec text
        for idx, row in df.iterrows():
            if pd.notna(row.get('spec_text')):
                spec_text = row['spec_text'].lower()
                
                # Track matches for logging
                matches = []
                
                # Check for each trim level
                for trim, value in trim_levels.items():
                    if trim in spec_text:
                        # Multiply the base value by the trim level value
                        # This makes multiple premium features compound
                        df.at[idx, 'spec_numeric'] *= value
                        matches.append(trim)
                
                # Cap the spec_numeric to a reasonable range
                df.at[idx, 'spec_numeric'] = min(5.0, df.at[idx, 'spec_numeric'])
                
        # Log some statistics about spec_numeric
        logger.info(f"spec_numeric stats: min={df['spec_numeric'].min():.2f}, max={df['spec_numeric'].max():.2f}, mean={df['spec_numeric'].mean():.2f}")
    else:
        # Default spec_numeric if no spec column
        df['spec_numeric'] = 1.0
    
    # Extract and encode fuel type (if available)
    if spec_column in df.columns:
        # Extract fuel type from spec
        df['fuel_type'] = 'Unknown'
        fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Plugin Hybrid']
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(spec_column)):
                spec_text = str(row[spec_column]).lower()
                for fuel in fuel_types:
                    if fuel.lower() in spec_text:
                        df.at[idx, 'fuel_type'] = fuel
                        break
    else:
        # Ensure fuel_type exists even if spec doesn't
        df['fuel_type'] = 'Unknown'
    
    # Extract and encode transmission (if available)
    if spec_column in df.columns:
        # Extract transmission from spec
        df['transmission'] = 'Unknown'
        transmission_types = ['Automatic', 'Manual', 'Semi-Auto', 'CVT', 'DSG']
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(spec_column)):
                spec_text = str(row[spec_column]).lower()
                for trans in transmission_types:
                    if trans.lower() in spec_text:
                        df.at[idx, 'transmission'] = trans
                        break
    else:
        # Ensure transmission exists even if spec doesn't
        df['transmission'] = 'Unknown'
    
    # Convert fuel_type and transmission from categorical to numeric values
    # Map categorical values to numeric for ML model compatibility
    if 'fuel_type' in df.columns:
        # Create a mapping of fuel types to numeric values
        fuel_map = {
            'Unknown': 0,
            'Petrol': 1,
            'Diesel': 2,
            'Hybrid': 3,
            'Electric': 4,
            'Plugin Hybrid': 5
        }
        # Apply the mapping - any values not in the map will become NaN
        df['fuel_type_numeric'] = df['fuel_type'].map(fuel_map).fillna(0)
    else:
        df['fuel_type_numeric'] = 0
    
    if 'transmission' in df.columns:
        # Create a mapping of transmission types to numeric values
        trans_map = {
            'Unknown': 0,
            'Manual': 1,
            'Automatic': 2,
            'Semi-Auto': 3,
            'CVT': 4,
            'DSG': 5
        }
        # Apply the mapping - any values not in the map will become NaN
        df['transmission_numeric'] = df['transmission'].map(trans_map).fillna(0)
    else:
        df['transmission_numeric'] = 0
    
    # Select only these specific features for the model
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
    
    # We don't include spec_text in the feature columns as it's a string
    # We'll handle it separately in the prediction functions if needed
    
    # Ensure all feature columns exist in the dataframe
    existing_features = [col for col in feature_columns if col in df.columns]
    
    # Drop rows with missing values in core features
    core_features = ['asking_price', 'mileage', 'age', 'market_value']
    df_features = df[existing_features].dropna(subset=[col for col in core_features if col in df.columns])
    
    logger.info(f"Prepared {len(df_features)} listings with complete feature data")
    logger.info(f"Features used: {existing_features}")
    
    return df_features


def estimate_profit_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate profit margins based on price and market data
    
    For dealer cars, we can calculate profit margin based on market value vs asking price
    This will be our target variable for training
    
    Args:
        df: DataFrame with car listing features
        
    Returns:
        DataFrame with estimated_profit_margin column added
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Calculate estimated profit margin using market value vs asking price
    # Profit margin = (Market Value - Asking Price) / Asking Price
    # A positive margin means the car is priced below market value (potential profit)
    result['estimated_profit_margin'] = (result['market_value'] - result['asking_price']) / result['asking_price']
    
    # Cap extremely high values (data quality issue)
    result['estimated_profit_margin'] = result['estimated_profit_margin'].clip(-0.5, 0.5)
    
    logger.info(f"Calculated profit margins: min={result['estimated_profit_margin'].min():.2f}, "
               f"max={result['estimated_profit_margin'].max():.2f}, "
               f"mean={result['estimated_profit_margin'].mean():.2f}")
    
    return result


# Model Training and Prediction Functions
def train_xgboost_model(df: pd.DataFrame, make: str, model: str = None) -> Tuple[xgb.Booster, StandardScaler]:
    """Train XGBoost model on dealer car listings
    
    Args:
        df: DataFrame with features and estimated_profit_margin
        make: Car manufacturer name for model identification
        model: Car model name for model identification
        
    Returns:
        Tuple of (trained XGBoost model, feature scaler)
    """
    # Check if we have enough data for training
    min_training_samples = 10
    if len(df) < min_training_samples:
        logger.error(f"Not enough training data: {len(df)} samples (minimum {min_training_samples})")
        raise ValueError(f"Insufficient training data: {len(df)} samples")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Prepare features and target
    # First, ensure all columns are numeric - especially check for 'Unknown' strings
    for col in df_copy.columns:
        if col != 'estimated_profit_margin':
            if df_copy[col].dtype == object:
                logger.warning(f"Column {col} has object type - converting to numeric")
                # Try to convert to numeric, setting errors to NaN
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                # Fill NaN values with the column mean or 0 if all are NaN
                if df_copy[col].isna().all():
                    df_copy[col] = 0
                else:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    # Display the final feature dtypes after conversion
    logger.info(f"Feature dtypes after conversion: {df_copy.dtypes.to_dict()}")
    
    X = df_copy.drop('estimated_profit_margin', axis=1)
    y = df_copy['estimated_profit_margin']
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    
    # Set XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train the model
    logger.info(f"Training XGBoost model on {len(X_train)} samples")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate model
    val_preds = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    
    logger.info(f"Model trained with validation RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save the model
    model_filename = f"{make}_{model or 'all'}_model.json"
    model_path = os.path.join(MODELS_PATH, model_filename)
    model.save_model(model_path)
    
    # Save the scaler
    scaler_filename = f"{make}_{model or 'all'}_scaler.pkl"
    scaler_path = os.path.join(MODELS_PATH, scaler_filename)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Model and scaler saved to {MODELS_PATH}")
    
    return model, scaler


def load_or_train_model(
    make: str, 
    model: str, 
    dealer_data: pd.DataFrame
) -> Tuple[Optional[xgb.Booster], Optional[StandardScaler]]:
    """Load existing model or train a new one if needed
    
    Args:
        make: Car manufacturer
        model: Car model
        dealer_data: DataFrame with dealer listings for training
        
    Returns:
        Tuple of (XGBoost model, feature scaler) or (None, None) if error
    """
    model_filename = f"{make}_{model or 'all'}_model.json"
    scaler_filename = f"{make}_{model or 'all'}_scaler.pkl"
    
    model_path = os.path.join(MODELS_PATH, model_filename)
    scaler_path = os.path.join(MODELS_PATH, scaler_filename)
    
    # Check if model and scaler exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Load existing model
            loaded_model = xgb.Booster()
            loaded_model.load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                loaded_scaler = pickle.load(f)
                
            logger.info(f"Loaded existing model for {make} {model or 'all'}")
            return loaded_model, loaded_scaler
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    # Train new model if needed
    logger.info(f"No existing model found for {make} {model or 'all'}, training new model")
    
    try:
        # Prepare training data
        features_df = prepare_features(dealer_data)
        training_df = estimate_profit_margins(features_df)
        
        # Train model
        trained_model, trained_scaler = train_xgboost_model(training_df, make, model)
        return trained_model, trained_scaler
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None


# Prediction and Deal Analysis Functions
def predict_profit_margins(
    private_listings: List[Dict[str, Any]],
    xgb_model: xgb.Booster,
    scaler: StandardScaler
) -> pd.DataFrame:
    """Predict profit margins for private seller listings
    
    Args:
        private_listings: List of car listings from private sellers
        xgb_model: Trained XGBoost model
        scaler: Feature scaler
        
    Returns:
        DataFrame with listings and predicted profit margins
    """
    # Prepare features for prediction
    private_df = pd.DataFrame(private_listings)
    
    # Extract features
    features_df = prepare_features(private_df)
    
    # Get the feature columns the model was trained on
    feature_columns = [col for col in features_df.columns if col != 'market_value']  # We exclude market_value as it's our target base
    
    # Ensure all columns are numeric before scaling (same process as in training)
    for col in feature_columns:
        if features_df[col].dtype == object:
            logger.warning(f"Column {col} has object type during prediction - converting to numeric")
            # Try to convert to numeric, setting errors to NaN
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            # Fill NaN values with 0 (we can't use mean here as we want to match training)
            features_df[col] = features_df[col].fillna(0)
    
    # Display the final feature dtypes after conversion for debugging
    logger.info(f"Prediction features dtypes: {features_df[feature_columns].dtypes.to_dict()}")
    
    # Scale features
    X = scaler.transform(features_df[feature_columns])
    dtest = xgb.DMatrix(X)
    
    # Make predictions
    predictions = xgb_model.predict(dtest)
    
    # Add predictions to DataFrame
    features_df['predicted_profit_margin'] = predictions
    
    # Calculate predicted market value based on predicted profit margin
    features_df['predicted_market_value'] = features_df['asking_price'] * (1 + features_df['predicted_profit_margin'])
    
    # Join with original data to include all listing details
    result = pd.merge(
        private_df,
        features_df[['predicted_profit_margin', 'predicted_market_value']],
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Calculate potential profit in monetary terms
    result['potential_profit'] = result['predicted_market_value'] - result['price_numeric']
    
    logger.info(f"Predicted profit margins for {len(result)} private listings")
    return result


def filter_profitable_deals(
    predictions_df: pd.DataFrame,
    min_margin: float = MIN_PROFIT_MARGIN
) -> pd.DataFrame:
    """Filter listings to keep only profitable deals
    
    Args:
        predictions_df: DataFrame with listings and predicted profit margins
        min_margin: Minimum profit margin threshold (default 15%)
        
    Returns:
        DataFrame with only profitable deals
    """
    # Filter to keep only listings with margin above threshold
    profitable = predictions_df[predictions_df['predicted_profit_margin'] >= min_margin]
    
    # Sort by predicted profit margin (highest first)
    profitable = profitable.sort_values('predicted_profit_margin', ascending=False)
    
    logger.info(f"Found {len(profitable)} profitable deals with margin >= {min_margin:.1%}")
    return profitable


# Database and Notification Functions
def save_to_database(profitable_deals: pd.DataFrame, make: str, model: str = None) -> None:
    """Save profitable deals to a simple database (pickle file)
    
    Args:
        profitable_deals: DataFrame with profitable deals
        make: Car manufacturer
        model: Car model
    """
    # Check if database exists
    if os.path.exists(DATABASE_PATH):
        try:
            # Load existing database
            with open(DATABASE_PATH, 'rb') as f:
                database = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            database = {}
    else:
        database = {}
    
    # Create database key
    timestamp = datetime.now().strftime("%Y-%m-%d")
    key = f"{make}_{model or 'all'}_{timestamp}"
    
    # Convert DataFrame to dict for storage
    deals_to_save = profitable_deals.to_dict('records')
    
    # Save to database
    database[key] = {
        'timestamp': timestamp,
        'make': make,
        'model': model,
        'deals': deals_to_save,
        'count': len(deals_to_save)
    }
    
    # Save database to file
    try:
        with open(DATABASE_PATH, 'wb') as f:
            pickle.dump(database, f)
        logger.info(f"Saved {len(deals_to_save)} profitable deals to database")
    except Exception as e:
        logger.error(f"Error saving database: {e}")


def alert_buyer(profitable_deals: pd.DataFrame, make: str, model: str = None) -> None:
    """Generate alerts for profitable deals
    
    In a real system, this might send emails, SMS, or push notifications
    For this test script, we'll just log the alerts
    
    Args:
        profitable_deals: DataFrame with profitable deals
        make: Car manufacturer
        model: Car model
    """
    if len(profitable_deals) == 0:
        logger.info(f"No profitable deals found for {make} {model or 'all'}")
        return
    
    # Generate alert message
    alert_message = f"\n{'='*60}\n"
    alert_message += f"PROFITABLE DEALS ALERT: {make.upper()} {model.upper() if model else 'ALL MODELS'}\n"
    alert_message += f"{'='*60}\n"
    alert_message += f"Found {len(profitable_deals)} potentially profitable deals:\n\n"
    
    # Add all profitable deals (no limit)
    for i, (_, deal) in enumerate(profitable_deals.iterrows()):
        alert_message += f"DEAL #{i+1}: {deal.get('year', '')} {deal.get('make', '')} {deal.get('model', '')}\n"
        alert_message += f"  Asking Price: £{deal.get('price_numeric', 0):.2f}\n"
        alert_message += f"  Predicted Market Value: £{deal.get('predicted_market_value', 0):.2f}\n"
        alert_message += f"  Predicted margin: {deal.get('predicted_profit_margin', 0):.1%}\n"
        alert_message += f"  Est. profit: £{deal.get('potential_profit', 0):.2f}\n"
        
        # Add fuel type and transmission if available
        if 'fuel_type' in deal and deal.get('fuel_type') != 'Unknown':
            alert_message += f"  Fuel: {deal.get('fuel_type')}\n"
            
        if 'transmission' in deal and deal.get('transmission') != 'Unknown':
            alert_message += f"  Transmission: {deal.get('transmission')}\n"
            
        # Add spec summary if available
        if 'spec' in deal and pd.notna(deal.get('spec')):
            spec = deal.get('spec')
            if len(spec) > 100:
                spec = spec[:97] + '...'
            alert_message += f"  Spec: {spec}\n"
            
        alert_message += f"  URL: {deal.get('url', '')}\n\n"
    
    # Log the alert
    logger.info(alert_message)
    
    # In a real system, we would send the alert via email/SMS/etc.
    # For this test script, we'll save it to a text file
    alert_file = f"alert_{make}_{model or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(alert_file, 'w') as f:
            f.write(alert_message)
        logger.info(f"Saved alert to {alert_file}")
    except Exception as e:
        logger.error(f"Error saving alert: {e}")


# Main Workflow Function
def process_car_model(make: str, model: str = None, max_pages: int = None) -> None:
    """Process a specific car make/model to find profitable deals
    
    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (optional, e.g., '3-series')
        max_pages: Maximum number of pages to scrape (None for all available pages)
    """
    logger.info(f"\n{'='*60}\nProcessing {make.upper()} {model.upper() if model else 'ALL MODELS'}\n{'='*60}")
    
    # Step 1: Scrape all listings
    all_listings = scrape_listings(make, model, max_pages)
    if not all_listings:
        logger.error(f"No listings found for {make} {model or 'all'}")
        return
    
    # Step 2: Separate dealer and private listings
    dealer_listings = filter_listings_by_seller_type(all_listings, "Dealer")
    private_listings = filter_listings_by_seller_type(all_listings, "Private")
    
    if len(dealer_listings) < 5:
        logger.warning(f"Insufficient dealer listings ({len(dealer_listings)}) for training. Need at least 5.")
        return
    
    if len(private_listings) == 0:
        logger.warning(f"No private listings found for {make} {model or 'all'}")
        return
    
    # Step 3: Enrich dealer listings with price markers
    enriched_dealer_listings = enrich_with_price_markers(dealer_listings)
    
    # Step 4: Load or train model
    xgb_model, scaler = load_or_train_model(make, model, enriched_dealer_listings)
    if xgb_model is None or scaler is None:
        logger.error(f"Failed to create model for {make} {model or 'all'}")
        return
    
    # Step 5: Predict on private listings
    predictions_df = predict_profit_margins(private_listings, xgb_model, scaler)
    
    # Step 6: Filter for profitable deals
    profitable_deals = filter_profitable_deals(predictions_df)
    
    # Step 7: Save and alert
    if len(profitable_deals) > 0:
        save_to_database(profitable_deals, make, model)
        alert_buyer(profitable_deals, make, model)
    else:
        logger.info(f"No profitable deals found for {make} {model or 'all'}")


def main():
    """Main entry point for the car deal ML analyzer"""
    # Define car makes and models to analyze
    # This can be expanded with more makes/models
    targets = [
        {"make": "bmw", "model": "3-series"},
        {"make": "ford", "model": "fiesta"},
        {"make": "toyota", "model": "prius"},
        # Add more makes/models as needed
    ]
    
    # Process command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Run with a single test make/model (using limited pages for test mode only)
            logger.info("Running in test mode with BMW 3-series")
            process_car_model("bmw", "3-series", max_pages=2)
            sys.exit(0)
        elif len(sys.argv) >= 3:
            # Process specific make/model from command line
            make = sys.argv[1]
            model = sys.argv[2] if len(sys.argv) > 2 else None
            logger.info(f"Processing {make} {model or 'all'} from command line")
            process_car_model(make, model)
            sys.exit(0)
    
    # Process all defined targets
    for target in targets:
        try:
            process_car_model(target["make"], target.get("model"))
        except Exception as e:
            logger.error(f"Error processing {target['make']} {target.get('model', 'all')}: {e}")


if __name__ == "__main__":
    main()