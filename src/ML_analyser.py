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
# Removed GradientBoostingRegressor - using XGBoost only
import pickle
import json
import logging
import warnings
import urllib3
import requests
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from playwright.sync_api import sync_playwright

# Set up proper import paths (only need to do this once)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import proxy rotation service
from services.proxy_rotation import ProxyManager

# Import the retail price scraper for price marker data
from services.retail_price_scraper import scrape_price_marker, batch_scrape_price_markers

# Import main scraping components from your core codebase
from services.network_requests import AutoTraderAPIClient
from services.data_adapter import NetworkDataAdapter

# Project paths - everything goes to archive folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.path.join(PROJECT_ROOT, 'archive')

# Set up logging
os.makedirs(ARCHIVE_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(message)s',  # Clean format without timestamps
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_analyser")

# Constants
MIN_PROFIT_MARGIN = 0.15  # 15% minimum profit margin
DATABASE_PATH = os.path.join(ARCHIVE_PATH, "car_deals_database.pkl")
MODELS_PATH = os.path.join(ARCHIVE_PATH, "ml_models")
CACHE_DIR = os.path.join(ARCHIVE_PATH, "cache")
DATA_OUTPUTS_PATH = os.path.join(ARCHIVE_PATH, "outputs")

# Create directories if they don't exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUTS_PATH, exist_ok=True)

# Global execution mode tracking
EXECUTION_MODE = "production"  # "test" or "production"

def set_execution_mode(mode: str):
    """Set the global execution mode for data saving
    
    Args:
        mode: Either "test" or "production"
    """
    global EXECUTION_MODE
    if mode not in ["test", "production"]:
        logger.warning(f"Invalid execution mode '{mode}', defaulting to 'production'")
        mode = "production"
    
    EXECUTION_MODE = mode
    logger.info(f"Execution mode set to: {mode}")


def get_execution_mode() -> str:
    """Get the current execution mode"""
    global EXECUTION_MODE
    return EXECUTION_MODE


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


def get_cache_filename(make: str, model: str = None, data_type: str = 'all_listings') -> str:
    """Generate a consistent cache filename based on make/model and data type
    
    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (e.g., '3-series')
        data_type: Type of data (all_listings or retail_prices)
        
    Returns:
        Cache filename
    """
    model_part = model or 'all'
    return os.path.join(CACHE_DIR, f"{make}_{model_part}_{data_type}.json")


def save_to_cache(data: List[Dict[str, Any]], make: str, model: str = None, data_type: str = 'all_listings') -> bool:
    """Save data to cache file
    
    Args:
        data: Data to save
        make: Car manufacturer
        model: Car model
        data_type: Type of data (all_listings or retail_prices)
        
    Returns:
        True if successful, False otherwise
    """
    cache_file = get_cache_filename(make, model, data_type)
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} items to cache: {cache_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving to cache {cache_file}: {e}")
        return False


def load_from_cache(make: str, model: str = None, data_type: str = 'all_listings') -> List[Dict[str, Any]]:
    """Load data from cache file
    
    Args:
        make: Car manufacturer
        model: Car model
        data_type: Type of data (all_listings or retail_prices)
        
    Returns:
        Cached data or empty list if cache not found or invalid
    """
    cache_file = get_cache_filename(make, model, data_type)
    if not os.path.exists(cache_file):
        print("📂 Cache: Not found")
        print()
        return []
        
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from cache: {cache_file}")
        return data
    except Exception as e:
        logger.error(f"Error loading from cache {cache_file}: {e}")
        return []


def scrape_listings(make: str, model: str = None, max_pages: int = None, use_cache: bool = True, verify_ssl: bool = True, use_proxy: bool = False) -> List[Dict[str, Any]]:
    """Scrape car listings for a given make and model
    
    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (e.g., '3-series')
        max_pages: Maximum number of pages to scrape (None for all available pages)
        use_cache: Whether to check for cached data before scraping
        verify_ssl: Whether to verify SSL certificates (set to False to ignore SSL errors)
        use_proxy: Whether to use proxy rotation for scraping
    
    Returns:
        List of dictionaries containing car listing details
    """
    # Try to load from cache first if use_cache is True
    if use_cache:
        cached_listings = load_from_cache(make, model, 'all_listings')
        if cached_listings:
            print(f"📂 Cache: Found {len(cached_listings)} listings")
            print()
            return cached_listings
    
    # Initialize proxy manager if proxy rotation is enabled
    proxy_manager = None
    if use_proxy:
        try:
            # Use absolute path for proxy configuration
            config_path = os.path.join(PROJECT_ROOT, 'config/proxies.json')
            archive_path = ARCHIVE_PATH
            
            proxy_manager = ProxyManager(config_path=config_path, archive_path=archive_path)
            # Proxy rotation enabled silently
        except Exception as e:
            logger.warning(f"Failed to initialize proxy rotation: {e}")
            proxy_manager = None
    
    # Get a proxy URL if proxy rotation is enabled
    proxy = None
    if proxy_manager:
        proxy = proxy_manager.get_proxy()
    
    # Initialize the AutoTrader API client from your main code
    api_client = AutoTraderAPIClient(verify_ssl=verify_ssl, proxy=proxy, proxy_manager=proxy_manager)
    
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
                'asking_price': car.get('price', 0),  # Changed from price_numeric to asking_price
                'mileage': car.get('mileage', 0),
                
                # Vehicle specifications - preserve individual fields
                'engine_size': car.get('engine_size'),
                'fuel_type': car.get('fuel_type'),
                'transmission': car.get('transmission'),
                'body_type': car.get('body_type'),
                'doors': car.get('doors'),
                
                # Additional data
                'seller_type': 'Dealer' if car.get('seller_type', '').upper() == 'TRADE' else 'Private',
                'url': car.get('url', ''),
                'image_url': car.get('image_url', ''),
                'image_url_2': car.get('image_url_2', ''),
                'location': car.get('location', ''),
                
                # Keep original spec unchanged from API (trim/variant info like "R-Line", "AMG Line")
                'spec': car.get('spec', ''),
                
                # Market value will be added later for dealer listings (from scraping) or private listings (from ML prediction)
                'market_value': None
            }
            
            listings.append(listing)
        
        print(f"📊 Results: {len(listings)} listings found")
        print()
        
        # Save to cache if we have data
        if listings:
            save_to_cache(listings, make, model, 'all_listings')
            
        return listings
        
    except Exception as e:
        logger.error(f"Error scraping listings: {e}")
        return []


def enrich_with_price_markers(listings: List[Dict[str, Any]], make: str = None, model: str = None, use_cache: bool = True, use_proxy: bool = False) -> List[Dict[str, Any]]:
    """Enrich dealer listings with price marker data using parallel processing
    
    Args:
        listings: List of car listings
        make: Car manufacturer (for cache filename)
        model: Car model (for cache filename)
        use_cache: Whether to check for cached retail prices
        
    Returns:
        Enriched listings with price_vs_market field added
    """
    # Try to load from cache first if use_cache is True and make/model are provided
    if use_cache and make is not None:
        cached_retail_prices = load_from_cache(make, model, 'retail_prices')
        if cached_retail_prices:
            logger.info(f"Using {len(cached_retail_prices)} cached retail prices for {make} {model or ''}")
            
            # Create a lookup dictionary from cached data
            retail_price_lookup = {}
            for item in cached_retail_prices:
                if 'url' in item and 'price_vs_market' in item:
                    retail_price_lookup[item['url']] = item['price_vs_market']
            
            # Apply cached retail prices to listings with matching URLs
            enriched_listings = []
            cache_hit_count = 0
            
            for listing in listings:
                url = listing.get('url', '')
                if url and url in retail_price_lookup:
                    # Use cached price_vs_market value
                    enriched_listing = {**listing, 'price_vs_market': retail_price_lookup[url]}
                    enriched_listings.append(enriched_listing)
                    cache_hit_count += 1
                else:
                    # No cached data for this URL
                    listing['price_vs_market'] = 0.0  # Default to market value
                    enriched_listings.append(listing)
            
            logger.info(f"Applied {cache_hit_count}/{len(listings)} cached retail prices")
            
            # If we have a good cache hit rate (>80%), use the cached data
            if len(listings) > 0 and cache_hit_count / len(listings) >= 0.8:
                return enriched_listings
            else:
                logger.info(f"Cache hit rate too low ({cache_hit_count}/{len(listings)}), fetching fresh data")
    
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
    
    # Initialize proxy manager if proxy rotation is enabled
    proxy_manager = None
    if use_proxy:
        try:
            # Use absolute path for proxy configuration
            config_path = os.path.join(PROJECT_ROOT, 'config/proxies.json')
            archive_path = ARCHIVE_PATH
            
            proxy_manager = ProxyManager(config_path=config_path, archive_path=archive_path)
            # Proxy rotation enabled silently
        except Exception as e:
            logger.warning(f"Failed to initialize proxy rotation: {e}")
    
    try:
        # Use sequential processing with 7 workers (no batching)
        # This eliminates resource conflicts and browser crashes
        global EXECUTION_MODE
        results = batch_scrape_price_markers(
            valid_urls,
            headless=True,
            progress_callback=progress_callback,
            test_mode=(EXECUTION_MODE == "test")
        )
        
        print(f"\n{'='*60}")
        print(f"PROCESSING RESULTS")
        print(f"{'='*60}")
        
        # Merge price marker data with original listings
        enriched_listings = []
        success_count = 0
        error_count = 0
        
        # Also create a list to store retail price data for caching
        retail_prices_cache = []
        
        for listing in valid_listings:
            url = listing['url']
            
            if url in results:
                price_marker = results[url]
                
                # Check if this is an error result
                if 'Error' in price_marker.get('marker_text', ''):
                    error_count += 1
                    # Only log errors in test mode to avoid spam
                    if EXECUTION_MODE == "test":
                        logger.warning(f"Error for {listing.get('make', '')} {listing.get('model', '')}: {price_marker.get('marker_text', 'Unknown error')}")
                    listing['price_vs_market'] = 0.0  # Default to market value
                    enriched_listings.append(listing)
                else:
                    # Successful result
                    success_count += 1
                    market_difference = price_marker['market_difference']
                    
                    # Add price marker data to the listing
                    enriched_listing = {**listing, 'price_vs_market': market_difference}
                    enriched_listings.append(enriched_listing)
                    
                    # Add to retail prices cache
                    retail_prices_cache.append({
                        'url': url,
                        'price_vs_market': market_difference,
                        'make': listing.get('make', ''),
                        'model': listing.get('model', '')
                    })
                    
                    # Log but don't spam console
                    if success_count % 10 == 0:  # Log every 10th success
                        logger.info(f"Enriched {success_count} listings with price markers")
            else:
                # URL was processed but no result
                error_count += 1
                # Only log errors in test mode to avoid spam
                if EXECUTION_MODE == "test":
                    logger.error(f"No price marker result for URL: {url}")
                listing['price_vs_market'] = 0.0  # Default to market value
                enriched_listings.append(listing)
        
        # Save retail prices to cache if we have data and make/model are provided
        if retail_prices_cache and make is not None:
            save_to_cache(retail_prices_cache, make, model, 'retail_prices')
        
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
        retail_prices_cache = []
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
                
                # Add to retail prices cache
                retail_prices_cache.append({
                    'url': listing['url'],
                    'price_vs_market': price_marker['market_difference'],
                    'make': listing.get('make', ''),
                    'model': listing.get('model', '')
                })
                
            except Exception as e:
                logger.error(f"Error enriching listing with price marker: {e}")
                # Add without price marker
                listing['price_vs_market'] = 0.0  # Default to market value
                enriched_listings.append(listing)
        
        print("\nSequential processing complete")
        
        # Save retail prices to cache if we have data and make/model are provided
        if retail_prices_cache and make is not None:
            save_to_cache(retail_prices_cache, make, model, 'retail_prices')
    
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
    required_columns = ['asking_price', 'mileage', 'year']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' missing from listings data")
            df[col] = np.nan
    
    # Handle legacy price_numeric column if it exists
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


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare training data with market_value as the target
    
    For dealer cars, we use the scraped market_value as our training target
    This is much more direct than predicting profit margins
    
    Args:
        df: DataFrame with car listing features including market_value
        
    Returns:
        DataFrame ready for training with market_value as target
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Ensure market_value is the target variable (already calculated from price_vs_market)
    # No need for additional calculations - market_value is our direct target
    
    logger.info(f"Training data prepared: min market_value=£{result['market_value'].min():,.0f}, "
               f"max market_value=£{result['market_value'].max():,.0f}, "
               f"mean market_value=£{result['market_value'].mean():,.0f}")
    
    return result


# Model Training and Prediction Functions
def train_xgboost_model(df: pd.DataFrame, make: str, model: str = None) -> Tuple[xgb.Booster, StandardScaler]:
    """Train XGBoost model on dealer car listings to predict market values
    
    Args:
        df: DataFrame with features and market_value
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
        if col != 'market_value':
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
    
    X = df_copy.drop('market_value', axis=1)
    y = df_copy['market_value']
    
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
        training_df = prepare_training_data(features_df)
        
        # Train model
        trained_model, trained_scaler = train_xgboost_model(training_df, make, model)
        return trained_model, trained_scaler
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None


# Prediction and Deal Analysis Functions
def predict_market_values(
    private_listings: List[Dict[str, Any]],
    xgb_model: xgb.Booster,
    scaler: StandardScaler
) -> pd.DataFrame:
    """Predict market values for private seller listings
    
    Args:
        private_listings: List of car listings from private sellers
        xgb_model: Trained XGBoost model
        scaler: Feature scaler
        
    Returns:
        DataFrame with listings and predicted market values
    """
    # Prepare features for prediction
    private_df = pd.DataFrame(private_listings)
    
    # Extract features - but don't include market_value since we're predicting it
    features_df = prepare_features(private_df)
    
    # Get feature columns for prediction (exclude target variable)
    feature_columns = ['asking_price', 'mileage', 'age', 'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric']
    
    # Check if all required columns exist in the DataFrame, add any missing ones
    for col in feature_columns:
        if col not in features_df.columns:
            logger.warning(f"Column {col} missing in prediction data, adding with default values")
            features_df[col] = 0
    
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
    
    # Scale features using the same scaler from training
    X_scaled = scaler.transform(features_df[feature_columns])
    
    # Make predictions using XGBoost
    dtest = xgb.DMatrix(X_scaled)
    
    # Predict market values directly
    predicted_market_values = xgb_model.predict(dtest)
    
    # Add predictions to DataFrame
    features_df['predicted_market_value'] = predicted_market_values
    
    # Calculate profit margin from the predicted market value
    features_df['predicted_profit_margin'] = (
        (features_df['predicted_market_value'] - features_df['asking_price']) / 
        features_df['asking_price']
    )
    
    # Join with original data to include all listing details
    result = pd.merge(
        private_df,
        features_df[['predicted_market_value', 'predicted_profit_margin']],
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Calculate potential profit in monetary terms
    result['potential_profit'] = result['predicted_market_value'] - result['asking_price']
    
    logger.info(f"Predicted market values for {len(result)} private listings")
    logger.info(f"Average predicted market value: £{result['predicted_market_value'].mean():,.0f}")
    logger.info(f"Average profit margin: {result['predicted_profit_margin'].mean():.2%}")
    
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
def save_comprehensive_data(
    make: str,
    model: str = None,
    all_listings: List[Dict[str, Any]] = None,
    retail_prices: List[Dict[str, Any]] = None,
    profitable_deals: pd.DataFrame = None
) -> None:
    """
    Save comprehensive data based on execution mode:
    - Test mode: saves all data (scraped cars, retail prices, final deals)
    - Production mode: saves only final deals
    
    Args:
        make: Car manufacturer
        model: Car model
        all_listings: All scraped listings
        retail_prices: All retail price data
        profitable_deals: Final profitable deals
    """
    global EXECUTION_MODE
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model or "all"
    
    if EXECUTION_MODE == "test":
        logger.info(f"💾 Saving comprehensive test data for {make} {model_name}")
        
        # Save all scraped listings
        if all_listings:
            all_listings_file = os.path.join(DATA_OUTPUTS_PATH, f"{timestamp}_{make}_{model_name}_all_listings.json")
            try:
                with open(all_listings_file, 'w') as f:
                    json.dump(all_listings, f, indent=2, default=str)
                logger.info(f"✅ Saved {len(all_listings)} scraped listings to {all_listings_file}")
            except Exception as e:
                logger.error(f"❌ Error saving all listings: {e}")
        
        # Save retail prices
        if retail_prices:
            retail_prices_file = os.path.join(DATA_OUTPUTS_PATH, f"{timestamp}_{make}_{model_name}_retail_prices.json")
            try:
                with open(retail_prices_file, 'w') as f:
                    json.dump(retail_prices, f, indent=2, default=str)
                logger.info(f"✅ Saved {len(retail_prices)} retail prices to {retail_prices_file}")
            except Exception as e:
                logger.error(f"❌ Error saving retail prices: {e}")
        
        # Save final deals
        if profitable_deals is not None and len(profitable_deals) > 0:
            deals_file = os.path.join(DATA_OUTPUTS_PATH, f"{timestamp}_{make}_{model_name}_profitable_deals.json")
            try:
                deals_data = profitable_deals.to_dict('records')
                with open(deals_file, 'w') as f:
                    json.dump(deals_data, f, indent=2, default=str)
                logger.info(f"✅ Saved {len(deals_data)} profitable deals to {deals_file}")
            except Exception as e:
                logger.error(f"❌ Error saving profitable deals: {e}")
                
    elif EXECUTION_MODE == "production":
        logger.info(f"💾 Saving production data (deals only) for {make} {model_name}")
        
        # Only save final deals in production mode
        if profitable_deals is not None and len(profitable_deals) > 0:
            save_to_database(profitable_deals, make, model)
    
    logger.info(f"💾 Data saving complete for {make} {model_name} in {EXECUTION_MODE} mode")


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
        alert_message += f"  Asking Price: £{deal.get('asking_price', 0):.2f}\n"
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
    alert_file = os.path.join(ARCHIVE_PATH, f"alert_{make}_{model or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    try:
        with open(alert_file, 'w') as f:
            f.write(alert_message)
        logger.info(f"Saved alert to {alert_file}")
    except Exception as e:
        logger.error(f"Error saving alert: {e}")


# Main Workflow Function
def process_car_model(make: str, model: str = None, max_pages: int = None, verify_ssl: bool = False) -> None:
    """Process a specific car make/model to find profitable deals
    
    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (optional, e.g., '3-series')
        max_pages: Maximum number of pages to scrape (None for all available pages)
        verify_ssl: Whether to verify SSL certificates
    """
    global EXECUTION_MODE
    
    # Clean header already printed in main(), so skip it here
    
    # Step 1: Scrape all listings (always use cache and proxy)
    all_listings = scrape_listings(make, model, max_pages, use_cache=True, verify_ssl=verify_ssl, use_proxy=True)
    if not all_listings:
        print(f"📊 Results: 0 listings found")
        print()
        print(f"❌ Error: No listings found for {make} {model or 'All Models'}")
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
    
    # Step 3: Enrich dealer listings with price markers (always use cache and proxy)
    enriched_dealer_listings = enrich_with_price_markers(dealer_listings, make, model, use_cache=True, use_proxy=True)
    
    # Collect retail price data for saving (if in test mode)
    retail_price_data = []
    if EXECUTION_MODE == "test":
        for listing in enriched_dealer_listings:
            if 'price_vs_market' in listing and listing['price_vs_market'] != 0:
                retail_price_data.append({
                    'url': listing.get('url', ''),
                    'make': listing.get('make', ''),
                    'model': listing.get('model', ''),
                    'price_vs_market': listing['price_vs_market'],
                    'asking_price': listing.get('asking_price', 0),
                    'timestamp': datetime.now().isoformat()
                })
    
    # Step 4: Load or train model
    xgb_model, scaler = load_or_train_model(make, model, enriched_dealer_listings)
    if xgb_model is None or scaler is None:
        logger.error(f"Failed to create model for {make} {model or 'all'}")
        return
    
    # Step 5: Predict on private listings
    predictions_df = predict_market_values(private_listings, xgb_model, scaler)
    
    # Step 6: Filter for profitable deals
    profitable_deals = filter_profitable_deals(predictions_df)
    
    # Step 7: Merge enriched data back into all_listings for comprehensive saving
    if EXECUTION_MODE == "test":
        # Create a mapping of enriched listings by URL for easy lookup
        enriched_lookup = {listing.get('url', ''): listing for listing in enriched_dealer_listings}
        
        # Create a mapping of ML predictions by URL for private listings
        predictions_lookup = {}
        if len(predictions_df) > 0:
            for _, prediction in predictions_df.iterrows():
                url = prediction.get('url', '')
                if url:
                    # Calculate price_vs_market from ML prediction
                    asking_price = prediction.get('asking_price', 0)
                    predicted_market_value = prediction.get('predicted_market_value', 0)
                    if asking_price > 0:
                        price_vs_market = predicted_market_value - asking_price
                        predictions_lookup[url] = price_vs_market
        
        # Update all_listings with enriched data
        for listing in all_listings:
            listing_url = listing.get('url', '')
            
            # For dealer listings: use scraped market value data
            if listing_url in enriched_lookup:
                enriched_listing = enriched_lookup[listing_url]
                if 'price_vs_market' in enriched_listing:
                    # Calculate market_value from price_vs_market and asking_price
                    listing['market_value'] = listing['asking_price'] + enriched_listing['price_vs_market']
            
            # For private listings: use ML predicted market value
            elif listing_url in predictions_lookup:
                # predictions_lookup contains market values for private listings
                market_value = None
                for _, pred in predictions_df.iterrows():
                    if pred.get('url') == listing_url:
                        listing['market_value'] = pred.get('predicted_market_value', 0)
                        break
                
            # If no market value available, use asking price as fallback
            if listing.get('market_value') is None:
                listing['market_value'] = listing['asking_price']
    
    # Step 8: Save comprehensive data based on execution mode
    save_comprehensive_data(
        make=make,
        model=model,
        all_listings=all_listings if EXECUTION_MODE == "test" else None,
        retail_prices=retail_price_data if retail_price_data else None,
        profitable_deals=profitable_deals if len(profitable_deals) > 0 else None
    )
    
    # Step 9: Alert for profitable deals
    if len(profitable_deals) > 0:
        alert_buyer(profitable_deals, make, model)
    else:
        logger.info(f"No profitable deals found for {make} {model or 'all'}")


def test_cache_functionality(make: str, model: str) -> None:
    """Test the data caching functionality without running the full model pipeline
    
    Args:
        make: Car manufacturer
        model: Car model
    """
    logger.info(f"\n{'='*60}\nTESTING CACHE FUNCTIONALITY FOR {make.upper()} {model.upper()}\n{'='*60}")
    
    # Check if cache files exist
    all_listings_cache = get_cache_filename(make, model, 'all_listings')
    retail_prices_cache = get_cache_filename(make, model, 'retail_prices')
    
    cache_exists = os.path.exists(all_listings_cache) and os.path.exists(retail_prices_cache)
    logger.info(f"Cache files exist: {cache_exists}")
    
    if cache_exists:
        # Test loading listings from cache
        all_listings = load_from_cache(make, model, 'all_listings')
        logger.info(f"Loaded {len(all_listings)} listings from cache")
        
        # Test loading retail prices from cache
        retail_prices = load_from_cache(make, model, 'retail_prices')
        logger.info(f"Loaded {len(retail_prices)} retail prices from cache")
        
        # Separate dealer and private listings to verify cache content
        dealer_count = sum(1 for listing in all_listings if listing.get('seller_type') == 'Dealer')
        private_count = sum(1 for listing in all_listings if listing.get('seller_type') == 'Private')
        logger.info(f"Found {dealer_count} dealer listings and {private_count} private listings in cache")
        
        # Test the enrichment with retail prices
        dealer_listings = filter_listings_by_seller_type(all_listings, "Dealer")
        enriched_listings = enrich_with_price_markers(dealer_listings, make, model)
        
        price_marker_count = sum(1 for listing in enriched_listings if listing.get('price_vs_market', 0) != 0)
        logger.info(f"Successfully enriched {price_marker_count}/{len(enriched_listings)} listings with price markers")
        
        logger.info(f"\n{'='*60}\nCACHE TEST RESULTS\n{'='*60}")
        logger.info(f"All listings cache: {'SUCCESS' if len(all_listings) > 0 else 'FAILED'}")
        logger.info(f"Retail prices cache: {'SUCCESS' if len(retail_prices) > 0 else 'FAILED'}")
        logger.info(f"Price marker application: {'SUCCESS' if price_marker_count > 0 else 'FAILED'}")
        
    else:
        logger.error(f"Cache files not found for {make} {model}")


def main():
    """Main entry point for the car deal ML analyzer"""
    # Define car makes and models to analyze
    # This can be expanded with more makes/models
    targets = [
        {"make": "bmw", "model": "3 series"},  # IMPORTANT: Use "3 series" format (with space), not "3-series"
        {"make": "ford", "model": "fiesta"},
        {"make": "toyota", "model": "prius"},
        # Add more makes/models as needed
        # AutoTrader formats to use:
        # BMW: "1 series", "3 series", "5 series"
        # Mercedes: "a class", "c class", "e class"
        # Audi: "a1", "a3", "a4", "q3", "q5"
    ]
    
    # Default settings - disable SSL verification for scraping reliability
    verify_ssl = False
    
    # Disable SSL verification globally by default
    # Use global urllib3 and requests modules
    global urllib3, requests
    
    # Disable SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Disable SSL verification for all requests globally
    try:
        # For older versions of urllib3
        if hasattr(urllib3.util.ssl_, 'DEFAULT_CIPHERS'):
            urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
    except (ImportError, AttributeError):
        # For newer versions, we can skip this as it's not necessary
        pass
        
    # Disable warnings
    requests.packages.urllib3.disable_warnings()
    
    # Patch the default HTTPS adapter to use unverified context
    old_merge_environment_settings = requests.Session.merge_environment_settings
    
    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings
        
    requests.Session.merge_environment_settings = merge_environment_settings
    
    # Also create a patched Session for all future requests
    orig_session = requests.Session
    
    class PatchedSession(orig_session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.verify = False
    
    requests.Session = PatchedSession
    
    # Set the default SSL verification for new requests to False
    requests.defaults = requests.defaults if hasattr(requests, 'defaults') else {}
    requests.defaults['verify'] = False
    
    # Process command line arguments
    if len(sys.argv) > 1:
        if "--test" in sys.argv:
            # Enable test mode for comprehensive data saving
            set_execution_mode("test")
            sys.argv.remove("--test")
        
        # Check for --model flag
        if "--model" in sys.argv:
            model_index = sys.argv.index("--model")
            if model_index + 1 < len(sys.argv):
                model = sys.argv[model_index + 1]
                make = "BMW"  # Default make when using --model
                
                print("\n" + "=" * 60)
                print("                    🚗 CAR DEALER BOT")
                print("=" * 60)
                print()
                print(f"🔍 Processing: {make} {model} ({get_execution_mode()} mode)")
                print()
                
                process_car_model(make, model, verify_ssl=verify_ssl)
                sys.exit(0)
        
        # Check remaining args after flag processing
        if len(sys.argv) >= 2:
            # Process specific make/model from command line
            make = sys.argv[1]
            model = sys.argv[2] if len(sys.argv) > 2 else None
            
            print("\n" + "=" * 60)
            print("                    🚗 CAR DEALER BOT")
            print("=" * 60)
            print()
            print(f"🔍 Processing: {make} {model or 'All Models'} ({get_execution_mode()} mode)")
            print()
            
            process_car_model(make, model, verify_ssl=verify_ssl)
            sys.exit(0)
    
    # Process all defined targets
    for target in targets:
        try:
            process_car_model(target["make"], target.get("model"), verify_ssl=verify_ssl)
        except Exception as e:
            logger.error(f"Error processing {target['make']} {target.get('model', 'all')}: {e}")


# ────────────────────────────────────────────────────────────
# Compatibility functions for scraper.py integration
# ────────────────────────────────────────────────────────────

def enhanced_analyse_listings(listings: List[Dict[str, Any]], cfg: Dict = None, verbose: bool = False) -> None:
    """
    Enhanced analysis using ML models for vehicle profit prediction.
    Compatible with the existing scraper.py interface.
    
    Args:
        listings: List of car listings to analyze
        cfg: Configuration dictionary (optional)
        verbose: Whether to output detailed analysis
    """
    if cfg is None:
        cfg = {
            'recon_cost': 300,
            'excellent_margin_pct': 0.25,
            'good_margin_pct': 0.20,
            'negotiation_margin_pct': 0.15,
            'min_cash_margin': 800,
        }
    
    if not listings:
        return
    
    if verbose:
        logger.info(f"🔧 Starting enhanced analysis of {len(listings)} listings...")
    
    # Step 1: Enrich with retail price markers first
    if verbose:
        logger.info(f"📊 Enriching with retail price markers...")
    
    # Extract make/model from first listing for context
    make = listings[0].get('make', '').lower() if listings else None
    model = listings[0].get('model', '').lower() if listings else None
    
    # Call retail price enrichment
    enriched_listings = enrich_with_price_markers(listings, make=make, model=model)
    
    # Update the original listings with enriched data
    listings.clear()
    listings.extend(enriched_listings)
    
    if verbose:
        logger.info(f"✅ Price marker enrichment complete")
    
    # Step 2: Continue with ML analysis
    if verbose:
        logger.info(f"🤖 Starting ML analysis...")
    
    # Group listings by make/model for efficient processing
    grouped_listings = {}
    for listing in listings:
        make = listing.get('make', '').lower()
        model = listing.get('model', '').lower()
        key = f"{make}_{model}"
        
        if key not in grouped_listings:
            grouped_listings[key] = []
        grouped_listings[key].append(listing)
    
    analyzed_listings = []
    
    # Process each make/model group
    for group_key, group_listings in grouped_listings.items():
        if len(group_listings) < 5:
            continue  # Skip groups with insufficient data
        
        make, model = group_key.split('_', 1)
        
        try:
            # Separate dealer and private listings
            dealer_listings = [l for l in group_listings if l.get('seller_type', '').upper() in ['TRADE', 'DEALER']]
            private_listings = [l for l in group_listings if l.get('seller_type', '').upper() == 'PRIVATE']
            
            if len(dealer_listings) < 3 or len(private_listings) < 1:
                continue
            
            # Convert to the format expected by ML functions
            dealer_data = []
            for listing in dealer_listings:
                dealer_data.append({
                    'make': listing.get('make', ''),
                    'model': listing.get('model', ''),
                    'year': listing.get('year', 2015),
                    'asking_price': listing.get('asking_price', listing.get('price', 0)),
                    'mileage': listing.get('mileage', 50000),
                    'seller_type': 'Dealer',
                    'url': listing.get('url', ''),
                    # Preserve original spec unchanged from API (trim/variant info)
                    'spec': listing.get('spec', '')
                })
            
            # Load or train model for this make/model
            xgb_model, scaler = load_or_train_model(make, model, dealer_data)
            
            if xgb_model is None or scaler is None:
                continue
            
            # Predict on private listings
            private_data = []
            for listing in private_listings:
                private_data.append({
                    'make': listing.get('make', ''),
                    'model': listing.get('model', ''),
                    'year': listing.get('year', 2015),
                    'asking_price': listing.get('asking_price', listing.get('price', 0)),
                    'mileage': listing.get('mileage', 50000),
                    'seller_type': 'Private',
                    'url': listing.get('url', ''),
                    # Preserve original spec unchanged from API (trim/variant info)
                    'spec': listing.get('spec', '')
                })
            
            predictions_df = predict_market_values(private_data, xgb_model, scaler)
            
            # Convert predictions back to listings format and add analysis results
            for i, (_, prediction) in enumerate(predictions_df.iterrows()):
                if i < len(private_listings):
                    original_listing = private_listings[i]
                    
                    # Add ML prediction results
                    predicted_margin = prediction.get('predicted_profit_margin', 0)
                    predicted_market_value = prediction.get('predicted_market_value', 0)
                    potential_profit = prediction.get('potential_profit', 0)
                    
                    original_listing['enhanced_retail_estimate'] = predicted_market_value
                    original_listing['enhanced_gross_margin_pct'] = predicted_margin
                    original_listing['enhanced_gross_cash_profit'] = potential_profit
                    
                    # Determine rating based on ML prediction
                    if predicted_margin >= cfg['excellent_margin_pct']:
                        rating = "Excellent Deal"
                    elif predicted_margin >= cfg['good_margin_pct']:
                        rating = "Good Deal"
                    elif predicted_margin >= cfg['negotiation_margin_pct'] and potential_profit >= cfg['min_cash_margin']:
                        rating = "Negotiation Target"
                    else:
                        rating = "Reject"
                    
                    original_listing['enhanced_rating'] = rating
                    original_listing['analysis_method'] = 'ml_xgboost'
                    
                    # Only keep profitable deals
                    if rating != "Reject":
                        analyzed_listings.append(original_listing)
        
        except Exception as e:
            logger.error(f"Error analyzing {make} {model}: {e}")
            continue
    
    # Replace original listings with analyzed results
    listings.clear()
    listings.extend(analyzed_listings)
    
    if verbose:
        logger.info(f"✅ ML analysis complete: {len(analyzed_listings)} profitable deals found")
    
    # Save data in test mode when called from scraper
    global EXECUTION_MODE
    if EXECUTION_MODE == "test" and analyzed_listings:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deals_file = os.path.join(DATA_OUTPUTS_PATH, f"{timestamp}_scraper_analyzed_deals.json")
        try:
            with open(deals_file, 'w') as f:
                json.dump(analyzed_listings, f, indent=2, default=str)
            logger.info(f"💾 Saved {len(analyzed_listings)} analyzed deals from scraper to {deals_file}")
        except Exception as e:
            logger.error(f"❌ Error saving scraper deals: {e}")


def enhanced_keep_listing(listing: Dict[str, Any], cfg: Dict = None) -> bool:
    """
    Enhanced filtering logic for ML-analyzed listings.
    Compatible with the existing scraper.py interface.
    
    Args:
        listing: Car listing dictionary
        cfg: Configuration dictionary (optional)
        
    Returns:
        bool: True if listing should be kept, False otherwise
    """
    if cfg is None:
        cfg = {
            'excellent_margin_pct': 0.25,
            'good_margin_pct': 0.20,
            'negotiation_margin_pct': 0.15,
            'min_cash_margin': 800,
        }
    
    # Basic seller type filter (private only for deals)
    seller_type = listing.get("seller_type", "").upper()
    if seller_type not in ["PRIVATE"]:
        return False
    
    # Use ML-enhanced rating only - no fallback to old methods
    rating = listing.get("enhanced_rating", "")
    
    # Keep profitable deals only
    return rating in {"Excellent Deal", "Good Deal", "Negotiation Target"}


def convert_network_request_to_listing(network_car: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert network request car data to the format expected by the ML analyzer.
    
    Args:
        network_car: Raw network request data
        
    Returns:
        Dict: Converted listing format
    """
    processed = network_car.get('processed', {})
    
    # Create listing format compatible with ML analyzer
    listing = {
        # Basic identification
        'deal_id': processed.get('deal_id'),
        'url': processed.get('url'),
        'make': processed.get('make'),
        'model': processed.get('model'),
        
        # Core pricing and vehicle data
        'year': processed.get('year', 0),
        'asking_price': processed.get('price', 0) or 0,
        'mileage': processed.get('mileage', 0),
        
        # Enhanced specification data
        'fuel_type': processed.get('fuel_type'),
        'transmission': processed.get('transmission'),
        'engine_size': processed.get('engine_size'),
        'body_type': processed.get('body_type'),
        'doors': processed.get('doors'),
        
        # Seller information
        'seller_type': processed.get('seller_type'),
        'location': processed.get('location', ''),
        
        # Metadata
        'number_of_images': processed.get('number_of_images'),
        'image_url': processed.get('image_url'),
        'image_url_2': processed.get('image_url_2'),
        'badges': processed.get('badges', []),
    }
    
    return listing


if __name__ == "__main__":
    main()