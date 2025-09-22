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

# Note: Legacy one-hot encoding removed - now using model-specific approach

# Set up proper import paths (only need to do this once)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import proxy rotation service
from services.stealth_orchestrator import ProxyManager

# Import the price scraper for price marker data
from services.price_scraper import scrape_price_marker, batch_scrape_price_markers

# Import main scraping components from your core codebase
from services.network_requests import AutoTraderAPIClient
from services.network_requests import NetworkDataAdapter

# Individual ML model components (no universal model needed)
from config.config import is_retail_scraping_due

# Project paths - everything goes to archive folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.path.join(PROJECT_ROOT, 'archive')

# Set up logging - suppress verbose logs, only show errors
os.makedirs(ARCHIVE_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.ERROR,  # Only show errors
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
DATA_OUTPUTS_PATH = os.path.join(ARCHIVE_PATH, "outputs")

# Create directories if they don't exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(DATA_OUTPUTS_PATH, exist_ok=True)

# Global execution mode tracking
EXECUTION_MODE = "production"  # "test" or "production"
USE_KNN_MODEL = False  # Global flag for KNN vs XGBoost model selection

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
    # Only log mode changes in test mode to avoid verbose output
    if mode == "test":
        logger.info(f"Execution mode set to: {mode}")


def get_execution_mode() -> str:
    """Get the current execution mode"""
    global EXECUTION_MODE
    return EXECUTION_MODE


def set_model_type(use_knn: bool):
    """Set whether to use KNN or XGBoost models

    Args:
        use_knn: True for KNN, False for XGBoost
    """
    global USE_KNN_MODEL
    USE_KNN_MODEL = use_knn


def get_model_type() -> str:
    """Get the current model type"""
    global USE_KNN_MODEL
    return "KNN" if USE_KNN_MODEL else "XGBoost"


# Feature Extraction Helper Functions
def extract_fuel_from_spec(spec_text: str) -> str:
    """
    Extract fuel type from specification text.

    Args:
        spec_text: Vehicle specification text

    Returns:
        Fuel type string or None if not found
    """
    if not spec_text:
        return None

    spec_lower = str(spec_text).lower()

    # Extended fuel type patterns including common automotive codes
    fuel_patterns = {
        'Diesel': ['diesel', 'tdi', 'hdi', 'dci', 'cdti', 'crdi', 'd4d', 'dtec', 'bluetec'],
        'Petrol': ['petrol', 'gasoline', 'tsi', 'tfsi', 'vti', 'gti', 'turbo', 'i-vtec', 'mpi'],
        'Hybrid': ['hybrid', 'hev'],
        'Plugin Hybrid': ['plugin hybrid', 'plug-in hybrid', 'phev', 'e-hybrid'],
        'Electric': ['electric', 'ev', 'bev', 'e-tron', 'i3', 'leaf', 'tesla']
    }

    # Check each fuel type pattern
    for fuel_type, patterns in fuel_patterns.items():
        for pattern in patterns:
            if pattern in spec_lower:
                return fuel_type

    # BMW/Audi specific patterns
    import re
    if re.search(r'\d{3}i\b', spec_lower):  # Matches patterns like 320i, 330i, etc. (petrol)
        return 'Petrol'
    if re.search(r'\d{3}d\b', spec_lower):  # Matches patterns like 320d, 330d, etc. (diesel)
        return 'Diesel'

    # Mercedes patterns (e.g., C200, E350)
    if re.search(r'[a-z]\d{3}\b', spec_lower) and 'tdi' not in spec_lower and 'hdi' not in spec_lower:
        return 'Petrol'

    return None


def extract_transmission_from_spec(spec_text: str) -> str:
    """
    Extract transmission type from specification text.

    Args:
        spec_text: Vehicle specification text

    Returns:
        Transmission type string or None if not found
    """
    if not spec_text:
        return None

    spec_lower = str(spec_text).lower()

    # Extended transmission patterns
    transmission_patterns = {
        'Automatic': ['automatic', 'auto', 'tiptronic', 'multitronic', 'steptronic', 'powershift'],
        'Manual': ['manual', 'stick', 'mt', '5dr', '3dr'],  # dr patterns often indicate manual
        'Semi-Auto': ['semi-auto', 'semi auto', 'automated manual', 'amt'],
        'CVT': ['cvt', 'continuously variable'],
        'DSG': ['dsg', 'dual clutch', 's tronic', 'pdk', 'edc']
    }

    # Check each transmission pattern
    for trans_type, patterns in transmission_patterns.items():
        for pattern in patterns:
            if pattern in spec_lower:
                return trans_type

    # Additional patterns with regex
    import re

    # Look for specific gear patterns (e.g., "6-speed auto", "5-speed manual")
    if re.search(r'\d-speed auto|\d speed auto', spec_lower):
        return 'Automatic'
    if re.search(r'\d-speed manual|\d speed manual', spec_lower):
        return 'Manual'

    # BMW/Audi xDrive, AWD systems often come with auto
    if 'xdrive' in spec_lower or 'quattro' in spec_lower or '4matic' in spec_lower:
        return 'Automatic'

    return None


def clean_spec_for_ml(spec_text: str) -> str:
    """
    Clean specification text for ML analysis by removing technical details
    and keeping only meaningful trim/variant information.

    Args:
        spec_text: Raw specification text

    Returns:
        Cleaned specification suitable for ML analysis
    """
    if not spec_text:
        return ""

    import re

    # Start with the original spec
    cleaned = str(spec_text).strip()

    # Remove engine size patterns at the beginning (e.g., "2.0", "1.6", "3.0")
    cleaned = re.sub(r'^\d+\.\d+\s+', '', cleaned)

    # Remove transmission indicators
    transmission_patterns = [
        r'\s+Auto\b', r'\s+Manual\b', r'\s+Automatic\b', r'\s+DSG\b',
        r'\s+CVT\b', r'\s+Tiptronic\b', r'\s+Multitronic\b'
    ]
    for pattern in transmission_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove Euro emissions and door patterns (e.g., "Euro 6 (s/s) 4dr", "Euro 5 3dr")
    cleaned = re.sub(r'\s+Euro\s+\d+\s*\([^)]*\)\s*\d+dr\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+Euro\s+\d+\s+\d+dr\s*$', '', cleaned, flags=re.IGNORECASE)

    # Remove standalone door patterns at the end (e.g., "5dr", "3dr", "4dr")
    cleaned = re.sub(r'\s+\d+dr\s*$', '', cleaned, flags=re.IGNORECASE)

    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())

    return cleaned


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




def scrape_listings(make: str, model: str = None, max_pages: int = None, verify_ssl: bool = True, use_proxy: bool = False) -> List[Dict[str, Any]]:
    """Scrape car listings for a given make and model

    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (e.g., '3-series')
        max_pages: Maximum number of pages to scrape (None for all available pages)
        verify_ssl: Whether to verify SSL certificates (set to False to ignore SSL errors)
        use_proxy: Whether to use proxy rotation for scraping

    Returns:
        List of dictionaries containing car listing details
    """
    # Group structure is handled by scraping.py orchestrator
    # No need to duplicate it here

    # This function is kept for backwards compatibility but shouldn't be called
    
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
        # Use parallel API scraping for maximum speed, fallback to sequential mileage splitting
        try:
            from config.config import PARALLEL_API_SCRAPING
            use_parallel = PARALLEL_API_SCRAPING.get('enabled', True)
        except ImportError:
            use_parallel = True  # Default to parallel if config unavailable

        # Scraping start is handled by orchestrator

        cars = api_client.get_all_cars_parallel(
            make=make,
            model=model,
            max_pages=max_pages,  # Passing None to get all pages
            use_parallel=use_parallel
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
                'doors': car.get('doors'),
                
                # Additional data
                'seller_type': 'Dealer' if car.get('seller_type', '').upper() == 'TRADE' else 'Private',
                'url': car.get('url', ''),
                'image_url': car.get('image_url', ''),
                'image_url_2': car.get('image_url_2', ''),
                'location': car.get('location', ''),
                
                # Keep original spec unchanged from API (trim/variant info like "R-Line", "AMG Line")
                'spec': car.get('spec', '')
            }

            # Apply fallback extraction for missing fuel_type and transmission from spec
            spec_text = car.get('spec', '')

            # Only extract fuel_type if API didn't provide it or it's empty/None
            if not listing.get('fuel_type'):
                extracted_fuel = extract_fuel_from_spec(spec_text)
                if extracted_fuel:
                    listing['fuel_type'] = extracted_fuel

            # Only extract transmission if API didn't provide it or it's empty/None
            if not listing.get('transmission'):
                extracted_transmission = extract_transmission_from_spec(spec_text)
                if extracted_transmission:
                    listing['transmission'] = extracted_transmission

            # Clean the spec text for ML analysis before caching
            listing['spec'] = clean_spec_for_ml(spec_text)

            listings.append(listing)
        
        # Scraping result is handled by orchestrator with proper timing

        return listings
        
    except Exception as e:
        logger.error(f"Error scraping listings: {e}")
        return []


def enrich_with_price_markers(listings: List[Dict[str, Any]], make: str = None, model: str = None, use_proxy: bool = False, quiet_mode: bool = False) -> List[Dict[str, Any]]:
    """Enrich dealer listings with price marker data using parallel processing

    Args:
        listings: List of car listings
        make: Car manufacturer (for identification)
        model: Car model (for identification)
        use_proxy: Whether to use proxy rotation for scraping
        quiet_mode: If True, suppress banner output for clean logging

    Returns:
        Enriched listings with price_vs_market field added
    """
    # Process retail prices from listings
    
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
    logger.debug(f"Starting to process {total_urls} URLs in parallel for price markers")

    # Initialize dynamic progress display (replaces static banners)
    progress_display = None
    # Disabled to prevent duplication with OutputManager
    # if not quiet_mode:
    #     from services.progress_display import DynamicProgressDisplay
    #     progress_display = DynamicProgressDisplay(total_urls, "🔍 Scraping Retail Prices")

    if not valid_urls:
        return []
    
    # Create a progress display function
    start_time = datetime.now()

    # Initialize retail price scraping display
    from src.output_manager import get_output_manager
    output_manager = get_output_manager()

    def progress_callback(completed, total):
        # Update via OutputManager for coordinated display
        if completed == 0:
            # First call - show initial status
            output_manager.progress_update(0, total, "retail_prices", 0, "Calculating...")
        else:
            # Calculate stats
            elapsed_time = (datetime.now() - start_time).total_seconds()
            speed = (completed * 60 / elapsed_time) if elapsed_time > 0 else 0

            # Calculate ETA
            if completed < total:
                avg_time_per_item = elapsed_time / completed
                remaining_items = total - completed
                eta_seconds = remaining_items * avg_time_per_item
                if eta_seconds < 60:
                    eta_str = f"{int(eta_seconds)}s"
                else:
                    eta_minutes = int(eta_seconds // 60)
                    eta_seconds_remainder = int(eta_seconds % 60)
                    eta_str = f"{eta_minutes}m {eta_seconds_remainder}s"
            else:
                eta_str = "Complete!"

            output_manager.progress_update(completed, total, "retail_prices", speed, eta_str)
    
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
        # Initialize progress display once here
        output_manager.progress_update(0, total_urls, "retail_prices", 0, "Starting...")

        results = batch_scrape_price_markers(
            valid_urls,
            headless=True,
            progress_callback=progress_callback,
            test_mode=(EXECUTION_MODE == "test")
        )

        # Merge price marker data with original listings
        enriched_listings = []
        success_count = 0
        error_count = 0

        for listing in valid_listings:
            url = listing['url']

            if url in results:
                price_marker = results[url]

                # Check if this is an error result or no valid price difference found
                if ('Error' in price_marker.get('marker_text', '') or
                    not price_marker.get('price_difference_found', False)):
                    error_count += 1
                    # Only log in test mode to avoid spam
                    if EXECUTION_MODE == "test":
                        if 'Error' in price_marker.get('marker_text', ''):
                            logger.warning(f"Scraping error for {listing.get('make', '')} {listing.get('model', '')}: {price_marker.get('marker_text', 'Unknown error')}")
                        else:
                            logger.debug(f"No valid price difference found for {listing.get('make', '')} {listing.get('model', '')}")
                    continue
                else:
                    # Successful result with valid price difference
                    success_count += 1
                    market_difference = price_marker['market_difference']

                    # Add price marker data to the listing
                    enriched_listing = {
                        **listing,
                        'price_vs_market': market_difference,
                        'market_value': price_marker.get('market_value', 0)
                    }
                    enriched_listings.append(enriched_listing)

            else:
                # URL was processed but no result - skip this listing
                error_count += 1
                # Only log errors in test mode to avoid spam
                if EXECUTION_MODE == "test":
                    logger.error(f"No price marker result for URL: {url}")
                # Skip this listing instead of defaulting to 0.0 - prevents bad training data
                logger.debug(f"Skipping listing without price marker result: {listing.get('make', '')} {listing.get('model', '')}")
                continue
        
        # Complete the progress display (replaces static completion banner)
        output_manager.progress_complete("retail_prices")
                
    except Exception as e:
        logger.error(f"Error in batch price marker processing: {e}")
        # Fall back to sequential processing as a backup
        logger.warning("Falling back to sequential processing")
        
        enriched_listings = []
        total = len(valid_listings)

        from src.output_manager import get_output_manager
        output_manager = get_output_manager()
        output_manager.warning("Falling back to sequential processing")

        for i, listing in enumerate(valid_listings):
            try:
                # Show progress for sequential processing too
                # Progress handled silently for sequential processing

                # Scrape price marker for this listing
                price_marker = scrape_price_marker(listing['url'])

                # Only process if we found a valid price difference
                if (not price_marker.get('price_difference_found', False) or
                    'Error' in price_marker.get('marker_text', '')):
                    # Skip this listing - no valid price difference
                    continue

                # Add price marker data to the listing
                enriched_listing = {
                    **listing,
                    'price_vs_market': price_marker['market_difference'],
                    'market_value': price_marker.get('market_value', 0)
                }
                enriched_listings.append(enriched_listing)

            except Exception as e:
                logger.error(f"Error enriching listing with price marker: {e}")
                # Skip listing with exception instead of defaulting to 0.0
                logger.debug(f"Skipping listing due to processing error: {listing.get('make', '')} {listing.get('model', '')}")
                continue
    
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
    logger.debug(f"Found seller types in data: {seller_types}")
    
    filtered = [listing for listing in listings if listing.get('seller_type') == seller_type]
    logger.debug(f"Filtered {len(filtered)} {seller_type.lower()} listings from {len(listings)} total")
    return filtered


# Legacy prepare_features function removed - now using model_specific_trainer.prepare_model_specific_features
# (The entire function with one-hot encoding has been replaced by model-specific approach)
    """Extract and prepare features for model training or prediction
    
    Args:
        listings: List of car listings
        
    Returns:
        DataFrame with prepared features
    """
    # LEGACY FUNCTION - DISABLED
    # This function has been replaced by model_specific_trainer.prepare_model_specific_features
    raise NotImplementedError("Legacy prepare_features function disabled. Use model_specific_trainer.prepare_model_specific_features instead.")
    
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
        
        # Create categorical spec encoding instead of hardcoded multipliers
        # Let the ML model learn the value relationships from training data

        # Define spec categories for consistent encoding
        spec_categories = {
            # Base/Standard trims
            'base': ['s', 'se', 'life', 'match', 'edition'],

            # Sport/Performance trims
            'sport': ['sport', 'r-line', 'gti', 'gtd', 'r', 'gte', 'm sport', 'amg'],

            # Luxury/Premium trims
            'luxury': ['luxury', 'premium', 'executive', 'lounge', 'se l'],

            # All-wheel drive variants
            'awd': ['4wd', 'awd', '4x4', 'quattro', '4motion'],

            # Technology/Comfort features
            'tech': ['panoramic', 'leather', 'navigation', 'tech pack', 'dsg']
        }

        # Create unique spec encoding - each spec gets its own ID
        # Use the already processed spec_text column that was created earlier
        spec_codes, spec_uniques = pd.factorize(df['spec_text'])
        df['spec_numeric'] = spec_codes + 1

        logger.info(f"Found {len(spec_uniques)} unique specs in legacy analyzer dataset")
        logger.info(f"spec_numeric stats: min={df['spec_numeric'].min():.2f}, max={df['spec_numeric'].max():.2f}, mean={df['spec_numeric'].mean():.2f}")
    else:
        # Default spec_numeric if no spec column
        df['spec_numeric'] = 1.0
    
    # Extract and encode fuel type (fallback for missing values only)
    if spec_column in df.columns:
        # Only fill missing fuel_type values, don't overwrite existing ones
        if 'fuel_type' not in df.columns:
            df['fuel_type'] = 'Unknown'

        fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Plugin Hybrid']

        for idx, row in df.iterrows():
            # Only extract if fuel_type is missing/None/Unknown
            if (not row.get('fuel_type') or row.get('fuel_type') == 'Unknown') and pd.notna(row.get(spec_column)):
                spec_text = str(row[spec_column]).lower()
                for fuel in fuel_types:
                    if fuel.lower() in spec_text:
                        df.at[idx, 'fuel_type'] = fuel
                        break
                # If still no fuel type found, keep as Unknown
                if not df.at[idx, 'fuel_type'] or df.at[idx, 'fuel_type'] == '':
                    df.at[idx, 'fuel_type'] = 'Unknown'
    else:
        # Ensure fuel_type exists even if spec doesn't
        if 'fuel_type' not in df.columns:
            df['fuel_type'] = 'Unknown'
    
    # Extract and encode transmission (fallback for missing values only)
    if spec_column in df.columns:
        # Only fill missing transmission values, don't overwrite existing ones
        if 'transmission' not in df.columns:
            df['transmission'] = 'Unknown'

        transmission_types = ['Automatic', 'Manual', 'Semi-Auto', 'CVT', 'DSG']

        for idx, row in df.iterrows():
            # Only extract if transmission is missing/None/Unknown
            if (not row.get('transmission') or row.get('transmission') == 'Unknown') and pd.notna(row.get(spec_column)):
                spec_text = str(row[spec_column]).lower()
                for trans in transmission_types:
                    if trans.lower() in spec_text:
                        df.at[idx, 'transmission'] = trans
                        break
                # If still no transmission found, keep as Unknown
                if not df.at[idx, 'transmission'] or df.at[idx, 'transmission'] == '':
                    df.at[idx, 'transmission'] = 'Unknown'
    else:
        # Ensure transmission exists even if spec doesn't
        if 'transmission' not in df.columns:
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

    # Add one-hot encodings for makes and models instead of ordinal encoding
    def add_one_hot_encodings(df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot encodings for makes and models."""
        df_encoded = df.copy()

        # Create one-hot encodings for makes
        for make in MAKE_ENCODING.keys():
            make_column = f"make_{make.lower().replace('-', '_').replace(' ', '_')}"
            df_encoded[make_column] = (df_encoded['make'] == make).astype(int)

        # Create one-hot encodings for models
        for model in MODEL_ENCODING.keys():
            model_column = f"model_{model.lower().replace('-', '_').replace(' ', '_')}"
            df_encoded[model_column] = (df_encoded['model'] == model).astype(int)

        logger.info(f"Added one-hot encodings for {len(MAKE_ENCODING)} makes and {len(MODEL_ENCODING)} models")
        return df_encoded

    # Apply one-hot encoding
    df = add_one_hot_encodings(df)

    # Get lists of one-hot encoded columns
    make_columns = [f"make_{make.lower().replace('-', '_').replace(' ', '_')}"
                   for make in MAKE_ENCODING.keys()]
    model_columns = [f"model_{model.lower().replace('-', '_').replace(' ', '_')}"
                    for model in MODEL_ENCODING.keys()]

    # Select features for the model (now including one-hot encoded make/model)
    feature_columns = [
        'asking_price',
        'mileage',
        'age',
        'market_value',
        'fuel_type_numeric',
        'transmission_numeric',
        'engine_size',
        'spec_numeric'
    ] + make_columns + model_columns
    
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
    
    # Train the model with warning suppression
    logger.info(f"Training XGBoost model on {len(X_train)} samples")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*UBJSON format.*")
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

    # Suppress XGBoost UBJSON format warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*UBJSON format.*")
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
    dealer_data: List[Dict[str, Any]],
    force_retrain: bool = False
) -> Tuple[Optional[Any], Optional[StandardScaler], Optional[dict]]:
    """Load existing model or automatically train if missing

    Args:
        make: Car manufacturer
        model: Car model
        dealer_data: List of dealer listings for training if needed
        force_retrain: Force retraining regardless of model age

    Returns:
        Tuple of (model, feature scaler, performance_metrics) or (None, None, None) if training fails
        Model type depends on USE_KNN_MODEL global flag
    """
    global USE_KNN_MODEL
    model_type = "KNN" if USE_KNN_MODEL else "XGBoost"

    # Check if model exists and if it's too old before loading
    model_age_days = get_model_age_days(make, model)

    if not force_retrain and model_age_days < 999:  # Model exists and not forcing retrain
        if model_age_days <= 14:  # Model is fresh
            # Load existing fresh model based on type
            if USE_KNN_MODEL:
                from services.knn_trainer import load_knn_model
                loaded_model, scaler, performance_metrics = load_knn_model(make, model)
            else:
                from services.model_specific_trainer import load_model_specific
                loaded_model, scaler, performance_metrics = load_model_specific(make, model)

            if loaded_model is not None and scaler is not None:
                return loaded_model, scaler, performance_metrics
            else:
                from src.output_manager import get_output_manager
                output_manager = get_output_manager()
                output_manager.ml_model_status(f"❌ Failed to load existing {model_type} model - will retrain")
        else:
            from src.output_manager import get_output_manager
            output_manager = get_output_manager()
            output_manager.ml_model_status(f"Found existing {model_type} model ({model_age_days} days old)")
            output_manager.ml_model_status("Model too old - retraining needed")
    else:
        pass

    # Proceed with automatic training (either no model or old model)
    # Check if we have enough dealer data for training
    if not dealer_data or len(dealer_data) < 5:
        from src.output_manager import get_output_manager
        output_manager = get_output_manager()
        output_manager.ml_model_status(f"❌ Insufficient dealer data (need 5, have {len(dealer_data) if dealer_data else 0})")
        output_manager.ml_model_complete()
        return None, None, None

    try:
        # Import training functions based on model type
        if USE_KNN_MODEL:
            from services.knn_trainer import train_knn_model
            train_function = train_knn_model
        else:
            from services.model_specific_trainer import train_model_specific
            train_function = train_model_specific

        # Attempt to train the model with timeout
        import threading
        training_result = [None]
        training_error = [None]

        def training_worker():
            try:
                training_result[0] = train_function(make, model, dealer_data)
            except Exception as e:
                training_error[0] = e

        # Start training in separate thread with timeout
        training_thread = threading.Thread(target=training_worker)
        training_thread.daemon = True
        training_thread.start()
        training_thread.join(timeout=600)  # 10 minutes timeout

        if training_thread.is_alive():
            from src.output_manager import get_output_manager
            output_manager = get_output_manager()
            output_manager.ml_model_status(f"❌ {model_type} model training timed out after 10 minutes")
            output_manager.ml_model_complete()
            return None, None, None
        elif training_error[0]:
            from src.output_manager import get_output_manager
            output_manager = get_output_manager()

            # Check if error is related to missing market_value data
            error_str = str(training_error[0]).lower()
            is_market_value_error = ('market_value' in error_str or
                                   'no valid features prepared' in error_str or
                                   'cannot train without real market prices' in error_str)

            if is_market_value_error and is_retail_scraping_due():
                output_manager.ml_model_status("❌ Training failed due to missing market_value data")
                output_manager.ml_model_status("⚠️ Retail scraping is due - triggering automatic retail price scraping")

                # Trigger automatic retail price scraping
                try:
                    from services.price_scraper import batch_scrape_price_markers

                    # Get listings for retail price scraping
                    retail_listings = [listing for listing in listings if listing.get('url')]
                    if retail_listings:
                        output_manager._print("")

                        # Scrape retail prices using batch scraper
                        retail_data = batch_scrape_price_markers(
                            [listing['url'] for listing in retail_listings],
                            headless=True,
                            test_mode=False
                        )

                        if retail_data:
                            # batch_scrape_price_markers returns a dict: {url: result_data}
                            # Transform it to the format expected by the code
                            url_to_market_value = {url: result.get('market_value')
                                                 for url, result in retail_data.items()
                                                 if result.get('market_value') and result.get('market_value') > 0}

                            # Update dealer_data with market values
                            updated_dealer_data = []
                            for listing in dealer_data:
                                url = listing.get('url', '')
                                if url in url_to_market_value:
                                    listing_copy = listing.copy()
                                    listing_copy['market_value'] = url_to_market_value[url]
                                    updated_dealer_data.append(listing_copy)
                                else:
                                    updated_dealer_data.append(listing)

                            output_manager.ml_model_status(f"Updated {len([d for d in updated_dealer_data if d.get('market_value')])} listings with market values")

                            # Retry training with updated data
                            output_manager.ml_model_status("Retrying model training with market value data...")

                            # Reset training variables for retry
                            training_result_retry = [None]
                            training_error_retry = [None]

                            def training_worker_retry():
                                try:
                                    training_result_retry[0] = train_function(make, model, updated_dealer_data)
                                except Exception as e:
                                    training_error_retry[0] = e

                            # Retry training
                            training_thread_retry = threading.Thread(target=training_worker_retry)
                            training_thread_retry.daemon = True
                            training_thread_retry.start()
                            training_thread_retry.join(timeout=600)

                            if training_thread_retry.is_alive():
                                output_manager.ml_model_status("❌ Retry training timed out")
                                output_manager.ml_model_complete()
                                return None, None
                            elif training_error_retry[0]:
                                output_manager.ml_model_status(f"❌ Retry training failed: {training_error_retry[0]}")
                                output_manager.ml_model_complete()
                                return None, None
                            elif training_result_retry[0]:
                                # Training succeeded after retail scraping
                                if USE_KNN_MODEL:
                                    from services.knn_trainer import load_knn_model
                                    trained_model, scaler, performance_metrics = load_knn_model(make, model)
                                else:
                                    from services.model_specific_trainer import load_model_specific
                                    trained_model, scaler, performance_metrics = load_model_specific(make, model)

                                if trained_model is not None and scaler is not None:
                                    output_manager.ml_model_status(f"✅ Successfully trained {model_type} model after retail price scraping")
                                    # Mark retail scraping as completed
                                    from config.config import mark_retail_scraping_completed
                                    mark_retail_scraping_completed()
                                    output_manager.ml_model_complete()
                                    return trained_model, scaler, performance_metrics
                                else:
                                    output_manager.ml_model_status(f"❌ {model_type} training succeeded but failed to load model")
                                    output_manager.ml_model_complete()
                                    return None, None, None
                            else:
                                output_manager.ml_model_status("❌ Retry training returned no result")
                                output_manager.ml_model_complete()
                                return None, None, None
                        else:
                            output_manager.ml_model_status("❌ Retail price scraping returned no data")
                            output_manager.ml_model_complete()
                            return None, None, None
                    else:
                        output_manager.ml_model_status("❌ No listings with URLs available for retail price scraping")
                        output_manager.ml_model_complete()
                        return None, None, None

                except Exception as scraping_error:
                    output_manager.ml_model_status(f"❌ Automatic retail price scraping failed: {scraping_error}")
                    output_manager.ml_model_complete()
                    return None, None, None
            else:
                # Original error handling for non-market_value errors
                output_manager.ml_model_status(f"❌ {model_type} training failed: {training_error[0]}")
                output_manager.ml_model_complete()
                return None, None, None
        else:
            success = training_result[0]

        if success:
            # Try to load the newly trained model
            if USE_KNN_MODEL:
                from services.knn_trainer import load_knn_model
                trained_model, scaler, performance_metrics = load_knn_model(make, model)
            else:
                from services.model_specific_trainer import load_model_specific
                trained_model, scaler, performance_metrics = load_model_specific(make, model)

            if trained_model is not None and scaler is not None:
                from src.output_manager import get_output_manager
                output_manager = get_output_manager()
                output_manager.ml_model_complete()
                return trained_model, scaler, performance_metrics
            else:
                from src.output_manager import get_output_manager
                output_manager = get_output_manager()
                output_manager.ml_model_status(f"❌ {model_type} training appeared successful but failed to load")
                output_manager.ml_model_complete()
                return None, None, None
        else:
            from src.output_manager import get_output_manager
            output_manager = get_output_manager()
            output_manager.ml_model_status(f"❌ Failed to train {model_type} model")
            output_manager.ml_model_complete()
            return None, None, None

    except Exception as e:
        from src.output_manager import get_output_manager
        output_manager = get_output_manager()
        output_manager.ml_model_status(f"❌ Error during {model_type} training: {str(e)}")
        output_manager.ml_model_complete()
        return None, None, None


def get_model_age_days(make: str, model: str) -> int:
    """Get the age of a model in days by checking its file modification time

    Args:
        make: Car manufacturer
        model: Car model

    Returns:
        Age in days, or 999 if model doesn't exist
    """
    try:
        from services.daily_trainer import DailyModelTrainingOrchestrator
        trainer = DailyModelTrainingOrchestrator()
        model_file, scaler_file = trainer.get_model_path(make, model)

        if model_file.exists():
            import os
            from datetime import datetime

            # Get file modification time
            mod_time = os.path.getmtime(model_file)
            mod_datetime = datetime.fromtimestamp(mod_time)

            # Calculate age in days
            age_days = (datetime.now() - mod_datetime).days
            return age_days
        else:
            return 999  # Model doesn't exist

    except Exception as e:
        return 999  # Error occurred, assume old


# Prediction and Deal Analysis Functions
def detect_model_feature_count(scaler: StandardScaler) -> int:
    """
    Detect if a model was trained with different feature counts:
    - 7 features: old models (with asking_price, no market context)
    - 11 features: intermediate models (with asking_price + market context) - caused overfitting
    - 10 features: intermediate models (no asking_price, with market context) - caused overfitting
    - 6 features: old numeric-only models (no asking_price, no market context) - simple and robust
    - 3 features: new categorical models (numeric features only scaled) - XGBoost native categorical

    Args:
        scaler: StandardScaler from the trained model

    Returns:
        int: Feature count expected by the model
    """
    try:
        # Check the scaler's expected feature count
        expected_features = scaler.n_features_in_

        if expected_features == 7:
            logger.debug("Detected old 7-feature model (legacy with asking_price)")
            return 7
        elif expected_features == 11:
            logger.debug("Detected intermediate 11-feature model (asking_price + market context)")
            return 11
        elif expected_features == 10:
            logger.debug("Detected intermediate 10-feature model (no asking_price, with market context)")
            return 10
        elif expected_features == 6:
            logger.debug("Detected old 6-feature model (simple and robust)")
            return 6
        elif expected_features == 3:
            logger.debug("Detected new categorical model (XGBoost native categorical support)")
            return 3
        else:
            logger.warning(f"Unexpected feature count {expected_features}, defaulting to 7-feature mode")
            return 7

    except AttributeError:
        # Older sklearn versions might not have n_features_in_
        logger.warning("Cannot detect feature count, defaulting to 7-feature mode for compatibility")
        return 7


def predict_market_values(
    private_listings: List[Dict[str, Any]],
    model: Any,
    scaler: StandardScaler,
    performance_metrics: dict = None
) -> pd.DataFrame:
    """Predict market values for private seller listings using model-specific features

    Args:
        private_listings: List of car listings from private sellers
        model: Trained model (XGBoost or KNN)
        scaler: Feature scaler
        performance_metrics: Model metadata (required for KNN)

    Returns:
        DataFrame with listings and predicted market values
    """
    # Check if we're using KNN model
    global USE_KNN_MODEL
    if USE_KNN_MODEL:
        # Use KNN prediction path
        return predict_with_knn_model(private_listings, model, scaler, performance_metrics)

    # Use model-specific feature preparation for XGBoost (no one-hot encoding)
    from services.model_specific_trainer import prepare_model_specific_features
    from src.output_manager import get_output_manager

    output_manager = get_output_manager()

    try:
        features_df = prepare_model_specific_features(private_listings)
    except Exception as e:
        logger.error(f"Prediction preprocessing failed: {e}")
        return pd.DataFrame()

    if features_df.empty:
        logger.warning("No valid features prepared for prediction")
        return pd.DataFrame()


    # Detect if this model expects 7 features (old) or 11 features (new)
    expected_feature_count = detect_model_feature_count(scaler)

    # Define feature columns based on model type
    if expected_feature_count == 7:
        # Legacy 7-feature models (with asking_price, no market context)
        feature_columns = ['asking_price', 'mileage', 'age', 'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric']
        logger.debug("Using legacy 7-feature mode for prediction (with asking_price)")
    elif expected_feature_count == 11:
        # Intermediate 11-feature models (with asking_price + market context) - overfitting prone
        feature_columns = [
            'asking_price', 'mileage', 'age', 'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric',
            'n_similar_training', 'price_percentile_rank', 'mileage_for_age_learned', 'spec_similarity_score'
        ]
        logger.debug("Using intermediate 11-feature mode for prediction (asking_price + market context)")
    elif expected_feature_count == 10:
        # Intermediate 10-feature models (no asking_price, with market context) - overfitting prone
        feature_columns = [
            'mileage', 'age', 'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric',
            'n_similar_training', 'price_percentile_rank', 'mileage_for_age_learned', 'spec_similarity_score'
        ]
        logger.debug("Using intermediate 10-feature mode for prediction (no asking_price, with market context)")
    elif expected_feature_count == 6:
        # Old 6-feature models (simple and robust)
        feature_columns = [
            'mileage', 'age', 'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric'
        ]
        logger.debug("Using old 6-feature mode for prediction (simple and robust)")
    elif expected_feature_count == 3:
        # New categorical models - handle differently
        logger.debug("Using new categorical model for prediction (XGBoost native categorical)")
        return predict_with_categorical_model(private_listings, xgb_model, scaler, features_df)
    else:
        # Default fallback
        feature_columns = [
            'mileage', 'age', 'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric'
        ]
        logger.debug("Using fallback 6-feature mode for prediction")

    # Check if all required columns exist in the DataFrame, add any missing ones
    for col in feature_columns:
        if col not in features_df.columns:
            logger.warning(f"Column {col} missing in prediction data, adding with default values")
            # Use appropriate defaults based on feature type
            if col in ['n_similar_training']:
                features_df[col] = 0  # No similar cars found
            elif col in ['price_percentile_rank', 'mileage_for_age_learned']:
                features_df[col] = 0.5  # Median position
            elif col in ['spec_similarity_score']:
                features_df[col] = 0.1  # Low similarity (rare spec)
            else:
                features_df[col] = 0  # Default for other features
    
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
    X_scaled = scaler.transform(features_df[feature_columns].values)
    
    # Make predictions using XGBoost
    dtest = xgb.DMatrix(X_scaled)

    # Predict market values directly
    predicted_market_values = model.predict(dtest)

    # Add predictions to DataFrame
    features_df['predicted_market_value'] = predicted_market_values
    
    # Calculate profit margin from the predicted market value
    features_df['predicted_profit_margin'] = (
        (features_df['predicted_market_value'] - features_df['asking_price']) /
        features_df['asking_price']
    )
    
    # Join with original data to include all listing details
    private_df = pd.DataFrame(private_listings)

    # Ensure asking_price is numeric in private_df
    private_df['asking_price'] = pd.to_numeric(private_df['asking_price'], errors='coerce')
    private_df['mileage'] = pd.to_numeric(private_df['mileage'], errors='coerce')

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


def predict_with_categorical_model(
    private_listings: List[Dict[str, Any]],
    xgb_model: xgb.Booster,
    scaler: StandardScaler,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """Predict market values using new categorical XGBoost models with native categorical support

    Args:
        private_listings: List of car listings from private sellers
        xgb_model: Trained XGBoost model with categorical support
        scaler: Feature scaler (only scales numeric features)
        features_df: Prepared features DataFrame

    Returns:
        DataFrame with listings and predicted market values
    """
    logger.debug("Using categorical model prediction path")

    # Define feature structure for categorical models
    numeric_features = ['mileage', 'age', 'engine_size']
    categorical_features = ['fuel_type_extracted', 'transmission_extracted', 'spec_normalized']

    # Check if categorical features exist, if not create defaults
    for cat_feature in categorical_features:
        if cat_feature not in features_df.columns:
            logger.warning(f"Missing categorical feature {cat_feature}, using defaults")
            if cat_feature == 'fuel_type_extracted':
                features_df[cat_feature] = 'petrol'
            elif cat_feature == 'transmission_extracted':
                features_df[cat_feature] = 'manual'
            elif cat_feature == 'spec_normalized':
                features_df[cat_feature] = ''

    # Scale only numeric features
    if all(col in features_df.columns for col in numeric_features):
        numeric_scaled = scaler.transform(features_df[numeric_features])
        numeric_df = pd.DataFrame(numeric_scaled, columns=numeric_features, index=features_df.index)

        # Convert categorical features to pandas categorical type
        categorical_df = features_df[categorical_features].copy()
        for col in categorical_features:
            categorical_df[col] = categorical_df[col].astype('category')

        # Combine scaled numeric with categorical features
        combined_df = pd.concat([numeric_df, categorical_df], axis=1)
    else:
        logger.error("Missing required numeric features for categorical model")
        return pd.DataFrame()

    # Create DMatrix with categorical support
    dtest = xgb.DMatrix(
        combined_df,
        enable_categorical=True
    )

    # Make predictions
    predicted_market_values = xgb_model.predict(dtest)

    # Add predictions to DataFrame
    features_df['predicted_market_value'] = predicted_market_values

    # Calculate profit margin
    features_df['predicted_profit_margin'] = (
        (features_df['predicted_market_value'] - features_df['asking_price']) /
        features_df['asking_price']
    )

    # Join with original data
    private_df = pd.DataFrame(private_listings)
    private_df['asking_price'] = pd.to_numeric(private_df['asking_price'], errors='coerce')
    private_df['mileage'] = pd.to_numeric(private_df['mileage'], errors='coerce')

    result = pd.merge(
        private_df,
        features_df[['predicted_market_value', 'predicted_profit_margin']],
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Calculate potential profit
    result['potential_profit'] = result['predicted_market_value'] - result['asking_price']

    logger.info(f"Predicted market values for {len(result)} private listings using categorical model")
    logger.info(f"Average predicted market value: £{result['predicted_market_value'].mean():,.0f}")
    logger.info(f"Average profit margin: {result['predicted_profit_margin'].mean():.2%}")

    return result


def predict_with_knn_model(
    private_listings: List[Dict[str, Any]],
    knn_model,
    scaler: StandardScaler,
    performance_metrics: dict = None
) -> pd.DataFrame:
    """Predict market values using KNN model with target-encoded categorical features

    Args:
        private_listings: List of car listings from private sellers
        knn_model: Trained KNN model
        scaler: Feature scaler
        performance_metrics: Model metadata containing target encodings

    Returns:
        DataFrame with listings and predicted market values
    """
    logger.debug("Using KNN model prediction path")

    # Prepare features with target encodings from model metadata
    from services.knn_trainer import prepare_knn_features

    # Extract target encodings from performance metrics
    target_encodings = None
    if performance_metrics and 'target_encodings' in performance_metrics:
        target_encodings = performance_metrics['target_encodings']
        logger.debug(f"Using saved target encodings: {list(target_encodings.keys())}")

    features_df, numeric_features, all_features = prepare_knn_features(
        private_listings,
        target_encodings=target_encodings
    )

    if features_df.empty:
        logger.warning("No valid features prepared for KNN prediction")
        return pd.DataFrame()

    # Get feature structure from model metadata
    if performance_metrics:
        feature_columns = performance_metrics.get('feature_columns', all_features)
        logger.debug(f"Using saved feature columns: {feature_columns}")
    else:
        feature_columns = all_features
        logger.warning("No performance metrics provided - using auto-detected features")

    # Ensure all required features exist
    missing_features = [col for col in feature_columns if col not in features_df.columns]
    if missing_features:
        logger.warning(f"Missing features for KNN prediction: {missing_features}")
        # Use only available features
        feature_columns = [col for col in feature_columns if col in features_df.columns]

    if not feature_columns:
        logger.error("No valid features available for KNN prediction")
        return pd.DataFrame()

    # Scale features
    X_scaled = scaler.transform(features_df[feature_columns].values)

    # Make predictions using KNN
    predicted_market_values = knn_model.predict(X_scaled)

    # Add predictions to DataFrame
    features_df['predicted_market_value'] = predicted_market_values

    # Calculate profit margin
    features_df['predicted_profit_margin'] = (
        (features_df['predicted_market_value'] - features_df['asking_price']) /
        features_df['asking_price']
    )

    # Join with original data
    private_df = pd.DataFrame(private_listings)
    private_df['asking_price'] = pd.to_numeric(private_df['asking_price'], errors='coerce')
    private_df['mileage'] = pd.to_numeric(private_df['mileage'], errors='coerce')

    result = pd.merge(
        private_df,
        features_df[['predicted_market_value', 'predicted_profit_margin']],
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Calculate potential profit
    result['potential_profit'] = result['predicted_market_value'] - result['asking_price']

    logger.info(f"Predicted market values for {len(result)} private listings using KNN model")
    logger.info(f"Average predicted market value: £{result['predicted_market_value'].mean():,.0f}")
    logger.info(f"Average profit margin: {result['predicted_profit_margin'].mean():.2%}")

    return result


def filter_profitable_deals(
    predictions_df: pd.DataFrame,
    min_margin: float = MIN_PROFIT_MARGIN,
    min_cash_profit: float = 800
) -> pd.DataFrame:
    """Filter listings to keep only profitable deals

    Args:
        predictions_df: DataFrame with listings and predicted profit margins
        min_margin: Minimum profit margin threshold (default 15%)
        min_cash_profit: Minimum absolute profit in £ (default £800)

    Returns:
        DataFrame with only profitable deals
    """
    # Filter to keep only listings that meet BOTH margin AND cash profit requirements
    margin_filter = predictions_df['predicted_profit_margin'] >= min_margin
    cash_filter = predictions_df['potential_profit'] >= min_cash_profit

    profitable = predictions_df[margin_filter & cash_filter]

    # Sort by predicted profit margin (highest first)
    profitable = profitable.sort_values('predicted_profit_margin', ascending=False)

    logger.info(f"Found {len(profitable)} profitable deals with margin >= {min_margin:.1%} AND profit >= £{min_cash_profit}")
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
    """Process a specific car make/model to find profitable deals using universal model
    
    Args:
        make: Car manufacturer (e.g., 'bmw')
        model: Car model (optional, e.g., '3-series')
        max_pages: Maximum number of pages to scrape (None for all available pages)
        verify_ssl: Whether to verify SSL certificates
    """
    global EXECUTION_MODE
    
    # NOTE: Automatic weekly training disabled - now handled by individual model retry logic
    # This prevents early retail scraping during training phase
    # Retail scraping now only happens when individual model training fails due to missing market_value
    #
    # # Check if weekly retail scraping is due
    # if is_retail_scraping_due():
    #     logger.info("🔄 Weekly retail scraping is due - running full training process...")
    #     from services.ML_trainer import run_weekly_training  # Import locally to avoid circular import
    #     training_success = run_weekly_training(
    #         max_pages_per_model=max_pages,
    #         verify_ssl=verify_ssl,
    #         test_mode=(EXECUTION_MODE == "test")
    #     )
    #
    #     if not training_success:
    #         logger.error("❌ Weekly training failed - falling back to individual model processing")
    #         # Fall back to original individual model approach
    #         _process_individual_model(make, model, max_pages, verify_ssl)
    #         return
    
    # NOTE: Automatic universal model training disabled - now use individual model approach
    # This prevents early retail scraping and uses the controlled retry logic instead
    #
    # # Load universal model for daily operations
    # if not is_universal_model_available():
    #     logger.warning("⚠️ No universal model available - running weekly training first...")
    #     from services.ML_trainer import run_weekly_training  # Import locally to avoid circular import
    #     training_success = run_weekly_training(
    #         max_pages_per_model=max_pages,
    #         verify_ssl=verify_ssl,
    #         test_mode=(EXECUTION_MODE == "test")
    #     )
    #
    #     if not training_success:
    #         logger.error("❌ Training failed - falling back to individual model processing")
    #         _process_individual_model(make, model, max_pages, verify_ssl)
    #         return

    # Always use individual model processing to ensure controlled retail scraping
    _process_individual_model(make, model, max_pages, verify_ssl)
    return
    
    # Step 4: Filter for profitable deals
    profitable_deals = filter_profitable_deals(predictions_df)
    
    # Step 5: Save and alert (only save profitable deals in production mode)
    save_comprehensive_data(
        make=make,
        model=model,
        all_listings=all_listings if EXECUTION_MODE == "test" else None,
        retail_prices=None,  # No retail prices in daily operations
        profitable_deals=profitable_deals if len(profitable_deals) > 0 else None
    )
    
    # Step 6: Alert for profitable deals
    if len(profitable_deals) > 0:
        alert_buyer(profitable_deals, make, model)
        logger.info(f"✅ Found {len(profitable_deals)} profitable deals using universal model")
    else:
        logger.info(f"No profitable deals found for {make} {model or 'all'}")


def _process_individual_model(make: str, model: str = None, max_pages: int = None, verify_ssl: bool = False) -> None:
    """Fallback function using original individual model processing logic"""
    global EXECUTION_MODE
    
    logger.info(f"🔧 Processing {make} {model or 'All Models'} using individual model approach (fallback)")
    
    # Step 1: Scrape all listings
    all_listings = scrape_listings(make, model, max_pages, verify_ssl=verify_ssl, use_proxy=True)
    if not all_listings:
        from src.output_manager import get_output_manager
        output_manager = get_output_manager()
        output_manager.error(f"No listings found for {make} {model or 'All Models'}")
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
    
    # Step 3: Check if existing model is available first (avoid unnecessary retail scraping)
    model_age_days = get_model_age_days(make, model)

    enriched_dealer_listings = dealer_listings  # Default to unprocessed dealer listings
    retail_price_data = []

    # Always enrich with retail price markers when training is needed (regardless of model age)
    # This ensures we have market_value data for ML training
    enriched_dealer_listings = enrich_with_price_markers(dealer_listings, make, model, use_proxy=True, quiet_mode=True)

    # Collect retail price data for saving (if in test mode)
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
    trained_model, scaler, performance_metrics = load_or_train_model(make, model, enriched_dealer_listings)
    if trained_model is None or scaler is None:
        logger.error(f"Failed to create model for {make} {model or 'all'}")
        return

    # Step 5: Predict on private listings
    predictions_df = predict_market_values(private_listings, trained_model, scaler, performance_metrics)
    
    # Step 6: Filter for profitable deals
    profitable_deals = filter_profitable_deals(predictions_df)
    
    # Step 7: Save and alert
    save_comprehensive_data(
        make=make,
        model=model,
        all_listings=all_listings if EXECUTION_MODE == "test" else None,
        retail_prices=retail_price_data if retail_price_data else None,
        profitable_deals=profitable_deals if len(profitable_deals) > 0 else None
    )
    
    # Step 8: Alert for profitable deals
    if len(profitable_deals) > 0:
        alert_buyer(profitable_deals, make, model)
    else:
        logger.info(f"No profitable deals found for {make} {model or 'all'}")




def get_make_from_model(model_name: str) -> Optional[str]:
    """
    Auto-detect make from model name using TARGET_VEHICLES_BY_MAKE configuration.
    
    Args:
        model_name: Vehicle model name (e.g., "3 series", "fiesta", "yaris")
        
    Returns:
        str: Vehicle make in lowercase (e.g., "bmw", "ford", "toyota") or None if not found
    """
    from config.config import TARGET_VEHICLES_BY_MAKE
    
    if not model_name:
        return None
        
    model_lower = model_name.lower().strip()
    
    for make, models in TARGET_VEHICLES_BY_MAKE.items():
        for model in models:
            if model.lower() == model_lower:
                return make.lower()
    
    return None


def main():
    """Main entry point for the car deal ML analyzer with universal model support"""
    import argparse
    from config.config import TARGET_VEHICLES_BY_MAKE, get_weekly_retail_config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Car Deal ML Analyzer with Universal Model")
    parser.add_argument("--test", action="store_true", help="Safe test mode: use with --model for individual testing")
    parser.add_argument("--model", type=str, help="Test specific model (e.g., '3 series', 'fiesta', 'yaris')")
    parser.add_argument("--weekly-training", action="store_true", help="Force weekly retail scraping and model training")
    parser.add_argument("--no-proxy", action="store_true", help="Disable proxy rotation")
    parser.add_argument("--knn", action="store_true", help="Use K-Nearest Neighbors instead of XGBoost for ML predictions")
    
    args = parser.parse_args()
    
    # Set execution mode
    if args.test:
        set_execution_mode("test")

    # Set model type
    if args.knn:
        set_model_type(True)

    # Get configuration
    retail_config = get_weekly_retail_config()
    verify_ssl = retail_config.get('verify_ssl', False)

    from src.output_manager import get_output_manager
    output_manager = get_output_manager()
    output_manager.startup_banner()
    output_manager.info(f"Mode: {get_execution_mode()}")
    output_manager.info(f"ML Model: {get_model_type()}")


    # Handle safe individual model testing
    if args.test and args.model:
        make = get_make_from_model(args.model)
        if not make:
            output_manager.error(f"Unknown model: {args.model}")
            output_manager.info("Available models:")
            for make_name, models in TARGET_VEHICLES_BY_MAKE.items():
                output_manager.info(f"  {make_name}: {', '.join(models)}")
            return

        output_manager.info(f"🧪 SAFE TEST MODE: Processing {make.upper()} {args.model.upper()} individually")
        output_manager.info("   - No impact on weekly schedule")
        output_manager.info("   - Uses individual ML approach")
        output_manager.info("   - Minimal data usage")
        
        _process_individual_model(make, args.model, max_pages=None, verify_ssl=verify_ssl)
        return
    
    # Handle weekly training override (legacy flag - now uses daily model-specific training)
    if args.weekly_training:
        output_manager.warning("⚠️  --weekly-training flag is deprecated")
        output_manager.info("🔄 Using daily model-specific training instead (more accurate)")
        # Continue with normal processing using model-specific approach
    
    # Default targets - sample from configuration for production runs
    targets = [
        {"make": "bmw", "model": "3 series"},
        {"make": "audi", "model": "a3"},
        {"make": "ford", "model": "fiesta"},
    ]
    
    output_manager.info(f"🎯 Processing {len(targets)} make/model combinations")

    # Using individual model-specific training approach
    output_manager.info("🤖 Using model-specific ML training approach for optimal accuracy")
    
    # Process all targets
    for i, target in enumerate(targets):
        try:
            output_manager.info(f"\n{'='*60}")
            output_manager.info(f"PROCESSING {i+1}/{len(targets)}: {target['make'].upper()} {target['model'].upper()}")
            output_manager.info(f"{'='*60}")

            process_car_model(
                make=target["make"],
                model=target["model"],
                max_pages=None,  # Always use maximum pages
                verify_ssl=verify_ssl
            )

        except Exception as e:
            output_manager.error(f"Error processing {target['make']} {target['model']}: {e}")
            logger.error(f"Error processing {target['make']} {target['model']}: {e}")
            continue
    
    output_manager.info(f"\n{'='*60}")
    output_manager.info(f"PROCESSING COMPLETE")
    output_manager.info(f"{'='*60}")
    output_manager.success(f"Successfully processed {len(targets)} make/model combinations")

    # Show schedule status
    from config.config import get_schedule_status
    schedule_status = get_schedule_status()
    output_manager.info(f"\n📅 Schedule Status:")
    output_manager.info(f"   Weekly retail scraping due: {'Yes' if schedule_status['is_due'] else 'No'}")
    if schedule_status['days_since_last']:
        output_manager.info(f"   Days since last scraping: {schedule_status['days_since_last']}")
    if schedule_status['next_scheduled']:
        output_manager.info(f"   Next scheduled: {schedule_status['next_scheduled']}")
    


# ────────────────────────────────────────────────────────────
# Compatibility functions for scraper.py integration
# ────────────────────────────────────────────────────────────

def enhanced_analyse_listings(listings: List[Dict[str, Any]], cfg: Dict = None, verbose: bool = False, training_mode: bool = False, export_predictions: bool = False, force_retrain: bool = False) -> None:
    """
    Enhanced analysis using universal ML model for vehicle profit prediction.
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
    
    # Always show ML model processing section with clean format, even if no listings
    from src.output_manager import get_output_manager
    output_manager = get_output_manager()
    output_manager.ml_model_processing_start()

    # First, check if we have existing models for the main make/model combinations
    # Extract the most common make/model to check for existing models
    if listings:
        # Get most common make/model combination
        make_model_counts = {}
        for listing in listings:
            make = listing.get('make', '').lower()
            model = listing.get('model', '').lower()
            key = f"{make}_{model}"
            make_model_counts[key] = make_model_counts.get(key, 0) + 1

        if make_model_counts:
            # Check the most common make/model for existing model
            most_common_key = max(make_model_counts, key=make_model_counts.get)
            make, model = most_common_key.split('_', 1)
            model_age_days = get_model_age_days(make, model)

            if not force_retrain and model_age_days < 999 and model_age_days <= 14:
                # Fresh model exists - no training needed (unless force_retrain is True)
                output_manager.ml_model_status(f"Using existing model ({model_age_days} days old)")
                # Skip retail scraping and use individual processing
                _enhanced_analyse_listings_individual(listings, cfg, verbose, output_manager, training_mode, export_predictions, force_retrain)
                return
            elif model_age_days < 999:
                # Model exists but is old (or force_retrain is True)
                if force_retrain:
                    output_manager.ml_model_status(f"Force retraining model ({model_age_days} days old)")
                else:
                    output_manager.ml_model_status(f"Training new model ({model_age_days} days old)")
            else:
                # No existing model
                output_manager.ml_model_status("Training new model (no existing model)")

    # Check if retail scraping is needed (either due by schedule OR missing market_value data)
    listings_need_market_value = listings and len([l for l in listings if l.get('market_value') and l.get('market_value') > 0]) == 0
    retail_scraping_needed = is_retail_scraping_due() or listings_need_market_value

    if retail_scraping_needed:

        # Auto-trigger retail price scraping
        try:
            from services.price_scraper import batch_scrape_price_markers

            # Get listings for retail price scraping
            retail_listings = [listing for listing in listings if listing.get('url')]
            if retail_listings:
                output_manager._print("")

                # Scrape retail prices using batch scraper
                retail_data = batch_scrape_price_markers(
                    [listing['url'] for listing in retail_listings],
                    headless=True,
                    test_mode=False
                )

                if retail_data:
                    # batch_scrape_price_markers returns a dict: {url: result_data}
                    # Transform it to the format expected by the code
                    url_to_market_value = {url: result.get('market_value')
                                         for url, result in retail_data.items()
                                         if result.get('market_value') and result.get('market_value') > 0}

                    # Update listings with market values
                    updated_count = 0
                    for listing in listings:
                        url = listing.get('url', '')
                        if url in url_to_market_value:
                            listing['market_value'] = url_to_market_value[url]
                            updated_count += 1

                    output_manager.ml_model_status(f"Updated {updated_count} listings with market values")
                    # Update retail scraping schedule
                    from config.config import mark_retail_scraping_complete
                    mark_retail_scraping_complete(success=True, total_vehicles=updated_count, notes="Auto-triggered during ML analysis")
                else:
                    output_manager.ml_model_status("❌ Retail scraping failed - no data retrieved")
            else:
                output_manager.ml_model_status("❌ No listings with URLs available for retail scraping")
        except Exception as e:
            output_manager.ml_model_status(f"❌ Retail scraping error: {str(e)}")
            logger.error(f"Auto retail scraping failed: {e}")

    if not listings:
        output_manager.ml_model_status("No listings provided for analysis")
        output_manager.ml_model_complete()

        # Always show analysis section even with no listings
        output_manager.analysis_start()
        output_manager.analysis_results(
            avg_market_value=0,
            r2=0.0,
            mape=0.0,
            sample_size=0,
            deals_found=0
        )
        return


    # Use individual model processing directly (no universal model)
    _enhanced_analyse_listings_individual(listings, cfg, verbose, output_manager, training_mode, export_predictions, force_retrain)

    # Analysis results will be shown by the individual processing function
    return


def _enhanced_analyse_listings_individual(listings: List[Dict[str, Any]], cfg: Dict, verbose: bool = False, output_manager = None, training_mode: bool = False, export_predictions: bool = False, force_retrain: bool = False) -> None:
    """
    Fallback enhanced analysis using individual make/model processing.
    This is the original enhanced_analyse_listings logic.
    """
    if output_manager is None:
        from src.output_manager import get_output_manager
        output_manager = get_output_manager()

    # Step 1: Only enrich with retail price markers if in training mode
    if training_mode:
        # Extract make/model from first listing for context
        make = listings[0].get('make', '').lower() if listings else None
        model = listings[0].get('model', '').lower() if listings else None

        # Call retail price enrichment (use quiet_mode to suppress verbose banners)
        enriched_listings = enrich_with_price_markers(listings, make=make, model=model, quiet_mode=True)

        # Update the original listings with enriched data
        listings.clear()
        listings.extend(enriched_listings)
    # In normal (non-training) mode, skip retail price scraping entirely

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
    all_predictions = []  # Store all predictions for analysis stats

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
            enriched_dealer_count = 0
            for listing in dealer_listings:
                market_value = listing.get('market_value', 0)
                price_vs_market = listing.get('price_vs_market', 0)

                # Count as enriched if we have either direct market_value or price_vs_market
                if market_value != 0 or price_vs_market != 0:
                    enriched_dealer_count += 1

                dealer_data.append({
                    'make': listing.get('make', ''),
                    'model': listing.get('model', ''),
                    'year': listing.get('year', 2015),
                    'asking_price': listing.get('asking_price', listing.get('price', 0)),
                    'mileage': listing.get('mileage', 50000),
                    'market_value': market_value,  # Direct market value (most accurate)
                    'price_vs_market': price_vs_market,  # Fallback for training
                    'seller_type': 'Dealer',
                    'url': listing.get('url', ''),
                    'spec': listing.get('spec', '')
                })

            # Load or train model for this make/model
            trained_model, scaler, performance_metrics = load_or_train_model(make, model, dealer_data, force_retrain)

            if trained_model is None or scaler is None:
                output_manager.ml_model_status(f"❌ Unable to create model for {make} {model} - skipping analysis")
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
                    'spec': listing.get('spec', '')
                })

            # Analyze private listings with trained model
            predictions_df = predict_market_values(private_data, trained_model, scaler, performance_metrics)

            # Convert predictions back to listings format and add analysis results
            processed_count = 0
            reject_count = 0
            for i, (_, prediction) in enumerate(predictions_df.iterrows()):
                if i < len(private_listings):
                    original_listing = private_listings[i]
                    processed_count += 1

                    # Add ML prediction results
                    predicted_margin = prediction.get('predicted_profit_margin', 0)
                    predicted_market_value = prediction.get('predicted_market_value', 0)
                    potential_profit = prediction.get('potential_profit', 0)


                    original_listing['predicted_market_value'] = predicted_market_value
                    original_listing['predicted_margin_pct'] = predicted_margin
                    original_listing['predicted_profit'] = potential_profit

                    # Determine rating based on ML prediction
                    # ALL deals must have minimum £800 profit to be kept
                    if potential_profit >= cfg['min_cash_margin']:
                        if predicted_margin >= cfg['excellent_margin_pct']:
                            rating = "Excellent Deal"
                        elif predicted_margin >= cfg['good_margin_pct']:
                            rating = "Good Deal"
                        elif predicted_margin >= cfg['negotiation_margin_pct']:
                            rating = "Negotiation Target"
                        else:
                            rating = "Reject"
                            reject_count += 1
                    else:
                        rating = "Reject"
                        reject_count += 1

                    original_listing['enhanced_rating'] = rating


                    # Store all predictions for analysis stats (regardless of profitability)
                    all_predictions.append({
                        'url': original_listing.get('url', ''),
                        'make': original_listing.get('make', ''),
                        'model': original_listing.get('model', ''),
                        'year': original_listing.get('year', ''),
                        'mileage': original_listing.get('mileage', 0),
                        'asking_price': round(original_listing.get('asking_price', 0), 2),
                        'predicted_market_value': round(predicted_market_value, 2),
                        'predicted_margin': round(predicted_margin, 4),  # Keep 4dp for margin as it's a percentage
                        'potential_profit': round(potential_profit, 2),
                        'rating': rating
                    })

                    # Only keep profitable deals for final output
                    if rating != "Reject":
                        analyzed_listings.append(original_listing)

        
        except Exception as e:
            output_manager.ml_model_status(f"❌ Error analyzing {make} {model}: {e}")
            continue

    # Complete the ML model processing section
    output_manager.ml_model_complete()

    # Replace original listings with analyzed results
    listings.clear()
    listings.extend(analyzed_listings)

    # Show analysis results based on ALL predictions (not just profitable ones)
    output_manager.analysis_start()
    if all_predictions:
        # Calculate stats from all predictions made
        total_predictions = len(all_predictions)
        avg_market_value = sum(p['predicted_market_value'] for p in all_predictions) / total_predictions
        avg_margin = sum(p['predicted_margin'] for p in all_predictions) / total_predictions
        deals_found = len(analyzed_listings)  # Number of deals that passed profitability filter


        # Show analysis results using OutputManager with real performance metrics
        # Get average performance metrics from all models used
        avg_r2 = 0.85  # Default fallback
        avg_mape = 10.0  # Default fallback

        # Try to get real metrics from model performance data
        models_used = set()
        total_r2 = 0
        total_mape = 0
        metrics_count = 0

        # Collect metrics from all predictions made
        for prediction in all_predictions:
            make_model_key = f"{prediction.get('make', 'unknown')}_{prediction.get('model', 'unknown')}"
            if make_model_key not in models_used:
                models_used.add(make_model_key)
                # Try to load metrics for this model
                try:
                    from services.model_specific_trainer import load_model_specific
                    _, _, performance_metrics = load_model_specific(prediction.get('make', ''), prediction.get('model', ''))
                    if performance_metrics:
                        total_r2 += performance_metrics.get('r2', 0.85)
                        total_mape += performance_metrics.get('mape', 10.0)
                        metrics_count += 1
                except Exception:
                    pass  # Use defaults

        # Calculate average metrics if we have any
        if metrics_count > 0:
            avg_r2 = total_r2 / metrics_count
            avg_mape = total_mape / metrics_count

        output_manager.analysis_results(
            avg_market_value=avg_market_value,
            r2=avg_r2,
            mape=avg_mape,
            sample_size=total_predictions,
            deals_found=deals_found
        )
    else:
        # Show analysis results even if no predictions made
        output_manager.analysis_results(
            avg_market_value=0,
            r2=0.0,
            mape=0.0,
            sample_size=0,
            deals_found=0
        )

    # Export predictions if requested
    if export_predictions and all_predictions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(DATA_OUTPUTS_PATH, f"{timestamp}_ml_predictions_export.json")
        try:
            import json
            with open(predictions_file, 'w') as f:
                json.dump(all_predictions, f, indent=2, default=str)
            output_manager.ml_model_status(f"📊 Exported {len(all_predictions)} ML predictions to {predictions_file}")
        except Exception as e:
            output_manager.ml_model_status(f"❌ Error exporting predictions: {e}")

    # Save data in test mode when called from scraper (keep logger for file operations)
    global EXECUTION_MODE
    if EXECUTION_MODE == "test" and analyzed_listings:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deals_file = os.path.join(DATA_OUTPUTS_PATH, f"{timestamp}_scraper_individual_analyzed_deals.json")
        try:
            import json
            with open(deals_file, 'w') as f:
                json.dump(analyzed_listings, f, indent=2, default=str)
            logger.info(f"💾 Saved {len(analyzed_listings)} individual analyzed deals to {deals_file}")
        except Exception as e:
            logger.error(f"❌ Error saving individual analyzed deals: {e}")


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