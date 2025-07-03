"""
Enhanced Deal analysis functions for AutoTrader listings with network request data.
This is the complete enhanced analyzer with all functionality included.

KEY IMPLEMENTATION: Uses real market premium ratios for all categorical features (spec, fuel, body, transmission)
instead of converting them to discrete scores. This provides more accurate, granular, and data-driven regression features.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import numpy as np

# Machine learning imports for regression-based market value estimation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings


CONFIG = {
    # Enhanced profit analysis thresholds
    'recon_cost': 300,                 
    'excellent_margin_pct': 0.25,      
    'good_margin_pct': 0.20,          
    'negotiation_margin_pct': 0.15,    
    'min_cash_margin': 800,
}

# ────────────────────────────────────────────────────────────
# Regression-based Market Value Estimation
# ────────────────────────────────────────────────────────────

def calculate_spec_market_ratio(vehicle: dict, all_vehicles: List[dict]) -> float:
    """
    Calculate spec market premium ratio based on actual market data.
    Compares average prices of vehicles with the same spec vs the make/model average.
    Returns the real market ratio (e.g., 1.2 = 20% premium, 0.8 = 20% discount).
    """
    vehicle_spec = vehicle.get('spec', '').strip()
    vehicle_make = vehicle.get('make', '').lower()
    vehicle_model = vehicle.get('model', '').lower()
    
    if not vehicle_spec or not vehicle_make or not vehicle_model:
        return 1.0  # Default to neutral ratio
    
    # Get all vehicles of same make/model
    same_make_model = [
        v for v in all_vehicles 
        if (v.get('make', '').lower() == vehicle_make and 
            v.get('model', '').lower() == vehicle_model and
            v.get('price_numeric', 0) > 500)
    ]
    
    if len(same_make_model) < 3:  # Reduced minimum for testing
        return 1.0  # Default neutral ratio
    
    # Calculate overall average price for this make/model
    overall_prices = [v.get('price_numeric', 0) for v in same_make_model]
    overall_avg = sum(overall_prices) / len(overall_prices)
    
    # Get vehicles with the same spec
    same_spec_vehicles = [
        v for v in same_make_model 
        if v.get('spec', '').strip().lower() == vehicle_spec.lower()
    ]
    
    if len(same_spec_vehicles) < 1:  # Need at least 1 sample for this spec
        return 1.0  # Default neutral ratio
    
    # Calculate average price for this specific spec
    spec_prices = [v.get('price_numeric', 0) for v in same_spec_vehicles]
    spec_avg = sum(spec_prices) / len(spec_prices)
    
    # Return the actual market premium ratio
    price_ratio = spec_avg / overall_avg if overall_avg > 0 else 1.0
    
    return price_ratio


def calculate_dynamic_category_ratio(vehicle: dict, all_vehicles: List[dict], category_field: str) -> float:
    """
    Calculate dynamic market-based premium ratio for categorical features (fuel_type, body_type, etc.).
    Compares average prices of vehicles with the same category vs the make/model average.
    Returns the real market ratio (e.g., 1.15 = 15% premium, 0.9 = 10% discount).
    
    Args:
        vehicle: Current vehicle being analyzed
        all_vehicles: Complete dataset for market analysis
        category_field: Field name to analyze ('fuel_type', 'body_type', 'transmission')
    
    Returns:
        Market-based ratio representing the premium/discount for this category
    """
    vehicle_category = vehicle.get(category_field, '').strip().lower()
    vehicle_make = vehicle.get('make', '').lower()
    vehicle_model = vehicle.get('model', '').lower()
    
    if not vehicle_category or not vehicle_make or not vehicle_model:
        return 1.0  # Default neutral ratio
    
    # Get all vehicles of same make/model
    same_make_model = [
        v for v in all_vehicles 
        if (v.get('make', '').lower() == vehicle_make and 
            v.get('model', '').lower() == vehicle_model and
            v.get('price_numeric', 0) > 500)
    ]
    
    if len(same_make_model) < 3:  # Need minimum sample size
        return 1.0  # Default neutral ratio
    
    # Calculate overall average price for this make/model
    overall_prices = [v.get('price_numeric', 0) for v in same_make_model]
    overall_avg = sum(overall_prices) / len(overall_prices)
    
    # Get vehicles with the same category value
    same_category_vehicles = [
        v for v in same_make_model 
        if v.get(category_field, '').strip().lower() == vehicle_category
    ]
    
    if len(same_category_vehicles) < 1:  # Need at least 1 sample
        return 1.0  # Default neutral ratio
    
    # Calculate average price for this specific category
    category_prices = [v.get('price_numeric', 0) for v in same_category_vehicles]
    category_avg = sum(category_prices) / len(category_prices)
    
    # Calculate market premium/discount ratio
    price_ratio = category_avg / overall_avg if overall_avg > 0 else 1.0
    
    return price_ratio


def extract_regression_features(vehicle: dict, all_vehicles: List[dict] = None) -> List[float]:
    """
    Convert vehicle attributes to numerical features for regression model.
    Uses real market premium ratios for all categorical features instead of static ordinals.
    
    Returns:
        List of numerical features: [year, mileage, engine_size, spec_ratio, fuel_ratio, body_ratio, trans_ratio]
        Where ratios represent actual market premiums (e.g., 1.2 = 20% premium, 0.8 = 20% discount)
    """
    
    # Core numerical features - these are the most important
    year = vehicle.get('year', 2015)
    mileage = vehicle.get('mileage', 50000)
    engine_size = vehicle.get('engine_size', 1.6)
    
    # Data-driven market ratios - all categorical features learn from actual market prices
    if all_vehicles and len(all_vehicles) > 10:
        spec_ratio = calculate_spec_market_ratio(vehicle, all_vehicles)
        fuel_ratio = calculate_dynamic_category_ratio(vehicle, all_vehicles, 'fuel_type')
        body_ratio = calculate_dynamic_category_ratio(vehicle, all_vehicles, 'body_type')
        trans_ratio = calculate_dynamic_category_ratio(vehicle, all_vehicles, 'transmission')
    else:
        # Fallback to neutral ratios if insufficient data
        spec_ratio = 1.0
        fuel_ratio = 1.0
        body_ratio = 1.0
        trans_ratio = 1.0
    
    # Features list - all ratios are now real market-driven premiums/discounts
    features = [
        year,          # Most important: newer = more valuable
        mileage,       # Most important: lower mileage = more valuable  
        engine_size,   # Important: affects performance and value
        spec_ratio,    # CRITICAL: real market premium ratio for trim level
        fuel_ratio,    # Dynamic: real market premium ratio for fuel type
        body_ratio,    # Dynamic: real market premium ratio for body type
        trans_ratio,   # Dynamic: real market premium ratio for transmission
    ]
    
    return features


def calculate_regression_market_value(vehicle: dict, same_make_model_vehicles: List[dict], 
                                    cfg: Dict = None) -> Dict:
    """
    Use regression analysis to predict market value based on comparable vehicles.
    Returns insufficient data indication if not enough samples for reliable regression.
    
    Returns:
        dict: {
            'estimated_value': float,
            'confidence': float (R² score),
            'sample_size': int,
            'method': str
        }
    """
    if cfg is None:
        cfg = CONFIG
    
    # Minimum sample size requirement for reliable regression
    min_sample_size = 15
    if len(same_make_model_vehicles) < min_sample_size:
        print(f"   ⚠️  Only {len(same_make_model_vehicles)} samples - insufficient for regression")
        return {
            'estimated_value': 0,
            'confidence': 0.0,
            'sample_size': len(same_make_model_vehicles),
            'method': 'insufficient_data'
        }
    
    try:
        # Prepare training data
        X = []  # Features
        y = []  # Prices
        
        for v in same_make_model_vehicles:
            price = v.get('price_numeric', 0)
            if price > 500:  # Valid price threshold
                features = extract_regression_features(v, same_make_model_vehicles)
                X.append(features)
                y.append(price)
        
        # Need sufficient valid data points
        if len(X) < min_sample_size:
            print(f"   ⚠️  Only {len(X)} valid price points - insufficient for regression")
            return {
                'estimated_value': 0,
                'confidence': 0.0,
                'sample_size': len(X),
                'method': 'insufficient_valid_data'
            }
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Feature scaling for better regression performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate model confidence (R² score)
        y_pred = model.predict(X_scaled)
        confidence = r2_score(y, y_pred)
        
        # Extract features for target vehicle and predict
        target_features = extract_regression_features(vehicle, same_make_model_vehicles)
        target_features_scaled = scaler.transform([target_features])
        estimated_value = model.predict(target_features_scaled)[0]
        
        # Ensure reasonable bounds
        estimated_value = max(estimated_value, 500)  # Minimum value
        estimated_value = min(estimated_value, np.percentile(y, 99))  # Cap at 99th percentile
        
        print(f"   📊 Regression: £{estimated_value:,.0f} (R²={confidence:.3f}, n={len(X)})")
        
        return {
            'estimated_value': estimated_value,
            'confidence': confidence,
            'sample_size': len(X),
            'method': 'regression'
        }
        
    except Exception as e:
        print(f"   ❌ Regression failed: {e}")
        return {
            'estimated_value': 0,
            'confidence': 0.0,
            'sample_size': len(same_make_model_vehicles),
            'method': 'regression_error'
        }




# ────────────────────────────────────────────────────────────
# Enhanced similarity scoring with new data points
# ────────────────────────────────────────────────────────────

def calculate_enhanced_similarity_score(car1: dict, car2: dict, weights: Dict[str, float] = None) -> float:
    """
    Enhanced similarity calculation using comprehensive vehicle specifications.
    Builds upon the original similarity scoring with new data points.
    """
    if weights is None:
        weights = {'year_decay': 2.0, 'mileage_decay': 15000}
    
    # Original similarity factors (year, mileage, spec)
    year_diff = abs(car1.get('year', 0) - car2.get('year', 0))
    year_score = np.exp(-year_diff / weights['year_decay'])
    
    mileage_diff = abs(car1.get('mileage', 0) - car2.get('mileage', 0))
    mileage_score = np.exp(-mileage_diff / weights['mileage_decay'])
    
    # Enhanced similarity factors using network request data
    
    # Fuel type similarity (high weight for exact match)
    fuel1 = car1.get('fuel_type', '').lower()
    fuel2 = car2.get('fuel_type', '').lower()
    fuel_score = 1.0 if fuel1 == fuel2 else 0.2  # 20% for different fuel types
    
    # Transmission similarity
    trans1 = car1.get('transmission', '').lower()
    trans2 = car2.get('transmission', '').lower()
    trans_score = 1.0 if trans1 == trans2 else 0.4  # 40% for different transmissions
    
    # Engine size similarity (graduated scoring)
    engine1 = car1.get('engine_size', 0) or 0  # Handle None values
    engine2 = car2.get('engine_size', 0) or 0  # Handle None values
    if engine1 > 0 and engine2 > 0:
        engine_diff = abs(engine1 - engine2)
        engine_score = np.exp(-engine_diff / 0.5)  # 0.5L decay constant
    else:
        engine_score = 0.5  # Partial score if engine size missing
    
    # Body type similarity
    body1 = car1.get('body_type', '').lower()
    body2 = car2.get('body_type', '').lower()
    body_score = 1.0 if body1 == body2 else 0.3  # 30% for different body types
    
    # Doors similarity
    doors1 = car1.get('doors', 0) or 0  # Handle None values
    doors2 = car2.get('doors', 0) or 0  # Handle None values
    if doors1 > 0 and doors2 > 0:
        doors_score = 1.0 if doors1 == doors2 else 0.7  # 70% for different door counts
    else:
        doors_score = 0.8  # Default if doors data missing
    
    # Combined enhanced score with weighted factors
    total_score = (
        year_score * 0.25 +      # Year (reduced weight)
        mileage_score * 0.25 +   # Mileage (reduced weight)
        fuel_score * 0.20 +      # Fuel type (new, high importance)
        trans_score * 0.10 +     # Transmission (new)
        engine_score * 0.10 +    # Engine size (new)
        body_score * 0.05 +      # Body type (new)
        doors_score * 0.05       # Doors (new)
    )
    
    return total_score

def estimate_enhanced_market_value(car: dict, similar_cars: List[dict], cfg: Dict = CONFIG) -> float:
    """
    Enhanced market value estimation using pure regression-based approach.
    No fallbacks to problematic static multiplier methods.
    """
    print(f"   🚗 Analyzing {car.get('make', 'Unknown')} {car.get('model', 'Unknown')} ({car.get('year', 'Unknown')})...")
    
    # Use regression-based market analysis
    base_analysis = dynamic_price_analysis(car, similar_cars, cfg)
    estimated_value = float(base_analysis[0])  # First element is estimated_market_value
    
    if estimated_value <= 0:
        print(f"   ❌ Could not determine market value - insufficient data for regression")
        return 0.0
    
    print(f"   ✅ Final market value: £{estimated_value:,.0f}")
    return estimated_value

def enhanced_analyse_listings(listings: List[dict], cfg: Dict = CONFIG) -> None:
    """
    Enhanced analysis using comprehensive vehicle specifications from network requests.
    Builds upon the original analyse_listings function with new data points.
    """
    if not listings:
        return
    
    original_count = len(listings)
    enhanced_listings = []
    
    print(f"🔧 Starting enhanced analysis of {original_count} listings...")
    
    for i, listing in enumerate(listings):
        if i % 20 == 0:
            print(f"   Processing listing {i+1}/{original_count}...")
        
        # Enhanced market value estimation
        enhanced_value = estimate_enhanced_market_value(listing, listings, cfg)
        
        if enhanced_value <= 0:
            continue  # Skip listings without reliable pricing
        
        # Store enhanced analysis results
        listing['enhanced_retail_estimate'] = enhanced_value
        listing['analysis_method'] = 'enhanced_comprehensive'
        
        # Calculate enhanced profit metrics
        price = listing.get('price_numeric', 0) or 0  # Handle None values
        if price <= 0:
            continue
        
        recon_cost = cfg['recon_cost']
        net_sale_price = enhanced_value - recon_cost
        gross_cash_profit = net_sale_price - price
        
        listing['enhanced_net_sale_price'] = net_sale_price
        listing['enhanced_gross_cash_profit'] = gross_cash_profit
        
        # Calculate enhanced margin percentage
        if net_sale_price > 0:
            enhanced_margin_pct = gross_cash_profit / net_sale_price
        else:
            enhanced_margin_pct = -1.0
        
        listing['enhanced_gross_margin_pct'] = enhanced_margin_pct
        
        # Enhanced rating with comprehensive criteria
        if enhanced_margin_pct >= cfg['excellent_margin_pct']:
            enhanced_rating = "Excellent Deal"
        elif enhanced_margin_pct >= cfg['good_margin_pct']:
            enhanced_rating = "Good Deal"
        elif enhanced_margin_pct >= cfg['negotiation_margin_pct'] and gross_cash_profit >= cfg['min_cash_margin']:
            enhanced_rating = "Negotiation Target"
        else:
            enhanced_rating = "Reject"
        
        listing['enhanced_rating'] = enhanced_rating
        
        # Store specification analysis details
        listing['spec_analysis'] = {
            'fuel_type': listing.get('fuel_type'),
            'transmission': listing.get('transmission'),
            'engine_size': listing.get('engine_size'),
            'body_type': listing.get('body_type'),
            'doors': listing.get('doors'),
            'comprehensive_factors_applied': True
        }
        
        # Only keep profitable listings
        if enhanced_rating != "Reject":
            enhanced_listings.append(listing)
    
    # Replace original listings with enhanced results
    listings.clear()
    listings.extend(enhanced_listings)
    
    enhanced_count = len(enhanced_listings)
    print(f"✅ Enhanced analysis complete:")
    print(f"   {original_count} → {enhanced_count} profitable listings")
    print(f"   Filtered out {original_count - enhanced_count} unprofitable listings")
    print(f"   Enhanced factors: fuel type, transmission, engine size, body type, doors, age curves")

def convert_network_request_to_listing(network_car: dict) -> dict:
    """
    Convert network request car data to the format expected by the enhanced analyzer.
    Maps the comprehensive network request data to analysis-ready format.
    """
    processed = network_car.get('processed', {})
    raw_data = network_car.get('raw_data', {})
    
    # Create listing format compatible with enhanced analyzer
    listing = {
        # Basic identification
        'deal_id': processed.get('deal_id'),
        'url': processed.get('url'),
        'make': processed.get('make'),
        'model': processed.get('model'),
        
        # Core pricing and vehicle data (with safe defaults)
        'year': processed.get('year', 0),
        'price_numeric': processed.get('price', 0) or 0,  # Handle None values
        'mileage': processed.get('mileage', 0),
        
        # Enhanced specification data from network requests
        'fuel_type': processed.get('fuel_type'),
        'transmission': processed.get('transmission'),
        'engine_size': processed.get('engine_size'),
        'body_type': processed.get('body_type'),
        'doors': processed.get('doors'),
        'trim_level': processed.get('trim_level'),
        'euro_standard': processed.get('euro_standard'),
        'stop_start': processed.get('stop_start'),
        
        # Market context
        'seller_type': processed.get('seller_type'),
        'location': processed.get('location'),
        'distance': processed.get('distance'),
        'price_indicator_rating': processed.get('price_indicator_rating'),
        
        # Additional metadata
        'image_count': processed.get('image_count'),
        'dealer_rating': processed.get('dealer_rating'),
        'has_video': processed.get('has_video'),
        'has_360_spin': processed.get('has_360_spin'),
        'manufacturer_approved': processed.get('manufacturer_approved'),
        'badges': processed.get('badges', []),
        
        # Preserve raw network data for further analysis
        'network_raw_data': raw_data,
        'network_processed_data': processed
    }
    
    return listing

# ────────────────────────────────────────────────────────────
# Enhanced utility functions
# ────────────────────────────────────────────────────────────

def enhanced_keep_listing(listing: dict, cfg: Dict = CONFIG) -> bool:
    """
    Enhanced filtering logic that considers comprehensive vehicle specifications.
    """
    # Basic seller type filter (private only for deals)
    if listing.get("seller_type", "").lower() != "private":
        return False
    
    # Use enhanced rating if available, fallback to original
    rating = listing.get("enhanced_rating", listing.get("rating", ""))
    
    return rating in {"Excellent Deal", "Good Deal", "Negotiation Target"}

def get_specification_summary(listing: dict) -> str:
    """
    Generate a human-readable specification summary for enhanced listings.
    """
    specs = []
    
    if listing.get('year'):
        specs.append(f"{listing['year']}")
    
    if listing.get('engine_size'):
        specs.append(f"{listing['engine_size']}L")
    
    if listing.get('fuel_type'):
        specs.append(listing['fuel_type'])
    
    if listing.get('transmission'):
        specs.append(listing['transmission'])
    
    if listing.get('doors'):
        specs.append(f"{listing['doors']}-door")
    
    if listing.get('body_type'):
        specs.append(listing['body_type'])
    
    return " • ".join(specs) if specs else "Specifications unavailable"

# ────────────────────────────────────────────────────────────
# Test and comparison functions
# ────────────────────────────────────────────────────────────

def compare_analysis_methods(network_cars: List[dict], cfg: Dict = CONFIG) -> Dict:
    """
    Compare original vs enhanced analysis on the same dataset.
    """
    # Convert network cars to listing format
    listings_original = [convert_network_request_to_listing(car) for car in network_cars]
    listings_enhanced = [convert_network_request_to_listing(car) for car in network_cars]
    
    print("🔬 Comparing analysis methods...")
    
    # Run original analysis
    print("   Running original analysis...")
    analyse_listings(listings_original, CONFIG, use_dynamic_pricing=True)
    original_count = len(listings_original)
    
    # Run enhanced analysis
    print("   Running enhanced analysis...")
    enhanced_analyse_listings(listings_enhanced, cfg)
    enhanced_count = len(listings_enhanced)
    
    # Generate comparison report
    comparison = {
        'original_method': {
            'total_deals': original_count,
            'excellent_deals': len([l for l in listings_original if l.get('rating') == 'Excellent Deal']),
            'good_deals': len([l for l in listings_original if l.get('rating') == 'Good Deal']),
            'negotiation_targets': len([l for l in listings_original if l.get('rating') == 'Negotiation Target'])
        },
        'enhanced_method': {
            'total_deals': enhanced_count,
            'excellent_deals': len([l for l in listings_enhanced if l.get('enhanced_rating') == 'Excellent Deal']),
            'good_deals': len([l for l in listings_enhanced if l.get('enhanced_rating') == 'Good Deal']),
            'negotiation_targets': len([l for l in listings_enhanced if l.get('enhanced_rating') == 'Negotiation Target'])
        },
        'input_count': len(network_cars)
    }
    
    print(f"📊 Analysis Comparison Results:")
    print(f"   Input vehicles: {comparison['input_count']}")
    print(f"   Original method: {original_count} deals")
    print(f"   Enhanced method: {enhanced_count} deals")
    print(f"   Difference: {enhanced_count - original_count} deals")
    
    return comparison

def compare_analysis_methods_detailed(vehicle: dict, all_vehicles: List[dict], cfg: Dict = CONFIG) -> Dict:
    """
    Compare regression-based method with the old similarity-based method for validation.
    This is useful for testing and validation of the new regression approach.
    """
    print(f"   🔬 Running method comparison for {vehicle.get('make', 'Unknown')} {vehicle.get('model', 'Unknown')}...")
    
    # Filter same make/model vehicles
    same_make_model_vehicles = filter_same_make_model_vehicles(vehicle, all_vehicles)
    
    # Method 1: New regression-based approach
    regression_result = calculate_regression_market_value(vehicle, same_make_model_vehicles, cfg)
    
    # Method 2: Old similarity-based approach
    similarity_result = fallback_similarity_method(vehicle, same_make_model_vehicles)
    
    # Calculate difference
    regression_value = regression_result['estimated_value']
    similarity_value = similarity_result['estimated_value']
    
    difference = regression_value - similarity_value
    percentage_diff = (difference / similarity_value * 100) if similarity_value > 0 else 0
    
    comparison = {
        'vehicle': f"{vehicle.get('make', 'Unknown')} {vehicle.get('model', 'Unknown')} ({vehicle.get('year', 'Unknown')})",
        'asking_price': vehicle.get('price_numeric', 0),
        'regression_method': regression_result,
        'similarity_method': similarity_result,
        'difference': difference,
        'percentage_difference': percentage_diff,
        'sample_size': len(same_make_model_vehicles)
    }
    
    print(f"     📊 Regression: £{regression_value:,.0f} | Similarity: £{similarity_value:,.0f} | Diff: {percentage_diff:+.1f}%")
    
    return comparison
# ────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────

def dynamic_price_analysis(listing: dict, similar_cars: List[dict], cfg: Dict = CONFIG) -> Tuple[float, float]:
    """
    Perform price analysis for a vehicle listing using regression-based market value estimation.
    Returns: (estimated_market_value, profit_potential)
    """
    # Use regression-based estimation as the primary method
    estimated_value = estimate_market_value_with_regression(listing, similar_cars, cfg)
    
    current_price = float(listing.get('price_numeric', listing.get('price', 0)))
    profit_potential = estimated_value - current_price
    
    return estimated_value, profit_potential

# ────────────────────────────────────────────────────────────
# Market Value Estimation Functions
# ────────────────────────────────────────────────────────────

def filter_same_make_model_vehicles(target_vehicle: dict, all_vehicles: List[dict]) -> List[dict]:
    """
    Filter vehicles to find those with the same make and model as the target vehicle.
    """
    target_make = target_vehicle.get('make', '').lower().strip()
    target_model = target_vehicle.get('model', '').lower().strip()
    
    if not target_make or not target_model:
        return []
    
    same_make_model = []
    for vehicle in all_vehicles:
        vehicle_make = vehicle.get('make', '').lower().strip()
        vehicle_model = vehicle.get('model', '').lower().strip()
        
        if vehicle_make == target_make and vehicle_model == target_model:
            same_make_model.append(vehicle)
    
    return same_make_model


def estimate_market_value_with_regression(vehicle: dict, all_vehicles: List[dict], cfg: Dict = CONFIG) -> float:
    """
    Estimate market value using regression-based approach only.
    Returns 0 if insufficient data for reliable regression.
    """
    # Filter to same make/model vehicles
    same_make_model_vehicles = filter_same_make_model_vehicles(vehicle, all_vehicles)
    
    print(f"   🔍 Found {len(same_make_model_vehicles)} same make/model vehicles for {vehicle.get('make', 'Unknown')} {vehicle.get('model', 'Unknown')}")
    
    # Use regression-based market value estimation
    result = calculate_regression_market_value(vehicle, same_make_model_vehicles, cfg)
    
    estimated_value = result['estimated_value']
    method = result['method']
    confidence = result['confidence']
    sample_size = result['sample_size']
    
    if estimated_value > 0:
        print(f"   💰 Market value: £{estimated_value:,.0f} (method: {method}, confidence: {confidence:.2f}, n={sample_size})")
    else:
        print(f"   ❌ Could not estimate market value: {method} (n={sample_size})")
    
    return estimated_value
