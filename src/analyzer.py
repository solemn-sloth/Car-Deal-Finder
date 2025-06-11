"""
Deal analysis functions for AutoTrader listings.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import numpy as np

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
CONFIG = dict(
    recon_cost=300,            # £ per car you normally spend on prep
    indie_factor=0.95,         # where you sit in the dealer price range
    bucket_low_pct=20,         # take 20th percentile of (trimmed) dealer ads
    bucket_trim_pct=10,        # drop top / bottom 10 % before percentile
    excellent_margin_pct=0.25, # ≥25% after recon ⇒ Excellent Deal
    good_margin_pct=0.20,      # ≥20% after recon ⇒ Good Deal
    negotiation_margin_pct=0.15,  # ≥15% ⇒ Negotiation Target
    min_cash_margin=500       # absolute profit floor (after recon)
)

# ────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────
def mileage_band(miles: int, step: int = 10_000, cap: int = 100_000) -> str:
    """
    Convert mileage to a standardized band string.
    E.g., 45000 → "40k-50k", 120000 → "100k+"
    """
    if miles >= cap:
        return f"{cap//1000}k+"
    band_start = (miles // step) * step
    band_end = band_start + step
    return f"{band_start//1000}k-{band_end//1000}k"

def spec_key(spec: str) -> str:
    """
    Normalize car spec string for grouping.
    """
    return re.sub(r'\W+', ' ', spec.lower().strip())

def build_bucket_key(v: dict) -> Tuple[int, str, str]:
    """
    Build a grouping key for similar vehicles.
    Returns (year, spec_normalized, mileage_band)
    """
    year = v.get('year', 0)
    spec = spec_key(v.get('spec', ''))
    mileage = mileage_band(v.get('mileage', 0))
    return (year, spec, mileage)

def robust_retail_price(prices: List[float],
                        low_pct: int,
                        trim_pct: int,
                        indie_factor: float) -> float:
    """
    Trim extreme prices, then return the chosen lower-percentile price adjusted
    for an independent dealer's position in the market.
    """
    if not prices:
        raise ValueError("Price list empty.")
    if len(prices) < 4:
        base = float(np.median(prices))
    else:
        lo, hi = np.percentile(prices, [trim_pct, 100 - trim_pct])
        core = [p for p in prices if lo <= p <= hi]
        base = float(np.percentile(core, low_pct))
    return base * indie_factor

# ────────────────────────────────────────────────────────────
# Dynamic similarity-based pricing with data-driven optimization
# ────────────────────────────────────────────────────────────
def analyze_optimal_similarity_weights(listings: List[dict]) -> Dict[str, float]:
    """
    Analyze actual data to determine optimal similarity decay rates.
    Returns optimal weights for year_decay, mileage_decay based on price correlation.
    """
    dealer_listings = [l for l in listings 
                      if l.get('seller_type', '').lower() == 'dealer' 
                      and l.get('price_numeric', 0) > 0
                      and l.get('year', 0) > 0
                      and l.get('mileage', 0) >= 0]
    
    if len(dealer_listings) < 20:
        # Use conservative defaults for small datasets
        return {'year_decay': 2.0, 'mileage_decay': 15000}
    
    # Test different decay rates and see which correlates best with price
    year_decays = [1.0, 1.5, 2.0, 3.0, 4.0]
    mileage_decays = [10000, 15000, 20000, 25000, 30000]
    
    best_correlation = -1
    best_params = {'year_decay': 2.0, 'mileage_decay': 15000}
    
    # Sample pairs to avoid O(n²) computation
    import random
    sample_pairs = []
    random.seed(42)  # For reproducible results
    for i in range(min(50, len(dealer_listings))):
        for j in range(i+1, min(i+20, len(dealer_listings))):
            sample_pairs.append((dealer_listings[i], dealer_listings[j]))
    
    for year_decay in year_decays:
        for mileage_decay in mileage_decays:
            similarities = []
            price_similarities = []
            
            for car1, car2 in sample_pairs:
                # Calculate similarity with these parameters
                year_diff = abs(car1['year'] - car2['year'])
                mileage_diff = abs(car1['mileage'] - car2['mileage'])
                
                year_score = np.exp(-year_diff / year_decay)
                mileage_score = np.exp(-mileage_diff / mileage_decay)
                similarity = (year_score * 0.5 + mileage_score * 0.5)
                
                price_diff = abs(car1['price_numeric'] - car2['price_numeric'])
                price_similarity = 1 / (1 + price_diff / 1000)  # Convert to 0-1 scale
                
                similarities.append(similarity)
                price_similarities.append(price_similarity)
            
            # Calculate correlation between feature similarity and price similarity
            if len(similarities) > 5:
                correlation = np.corrcoef(similarities, price_similarities)[0, 1]
                if not np.isnan(correlation) and correlation > best_correlation:
                    best_correlation = correlation
                    best_params = {'year_decay': year_decay, 'mileage_decay': mileage_decay}
    
    return best_params

def calculate_similarity_score(car1: dict, car2: dict, weights: Dict[str, float] = None) -> float:
    """
    Calculate how similar two cars are (0-1 score).
    Higher = more similar = more weight in price calculation.
    """
    if weights is None:
        weights = {'year_decay': 2.0, 'mileage_decay': 15000}
    
    # Year similarity (using data-driven decay)
    year_diff = abs(car1['year'] - car2['year'])
    year_score = np.exp(-year_diff / weights['year_decay'])
    
    # Mileage similarity (using data-driven decay)
    mileage_diff = abs(car1['mileage'] - car2['mileage'])
    mileage_score = np.exp(-mileage_diff / weights['mileage_decay'])
    
    # Spec similarity (binary for now, could be enhanced)
    spec1 = spec_key(car1.get('spec', ''))
    spec2 = spec_key(car2.get('spec', ''))
    spec_score = 1.0 if spec1 == spec2 else 0.3  # 30% weight for different specs
    
    # Combined score (weighted average)
    total_score = (year_score * 0.3 + mileage_score * 0.4 + spec_score * 0.3)
    
    return total_score

def dynamic_price_analysis_by_year(target: dict, all_listings: List[dict], cfg: Dict) -> dict:
    """
    Analyze price focusing on same-year vehicles first, then expanding if needed.
    Uses year-based batching for more accurate comparisons.
    """
    target_year = target.get('year', 0)
    if target_year == 0:
        return {'price': 0, 'method': 'no_year_data', 'sample_size': 0}
    
    # Get dealer listings
    dealer_listings = [l for l in all_listings 
                      if l.get('seller_type', '').lower() == 'dealer' 
                      and l.get('price_numeric', 0) > 0
                      and l.get('year', 0) > 0]
    
    if not dealer_listings:
        return {'price': 0, 'method': 'no_dealer_data', 'sample_size': 0}
    
    # Optimize similarity weights for this dataset
    similarity_weights = analyze_optimal_similarity_weights(dealer_listings)
    
    # Year-based batch analysis - expand gradually if needed
    year_ranges = [
        [target_year],                    # Exact year first
        [target_year-1, target_year, target_year+1],  # ±1 year
        [target_year-2, target_year-1, target_year, target_year+1, target_year+2],  # ±2 years
        list(range(target_year-3, target_year+4))  # ±3 years (fallback)
    ]
    
    for i, year_range in enumerate(year_ranges):
        # Filter to this year range
        year_filtered = [l for l in dealer_listings if l.get('year', 0) in year_range]
        
        if len(year_filtered) < 3:
            continue  # Not enough data, try wider range
        
        # Calculate similarity scores within this year range
        similarities = []
        for listing in year_filtered:
            score = calculate_similarity_score(target, listing, similarity_weights)
            if score > 0.1:  # Minimum similarity threshold
                similarities.append({
                    'listing': listing,
                    'score': score,
                    'price': listing['price_numeric'],
                    'year': listing['year'],
                    'mileage': listing['mileage']
                })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        if len(similarities) < 3:
            continue  # Try wider year range
        
        # Method selection based on data quality and quantity
        method_used = f"year_batch_{i+1}_{'exact' if i == 0 else f'plus_minus_{i}'}"
        
        # Method 1: Weighted average (for strong similarity or small datasets)
        top_similar = similarities[:min(20, len(similarities))]
        
        if len(similarities) <= 15 or sum(s['score'] for s in top_similar[:5]) > 3.5:
            # Use weighted average
            total_weight = sum(s['score'] for s in top_similar)
            weighted_price = sum(s['price'] * s['score'] for s in top_similar) / total_weight
            
            # Apply conservative adjustment
            conservative_price = weighted_price * 0.95 * cfg['indie_factor']
            
            return {
                'price': conservative_price,
                'method': f'{method_used}_weighted_avg',
                'sample_size': len(top_similar),
                'year_range': year_range,
                'avg_similarity': total_weight / len(top_similar),
                'similarity_weights': similarity_weights
            }
        
        # Method 2: Regression with normalized features (for larger datasets)
        if len(similarities) >= 10:
            try:
                # Build regression using similar cars with normalized features
                X = []
                y = []
                weights = []
                
                # Collect data for normalization
                years = [sim['listing']['year'] for sim in similarities[:30]]
                mileages = [sim['listing']['mileage'] for sim in similarities[:30]]
                
                # Calculate normalization parameters
                year_mean, year_std = np.mean(years), np.std(years)
                mileage_mean, mileage_std = np.mean(mileages), max(np.std(mileages), 1000)  # Avoid div by zero
                
                # Build normalized feature matrix
                for sim in similarities[:30]:
                    listing = sim['listing']
                    # Normalize features to similar scales
                    year_norm = (listing['year'] - year_mean) / year_std if year_std > 0 else 0
                    mileage_norm = (listing['mileage'] - mileage_mean) / mileage_std
                    
                    X.append([
                        year_norm,
                        mileage_norm,
                        year_norm * mileage_norm  # Interaction term
                    ])
                    y.append(listing['price_numeric'])
                    weights.append(sim['score'])  # Weight by similarity
                
                X = np.array(X)
                y = np.array(y)
                weights = np.array(weights)
                
                # Add intercept
                X = np.column_stack([np.ones(len(X)), X])
                
                # Weighted least squares with numerical stability
                W = np.diag(weights)
                try:
                    XTW = X.T @ W
                    coeffs = np.linalg.solve(XTW @ X, XTW @ y)
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse for numerical stability
                    coeffs = np.linalg.pinv(XTW @ X) @ (XTW @ y)
                
                # Predict for target (normalize target features the same way)
                target_year_norm = (target['year'] - year_mean) / year_std if year_std > 0 else 0
                target_mileage_norm = (target['mileage'] - mileage_mean) / mileage_std
                
                target_features = np.array([
                    1,
                    target_year_norm,
                    target_mileage_norm,
                    target_year_norm * target_mileage_norm
                ])
                
                predicted = target_features @ coeffs
                
                # Calculate conservative adjustment based on residuals
                residuals = y - (X @ coeffs)
                weighted_residuals = residuals * np.sqrt(weights)
                
                # Use lower percentile for conservative estimate
                negative_residuals = weighted_residuals[weighted_residuals < 0]
                if len(negative_residuals) > 0:
                    conservative_adjustment = np.percentile(negative_residuals, 20)
                else:
                    conservative_adjustment = -np.std(residuals) * 0.5  # Half std dev down
                
                final_price = (predicted + conservative_adjustment) * cfg['indie_factor']
                
                # Sanity check against data range
                price_range = (np.percentile(y, 10), np.percentile(y, 90))
                if price_range[0] * 0.7 <= final_price <= price_range[1] * 1.3:
                    return {
                        'price': final_price,
                        'method': f'{method_used}_regression',
                        'sample_size': len(similarities),
                        'year_range': year_range,
                        'effective_sample': sum(weights),
                        'similarity_weights': similarity_weights,
                        'r_squared': 1 - np.var(residuals) / np.var(y) if np.var(y) > 0 else 0
                    }
            except Exception as e:
                # Log error but continue to fallback method
                pass
        
        # Method 3: Fallback to percentile of similar cars
        similar_prices = [s['price'] for s in similarities[:15]]
        conservative_price = np.percentile(similar_prices, cfg['bucket_low_pct']) * cfg['indie_factor']
        
        return {
            'price': conservative_price,
            'method': f'{method_used}_percentile',
            'sample_size': len(similar_prices),
            'year_range': year_range,
            'similarity_weights': similarity_weights
        }
    
    # If we get here, no year range had enough data
    return {'price': target.get('price_numeric', 0) * 1.15, 
            'method': 'insufficient_year_data', 
            'sample_size': 0}
def dynamic_price_analysis(target: dict, all_listings: List[dict], cfg: Dict) -> dict:
    """
    Dynamically analyze price based on similar cars without fixed buckets.
    Falls back to simpler method if year-based analysis fails.
    """
    # Try year-based analysis first (more accurate)
    year_result = dynamic_price_analysis_by_year(target, all_listings, cfg)
    
    # If year-based analysis succeeded, use it
    if year_result['price'] > 0 and year_result['method'] != 'insufficient_year_data':
        return year_result
    
    # Fallback to original similarity-based method
    dealer_listings = [l for l in all_listings 
                      if l.get('seller_type', '').lower() == 'dealer' 
                      and l.get('price_numeric', 0) > 0]
    
    if not dealer_listings:
        return {'price': 0, 'method': 'no_data', 'sample_size': 0}
    
    # Get optimal weights
    similarity_weights = analyze_optimal_similarity_weights(dealer_listings)
    
    # Calculate similarity scores for all dealer listings
    similarities = []
    for listing in dealer_listings:
        score = calculate_similarity_score(target, listing, similarity_weights)
        if score > 0.1:  # Minimum similarity threshold
            similarities.append({
                'listing': listing,
                'score': score,
                'price': listing['price_numeric'],
                'year': listing['year'],
                'mileage': listing['mileage']
            })
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x['score'], reverse=True)
    
    # Need minimum number of similar cars
    if len(similarities) < 3:
        return {'price': target.get('price_numeric', 0) * 1.15, 
                'method': 'insufficient_similar', 
                'sample_size': len(similarities)}
    
    # Method 1: Weighted average of similar cars (if < 15 very similar cars)
    top_similar = similarities[:20]  # Consider top 20 most similar
    
    if sum(s['score'] for s in top_similar[:5]) > 3.5:  # Strong similarity in top 5
        # Use weighted average
        total_weight = sum(s['score'] for s in top_similar)
        weighted_price = sum(s['price'] * s['score'] for s in top_similar) / total_weight
        
        # Apply conservative adjustment
        conservative_price = weighted_price * 0.95 * cfg['indie_factor']
        
        return {
            'price': conservative_price,
            'method': 'weighted_similar',
            'sample_size': len(top_similar),
            'avg_similarity': total_weight / len(top_similar),
            'similarity_weights': similarity_weights
        }
    
    # Method 2: Regression on similar cars (if enough data)
    if len(similarities) >= 10:
        try:
            # Build regression using similar cars with normalized features
            X = []
            y = []
            weights = []
            
            # Collect data for normalization
            years = [sim['listing']['year'] for sim in similarities[:30]]
            mileages = [sim['listing']['mileage'] for sim in similarities[:30]]
            
            # Calculate normalization parameters
            year_mean, year_std = np.mean(years), np.std(years)
            mileage_mean, mileage_std = np.mean(mileages), max(np.std(mileages), 1000)
            
            for sim in similarities[:30]:  # Use top 30
                listing = sim['listing']
                # Use normalized features
                year_norm = (listing['year'] - year_mean) / year_std if year_std > 0 else 0
                mileage_norm = (listing['mileage'] - mileage_mean) / mileage_std
                
                X.append([
                    year_norm,
                    mileage_norm,
                    year_norm * mileage_norm  # Interaction term
                ])
                y.append(listing['price_numeric'])
                weights.append(sim['score'])  # Weight by similarity
            
            X = np.array(X)
            y = np.array(y)
            weights = np.array(weights)
            
            # Add intercept
            X = np.column_stack([np.ones(len(X)), X])
            
            # Weighted least squares with numerical stability
            W = np.diag(weights)
            try:
                XTW = X.T @ W
                coeffs = np.linalg.solve(XTW @ X, XTW @ y)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.pinv(XTW @ X) @ (XTW @ y)
            
            # Predict for target (normalize target features the same way)
            target_year_norm = (target['year'] - year_mean) / year_std if year_std > 0 else 0
            target_mileage_norm = (target['mileage'] - mileage_mean) / mileage_std
            
            target_features = np.array([
                1,
                target_year_norm,
                target_mileage_norm,
                target_year_norm * target_mileage_norm
            ])
            
            predicted = target_features @ coeffs
            
            # Calculate prediction interval based on residuals
            residuals = y - (X @ coeffs)
            weighted_residuals = residuals * np.sqrt(weights)
            
            # Use lower percentile for conservative estimate
            negative_residuals = weighted_residuals[weighted_residuals < 0]
            if len(negative_residuals) > 0:
                conservative_adjustment = np.percentile(negative_residuals, 20)
            else:
                conservative_adjustment = -np.std(residuals) * 0.5
            
            final_price = (predicted + conservative_adjustment) * cfg['indie_factor']
            
            # Sanity check
            price_range = (np.percentile(y, 10), np.percentile(y, 90))
            if price_range[0] * 0.8 <= final_price <= price_range[1] * 1.2:
                return {
                    'price': final_price,
                    'method': 'similarity_regression',
                    'sample_size': len(similarities),
                    'effective_sample': sum(weights),
                    'similarity_weights': similarity_weights
                }
        except Exception as e:
            pass  # Fall through to next method
    
    # Method 3: Fallback to percentile of similar cars
    similar_prices = [s['price'] for s in similarities[:15]]
    conservative_price = np.percentile(similar_prices, cfg['bucket_low_pct']) * cfg['indie_factor']
    
    return {
        'price': conservative_price,
        'method': 'similar_percentile',
        'sample_size': len(similar_prices),
        'similarity_weights': similarity_weights
    }

# ────────────────────────────────────────────────────────────
# Main analysis routine
# ────────────────────────────────────────────────────────────
def analyse_listings(listings: List[dict],
                     cfg: Dict = CONFIG,
                     use_dynamic_pricing: bool = True) -> None:
    """
    Mutates each listing dictionary, adding:
        • bucket_key (if using buckets)
        • retail_estimate
        • net_sale_price
        • gross_cash_profit
        • gross_margin_pct
        • rating  (Excellent Deal / Good Deal / Negotiation Target / Reject)
        • pricing_method
        • comparison_sample_size
    
    Args:
        listings: List of vehicle dictionaries to analyze
        cfg: Configuration dictionary with analysis parameters
        use_dynamic_pricing: Whether to use dynamic similarity-based pricing
    """
    if not listings:
        return
    
    if use_dynamic_pricing:
        # Use dynamic similarity-based pricing
        for listing in listings:
            # Get dynamic price estimate
            price_result = dynamic_price_analysis(listing, listings, cfg)
            
            listing['retail_estimate'] = price_result['price']
            listing['pricing_method'] = price_result['method']
            listing['comparison_sample_size'] = price_result['sample_size']
            
            # Add similarity info if available
            if 'avg_similarity' in price_result:
                listing['avg_similarity'] = price_result['avg_similarity']
            if 'effective_sample' in price_result:
                listing['effective_sample'] = price_result['effective_sample']
    else:
        # Use original bucket-based approach
        # Step 1: Build comparison buckets
        buckets = {}
        for listing in listings:
            bucket_key = build_bucket_key(listing)
            listing['bucket_key'] = bucket_key
            
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(listing)
        
        # Step 2: Calculate retail estimates for each bucket
        bucket_estimates = {}
        for bucket_key, bucket_listings in buckets.items():
            # Get dealer prices (exclude private sales)
            dealer_prices = [
                v['price_numeric'] for v in bucket_listings
                if v.get('seller_type', '').lower() == 'dealer' and v.get('price_numeric', 0) > 0
            ]
            
            if dealer_prices:
                retail_estimate = robust_retail_price(
                    dealer_prices,
                    cfg['bucket_low_pct'],
                    cfg['bucket_trim_pct'],
                    cfg['indie_factor']
                )
                bucket_estimates[bucket_key] = {
                    'retail_estimate': retail_estimate,
                    'sample_size': len(dealer_prices)
                }
            else:
                bucket_estimates[bucket_key] = {
                    'retail_estimate': 0.0,
                    'sample_size': 0
                }
        
        # Step 3: Apply estimates to each listing
        for listing in listings:
            bucket_key = listing['bucket_key']
            estimates = bucket_estimates[bucket_key]
            
            listing['retail_estimate'] = estimates['retail_estimate']
            listing['comparison_sample_size'] = estimates['sample_size']
            listing['pricing_method'] = 'bucket'
    
    # Step 4: Calculate profit metrics and ratings for all listings
    for listing in listings:
        price = listing.get('price_numeric', 0)
        retail_estimate = listing.get('retail_estimate', 0)
        
        # Use retail estimate if available, otherwise fall back to price-based estimate
        if retail_estimate <= 0:
            listing['retail_estimate'] = price * 1.15  # Assume 15% markup potential
            retail_estimate = listing['retail_estimate']
        
        # Calculate profit metrics
        recon_cost = cfg['recon_cost']
        net_sale_price = retail_estimate - recon_cost
        gross_cash_profit = net_sale_price - price
        
        listing['net_sale_price'] = net_sale_price
        listing['gross_cash_profit'] = gross_cash_profit
        
        # Calculate margin percentage
        if net_sale_price > 0:
            gross_margin_pct = gross_cash_profit / net_sale_price
        else:
            gross_margin_pct = -1.0  # Invalid
        
        listing['gross_margin_pct'] = gross_margin_pct
        
        # New rating logic with three tiers
        if gross_margin_pct >= cfg['excellent_margin_pct']:
            rating = "Excellent Deal"
        elif gross_margin_pct >= cfg['good_margin_pct']:
            rating = "Good Deal"
        elif gross_margin_pct >= cfg['negotiation_margin_pct'] and gross_cash_profit >= cfg['min_cash_margin']:
            rating = "Negotiation Target"
        else:
            rating = "Reject"
        
        listing['rating'] = rating

def keep_listing(v: dict,
                 cfg: Dict = CONFIG) -> bool:
    """
    Return True if the listing should be shown to you.
    """
    if v["seller_type"].lower() != "private":
        return False
    # Distance filtering removed - now shows all private listings regardless of distance
    return v.get("rating") in {"Excellent Deal", "Good Deal", "Negotiation Target"}

def comparison_url(v: dict) -> str:
    """
    Generate AutoTrader comparison URL for a vehicle.
    """
    make = v.get('make', '').replace(' ', '%20')
    model = v.get('model', '').replace(' ', '%20')
    year = v.get('year', '')
    mileage = v.get('mileage', 0)
    
    mileage_low = max(0, mileage - 20000)
    mileage_high = mileage + 20000
    year_low = max(2010, year - 2)
    year_high = year + 2