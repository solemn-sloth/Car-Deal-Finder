"""
Data adapter to convert network request API data to analyzer-compatible format.
Bridges the gap between the GraphQL API response format and the existing analyzer expectations.
"""
import re
from typing import Dict, List, Optional

class NetworkDataAdapter:
    """
    Converts vehicle data from network request API format to analyzer-compatible format.
    """
    
    @staticmethod
    def convert_vehicle(api_vehicle: Dict) -> Dict:
        """
        Convert a single vehicle from API format to analyzer format.
        
        Args:
            api_vehicle: Vehicle data from GraphQL API
            
        Returns:
            Dict: Vehicle data in analyzer-compatible format
            
        Raises:
            ValueError: If vehicle data is invalid/empty
        """
        # Handle None input
        if api_vehicle is None:
            raise ValueError("None vehicle object")
            
        # Skip completely empty objects that the API sometimes returns
        if not api_vehicle or api_vehicle == {}:
            raise ValueError("Empty vehicle object")
        
        # Check for required minimum fields
        advert_id = api_vehicle.get('advertId')
        if not advert_id:
            raise ValueError("Vehicle missing advertId")
        
        # Extract basic data with safe defaults and None handling
        title = api_vehicle.get('title') or ''
        subtitle = api_vehicle.get('subTitle') or ''
        price_str = api_vehicle.get('price') or '£0'
        location = api_vehicle.get('location') or ''
        seller_name = api_vehicle.get('sellerName') or ''
        seller_type = api_vehicle.get('sellerType') or 'TRADE'
        
        # Extract year and mileage from badges with enhanced fallback
        year = 0
        mileage = 0
        
        badges = api_vehicle.get('badges') or []
        for badge in badges:
            if not isinstance(badge, dict) or badge is None:
                continue
                
            badge_text = badge.get('displayText') or ''
            if not badge_text:
                continue
            
            # Extract year from "2013 (13 reg)" format
            if 'reg' in badge_text.lower():
                year_match = re.search(r'(\d{4})', badge_text)
                if year_match:
                    year = int(year_match.group(1))
            
            # Extract mileage from "99,680 miles" format
            if 'miles' in badge_text.lower():
                mileage_match = re.search(r'([\d,]+)', badge_text)
                if mileage_match:
                    mileage_str = mileage_match.group(1).replace(',', '')
                    try:
                        mileage = int(mileage_str)
                    except ValueError:
                        pass
        
        # Extract price numeric value with better error handling
        price_numeric = 0
        if price_str:
            price_match = re.search(r'[\d,]+', price_str.replace('£', ''))
            if price_match:
                price_str_clean = price_match.group().replace(',', '')
                try:
                    price_numeric = int(price_str_clean)
                except ValueError:
                    # If we can't parse price, try to get it from tracking context
                    pass
        
        # Get tracking context data (safely handle missing/None context)
        tracking_context = api_vehicle.get('trackingContext') or {}
        advert_context = tracking_context.get('advertContext') or {}
        
        # Override with tracking context data if available, otherwise use extracted values
        if 'year' in advert_context and advert_context['year']:
            year = advert_context['year']
        if 'price' in advert_context and advert_context['price']:
            price_numeric = advert_context['price']
        
        # If we still don't have year/mileage, try to extract from title
        full_title = f"{title} {subtitle}".strip()
        if year == 0:
            year_match = re.search(r'(\d{4})', full_title)
            if year_match:
                try:
                    potential_year = int(year_match.group(1))
                    if 2000 <= potential_year <= 2025:  # Reasonable year range
                        year = potential_year
                except ValueError:
                    pass
        
        # If no price found anywhere, this vehicle is probably invalid
        if price_numeric <= 0:
            raise ValueError(f"Vehicle {deal_url or advert_id} has no valid price")
        
        # If no year found, this is suspicious but we'll allow it with warning
        if year < 2000:
            print(f"⚠️  Warning: Vehicle {deal_url or advert_id} has invalid year: {year}")
        
        # Extract vehicle specifications from title/subtitle
        specs = NetworkDataAdapter._extract_specs_from_title(full_title)
        
        # Get make/model from tracking context or try to extract from title
        make = advert_context.get('make') or ''
        model = advert_context.get('model') or ''
        
        if not make and title:
            # Try to extract make from title (first word usually)
            title_words = title.split()
            if title_words:
                make = title_words[0]
        
        if not model and len(title.split()) > 1:
            # Try to extract model from title (second word usually)
            title_words = title.split()
            if len(title_words) > 1:
                model = title_words[1]
        
        # Convert seller type
        seller_type_mapped = 'dealer' if seller_type == 'TRADE' else 'private'
        
        # Build the deal URL from advert_id
        deal_url = f"https://www.autotrader.co.uk/car-details/{advert_id}" if advert_id else ""
        
        # Create analyzer-compatible vehicle record
        analyzer_vehicle = {
            # Required core fields
            'url': deal_url,
            'title': title,
            'subtitle': subtitle,
            'full_title': full_title,
            'price': price_str,
            'price_numeric': price_numeric,
            'year': year,
            'mileage': mileage,
            'location': location,
            'seller_type': seller_type_mapped,
            'seller_name': seller_name,
            
            # Vehicle specifications
            'make': make,
            'model': model,
            'engine_size': specs.get('engine_size'),
            'fuel_type': specs.get('fuel_type'),
            'transmission': specs.get('transmission'),
            'body_type': specs.get('body_type'),
            'doors': specs.get('doors'),
            
            # Additional API data (safely handle None values)
            'dealer_review_rating': (api_vehicle.get('dealerReview') or {}).get('overallReviewRating'),
            'dealer_review_count': (api_vehicle.get('dealerReview') or {}).get('numberOfReviews'),
            'attention_grabber': api_vehicle.get('attentionGrabber'),
            'distance': api_vehicle.get('formattedDistance'),
            'number_of_images': api_vehicle.get('numberOfImages') or 0,
            'has_video': api_vehicle.get('hasVideo') or False,
            'has_360_spin': api_vehicle.get('has360Spin') or False,
            'manufacturer_approved': api_vehicle.get('manufacturerApproved') or False,
            
            # Image URLs (processed by network scraper)
            'image_url': api_vehicle.get('image_url', ''),
            'image_url_2': api_vehicle.get('image_url_2', '')
        }
        
        return analyzer_vehicle
    
    @staticmethod
    def _extract_specs_from_title(title: str) -> Dict[str, Optional[str]]:
        """
        Extract vehicle specifications from title/subtitle text.
        
        Args:
            title: Combined title and subtitle text
            
        Returns:
            Dict with extracted specs
        """
        title_lower = title.lower()
        specs = {}
        
        # Extract engine size (1.0, 1.2L, 2.0 TDI, etc.)
        engine_patterns = [
            r'(\d+\.\d+)l?\s*(?:litre|liter)?',
            r'(\d+)l\s',
            r'(\d+\.\d+)\s*(?:tdi|vvt|turbo)',
        ]
        
        for pattern in engine_patterns:
            match = re.search(pattern, title_lower)
            if match:
                try:
                    specs['engine_size'] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract fuel type
        fuel_keywords = {
            'petrol': 'Petrol',
            'diesel': 'Diesel', 
            'hybrid': 'Hybrid',
            'electric': 'Electric',
            'plugin': 'Plug-in Hybrid',
            'plug-in': 'Plug-in Hybrid',
            'tdi': 'Diesel',
            'hdi': 'Diesel',
            'cdti': 'Diesel',
        }
        
        for keyword, fuel_type in fuel_keywords.items():
            if keyword in title_lower:
                specs['fuel_type'] = fuel_type
                break
        
        # Extract transmission
        transmission_keywords = {
            'manual': 'Manual',
            'automatic': 'Automatic',
            'auto': 'Automatic',
            'cvt': 'CVT',
            'semi-automatic': 'Semi-Automatic',
            'semi auto': 'Semi-Automatic',
            'tiptronic': 'Automatic',
            'dsg': 'Automatic',
        }
        
        for keyword, trans_type in transmission_keywords.items():
            if keyword in title_lower:
                specs['transmission'] = trans_type
                break
        
        # Extract body type
        body_keywords = {
            'hatchback': 'Hatchback',
            'saloon': 'Saloon',
            'estate': 'Estate',
            'coupe': 'Coupe',
            'convertible': 'Convertible',
            'suv': 'SUV',
            'mpv': 'MPV',
            '4x4': 'SUV',
            'crossover': 'SUV',
        }
        
        for keyword, body_type in body_keywords.items():
            if keyword in title_lower:
                specs['body_type'] = body_type
                break
        
        # Extract number of doors
        door_patterns = [
            r'(\d)\s*dr\b',
            r'(\d)\s*door',
            r'(\d)\s*-door',
        ]
        
        for pattern in door_patterns:
            match = re.search(pattern, title_lower)
            if match:
                try:
                    specs['doors'] = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        return specs
    
    @staticmethod
    def convert_vehicle_list(api_vehicles: List[Dict]) -> List[Dict]:
        """
        Convert a list of vehicles from API format to analyzer format.
        
        Args:
            api_vehicles: List of vehicles from GraphQL API
            
        Returns:
            List of vehicles in analyzer-compatible format
        """
        converted = []
        skipped_empty = 0
        skipped_invalid = 0
        conversion_errors = []
        
        for i, api_vehicle in enumerate(api_vehicles):
            try:
                analyzer_vehicle = NetworkDataAdapter.convert_vehicle(api_vehicle)
                
                # Validate converted vehicle has minimum required data
                if (analyzer_vehicle['price_numeric'] > 0 and 
                    analyzer_vehicle['year'] >= 2000 and 
                    analyzer_vehicle['mileage'] >= 0 and
                    analyzer_vehicle['url']):
                    converted.append(analyzer_vehicle)
                else:
                    skipped_invalid += 1
                    print(f"⚠️  Skipped vehicle {analyzer_vehicle.get('url', 'unknown')}: Invalid data (price={analyzer_vehicle['price_numeric']}, year={analyzer_vehicle['year']}, mileage={analyzer_vehicle['mileage']})")
                    
            except ValueError as e:
                if "Empty vehicle object" in str(e):
                    skipped_empty += 1
                else:
                    skipped_invalid += 1
                    conversion_errors.append(f"Vehicle {i}: {str(e)}")
            except Exception as e:
                skipped_invalid += 1
                conversion_errors.append(f"Vehicle {i} (unknown error): {str(e)}")
        
        # Report conversion statistics
        total_input = len(api_vehicles)
        total_converted = len(converted)
        total_skipped = skipped_empty + skipped_invalid
        
        print(f"📊 Conversion Summary:")
        print(f"   • Input vehicles: {total_input}")
        print(f"   • Successfully converted: {total_converted}")
        print(f"   • Skipped empty objects: {skipped_empty}")
        print(f"   • Skipped invalid data: {skipped_invalid}")
        print(f"   • Conversion rate: {total_converted/max(1, total_input-skipped_empty)*100:.1f}% (excluding empty objects)")
        
        if conversion_errors:
            print(f"⚠️  Conversion errors:")
            for error in conversion_errors[:5]:  # Show first 5 errors
                print(f"      • {error}")
            if len(conversion_errors) > 5:
                print(f"      ... and {len(conversion_errors)-5} more errors")
        
        return converted

# ────────────────────────────────────────────────────────────
# Testing and validation
# ────────────────────────────────────────────────────────────

def test_data_adapter():
    """Test the data adapter with real API data"""
    print("🧪 Testing Network Data Adapter...")
    
    from networkrequest_scraper import AutoTraderAPIClient
    
    # Get some test data
    client = AutoTraderAPIClient()
    api_vehicles = client.get_all_cars("Toyota", "AYGO", max_pages=1)
    
    if not api_vehicles:
        print("❌ No test data available")
        return
    
    print(f"📊 Testing with {len(api_vehicles)} API vehicles")
    
    # Convert using adapter
    converted_vehicles = NetworkDataAdapter.convert_vehicle_list(api_vehicles)
    
    print(f"✅ Successfully converted {len(converted_vehicles)} vehicles")
    
    if converted_vehicles:
        print(f"\n🔍 Sample converted vehicle:")
        sample = converted_vehicles[0]
        
        # Show key fields
        key_fields = ['url', 'title', 'price_numeric', 'year', 'mileage', 
                     'seller_type', 'engine_size', 'fuel_type', 'transmission']
        
        for field in key_fields:
            value = sample.get(field, 'N/A')
            print(f"  • {field}: {value}")
    
    return converted_vehicles

if __name__ == "__main__":
    test_data_adapter()
