"""
Data adapter to convert network request API data to analyzer-compatible format.
Bridges the gap between the API client processed format and the existing analyzer expectations.
"""
import re
from typing import Dict, List, Optional

class NetworkDataAdapter:
    """
    Converts vehicle data from API client processed format to analyzer-compatible format.
    Updated to work with the processed format from AutoTraderAPIClient.get_all_cars()
    """
    
    @staticmethod
    def _extract_fuel_type_from_title(vehicle_data: Dict) -> str:
        """
        Extract fuel type from vehicle title/subtitle when API field is missing.
        This is critical fallback logic since fuel type is important for analysis.
        """
        # Build full title from available parts
        title_parts = []
        if vehicle_data.get('title'):
            title_parts.append(vehicle_data['title'])
        if vehicle_data.get('subtitle'):
            title_parts.append(vehicle_data['subtitle'])
        
        full_title = ' '.join(title_parts).lower()
        
        if not full_title:
            return 'Unknown'
        
        # Comprehensive fuel type keyword matching
        fuel_keywords = {
            # Primary fuel types
            'petrol': 'Petrol',
            'diesel': 'Diesel',
            'hybrid': 'Hybrid',
            'electric': 'Electric',
            'plugin': 'Plug-in Hybrid',
            'plug-in': 'Plug-in Hybrid',
            
            # Diesel-specific patterns
            'tdi': 'Diesel',
            'hdi': 'Diesel',
            'cdti': 'Diesel',
            'dci': 'Diesel',
            'crdi': 'Diesel',
            'multijet': 'Diesel',
            'bluetec': 'Diesel',
            'bluemotion': 'Diesel',
            
            # Petrol-specific patterns
            'tsi': 'Petrol',
            'fsi': 'Petrol',
            'gti': 'Petrol',
            'vvt': 'Petrol',
            'vtec': 'Petrol',
            'tfsi': 'Petrol',
            '16v': 'Petrol',
            'turbo': 'Petrol',
            
            # Hybrid patterns
            'hybrid': 'Hybrid',
            'h-': 'Hybrid',
            'e-cvt': 'Hybrid',
            'synergy': 'Hybrid',
            
            # Electric patterns
            'electric': 'Electric',
            'ev': 'Electric',
            'e-': 'Electric',
            'zero emission': 'Electric',
        }
        
        # Check for fuel type indicators
        for keyword, fuel_type in fuel_keywords.items():
            if keyword in full_title:
                return fuel_type
        
        # Model-specific fallback patterns
        if any(pattern in full_title for pattern in ['116d', '118d', '120d', '318d', '320d', '520d']):
            return 'Diesel'
        
        # Default to Unknown if no matches found
        return 'Unknown'
    
    @staticmethod
    def _safe_string_field(value: any, default: str = '') -> str:
        """
        Safely convert a field to string, handling None values.
        """
        if value is None:
            return default
        return str(value).strip() if str(value).strip() else default
    
    @staticmethod
    def convert_vehicle(processed_vehicle: Dict) -> Dict:
        """
        Convert a single vehicle from API client processed format to analyzer format.
        
        Args:
            processed_vehicle: Vehicle data already processed by AutoTraderAPIClient
            
        Returns:
            Dict: Vehicle data in analyzer-compatible format
            
        Raises:
            ValueError: If vehicle data is invalid/empty
        """
        # Handle None input
        if processed_vehicle is None:
            raise ValueError("None vehicle object")
            
        # Skip completely empty objects
        if not processed_vehicle or processed_vehicle == {}:
            raise ValueError("Empty vehicle object")
        
        # Check for required deal_id (this is what your API client provides)
        deal_id = processed_vehicle.get('deal_id')
        if not deal_id:
            available_keys = list(processed_vehicle.keys()) if isinstance(processed_vehicle, dict) else []
            raise ValueError(f"Vehicle missing deal_id. Available keys: {available_keys}")
        
        # Validate essential data
        year = processed_vehicle.get('year', 0)
        mileage = processed_vehicle.get('mileage', 0)
        price = processed_vehicle.get('price', 0)
        
        if year == 0 or mileage == 0 or price == 0:
            raise ValueError(f"Invalid vehicle data: price={price}, year={year}, mileage={mileage}")
        
        try:
            # Handle price properly
            if isinstance(price, int):
                price_numeric = price
                price_display = str(price)
            else:
                # Try to extract numeric value if it's a string
                price_str = str(price)
                price_numeric = int(re.sub(r'[^\d]', '', price_str)) if price_str else 0
                price_display = price_str
            
            # Apply fallback logic for critical fields
            # CRITICAL: Handle fuel_type with fallback logic
            fuel_type = processed_vehicle.get('fuel_type')
            if not fuel_type:  # None or empty string
                fuel_type = NetworkDataAdapter._extract_fuel_type_from_title(processed_vehicle)
                print(f"   🔧 Fuel type fallback applied: {fuel_type} (from title analysis)")
            
            # Handle other fields that might be None
            transmission = NetworkDataAdapter._safe_string_field(processed_vehicle.get('transmission'), 'Unknown')
            body_type = NetworkDataAdapter._safe_string_field(processed_vehicle.get('body_type'), '')  # Empty string for analyzer to skip
            
            # Map the processed data to analyzer-compatible format
            analyzer_vehicle = {
                # Core identification - CRITICAL: Include deal_id
                'deal_id': deal_id,
                'url': processed_vehicle.get('url', ''),
                
                # Vehicle details
                'title': processed_vehicle.get('title', ''),
                'subtitle': processed_vehicle.get('subtitle', ''),
                'full_title': f"{processed_vehicle.get('title', '')} {processed_vehicle.get('subtitle', '')}".strip(),
                
                # Pricing
                'price': price_display,
                'price_numeric': price_numeric,
                'price_raw': processed_vehicle.get('price_raw', price_display),
                
                # Key specs
                'year': year,
                'mileage': mileage,
                'make': processed_vehicle.get('make', ''),
                'model': processed_vehicle.get('model', ''),
                
                # Location and seller
                'location': processed_vehicle.get('location', ''),
                'vehicle_location': processed_vehicle.get('vehicle_location', ''),
                'distance': processed_vehicle.get('distance', ''),
                'seller_type': 'dealer' if processed_vehicle.get('seller_type') == 'TRADE' else 'private',
                'seller_name': processed_vehicle.get('seller_name', ''),
                
                # Technical specifications - NOW WITH FALLBACK LOGIC
                'engine_size': processed_vehicle.get('engine_size'),
                'fuel_type': fuel_type,  # Now guaranteed to have a value (never None)
                'transmission': transmission,  # Now guaranteed to be string (never None)
                'body_type': body_type,  # Now guaranteed to be string (empty if unknown)
                'doors': processed_vehicle.get('doors'),
                'euro_standard': processed_vehicle.get('euro_standard'),
                'trim_level': processed_vehicle.get('trim_level'),
                'stop_start': processed_vehicle.get('stop_start', False),
                
                # Dealer information
                'dealer_review_rating': processed_vehicle.get('dealer_rating'),
                'dealer_review_count': processed_vehicle.get('dealer_review_count'),
                'dealer_logo': processed_vehicle.get('dealer_logo', ''),
                'dealer_link': processed_vehicle.get('dealer_link', ''),
                
                # Media and presentation
                'attention_grabber': processed_vehicle.get('attention_grabber', ''),
                'number_of_images': processed_vehicle.get('image_count', 0),
                'has_video': processed_vehicle.get('has_video', False),
                'has_360_spin': processed_vehicle.get('has_360_spin', False),
                'image_url': processed_vehicle.get('image_url', ''),
                'image_url_2': processed_vehicle.get('image_url_2', ''),
                'description': processed_vehicle.get('description'),
                
                # Features and badges
                'badges': processed_vehicle.get('badges', []),
                'manufacturer_approved': processed_vehicle.get('manufacturer_approved', False),
                'pre_reg': processed_vehicle.get('pre_reg', False),
                
                # Listing metadata
                'listing_type': processed_vehicle.get('listing_type'),
                'condition': processed_vehicle.get('condition'),
                'price_indicator_rating': processed_vehicle.get('price_indicator_rating'),
                'rrp': processed_vehicle.get('rrp'),
                'discount': processed_vehicle.get('discount'),
                'finance_info': processed_vehicle.get('finance_info'),
                'fpa_link': processed_vehicle.get('fpa_link', ''),
            }
            
            return analyzer_vehicle
            
        except Exception as e:
            raise ValueError(f"Error converting vehicle {deal_id}: {str(e)}")
    
    @staticmethod
    def convert_vehicle_list(vehicle_list: List[Dict]) -> List[Dict]:
        """
        Convert a list of vehicles from API client processed format to analyzer format.
        
        Args:
            vehicle_list: List of vehicles already processed by AutoTraderAPIClient
            
        Returns:
            List of converted vehicles in analyzer format
        """
        converted_vehicles = []
        skipped_empty = 0
        skipped_invalid = 0
        fuel_fallbacks_applied = 0
        
        print(f"🔄 Converting {len(vehicle_list)} vehicles from API client format...")
        
        for i, vehicle_data in enumerate(vehicle_list):
            # Skip empty objects
            if not vehicle_data:
                skipped_empty += 1
                continue
            
            print(f"🔍 Vehicle {i+1}: {type(vehicle_data).__name__} input (keys: {len(vehicle_data) if isinstance(vehicle_data, dict) else 'N/A'})")
            
            try:
                # Track fuel fallbacks
                original_fuel = vehicle_data.get('fuel_type')
                
                converted = NetworkDataAdapter.convert_vehicle(vehicle_data)
                
                # Count fuel fallbacks
                if not original_fuel and converted.get('fuel_type') != 'Unknown':
                    fuel_fallbacks_applied += 1
                
                converted_vehicles.append(converted)
                print(f"   ✅ Successfully converted vehicle {i+1} (deal_id: {converted.get('deal_id', 'N/A')})")
                
            except ValueError as e:
                skipped_invalid += 1
                print(f"   ⚠️  Skipped vehicle {i+1}: {str(e)}")
                
            except Exception as e:
                skipped_invalid += 1
                print(f"   ❌ Error converting vehicle {i+1}: {str(e)}")
        
        success_rate = len(converted_vehicles) / max(1, len(vehicle_list) - skipped_empty) * 100
        
        print(f"\n📊 Conversion Summary:")
        print(f"   • Input vehicles: {len(vehicle_list)}")
        print(f"   • Successfully converted: {len(converted_vehicles)}")
        print(f"   • Skipped empty objects: {skipped_empty}")
        print(f"   • Skipped invalid data: {skipped_invalid}")
        print(f"   • Fuel type fallbacks applied: {fuel_fallbacks_applied}")
        print(f"   • Conversion rate: {success_rate:.1f}% (excluding empty objects)")
        
        return converted_vehicles


# Test function to verify the adapter works
def test_adapter():
    """Test the adapter with sample data"""
    # Sample data in the format your API client provides
    test_vehicle = {
        "deal_id": "202505202601273",
        "url": "https://www.autotrader.co.uk/car-details/202505202601273",
        "make": "TOYOTA",
        "model": "Yaris",
        "year": 2020,
        "price": 13715,
        "price_raw": "£13,715",
        "mileage": 20900,
        "location": "Bolton",
        "distance": "154 miles",
        "seller_name": "TMMC The Manchester Motor Company Ltd",
        "seller_type": "TRADE",
        "title": "Toyota Yaris",
        "subtitle": "1.5 VVT-h Design E-CVT Euro 6 (s/s) 5dr",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "engine_size": 1.5,
        "doors": 5,
        "image_count": 42,
        "has_video": True,
        "dealer_rating": 4.1,
        "dealer_review_count": 127,
        "manufacturer_approved": False,
    }
    
    # Test case with None values (like your ABARTH 595 issue)
    test_vehicle_with_none = {
        "deal_id": "202505222699281",
        "url": "https://www.autotrader.co.uk/car-details/202505222699281",
        "make": "ABARTH",
        "model": "595",
        "year": 2018,
        "price": 6899,
        "price_raw": "£6,899",
        "mileage": 88803,
        "location": "York",
        "distance": "58 miles",
        "seller_name": "Prestige Car Supermarket",
        "seller_type": "TRADE",
        "title": "Abarth 595",
        "subtitle": "1.4 T-Jet 70th Euro 6 3dr",
        "fuel_type": None,  # This will trigger fallback
        "transmission": "Manual",
        "body_type": None,  # This will be handled gracefully
        "engine_size": 1.4,
        "doors": 3,
        "image_count": 33,
        "has_video": True,
        "dealer_rating": 4.4,
        "dealer_review_count": 940,
        "manufacturer_approved": False,
    }
    
    print("🧪 Testing with normal data...")
    try:
        converted = NetworkDataAdapter.convert_vehicle(test_vehicle)
        print("✅ Test conversion successful!")
        print(f"   • deal_id: {converted['deal_id']}")
        print(f"   • fuel_type: {converted['fuel_type']}")
        print(f"   • transmission: {converted['transmission']}")
        print(f"   • body_type: '{converted['body_type']}'")
    except Exception as e:
        print(f"❌ Test conversion failed: {str(e)}")
        return False
    
    print("\n🧪 Testing with None values (ABARTH 595 scenario)...")
    try:
        converted = NetworkDataAdapter.convert_vehicle(test_vehicle_with_none)
        print("✅ Test conversion with None values successful!")
        print(f"   • deal_id: {converted['deal_id']}")
        print(f"   • fuel_type: {converted['fuel_type']} (fallback from title)")
        print(f"   • transmission: {converted['transmission']}")
        print(f"   • body_type: '{converted['body_type']}' (empty string for analyzer to skip)")
        return True
    except Exception as e:
        print(f"❌ Test conversion with None values failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🧪 Testing NetworkDataAdapter with Hybrid Solution")
    print("="*60)
    test_adapter()