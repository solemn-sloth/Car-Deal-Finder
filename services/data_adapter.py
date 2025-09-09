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
            
            # Handle other fields that might be None
            transmission = NetworkDataAdapter._safe_string_field(processed_vehicle.get('transmission'), 'Unknown')
            body_type = NetworkDataAdapter._safe_string_field(processed_vehicle.get('body_type'), '')  # Empty string for analyzer to skip
            
            # Map the processed data to analyzer-compatible format
            analyzer_vehicle = {
                # Core identification - CRITICAL: Include deal_id
                'deal_id': deal_id,
                'url': processed_vehicle.get('url', ''),
                
                # Pricing
                'price': price_display,
                'price_numeric': price_numeric,
                
                # Key specs
                'year': year,
                'mileage': mileage,
                'make': processed_vehicle.get('make', ''),
                'model': processed_vehicle.get('model', ''),
                
                # Seller info - needed for filtering private sellers
                'seller_type': 'dealer' if processed_vehicle.get('seller_type') == 'TRADE' else 'private',
                'location': processed_vehicle.get('location', ''),
                
                # Technical specifications - essential for analysis
                'engine_size': processed_vehicle.get('engine_size'),
                'fuel_type': fuel_type,  # Now guaranteed to have a value (never None)
                'transmission': transmission,  # Now guaranteed to be string (never None)
                'body_type': body_type,  # Now guaranteed to be string (empty if unknown)
                'doors': processed_vehicle.get('doors'),
                
                # Required metadata fields only
                'number_of_images': processed_vehicle.get('image_count', 0),
                'image_url': processed_vehicle.get('image_url', ''),
                'image_url_2': processed_vehicle.get('image_url_2', ''),
                'badges': processed_vehicle.get('badges', []),
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
        fuel_fallbacks_applied = 0
        
        for vehicle_data in vehicle_list:
            # Skip empty objects
            if not vehicle_data:
                continue
                
            try:
                # Track fuel fallbacks
                original_fuel = vehicle_data.get('fuel_type')
                
                converted = NetworkDataAdapter.convert_vehicle(vehicle_data)
                
                # Count fuel fallbacks
                if not original_fuel and converted.get('fuel_type') != 'Unknown':
                    fuel_fallbacks_applied += 1
                
                converted_vehicles.append(converted)
                
            except Exception:
                pass
        
        # Silent operation - messages moved to scrape_grouping.py
        
        return converted_vehicles


