"""
Data adapter to convert network request API data to analyzer-compatible format.
Bridges the gap between the GraphQL API response format and the existing analyzer expectations.
"""
import re
from typing import Dict, List, Optional
from network_scraper import convert_api_car_to_deal

class NetworkDataAdapter:
    """
    Converts vehicle data from network request API format to analyzer-compatible format.
    Now uses the robust convert_api_car_to_deal() function for consistent processing.
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
        
        try:
            # Use the robust convert_api_car_to_deal function from network_scraper
            # This ensures consistent processing including image URLs, specs, and fallback logic
            converted_deal = convert_api_car_to_deal(api_vehicle)
            
            if not converted_deal or 'processed' not in converted_deal:
                raise ValueError(f"Failed to convert vehicle {advert_id}")
            
            processed_data = converted_deal['processed']
            
            # Map the processed data to analyzer-compatible format
            analyzer_vehicle = {
                # Core fields (directly mapped)
                'url': processed_data.get('url', ''),
                'title': processed_data.get('title', ''),
                'subtitle': processed_data.get('subtitle', ''),
                'full_title': f"{processed_data.get('title', '')} {processed_data.get('subtitle', '')}".strip(),
                'price': processed_data.get('price_raw', ''),
                'price_numeric': processed_data.get('price', 0),
                'year': processed_data.get('year', 0),
                'mileage': processed_data.get('mileage', 0),
                'location': processed_data.get('location', ''),
                'seller_type': 'dealer' if processed_data.get('seller_type') == 'TRADE' else 'private',
                'seller_name': processed_data.get('seller_name', ''),
                
                # Vehicle specifications (with fallback logic already applied)
                'make': processed_data.get('make', ''),
                'model': processed_data.get('model', ''),
                'engine_size': processed_data.get('engine_size'),
                'fuel_type': processed_data.get('fuel_type'),
                'transmission': processed_data.get('transmission'),
                'body_type': processed_data.get('body_type'),
                'doors': processed_data.get('doors'),
                
                # Additional fields
                'dealer_review_rating': processed_data.get('dealer_rating'),
                'dealer_review_count': processed_data.get('dealer_review_count'),
                'attention_grabber': processed_data.get('attention_grabber'),
                'distance': processed_data.get('distance'),
                'number_of_images': processed_data.get('image_count', 0),
                'has_video': processed_data.get('has_video', False),
                'has_360_spin': processed_data.get('has_360_spin', False),
                'manufacturer_approved': processed_data.get('manufacturer_approved', False),
                
                # Image URLs (now properly processed)
                'image_url': processed_data.get('image_url', ''),
                'image_url_2': processed_data.get('image_url_2', ''),
                
                # Additional processed fields that might be useful
                'description': processed_data.get('description', ''),
                'badges': processed_data.get('badges', []),
                'listing_type': processed_data.get('listing_type'),
                'condition': processed_data.get('condition'),
                'price_indicator_rating': processed_data.get('price_indicator_rating'),
                'rrp': processed_data.get('rrp'),
                'discount': processed_data.get('discount'),
                'finance_info': processed_data.get('finance_info'),
                'pre_reg': processed_data.get('pre_reg', False),
                'vehicle_location': processed_data.get('vehicle_location', ''),
                'dealer_logo': processed_data.get('dealer_logo', ''),
                'dealer_link': processed_data.get('dealer_link', ''),
                'fpa_link': processed_data.get('fpa_link', ''),
                'euro_standard': processed_data.get('euro_standard'),
                'trim_level': processed_data.get('trim_level'),
                'stop_start': processed_data.get('stop_start', False),
            }
            
            return analyzer_vehicle
            
        except Exception as e:
            raise ValueError(f"Error processing vehicle {advert_id}: {str(e)}")
    
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
    
    from network_scraper import AutoTraderAPIClient
    
    # Get some test data (raw API format)
    client = AutoTraderAPIClient()
    raw_api_data = client.search_cars("Toyota", "AYGO", page=1)
    
    if not raw_api_data:
        print("❌ No test data available")
        return
    
    # Extract listings from response
    search_results = raw_api_data[0].get('data', {}).get('searchResults', {})
    api_vehicles = search_results.get('listings', [])
    
    if not api_vehicles:
        print("❌ No vehicles in API response")
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
                     'seller_type', 'engine_size', 'fuel_type', 'transmission',
                     'image_url', 'image_url_2']
        
        for field in key_fields:
            value = sample.get(field, 'N/A')
            print(f"  • {field}: {value}")
    
    return converted_vehicles

if __name__ == "__main__":
    test_data_adapter()