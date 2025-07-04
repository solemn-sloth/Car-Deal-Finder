import requests
import json
import time
import random
from typing import List, Dict, Optional

def process_autotrader_image_url(raw_url: str, size: str = "w480h360") -> str:
    """
    Process AutoTrader image URL by replacing {resize} placeholder with proper format.
    
    Args:
        raw_url: Raw URL with {resize} placeholder
        size: Size format (w{width}h{height}) or empty string for original
    
    Returns:
        Processed URL ready for use
    """
    if not raw_url:
        return ""
    
    if "{resize}" in raw_url:
        if size:
            return raw_url.replace("{resize}", size)
        else:
            # Remove the {resize} part entirely for original size
            return raw_url.replace("/{resize}", "")
    
    return raw_url

class AutoTraderAPIClient:
    def __init__(self):
        self.base_url = "https://www.autotrader.co.uk/at-gateway"
        self.session = requests.Session()
        self.setup_headers()
    
    def setup_headers(self):
        """Set up the headers based on your curl request"""
        self.session.headers.update({
            'accept': '*/*',
            'accept-language': 'en-GB,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://www.autotrader.co.uk',
            'referer': 'https://www.autotrader.co.uk/',
            'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
            'x-sauron-app-name': 'sauron-search-results-app',
            'x-sauron-app-version': '5899c683e3'
        })
    
    def search_cars(self, make: str, model: str, postcode: str = "M15 4FN", 
                   min_year: int = 2010, max_year: int = 2023, 
                   max_mileage: int = 100000, page: int = 1) -> Dict:
        """
        Search for cars using AutoTrader's GraphQL API
        """
        
        # Generate a unique search ID (you might want to make this more sophisticated)
        search_id = f"fc840876-f46d-48b7-8d5f-{random.randint(100000000000, 999999999999)}"
        
        # Build the GraphQL query payload
        payload = [
            {
                "operationName": "SearchResultsListingsGridQuery",
                "variables": {
                    "filters": [
                        {"filter": "is_writeoff", "selected": ["exclude"]},
                        {"filter": "make", "selected": [make.upper()]},
                        {"filter": "max_mileage", "selected": [str(max_mileage)]},
                        {"filter": "max_year_manufactured", "selected": [str(max_year)]},
                        {"filter": "min_year_manufactured", "selected": [str(min_year)]},
                        {"filter": "model", "selected": [model]},
                        {"filter": "postcode", "selected": [postcode]},
                        {"filter": "price_search_type", "selected": ["total"]}
                    ],
                    "channel": "cars",
                    "page": page,
                    "sortBy": "relevance",
                    "listingType": None,
                    "searchId": search_id,
                    "showIntercept": False,
                    "useGridLayout": True
                },
                "query": """query SearchResultsListingsGridQuery($filters: [FilterInput!]!, $channel: Channel!, $page: Int, $sortBy: SearchResultsSort, $listingType: [ListingType!], $searchId: String!, $showIntercept: Boolean!, $useGridLayout: Boolean!) {
                  searchResults(
                    input: {facets: [], filters: $filters, channel: $channel, page: $page, sortBy: $sortBy, listingType: $listingType, searchId: $searchId, useGridLayout: $useGridLayout}
                  ) {
                    intercept @include(if: $showIntercept) {
                      interceptType
                      title
                      subtitle
                      buttonText
                    }
                    listings {
                      ... on SearchListing {
                        type
                        advertId
                        title
                        subTitle
                        attentionGrabber
                        price
                        vehicleLocation
                        location
                        formattedDistance
                        discount
                        description
                        images
                        numberOfImages
                        priceIndicatorRating
                        rrp
                        dealerLogo
                        manufacturerApproved
                        sellerName
                        sellerType
                        dealerLink
                        dealerReview {
                          overallReviewRating
                          numberOfReviews
                        }
                        fpaLink
                        hasVideo
                        has360Spin
                        hasDigitalRetailing
                        preReg
                        finance {
                          monthlyPrice {
                            priceFormattedAndRounded
                          }
                          quoteSubType
                          representativeValues {
                            financeKey
                            financeValue
                          }
                        }
                        badges {
                          type
                          displayText
                        }
                        trackingContext {
                          advertContext {
                            id
                            make
                            model
                            year
                            condition
                            price
                          }
                        }
                      }
                    }
                    page {
                      number
                      count
                      results {
                        count
                      }
                    }
                  }
                }"""
            }
        ]
        
        try:
            response = self.session.post(
                self.base_url + "?opname=SearchResultsListingsGridQuery",
                json=payload,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            if response.status_code != 200:
                print(f"Response content: {response.text[:500]}...")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None
    
    def get_all_cars(self, make: str, model: str, postcode: str = "M15 4FN",
                     min_year: int = 2010, max_year: int = 2023, 
                     max_mileage: int = 100000, max_pages: int = 5) -> List[Dict]:
        """
        Get all cars from multiple pages
        """
        all_cars = []
        page = 1
        
        while page <= max_pages:
            print(f"Fetching page {page} for {make} {model}...")
            
            data = self.search_cars(make, model, postcode, min_year, max_year, max_mileage, page)
            
            if not data or len(data) == 0:
                print(f"No data returned for page {page}")
                break
                
            # Extract listings from the first response
            search_results = data[0].get('data', {}).get('searchResults', {})
            listings = search_results.get('listings', [])
            
            if not listings:
                print(f"No more listings found on page {page}")
                break
                
            # Process each listing and include ALL processed listings
            processed_listings = []
            for listing in listings:
                processed_deal = convert_api_car_to_deal(listing)
                if processed_deal and processed_deal.get('processed'):
                    # Return the processed data which includes fixed image URLs
                    processed_listings.append(processed_deal['processed'])
            
            all_cars.extend(processed_listings)
            
            print(f"Found {len(processed_listings)} processed listings on page {page}")
            
            # Check if there are more pages
            page_info = search_results.get('page', {})
            current_page = page_info.get('number', 1)
            total_pages = page_info.get('count', 1)
            
            if current_page >= total_pages:
                print(f"Reached last page ({total_pages})")
                break
                
            page += 1
            
            # Be respectful - add a small delay between requests
            time.sleep(random.uniform(1, 3))
        
        return all_cars

def convert_api_car_to_deal(car_data: Dict) -> Dict:
    """
    Convert AutoTrader API car data - capturing ALL available data points
    """
    try:
        if car_data is None:
            return None
            
        # Return the complete raw data with some processing for readability
        deal = {
            # Raw data (complete)
            'raw_data': car_data,
            
            # Processed/extracted fields for easier access
            'processed': {}
        }
        
        # Only process if car_data has content
        if not car_data:
            return deal
            
        # Extract tracking context for additional info
        tracking_context = car_data.get('trackingContext', {})
        advert_context = tracking_context.get('advertContext', {})
        
        # Build the deal URL with search parameters
        advert_id = car_data.get('advertId', '')
        if advert_id:
            # Extract car details
            make = advert_context.get('make', '').lower()
            model = advert_context.get('model', '').lower()
            
            # Get year - use from context or extract from title
            year = advert_context.get('year')
            if not year:
                title = car_data.get('title', '')
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', title) if title else None
                year = int(year_match.group()) if year_match else ''
            
            # Get mileage and round up to nearest 10,000
            mileage = 0
            try:
                mileage = int(car_data.get('mileage', 0) or 0)
            except (ValueError, TypeError):
                pass
            
            max_mileage = ((mileage // 10000) + 1) * 10000  # Round up to nearest 10,000
            
            # Build the URL with parameters
            base_url = f"https://www.autotrader.co.uk/car-details/{advert_id}"
            params = {
                'sort': 'price-asc',
                'advertising-location': 'at_cars',
                'exclude-writeoff-categories': 'on',
                'make': make,
                'model': model,
                'postcode': 'M15 4FN',
                'year-from': year,
                'year-to': year,
                'maximum-mileage': max_mileage,
                'fromsra': ''
            }
            
            # Convert params to query string
            from urllib.parse import urlencode
            query_string = urlencode({k: v for k, v in params.items() if v is not None and v != ''})
            deal_url = f"{base_url}?{query_string}"
        else:
            deal_url = ""
        
        # Extract price
        price_str = car_data.get('price', '').replace('£', '').replace(',', '') if car_data.get('price') else ''
        try:
            price = int(price_str) if price_str.isdigit() else 0
        except:
            price = 0
        
        # Extract year from title or tracking context
        year = advert_context.get('year')
        if not year:
            # Try to extract from title
            title = car_data.get('title', '')
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', title) if title else None
            year = int(year_match.group()) if year_match else None
        
        # Extract mileage from badges, description, or subtitle
        mileage = 0
        subtitle = car_data.get('subTitle', '') or ''
        description = car_data.get('description', '') or ''
        
        # First try to get mileage from badges
        badges = car_data.get('badges', [])
        for badge in badges:
            badge_text = badge.get('displayText', '') if badge else ''
            if 'miles' in badge_text.lower():
                import re
                mileage_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*miles?', badge_text, re.IGNORECASE)
                if mileage_match:
                    mileage = int(mileage_match.group(1).replace(',', ''))
                    break
        
        # If not found in badges, try subtitle and description
        if mileage == 0:
            mileage_text = subtitle + ' ' + description
            import re
            mileage_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*miles?', mileage_text, re.IGNORECASE)
            if mileage_match:
                mileage = int(mileage_match.group(1).replace(',', ''))
        
        # Extract vehicle specifications from subtitle with enhanced fallback logic
        vehicle_specs = extract_vehicle_specs(subtitle, car_data.get('title', ''))
        
        # Store processed data for easy access
        deal['processed'] = {
            'deal_id': advert_id,
            'url': deal_url,
            'make': advert_context.get('make', '').upper() if advert_context.get('make') else '',
            'model': advert_context.get('model', '') if advert_context.get('model') else '',
            'year': year,
            'price': price,
            'price_raw': car_data.get('price', ''),
            'mileage': mileage,
            'location': car_data.get('location', ''),
            'distance': car_data.get('formattedDistance', ''),
            'seller_name': car_data.get('sellerName', ''),
            'seller_type': car_data.get('sellerType', ''),
            'title': car_data.get('title', ''),
            'subtitle': car_data.get('subTitle', ''),
            'description': car_data.get('description', ''),
            'attention_grabber': car_data.get('attentionGrabber', ''),
            'image_url': process_autotrader_image_url(
                car_data.get('images', [''])[0] if car_data.get('images') else '', 
                "w480h360"
            ),
            'image_url_2': process_autotrader_image_url(
                car_data.get('images', [''])[1] if car_data.get('images') and len(car_data.get('images', [])) > 1 else '', 
                "w480h360"
            ),
            'image_count': car_data.get('numberOfImages', 0),
            'dealer_rating': car_data.get('dealerReview', {}).get('overallReviewRating') if car_data.get('dealerReview') else None,
            'dealer_review_count': car_data.get('dealerReview', {}).get('numberOfReviews') if car_data.get('dealerReview') else None,
            'price_indicator_rating': car_data.get('priceIndicatorRating'),
            'badges': [badge.get('displayText', '') for badge in badges if badge],
            'has_video': car_data.get('hasVideo', False),
            'has_360_spin': car_data.get('has360Spin', False),
            'has_digital_retailing': car_data.get('hasDigitalRetailing', False),
            'manufacturer_approved': car_data.get('manufacturerApproved', False),
            'pre_reg': car_data.get('preReg', False),
            'vehicle_location': car_data.get('vehicleLocation', ''),
            'dealer_logo': car_data.get('dealerLogo', ''),
            'dealer_link': car_data.get('dealerLink', ''),
            'fpa_link': car_data.get('fpaLink', ''),
            'rrp': car_data.get('rrp'),
            'discount': car_data.get('discount'),
            'finance_info': car_data.get('finance'),
            'listing_type': car_data.get('type'),
            'condition': advert_context.get('condition') if advert_context else None,
            'advertiser_id': advert_context.get('advertiserId') if advert_context else None,
            'advertiser_type': advert_context.get('advertiserType') if advert_context else None,
            'vehicle_category': advert_context.get('vehicleCategory') if advert_context else None,
            
            # Vehicle specifications extracted from subtitle (with enhanced fallback)
            'engine_size': vehicle_specs.get('engine_size'),
            'fuel_type': vehicle_specs.get('fuel_type'),
            'transmission': vehicle_specs.get('transmission'),
            'doors': vehicle_specs.get('doors'),
            'euro_standard': vehicle_specs.get('euro_standard'),
            'body_type': vehicle_specs.get('body_type'),
            'trim_level': vehicle_specs.get('trim_level'),
            'stop_start': vehicle_specs.get('stop_start'),
        }
        
        return deal
        
    except Exception as e:
        print(f"Error converting car data: {e}")
        return {'raw_data': car_data, 'error': str(e), 'processed': {}}

def extract_vehicle_specs(subtitle: str, title: str = '') -> dict:
    """
    Extract vehicle specifications from the subtitle and title with enhanced fallback logic.
    
    Args:
        subtitle: Vehicle subtitle from API
        title: Vehicle title from API (fallback source)
        
    Returns:
        Dict containing extracted specifications
    """
    specs = {
        'engine_size': None,
        'fuel_type': None,
        'transmission': None,
        'doors': None,
        'euro_standard': None,
        'body_type': None,
        'trim_level': None,
        'stop_start': False
    }
    
    # Combine subtitle and title for more comprehensive extraction
    full_text = f"{subtitle} {title}".strip()
    
    if not full_text:
        return specs
    
    import re
    text_lower = full_text.lower()
    
    # Engine size (e.g., "1.4", "2.0", "1.0")
    engine_match = re.search(r'(\d+\.\d+)', subtitle)
    if engine_match:
        specs['engine_size'] = float(engine_match.group(1))
    
    # Enhanced fuel type detection with better fallback patterns
    fuel_patterns = {
        # Direct fuel indicators
        'diesel': ['diesel', 'tdi', 'hdi', 'cdti', 'dti', 'crdi', 'bluetec', 'biturbo diesel'],
        'petrol': ['petrol', 'tsi', 'tfsi', 'fsi', 'vvt', 'vtech', 'turbo petrol', '16v'],
        'hybrid': ['hybrid', 'plugin hybrid', 'plug-in hybrid', 'self-charging hybrid'],
        'electric': ['electric', 'ev', 'bev', 'pure electric', 'zero emission'],
    }
    
    for fuel_type, patterns in fuel_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                specs['fuel_type'] = fuel_type.capitalize()
                break
        if specs['fuel_type']:
            break
    
    # If no fuel type found, try common model-specific patterns
    if not specs['fuel_type']:
        # Common diesel model suffixes
        diesel_patterns = ['116d', '118d', '120d', '318d', '320d', '520d', '116d', '118d']
        for pattern in diesel_patterns:
            if pattern in text_lower:
                specs['fuel_type'] = 'Diesel'
                break
        
        # If still no fuel type and contains common petrol indicators
        if not specs['fuel_type'] and any(x in text_lower for x in ['16v', 'tsi', 'fsi', 'vvt']):
            specs['fuel_type'] = 'Petrol'
    
    # Enhanced transmission detection
    transmission_patterns = {
        'automatic': ['automatic', 'auto', 'dsg', 'tiptronic', 's-tronic', 'cvt', 'powershift', 'edc'],
        'manual': ['manual', 'man', 'stick shift', '6-speed manual', '5-speed manual'],
    }
    
    for trans_type, patterns in transmission_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                specs['transmission'] = trans_type.capitalize()
                break
        if specs['transmission']:
            break
    
    # Default to manual if no transmission specified (most common)
    if not specs['transmission']:
        specs['transmission'] = 'Manual'
    
    # Doors (e.g., "3dr", "5dr")
    doors_match = re.search(r'(\d+)dr', text_lower)
    if doors_match:
        specs['doors'] = int(doors_match.group(1))
    
    # Euro standard (e.g., "Euro 5", "Euro 6")
    euro_match = re.search(r'euro (\d+)', text_lower)
    if euro_match:
        specs['euro_standard'] = f"Euro {euro_match.group(1)}"
    
    # Stop/Start technology
    if '(s/s)' in text_lower or 'start/stop' in text_lower:
        specs['stop_start'] = True
    
    # Body type detection
    body_patterns = {
        'coupe': ['coupe', 'coup'],
        'estate': ['estate', 'touring', 'avant', 'variant', 'sw'],
        'hatchback': ['hatchback', 'hatch'],
        'saloon': ['saloon', 'sedan', 'limousine'],
        'suv': ['suv', '4x4', 'crossover', 'x-drive', 'quattro'],
        'convertible': ['convertible', 'cabriolet', 'cabrio', 'roadster'],
        'mpv': ['mpv', 'people carrier', 'tourer'],
    }
    
    for body_type, patterns in body_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                specs['body_type'] = body_type.capitalize()
                break
        if specs['body_type']:
            break
    
    # Trim level (extract common trim names)
    trim_patterns = ['se', 'sport', 'fr', 'toca', 'style', 'reference', 'copa', 'ecomotive', 'xcellence', 
                    'comfort', 'elegance', 'luxury', 'premium', 'gt', 'gti', 'rs', 'amg', 'm sport']
    for trim in trim_patterns:
        if trim in text_lower:
            specs['trim_level'] = trim.upper()
            break
    
    return specs

# Test function
def test_api_scraping():
    """Test the API scraping with SEAT Ibiza"""
    print("🧪 Starting API scraping test...")
    
    client = AutoTraderAPIClient()
    
    print("Testing API scraping for SEAT Ibiza...")
    cars = client.get_all_cars("SEAT", "Ibiza", max_pages=3)
    
    print(f"\nFound {len(cars)} cars total")
    
    if len(cars) == 0:
        print("❌ No cars found - checking first API response...")
        # Test a single request to see what we get
        response = client.search_cars("SEAT", "Ibiza", page=1)
        print(f"Raw response: {response}")
        return []
    
    # Convert to our deal format
    deals = []
    for car in cars:
        deal = convert_api_car_to_deal(car)
        if deal:
            deals.append(deal)
    
    print(f"Converted {len(deals)} cars to deal format")
    
    # Show first few deals with all data
    for i, deal in enumerate(deals[:3]):
        if not deal or 'processed' not in deal:
            continue
            
        p = deal['processed']
        print(f"\n{'='*50}")
        print(f"Deal {i+1}: {p.get('title', 'Unknown')}")
        print(f"{'='*50}")
        
        # Basic info
        print(f"🚗 Make/Model: {p.get('make', 'N/A')} {p.get('model', 'N/A')} ({p.get('year', 'N/A')})")
        print(f"💰 Price: {p.get('price_raw', 'N/A')} (numeric: £{p.get('price', 0):,})")
        print(f"🛣️  Mileage: {p.get('mileage', 0):,} miles")
        print(f"📍 Location: {p.get('location', 'N/A')} ({p.get('distance', 'N/A')})")
        print(f"🔗 URL: {p.get('url', 'N/A')}")
        
        # Vehicle specifications
        print(f"\n🔧 Vehicle Specs:")
        print(f"   Engine: {p.get('engine_size', 'N/A')}L {p.get('fuel_type', 'N/A')}")
        print(f"   Transmission: {p.get('transmission', 'N/A')}")
        print(f"   Body Type: {p.get('body_type', 'N/A')}")
        print(f"   Doors: {p.get('doors', 'N/A')}")
        
        # Images
        print(f"\n📸 Images:")
        print(f"   Primary: {p.get('image_url', 'N/A')}")
        print(f"   Secondary: {p.get('image_url_2', 'N/A')}")
        print(f"   Count: {p.get('image_count', 0)}")
        
    return deals

if __name__ == "__main__":
    test_api_scraping()