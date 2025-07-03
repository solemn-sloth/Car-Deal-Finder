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
        
        # Extract vehicle specifications from subtitle
        vehicle_specs = extract_vehicle_specs(subtitle)
        
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
            
            # Vehicle specifications extracted from subtitle
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

def extract_vehicle_specs(subtitle: str) -> dict:
    """
    Extract vehicle specifications from the subtitle
    Examples:
    - "1.4 16V SE Copa Sport Coupe Euro 5 3dr" 
    - "1.0 TSI SE Euro 6 (s/s) 5dr"
    - "1.2 TDI Ecomotive CR SE Euro 5 (s/s) 5dr"
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
    
    if not subtitle:
        return specs
    
    import re
    subtitle_lower = subtitle.lower()
    
    # Engine size (e.g., "1.4", "2.0", "1.0")
    engine_match = re.search(r'(\d+\.\d+)', subtitle)
    if engine_match:
        specs['engine_size'] = float(engine_match.group(1))
    
    # Fuel type detection
    if 'tdi' in subtitle_lower or 'diesel' in subtitle_lower:
        specs['fuel_type'] = 'Diesel'
    elif 'tsi' in subtitle_lower or 'petrol' in subtitle_lower or 'tfsi' in subtitle_lower:
        specs['fuel_type'] = 'Petrol'
    elif 'hybrid' in subtitle_lower:
        specs['fuel_type'] = 'Hybrid'
    elif 'electric' in subtitle_lower or 'ev' in subtitle_lower:
        specs['fuel_type'] = 'Electric'
    else:
        # Default for common patterns
        if any(x in subtitle_lower for x in ['16v', 'tsi', 'fsi']):
            specs['fuel_type'] = 'Petrol'
    
    # Transmission detection
    if 'dsg' in subtitle_lower or 'auto' in subtitle_lower or 'automatic' in subtitle_lower:
        specs['transmission'] = 'Automatic'
    elif 'manual' in subtitle_lower or 'man' in subtitle_lower:
        specs['transmission'] = 'Manual'
    else:
        # Most cars are manual unless specified
        specs['transmission'] = 'Manual'
    
    # Doors (e.g., "3dr", "5dr")
    doors_match = re.search(r'(\d+)dr', subtitle_lower)
    if doors_match:
        specs['doors'] = int(doors_match.group(1))
    
    # Euro standard (e.g., "Euro 5", "Euro 6")
    euro_match = re.search(r'euro (\d+)', subtitle_lower)
    if euro_match:
        specs['euro_standard'] = f"Euro {euro_match.group(1)}"
    
    # Stop/Start technology
    if '(s/s)' in subtitle_lower or 'start/stop' in subtitle_lower:
        specs['stop_start'] = True
    
    # Body type detection
    if 'coupe' in subtitle_lower:
        specs['body_type'] = 'Coupe'
    elif 'estate' in subtitle_lower or 'st' in subtitle_lower:
        specs['body_type'] = 'Estate'
    elif 'hatchback' in subtitle_lower:
        specs['body_type'] = 'Hatchback'
    elif 'saloon' in subtitle_lower or 'sedan' in subtitle_lower:
        specs['body_type'] = 'Saloon'
    elif 'suv' in subtitle_lower:
        specs['body_type'] = 'SUV'
    elif 'convertible' in subtitle_lower or 'cabriolet' in subtitle_lower:
        specs['body_type'] = 'Convertible'
    
    # Trim level (extract common trim names)
    trim_patterns = ['se', 'sport', 'fr', 'toca', 'style', 'reference', 'copa', 'ecomotive', 'xcellence']
    for trim in trim_patterns:
        if trim in subtitle_lower:
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
        
        # Listing details
        print(f"\n📋 Listing Details:")
        print(f"   Type: {p.get('listing_type', 'N/A')}")
        print(f"   Subtitle: {p.get('subtitle', 'N/A')}")
        print(f"   Attention: {p.get('attention_grabber', 'N/A')}")
        print(f"   Description: {p.get('description', 'N/A')}")
        
        # Seller info
        print(f"\n🏪 Seller:")
        print(f"   Name: {p.get('seller_name', 'N/A')}")
        print(f"   Type: {p.get('seller_type', 'N/A')}")
        print(f"   Rating: {p.get('dealer_rating', 'N/A')}/5 ({p.get('dealer_review_count', 0)} reviews)")
        
        # Vehicle condition & features
        print(f"\n🔧 Vehicle Info:")
        print(f"   Condition: {p.get('condition', 'N/A')}")
        print(f"   Engine: {p.get('engine_size', 'N/A')}L {p.get('fuel_type', 'N/A')}")
        print(f"   Transmission: {p.get('transmission', 'N/A')}")
        print(f"   Doors: {p.get('doors', 'N/A')}")
        print(f"   Body Type: {p.get('body_type', 'N/A')}")
        print(f"   Trim: {p.get('trim_level', 'N/A')}")
        print(f"   Euro Standard: {p.get('euro_standard', 'N/A')}")
        print(f"   Stop/Start: {p.get('stop_start', False)}")
        print(f"   Price Rating: {p.get('price_indicator_rating', 'N/A')}")
        print(f"   RRP: {p.get('rrp', 'N/A')}")
        print(f"   Discount: {p.get('discount', 'N/A')}")
        print(f"   Pre-reg: {p.get('pre_reg', False)}")
        print(f"   Manufacturer Approved: {p.get('manufacturer_approved', False)}")
        
        # Media & features
        print(f"\n📸 Media:")
        print(f"   Images: {p.get('image_count', 0)}")
        print(f"   Has Video: {p.get('has_video', False)}")
        print(f"   Has 360° Spin: {p.get('has_360_spin', False)}")
        print(f"   Digital Retailing: {p.get('has_digital_retailing', False)}")
        
        # Badges
        print(f"\n🏷️  Badges: {', '.join(p.get('badges', []))}")
        
        # Finance info
        finance = p.get('finance_info')
        if finance:
            print(f"\n💳 Finance Available:")
            if finance.get('monthlyPrice'):
                print(f"   Monthly: {finance['monthlyPrice'].get('priceFormattedAndRounded', 'N/A')}")
            print(f"   Quote Type: {finance.get('quoteSubType', 'N/A')}")
        
        # Raw data sample (first few keys)
        print(f"\n📊 Raw Data Keys Available:")
        raw_keys = list(deal.get('raw_data', {}).keys()) if deal.get('raw_data') else []
        print(f"   {', '.join(raw_keys[:10])}{'...' if len(raw_keys) > 10 else ''}")
        
    print(f"\n{'='*50}")
    print(f"📈 SUMMARY:")
    print(f"   Total vehicles: {len(deals)}")
    print(f"   Successfully processed: {len([d for d in deals if d and 'processed' in d])}")
    print(f"   Errors: {len([d for d in deals if d and 'error' in d])}")
    
    # Show unique listing types
    listing_types = set()
    seller_types = set()
    price_ratings = set()
    
    for deal in deals:
        if deal and 'processed' in deal:
            p = deal['processed']
            if p.get('listing_type'):
                listing_types.add(p['listing_type'])
            if p.get('seller_type'):
                seller_types.add(p['seller_type'])
            if p.get('price_indicator_rating'):
                price_ratings.add(p['price_indicator_rating'])
    
    print(f"\n🔍 Data Analysis:")
    print(f"   Listing Types: {', '.join(sorted(listing_types))}")
    print(f"   Seller Types: {', '.join(sorted(seller_types))}")
    print(f"   Price Ratings: {', '.join(sorted(price_ratings))}")
    
    return deals

if __name__ == "__main__":
    test_api_scraping()