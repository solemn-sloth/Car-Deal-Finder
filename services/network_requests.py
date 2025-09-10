import requests
import requests.adapters
import json
import time
import random
import ssl
from typing import List, Dict, Optional
from requests.packages.urllib3.util.retry import Retry

class CustomHTTPAdapter(requests.adapters.HTTPAdapter):
    """Custom adapter with specific cipher suites and TLS versions to mimic browser TLS fingerprints"""
    def init_poolmanager(self, *args, **kwargs):
        # Use common client-side cipher suites
        kwargs['ssl_context'] = self._create_ssl_context()
        return super().init_poolmanager(*args, **kwargs)
    
    def _create_ssl_context(self):
        context = ssl.create_default_context()
        # Use cipher suites common to Chrome browsers
        context.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384')
        return context

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
    def __init__(self, connection_pool_size=10, optimize_connection=True, proxy=None, proxy_manager=None):
        """
        Initialize the AutoTrader API client with configurable connection pooling.
        
        Args:
            connection_pool_size: Size of the connection pool (default: 10)
            optimize_connection: Whether to use connection pooling and other optimizations (default: True)
            proxy: Optional proxy URL (e.g., "http://user:pass@host:port")
            proxy_manager: Optional ProxyManager instance for rotation
        """
        self.base_url = "https://www.autotrader.co.uk/at-gateway"
        self.proxy_manager = proxy_manager
        
        # Create session with optimized connection parameters
        self.session = requests.Session()
        
        # Configure proxy if provided directly
        if proxy:
            self.session.proxies = {
                'http': proxy,
                'https': proxy
            }
        # Or use the proxy manager to get a proxy
        elif self.proxy_manager:
            proxy_url = self.proxy_manager.get_proxy()
            if proxy_url:
                self.session.proxies = {
                    'http': proxy_url,
                    'https': proxy_url
                }
        
        if optimize_connection:
            # Configure connection pooling with optimized parameters
            self.connection_pool_size = connection_pool_size
            
            # Create retry strategy with exponential backoff
            from requests.packages.urllib3.util.retry import Retry
            retry_strategy = Retry(
                total=3,  # Maximum number of retries
                status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
                backoff_factor=0.5,  # Exponential backoff factor
                allowed_methods=["GET", "POST"]  # Allow retries for GET and POST
            )
            
            # Configure adapter with retry strategy and connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=connection_pool_size,
                pool_maxsize=connection_pool_size,
                max_retries=retry_strategy,  # Apply retry strategy
                pool_block=False  # Don't block when pool is exhausted
            )
            
            # Apply connection pooling to both http and https
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
            
            # Set keep-alive and connection timeouts
            self.session.keep_alive = True
            self.timeout = 30
        else:
            # Standard configuration without pooling optimizations
            self.timeout = 30
            
        self.setup_headers()
    
    def randomize_headers(self):
        """Generate randomized browser-like headers to avoid fingerprinting"""
        # Browser versions - use a mix of recent Chrome, Firefox, and Edge versions
        chrome_versions = ["116.0.5845.110", "117.0.5938.132", "118.0.5993.88", "119.0.6045.123"]
        firefox_versions = ["116.0", "117.0", "118.0", "119.0"]
        edge_versions = ["116.0.1938.69", "117.0.2045.47", "118.0.2088.69", "119.0.2151.44"]
        
        # OS platforms with realistic variations
        os_platforms = [
            "Windows NT 10.0; Win64; x64", 
            "Macintosh; Intel Mac OS X 10_15_7",
            "Macintosh; Intel Mac OS X 11_6_0",
            "X11; Linux x86_64",
            "X11; Ubuntu; Linux x86_64"
        ]
        
        # Select random platform and browser type
        platform = random.choice(os_platforms)
        browser_type = random.choice(["chrome", "firefox", "edge"])
        
        # Build user agent based on browser type
        user_agent = ""
        if browser_type == "chrome":
            version = random.choice(chrome_versions)
            user_agent = f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
            browser_version = version.split(".")[0]
            ua_brand = f'"Google Chrome";v="{browser_version}", "Chromium";v="{browser_version}", "Not=A?Brand";v="99"'
        elif browser_type == "firefox":
            version = random.choice(firefox_versions)
            user_agent = f"Mozilla/5.0 ({platform}; rv:{version}) Gecko/20100101 Firefox/{version}"
            browser_version = version.split(".")[0]
            ua_brand = f'"Firefox";v="{browser_version}", "Not=A?Brand";v="99"'
        else:  # edge
            version = random.choice(edge_versions)
            user_agent = f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36 Edg/{version}"
            browser_version = version.split(".")[0]
            ua_brand = f'"Microsoft Edge";v="{browser_version}", "Chromium";v="{browser_version}", "Not=A?Brand";v="99"'
        
        # Random accept language variations
        accept_langs = [
            "en-GB,en-US;q=0.9,en;q=0.8",
            "en-US,en;q=0.9",
            "en-GB,en;q=0.8,fr;q=0.7",
            "en;q=0.9,en-GB;q=0.8"
        ]
        
        # Generate headers dictionary
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice(accept_langs),
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "sec-ch-ua": ua_brand,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": f'"{platform.split(";")[0] if ";" in platform else platform}"',
            "DNT": "1" if random.choice([True, False]) else "0",
            "Cache-Control": random.choice(["max-age=0", "no-cache"]),
            "Content-Type": "application/json",
            "Origin": "https://www.autotrader.co.uk",
            "Referer": "https://www.autotrader.co.uk/car-search?make=kia&model=sportage&postcode=SW1A1AA",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-sauron-app-name": "sauron-search-results-app",
            "x-sauron-app-version": f"{random.randint(1000000, 9999999)}"
        }
        
        # Add randomized ordering to further reduce fingerprinting
        shuffled_headers = {}
        keys = list(headers.keys())
        random.shuffle(keys)
        
        for key in keys:
            shuffled_headers[key] = headers[key]
            
        return shuffled_headers
    
    def setup_headers(self):
        """Set up the headers with randomized fingerprinting-resistant values"""
        # Apply randomized headers
        self.session.headers.update(self.randomize_headers())
        
        # Configure the custom TLS adapter for HTTPS connections
        self.session.mount('https://', CustomHTTPAdapter())
    
    def search_cars(self, make: str, model: str, postcode: str = "M15 4FN", 
                   min_year: int = 2010, max_year: int = 2023, 
                   max_mileage: int = 100000, page: int = 1, private_only: bool = True) -> Dict:
        """
        Search for cars using AutoTrader's GraphQL API
        """
        # Randomize headers for each request to avoid fingerprinting
        self.session.headers.update(self.randomize_headers())
        
        # Generate a unique search ID with realistic variation
        search_id = f"search-{random.randint(100000000000, 999999999999)}-{int(time.time())}"
        
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
            # First visit the search page to establish cookies and referrer legitimacy
            search_url = f"https://www.autotrader.co.uk/car-search?make={make.lower()}&model={model.lower()}&postcode={postcode}"
            self.session.get(search_url, timeout=self.timeout)
            
            # Increased delay between requests to reduce server load and avoid detection
            time.sleep(random.uniform(0.5, 0.8))
            
            # Now make the actual API request
            response = self.session.post(
                self.base_url + "?opname=SearchResultsListingsGridQuery",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"Response error: {response.status_code}")
                
                # If we have a proxy manager and get a 403 error, blacklist this proxy
                if response.status_code == 403 and self.proxy_manager and self.session.proxies:
                    proxy_url = self.session.proxies.get('https') or self.session.proxies.get('http')
                    if proxy_url:
                        ip = self.proxy_manager.extract_ip(proxy_url)
                        if ip:
                            self.proxy_manager.blacklist_ip(ip, reason=f"HTTP {response.status_code}")
                            # Try to get a new proxy for future requests
                            new_proxy = self.proxy_manager.get_proxy()
                            if new_proxy:
                                self.session.proxies = {'http': new_proxy, 'https': new_proxy}
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            
            # Similar error handling for exceptions
            if hasattr(e, 'response') and e.response and e.response.status_code == 403:
                if self.proxy_manager and self.session.proxies:
                    proxy_url = self.session.proxies.get('https') or self.session.proxies.get('http')
                    if proxy_url:
                        ip = self.proxy_manager.extract_ip(proxy_url)
                        if ip:
                            self.proxy_manager.blacklist_ip(ip, reason="HTTP 403 Exception")
            
            return None
    
    def get_all_cars(self, make: str, model: str, postcode: str = "M15 4FN",
                     min_year: int = 2010, max_year: int = 2023, 
                     max_mileage: int = 100000, max_pages: int = None) -> List[Dict]:
        """
        Get all cars from multiple pages
        """
        all_cars = []
        page = 1
        
        # Add a blank line before scraping
        print("")
        # Just print "Scraping..." without car name
        print(f"🔍 Scraping...", end="")
        
        # If max_pages is None, keep scraping until there are no more results
        # Otherwise, respect the max_pages limit
        while max_pages is None or page <= max_pages:
            # Print a dot for each page fetched (on same line)
            print(".", end="", flush=True)
            
            data = self.search_cars(make, model, postcode, min_year, max_year, max_mileage, page)
            
            if not data or len(data) == 0:
                break
                
            # Extract listings from the first response
            search_results = data[0].get('data', {}).get('searchResults', {})
            listings = search_results.get('listings', [])
            
            if not listings:
                break
                
            # Process each listing and include ALL processed listings
            processed_listings = []
            for listing in listings:
                processed_deal = convert_api_car_to_deal(listing)
                if processed_deal and processed_deal.get('processed'):
                    # Return the processed data which includes fixed image URLs
                    processed_listings.append(processed_deal['processed'])
            
            all_cars.extend(processed_listings)
            
            # Check if there are more pages
            page_info = search_results.get('page', {})
            current_page = page_info.get('number', 1)
            total_pages = page_info.get('count', 1)
            
            if current_page >= total_pages:
                break
                
            page += 1
            
            # Increased delay between page requests to reduce detection risk
            time.sleep(random.uniform(0.5, 0.8))  # Balanced for both speed and stealthiness
        
        # No found message needed - just finish the dots
        print()  # Print a newline after the dots
        
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
        
        # Extract price - handle both string and integer values
        price_value = car_data.get('price')
        if price_value is not None:
            if isinstance(price_value, (int, float)):
                # Price is already a number
                price = int(price_value)
            elif isinstance(price_value, str):
                # Price is a string, clean it
                price_str = price_value.replace('£', '').replace(',', '').strip()
                try:
                    price = int(price_str) if price_str.isdigit() else 0
                except ValueError:
                    price = 0
            else:
                price = 0
        else:
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
        
        # Store processed data for easy access - streamlined to only include essential fields
        deal['processed'] = {
            # Core identification
            'deal_id': advert_id,
            'url': deal_url,
            
            # Core vehicle data
            'make': advert_context.get('make', '').upper() if advert_context.get('make') else '',
            'model': advert_context.get('model', '') if advert_context.get('model') else '',
            'year': year,
            'price': price,
            'price_raw': car_data.get('price', ''),
            'mileage': mileage,
            
            # Location data
            'location': car_data.get('location', ''),
            'distance': car_data.get('formattedDistance', ''),
            'vehicle_location': car_data.get('vehicleLocation', ''),
            
            # Seller info
            'seller_type': car_data.get('sellerType', ''),
            
            # Basic listing info
            'title': car_data.get('title', ''),
            'listing_type': car_data.get('type'),
            
            # Media
            'image_url': process_autotrader_image_url(
                car_data.get('images', [''])[0] if car_data.get('images') else '', 
                "w480h360"
            ),
            'image_url_2': process_autotrader_image_url(
                car_data.get('images', [''])[1] if car_data.get('images') and len(car_data.get('images', [])) > 1 else '', 
                "w480h360"
            ),
            'image_count': car_data.get('numberOfImages', 0),
            'badges': [badge.get('displayText', '') for badge in badges if badge],
            
            # Vehicle specifications extracted from subtitle (with enhanced fallback)
            'engine_size': vehicle_specs.get('engine_size'),
            'fuel_type': vehicle_specs.get('fuel_type'),
            'transmission': vehicle_specs.get('transmission'),
            'doors': vehicle_specs.get('doors'),
            'body_type': vehicle_specs.get('body_type'),
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
        'body_type': None
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