import requests
import requests.adapters
import json
import time
import random
import ssl
import secrets
from typing import List, Dict, Optional
from requests.packages.urllib3.util.retry import Retry

class CustomHTTPAdapter(requests.adapters.HTTPAdapter):
    """Custom adapter with specific cipher suites and TLS versions to mimic browser TLS fingerprints"""
    def __init__(self, *args, verify_ssl=True, **kwargs):
        self.verify_ssl = verify_ssl
        super().__init__(*args, **kwargs)
        
    def init_poolmanager(self, *args, **kwargs):
        # Use common client-side cipher suites
        kwargs['ssl_context'] = self._create_ssl_context()
        return super().init_poolmanager(*args, **kwargs)
    
    def _create_ssl_context(self):
        context = ssl.create_default_context()
        
        # CloudFlare specifically looks for certain cipher suites and TLS features
        # Randomize cipher order to avoid static TLS fingerprint detection
        base_ciphers = [
            'ECDHE-ECDSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-ECDSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES256-GCM-SHA384',
            'ECDHE-ECDSA-CHACHA20-POLY1305',
            'ECDHE-RSA-CHACHA20-POLY1305',
            'ECDHE-RSA-AES128-SHA',
            'ECDHE-RSA-AES256-SHA'
        ]
        
        # Randomize cipher order for each new connection to avoid fingerprinting
        randomized_ciphers = base_ciphers.copy()
        secrets.SystemRandom().shuffle(randomized_ciphers)
        
        context.set_ciphers(':'.join(randomized_ciphers))
        
        # Set TLS version to 1.2 (CloudFlare sometimes has issues with TLS 1.3)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_NO_COMPRESSION  # Disable TLS compression for better mimicking browsers
        
        # Disable SSL verification if requested
        if not self.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            # Ensure hostname checking is enabled when SSL verification is on
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        
        # Apply elliptic curves that browsers use - randomize order
        try:
            import ctypes
            libssl = ctypes.cdll.LoadLibrary(ssl._ssl.__file__)
            libssl.SSL_CTX_set1_curves_list.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            libssl.SSL_CTX_set1_curves_list.restype = ctypes.c_int
            
            # Randomize elliptic curve order to avoid static fingerprint
            base_curves = ["X25519", "secp256r1", "prime256v1", "secp384r1"]
            randomized_curves = base_curves.copy()
            secrets.SystemRandom().shuffle(randomized_curves)
            
            curves = ":".join(randomized_curves).encode()
            libssl.SSL_CTX_set1_curves_list(context._ctx, curves)
        except (AttributeError, OSError):
            # If we can't set up the curves, continue anyway
            pass
            
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
    def __init__(self, connection_pool_size=10, optimize_connection=True, proxy=None, proxy_manager=None, verify_ssl=True):
        """Initialize the AutoTrader API client with configurable connection pooling.
        
        Args:
            connection_pool_size: Size of the connection pool (default: 10)
            optimize_connection: Whether to use connection pooling and other optimizations (default: True)
            proxy: Optional proxy URL (e.g., "http://user:pass@host:port")
            proxy_manager: Optional ProxyManager instance for rotation
            verify_ssl: Whether to verify SSL certificates (set to False to ignore SSL errors)
        """
        self.base_url = "https://www.autotrader.co.uk/at-gateway"
        self.proxy_manager = proxy_manager
        self._first_request_made = False  # Track if initial request has been made
        self._session_start_time = time.time()  # Track session start for behavior
        
        # Create session with optimized connection parameters
        self.session = requests.Session()
        
        # Set SSL verification (can be disabled for testing)
        self.session.verify = verify_ssl
        
        # Add initial CloudFlare cookies to help bypass protection
        self.session.cookies.set('cf_clearance', self._generate_cf_clearance(), 
                               domain='.autotrader.co.uk', path='/')
        self.session.cookies.set('cf_bm', self._generate_cf_bm(), 
                               domain='.autotrader.co.uk', path='/')
        self.session.cookies.set('cf_chl_2', self._generate_cf_chl(), 
                               domain='.autotrader.co.uk', path='/')
        self.session.cookies.set('cf_chl_prog', 'x19', 
                               domain='.autotrader.co.uk', path='/')
        
        # Add CloudFlare browser verification cookies
        self.session.cookies.set('_cfuvid', self._generate_cf_uvid(), 
                               domain='.autotrader.co.uk', path='/')
        self.session.cookies.set('_pxvid', self._generate_px_vid(),
                               domain='.autotrader.co.uk', path='/')
        
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
        
    def _generate_cf_clearance(self):
        """Generate a plausible CloudFlare clearance cookie value
        
        Format matches the pattern observed in real CloudFlare cookies:
        [random hex]-[unix timestamp]-[0-1]-[1-13]-[random digits]
        """
        import time, uuid
        # More realistic timestamp - recent past (last 7 days)
        timestamp = str(int(time.time() - random.randint(3600, 604800)))  # Between 1 hour and 7 days ago
        random_part = uuid.uuid4().hex[:20]  # Get a random string of correct length
        
        # Last parts are typically encoding information about the client
        # Format: {random}-{timestamp}-{0/1}-{1-13}-{random}
        return f"{random_part}-{timestamp}-{random.randint(0,1)}-{random.randint(1,13)}-{random.randint(100000, 999999)}"
    
    def _generate_cf_bm(self):
        """Generate a plausible CloudFlare bot management cookie (cf_bm)
        
        Format: base64-like string with specific length and pattern
        """
        import string, base64, time
        
        # Start with a valid-looking timestamp and device identifier
        timestamp = int(time.time())
        device_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
        
        # Generate some plausible binary data and encode it
        data = f"{timestamp}:{device_id}:some-verification-data-{random.randint(1000, 9999)}"
        
        # Make it look base64-encoded but ensure it has the right pattern with hyphens
        encoded = base64.b64encode(data.encode()).decode()
        parts = [encoded[i:i+12] for i in range(0, len(encoded), 12)]
        
        # CloudFlare bot management cookies typically have this format
        return f"{parts[0]}-{parts[1][:8]}-{int(time.time())}-{random.randint(1,9)}-q"
        
    def _generate_cf_uvid(self):
        """Generate a plausible CloudFlare unique visitor ID cookie
        
        Format: UUID-like string that CloudFlare uses for visitor identification
        """
        import uuid
        # Real _cfuvid cookies are UUIDs with a custom prefix
        return f"cfuv-{uuid.uuid4()}"
        
    def _generate_px_vid(self):
        """Generate a plausible PerimeterX visitor ID
        
        Format: Base64-like string used by PerimeterX (CloudFlare's bot detection partner)
        """
        import string, time
        # PerimeterX uses similar format but with different structure
        chars = string.ascii_lowercase + string.digits
        
        # Format is typically: [timestamp]-[random]-[random]
        timestamp = hex(int(time.time()))[2:]
        random_part1 = ''.join(random.choice(chars) for _ in range(8))
        random_part2 = ''.join(random.choice(chars) for _ in range(6))
        
        return f"{timestamp}-{random_part1}-{random_part2}"
    
    def _generate_cf_chl(self):
        """Generate a plausible CloudFlare challenge cookie
        
        Format is typically base64-like with specific patterns and length
        """
        import string
        
        # More precise character set matching actual CF cookies
        chars = string.ascii_letters + string.digits + '_-'
        
        # More accurate length - real cookies have more consistent lengths
        length = random.randint(85, 95)  # These cookies have a more consistent length
        
        # Generate with proper structure (segments separated by underscores)
        segments = [
            ''.join(random.choice(chars) for _ in range(random.randint(20, 25))),
            ''.join(random.choice(chars) for _ in range(random.randint(25, 30))),
            ''.join(random.choice(chars) for _ in range(random.randint(30, 35)))
        ]
        
        return '_'.join(segments)
    
    def randomize_headers(self):
        """Generate CloudFlare-optimized browser-like headers to avoid fingerprinting"""
        # Browser versions - use Chrome as it has the best compatibility with CloudFlare
        # Focus on recent stable versions that CloudFlare is least likely to block
        chrome_versions = ["118.0.5993.88", "119.0.6045.123", "120.0.6099.129"]
        
        # OS platforms - Focus on common Windows and Mac platforms
        os_platforms = [
            "Windows NT 10.0; Win64; x64",  # Most common Windows platform
            "Macintosh; Intel Mac OS X 10_15_7"  # Common Mac platform
        ]
        
        # Select random platform - prefer Windows slightly (more common)
        platform = random.choice(os_platforms)
        
        # Use Chrome only as it has best CloudFlare compatibility
        version = random.choice(chrome_versions)
        user_agent = f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
        browser_version = version.split(".")[0]
        ua_brand = f'"Google Chrome";v="{browser_version}", "Chromium";v="{browser_version}", "Not=A?Brand";v="99"'
        
        # CloudFlare often checks these specific header values
        accept_langs = [
            "en-GB,en-US;q=0.9,en;q=0.8",
            "en-US,en;q=0.9",
            "en-GB,en;q=0.9",
            "en-US,en-GB;q=0.9,en;q=0.8"
        ]
        
        # Device memory values typical of real browsers (in GB)
        device_memory = random.choice(["4", "8", "16"])
        
        # Screen color depth (typical values)
        color_depth = random.choice(["24", "30", "32"])
        
        # Viewport width (typical values)
        viewport_width = random.choice(["1280", "1366", "1440", "1536", "1920"])
        viewport_height = random.choice(["720", "768", "900", "864", "1080"])
        
        # DPR (Device Pixel Ratio) - common values
        dpr = random.choice(["1", "1.25", "1.5", "2", "2.5"])
        
        # Generate headers dictionary optimized for CloudFlare
        headers = {
            # Essential headers CloudFlare always checks
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": random.choice(accept_langs),
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            
            # Security headers that CloudFlare checks
            "sec-ch-ua": ua_brand,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": f'"{platform.split(";")[0] if ";" in platform else platform}"',
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "navigate",
            "sec-fetch-user": "?1",
            "sec-fetch-dest": "document",
            
            # Client hints that real browsers send - these help bypass CloudFlare
            "sec-ch-ua-arch": random.choice(['x86', 'arm']),
            "sec-ch-ua-full-version": version, # Full browser version
            "sec-ch-ua-platform-version": random.choice(['10.0.0', '11.0.0', '12.0.0', '15.6.1']),
            "sec-ch-ua-model": "",  # Desktop browsers don't have a model
            "sec-ch-ua-bitness": random.choice(['64', '32']),
            "sec-ch-ua-wow64": "?0",  # Not Windows-on-Windows
            "sec-ch-device-memory": device_memory,
            
            # Additional browser capability indicators
            "device-memory": device_memory,
            "dpr": dpr,
            "viewport-width": viewport_width,
            "rtt": random.choice(["50", "100", "150"]),  # Network round-trip time
            "downlink": random.choice(["10", "5.75", "1.75"]),  # Connection speed
            "ect": random.choice(["4g", "3g"]),  # Effective connection type
            
            # CloudFlare sometimes checks for these
            "Referer": "https://www.autotrader.co.uk/",
            "Origin": "https://www.autotrader.co.uk",
            
            # Cache and cookie settings
            "Cache-Control": "max-age=0",  # CloudFlare prefers this over no-cache
            
            # Additional AutoTrader-specific headers
            "x-sauron-app-name": "sauron-search-results-app",
            "x-sauron-app-version": f"{random.randint(1000000, 9999999)}"
        }
        
        # CloudFlare prefers headers in a specific order, so we won't shuffle them
        return headers
    
    def setup_headers(self):
        """Set up the headers with randomized fingerprinting-resistant values"""
        # Apply randomized headers
        self.session.headers.update(self.randomize_headers())
        
        # Configure the custom TLS adapter for HTTPS connections
        self.session.mount('https://', CustomHTTPAdapter(verify_ssl=self.session.verify))
    
    def _evolve_session_headers(self):
        """Subtly evolve headers during session to simulate organic browsing"""
        # Update cache-control occasionally to simulate browser behavior
        if random.random() < 0.3:  # 30% chance
            cache_options = ["max-age=0", "no-cache", "no-store, no-cache, must-revalidate"]
            self.session.headers['Cache-Control'] = random.choice(cache_options)
        
        # Vary some client hints slightly
        if random.random() < 0.4:  # 40% chance  
            device_memory = random.choice(["4", "8", "16"])
            self.session.headers['sec-ch-device-memory'] = device_memory
            
        # Update viewport width occasionally
        if random.random() < 0.2:  # 20% chance
            viewport_width = random.choice(["1280", "1366", "1440", "1536", "1920"])
            self.session.headers['viewport-width'] = viewport_width
    
    def search_cars(self, make: str, model: str, postcode: str = "M15 4FN", 
                   min_year: int = 2010, max_year: int = 2023, 
                   max_mileage: int = 100000, page: int = 1, private_only: bool = True,
                   max_retries: int = 2) -> Dict:
        """
        Search for cars using AutoTrader's GraphQL API
        
        Args:
            make: Car manufacturer
            model: Car model
            postcode: UK postcode for search location
            min_year: Minimum manufacturing year
            max_year: Maximum manufacturing year
            max_mileage: Maximum mileage
            page: Results page number
            private_only: Whether to include only private seller listings
            max_retries: Maximum number of retries for CloudFlare errors
        
        Returns:
            Dict containing search results data
        """
        # Randomize headers for each request to avoid fingerprinting
        self.session.headers.update(self.randomize_headers())
        
        # Track retry attempts
        retry_count = 0
        
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
        
        while retry_count <= max_retries:
            try:
                # First visit the search page to establish cookies and referrer legitimacy
                search_url = f"https://www.autotrader.co.uk/car-search?make={make.lower()}&model={model.lower()}&postcode={postcode}"
                
                # Add more random query parameters to appear more human-like
                from urllib.parse import urlencode
                import datetime
                
                # Generate random query parameters that look legitimate
                random_params = {
                    # Add a random include parameter
                    'include': random.choice(['', 'derivativeData', 'extendedData']),
                    
                    # Add a random sort parameter
                    'sort': random.choice(['', 'relevance', 'price-asc', 'price-desc', 'year-desc']),
                    
                    # Add a random tracking parameter that looks realistic
                    '_': str(int(datetime.datetime.now().timestamp() * 1000)),
                    
                    # Sometimes add a random utm parameter to look like a marketing link
                    'utm_source': random.choice(['', 'direct', 'search', 'email']) if random.random() > 0.7 else ''
                }
                
                # Remove empty parameters
                random_params = {k: v for k, v in random_params.items() if v}
                
                # Add parameters to URL if any exist
                if random_params:
                    param_string = urlencode(random_params)
                    search_url = f"{search_url}&{param_string}"
                
                # Randomize accept and content-type headers for this specific request
                search_headers = {
                    'Accept': random.choice([
                        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    ]),
                    'Accept-Language': random.choice([
                        'en-GB,en;q=0.9',
                        'en-US,en;q=0.8',
                        'en-GB,en-US;q=0.9,en;q=0.8'
                    ])
                }
                
                # Generate random but plausible cookies on each retry
                if retry_count > 0:
                    # Clear existing cookies first to start fresh
                    self.session.cookies.clear(domain='.autotrader.co.uk')
                    
                    # Add regenerated CloudFlare cookies
                    self.session.cookies.set('cf_clearance', self._generate_cf_clearance(), 
                                          domain='.autotrader.co.uk', path='/')
                    self.session.cookies.set('cf_bm', self._generate_cf_bm(), 
                                          domain='.autotrader.co.uk', path='/')
                    self.session.cookies.set('cf_chl_2', self._generate_cf_chl(), 
                                          domain='.autotrader.co.uk', path='/')
                    self.session.cookies.set('cf_chl_prog', f"x{random.randint(10, 99)}", 
                                          domain='.autotrader.co.uk', path='/')
                    self.session.cookies.set('_cfuvid', self._generate_cf_uvid(), 
                                          domain='.autotrader.co.uk', path='/')
                    self.session.cookies.set('_pxvid', self._generate_px_vid(),
                                          domain='.autotrader.co.uk', path='/')
                    
                    # Add human-like delay between retries
                    time.sleep(random.uniform(2.5, 4.5))
                    
                    # Get a completely fresh set of headers
                    self.session.headers.clear()
                    self.session.headers.update(self.randomize_headers())
                
                # Make the initial search page request with custom headers
                initial_response = self.session.get(search_url, timeout=self.timeout, headers=search_headers)
                
                # Check for CloudFlare challenge page
                if "Just a moment" in initial_response.text and retry_count < max_retries:
                    print(f"Detected CloudFlare challenge page on attempt {retry_count+1}. Retrying with new cookies...")
                    retry_count += 1
                    continue
                
                # Human-like behavior simulation - only on first request
                if not self._first_request_made:
                    # Session warmup: let cookies mature for 2-5 seconds
                    warmup_delay = random.uniform(2.0, 5.0)
                    time.sleep(warmup_delay)
                    
                    # Simulate realistic page processing time (3-8 seconds total)
                    reading_delay = random.uniform(3.0, 8.0)
                    time.sleep(reading_delay)
                    
                    # Add brief "decision" pause before API call
                    thinking_delay = random.uniform(0.5, 1.5) 
                    time.sleep(thinking_delay)
                    
                    # Evolve headers slightly to simulate organic browsing
                    self._evolve_session_headers()
                else:
                    # Subsequent requests: minimal delay to maintain speed
                    time.sleep(random.uniform(0.3, 0.8))
                
                # Add query parameters to the API request to look more legitimate
                api_params = {
                    'opname': 'SearchResultsListingsGridQuery',
                    # Add random cache buster parameter
                    'cb': str(random.randint(100000, 999999)),
                    # Add random tracking parameter
                    'seq': str(random.randint(1, 5))
                }
                
                # Create the API URL with parameters
                from urllib.parse import urlencode
                api_url = f"{self.base_url}?{urlencode(api_params)}"
                
                # Generate more specific headers for the API request
                api_headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json, text/plain, */*',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Origin': 'https://www.autotrader.co.uk',
                    'Referer': search_url,
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin',
                    # Add CloudFlare-specific header
                    'X-Requested-With': 'XMLHttpRequest'
                }
                
                # Now make the actual API request with the custom headers
                response = self.session.post(
                    api_url,
                    json=payload,
                    timeout=self.timeout,
                    headers=api_headers
                )
                
                # Mark first request as completed for future timing optimization
                if not self._first_request_made:
                    self._first_request_made = True
                
                # Exit retry loop if successful
                break
                
            except requests.exceptions.RequestException as e:
                # Handle proxy-related errors
                if self.proxy_manager and self.session.proxies:
                    proxy_url = self.session.proxies.get('https') or self.session.proxies.get('http')
                    if proxy_url and ("403" in str(e) or "proxy" in str(e).lower()):
                        ip = self.proxy_manager.extract_ip(proxy_url)
                        if ip:
                            self.proxy_manager.blacklist_ip(ip, reason=f"Error: {str(e)[:100]}")
                            # Get a new proxy for future requests
                            new_proxy = self.proxy_manager.get_proxy()
                            if new_proxy:
                                self.session.proxies = {'http': new_proxy, 'https': new_proxy}
                
                # Check if we should retry
                if retry_count < max_retries and ("403" in str(e) or "timeout" in str(e).lower()):
                    print(f"Request error on attempt {retry_count+1}: {str(e)[:100]}. Retrying...")
                    retry_count += 1
                    continue
                else:
                    # Log the error and raise it
                    print(f"Fatal request error: {e}")
                    raise
        
        # After the retry loop, process the response
        if 'response' in locals():  # Check if response variable exists
            # Log detailed response information for debugging
            if response.status_code != 200:
                print(f"Response error: {response.status_code}")
                
                # Try to extract and log detailed error information
                try:
                    error_content = response.text[:500]  # Get first 500 chars to avoid huge output
                    import logging
                    logging.info(f"Error response content: {error_content}")
                except:
                    pass
                
                # Check for CloudFlare challenge page in response
                if response.status_code == 403 and "Just a moment" in response.text:
                    print(f"Detected CloudFlare challenge in response. This should have been caught earlier.")
                    # We can't continue here as we're outside the loop - just log it
                
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
            
            # Log successful response info
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    
                    # Check for empty data in successful response
                    if not response_json or len(response_json) == 0:
                        print("Warning: Received empty JSON response with status 200")
                    elif 'errors' in response_json:
                        print(f"Warning: Response contains errors: {response_json['errors']}")
                    
                    return response_json
                except Exception as e:
                    print(f"Error parsing JSON response: {e}")
                    # Log the first part of the response content
                    try:
                        content = response.text[:500]
                        print(f"Response content snippet: {content}")
                    except:
                        pass
                    return None
            
            response.raise_for_status()
            return None  # This will only be reached if raise_for_status doesn't raise an exception
        
        # If we couldn't even get a response object
        return None
    
    def get_all_cars(self, make: str, model: str, postcode: str = "M15 4FN",
                     min_year: int = 2010, max_year: int = 2023, 
                     max_mileage: int = 100000, max_pages: int = None) -> List[Dict]:
        """
        Get all cars from multiple pages
        
        Args:
            make: Car manufacturer
            model: Car model (use format like "3 series" not "3-series")
            postcode: UK postcode for search location
            min_year: Minimum manufacturing year
            max_year: Maximum manufacturing year
            max_mileage: Maximum mileage
            max_pages: Maximum number of pages to fetch (None for all)
            
        Returns:
            List of car listings data
        """
        all_cars = []
        page = 1
        
        print(f"🔄 Scraping {make} {model}...", end="", flush=True)
        
        # If max_pages is None, keep scraping until there are no more results
        # Otherwise, respect the max_pages limit
        while max_pages is None or page <= max_pages:
            # Print a dot for each page fetched (on same line)
            print(".", end="", flush=True)
            
            # Use the search_cars method with retry mechanism
            try:
                data = self.search_cars(
                    make=make, 
                    model=model, 
                    postcode=postcode, 
                    min_year=min_year, 
                    max_year=max_year, 
                    max_mileage=max_mileage, 
                    page=page,
                    max_retries=2  # Allow up to 2 retries for CloudFlare issues
                )
            except Exception as e:
                print(f"\n❌ AutoTrader API access blocked by Cloudflare")
                break
            
            if not data or len(data) == 0:
                print(f"\n❌ No data returned from API on page {page}")
                break

            # Extract listings from the first response
            if isinstance(data, list) and len(data) > 0:
                search_results = data[0].get('data', {}).get('searchResults', {})
                listings = search_results.get('listings', [])
            else:
                listings = []
            
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
            'subtitle': subtitle,  # Keep original subtitle
            'spec': subtitle,  # Preserve original spec/subtitle unchanged
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