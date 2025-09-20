import requests
import requests.adapters
import json
import time
import random
import ssl
import secrets
import logging
import sys
import os
from typing import List, Dict, Optional, Callable, Any
from requests.packages.urllib3.util.retry import Retry
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from services.stealth_orchestrator import FingerprintGenerator, CustomHTTPAdapter

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.output_manager import get_output_manager

# Disable SSL warnings globally to avoid CloudFlare issues
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# CustomHTTPAdapter is now imported from stealth_orchestrator.py for consistency

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


# ──────────────────────────────────────────────────────────────────
# Parallel API Scraping Classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class APITask:
    """Represents an API scraping task for a specific mileage range."""
    make: str
    model: str
    postcode: str
    min_year: int
    max_year: int
    min_mileage: int
    max_mileage: int
    max_pages: Optional[int] = None

    def __str__(self):
        return f"{self.make}_{self.model}_{self.min_mileage}-{self.max_mileage}"


class ParallelCoordinator:
    """
    Coordinates parallel API workers with anti-detection features.
    Manages worker coordination, proxy distribution, rate limiting, and progress tracking.
    """

    def __init__(self, target_rate_per_minute: int = 300, max_workers: int = 5):
        """
        Initialize coordinator.

        Args:
            target_rate_per_minute: Global rate limit for all workers combined
            max_workers: Maximum number of parallel workers
        """
        self.target_rate_per_minute = target_rate_per_minute
        self.max_workers = max_workers
        self.work_queue = queue.Queue()
        self.results = {}
        self.completed_count = 0
        self.failed_workers = set()
        self.results_lock = threading.Lock()
        self.last_result_status = "⚠️ Starting up"

    def add_work(self, tasks: List[APITask]):
        """Add tasks to work queue for parallel distribution."""
        for task in tasks:
            self.work_queue.put(task)

    def get_next_work(self, worker_id: int, timeout: float = 1) -> Optional[APITask]:
        """Get next task for worker, respecting worker health."""
        if worker_id in self.failed_workers:
            return None

        try:
            return self.work_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def report_result(self, task_id: str, result: Dict[str, Any], worker_id: int):
        """Report result from worker, track failures for intelligent backoff."""
        with self.results_lock:
            self.results[task_id] = result
            self.completed_count += 1

            # Store result status for progress reporting
            self.last_result_status = self._get_result_status(result)

            # Check if this worker should be paused due to detection
            if self._is_worker_detected(result):
                self.failed_workers.add(worker_id)
                logger.debug(f"Worker {worker_id} detected as blocked, pausing")

        # Mark the task as done in the queue
        try:
            self.work_queue.task_done()
        except ValueError:
            # Handle race condition where task_done() is called too many times
            logger.debug(f"task_done() called too many times - worker {worker_id} likely processed same task")

    def _get_result_status(self, result: Dict[str, Any]) -> str:
        """Get API-specific status indicator."""
        if not result.get('success', False):
            error = result.get('error', 'Unknown error')
            if 'cloudflare' in error.lower():
                return "❌ CloudFlare blocked"
            elif '403' in error:
                return "❌ 403 Forbidden"
            elif 'timeout' in error.lower():
                return "❌ Request timeout"
            else:
                return f"❌ {error[:40]}"

        car_count = result.get('metadata', {}).get('car_count', 0)
        mileage_range = result.get('metadata', {}).get('mileage_range', '')
        return f"✅ {car_count} cars ({mileage_range})"

    def get_total_listings_scraped(self) -> int:
        """Get total number of listings scraped across all completed tasks."""
        with self.results_lock:
            total_listings = 0
            for result in self.results.values():
                if result.get('success', False):
                    car_count = result.get('metadata', {}).get('car_count', 0)
                    total_listings += car_count
            return total_listings

    def _is_worker_detected(self, result: Dict[str, Any]) -> bool:
        """Check if API worker was detected/blocked."""
        if result.get('success', True):
            return False

        error = result.get('error', '').lower()
        return any(keyword in error for keyword in [
            'cloudflare', '403', 'forbidden', 'blocked', 'challenge',
            'too many requests', 'rate limit'
        ])

    def process_batch(self, tasks: List[APITask], progress_callback: Optional[Callable] = None,
                     test_mode: bool = False, max_tasks_per_worker: int = 1) -> Dict[str, Any]:
        """
        Process batch of tasks using parallel workers.

        Args:
            tasks: List of tasks to process
            progress_callback: Optional callback for progress updates
            test_mode: Skip delays in test mode
            max_tasks_per_worker: Maximum tasks per worker before restart

        Returns:
            Dictionary mapping task IDs to results
        """
        total_tasks = len(tasks)

        logger.debug(f"🚀 Starting parallel coordination with {self.max_workers} workers")
        logger.debug(f"📊 Processing {total_tasks} tasks")

        # Initialize proxy manager if available
        proxy_manager = None
        try:
            from services.stealth_orchestrator import ProxyManager
            proxy_manager = ProxyManager()
        except ImportError:
            logger.warning("ProxyManager not available, workers will use no proxy")

        # Add work to queue
        self.add_work(tasks)

        def worker_thread(worker_id: int):
            """Worker thread that processes API tasks."""
            try:
                # Get dedicated proxy for this worker
                worker_proxy = None
                if proxy_manager:
                    worker_proxy = proxy_manager.get_proxy_for_worker(worker_id)

                # Initialize API client for this worker with unified fingerprinting
                api_client = AutoTraderAPIClient(
                    proxy=worker_proxy,
                    verify_ssl=False,
                    worker_id=worker_id  # Pass worker_id for consistent fingerprinting
                )

                # Fingerprint is automatically set up in AutoTraderAPIClient constructor

                logger.debug(f"API Worker {worker_id} initialized with proxy: {worker_proxy and worker_proxy[:20]+'...'}")

                # Process tasks from queue
                tasks_processed = 0
                consecutive_empty_gets = 0

                while tasks_processed < max_tasks_per_worker:
                    task = self.get_next_work(worker_id, timeout=0.2)
                    if task is None:
                        consecutive_empty_gets += 1
                        if consecutive_empty_gets >= 3:  # No work available
                            break
                        continue

                    consecutive_empty_gets = 0

                    try:
                        logger.debug(f"Worker {worker_id} processing: {task.make} {task.model} "
                                   f"({task.min_mileage}-{task.max_mileage} miles)")

                        start_time = time.time()

                        # Use the API client for this mileage range
                        cars = api_client.get_all_cars(
                            make=task.make,
                            model=task.model,
                            postcode=task.postcode,
                            min_year=task.min_year,
                            max_year=task.max_year,
                            min_mileage=task.min_mileage,
                            max_mileage=task.max_mileage,
                            max_pages=task.max_pages
                        )

                        processing_time = time.time() - start_time

                        result = {
                            'success': True,
                            'data': cars,
                            'metadata': {
                                'worker_id': worker_id,
                                'mileage_range': f"{task.min_mileage}-{task.max_mileage}",
                                'car_count': len(cars),
                                'processing_time': round(processing_time, 2),
                                'proxy_used': worker_proxy and worker_proxy[:20] + '...' or 'none'
                            }
                        }

                        task_id = str(task)
                        self.report_result(task_id, result, worker_id)

                        logger.debug(f"Worker {worker_id} completed: {len(cars)} cars in {processing_time:.1f}s")

                        # Update progress callback with total listings scraped
                        if progress_callback:
                            total_listings = self.get_total_listings_scraped()
                            progress_callback(total_listings, total_tasks, self.last_result_status)

                        tasks_processed += 1

                        # Rate limiting (skip in test mode)
                        if not test_mode:
                            time.sleep(random.uniform(0.05, 0.15))

                    except Exception as e:
                        # Check if this is a proxy-related error that should trigger IP blacklisting
                        error_str = str(e).lower()
                        should_blacklist = any(keyword in error_str for keyword in [
                            'cloudflare', '403', 'forbidden', 'blocked', 'challenge',
                            'too many requests', 'rate limit', 'access denied'
                        ])

                        if should_blacklist and proxy_manager and worker_proxy:
                            # Extract IP from proxy URL and blacklist it
                            import re
                            ip_match = re.search(r'@([^:]+):', worker_proxy)
                            if ip_match:
                                blocked_ip = ip_match.group(1)
                                proxy_manager.blacklist_ip(blocked_ip, reason=f"Worker {worker_id} error: {str(e)[:50]}")
                                logger.warning(f"Worker {worker_id}: Blacklisted IP {blocked_ip} due to error: {str(e)[:50]}")

                                # Try to get a new proxy for this worker
                                new_proxy = proxy_manager.get_proxy()
                                if new_proxy:
                                    logger.info(f"Worker {worker_id}: Switched to new IP after blacklisting")
                                    # Create new API client with fresh IP and fingerprint
                                    api_client = AutoTraderAPIClient(
                                        proxy=new_proxy,
                                        verify_ssl=False,
                                        worker_id=worker_id
                                    )
                                    worker_proxy = new_proxy

                        error_result = {
                            'success': False,
                            'error': str(e),
                            'data': [],
                            'metadata': {
                                'worker_id': worker_id,
                                'mileage_range': f"{task.min_mileage}-{task.max_mileage}",
                                'car_count': 0,
                                'processing_time': 0,
                                'proxy_used': worker_proxy and worker_proxy[:20] + '...' or 'none',
                                'ip_blacklisted': should_blacklist
                            }
                        }
                        task_id = str(task)
                        self.report_result(task_id, error_result, worker_id)
                        logger.error(f"Worker {worker_id} error processing task: {e}")

                # Cleanup worker resources
                if hasattr(api_client, 'session'):
                    api_client.session.close()
                logger.debug(f"API Worker {worker_id} finished, processed {tasks_processed} tasks")

            except Exception as e:
                logger.error(f"Worker {worker_id} thread error: {e}")

        # Start all worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit worker threads
            futures = [
                executor.submit(worker_thread, worker_id)
                for worker_id in range(1, self.max_workers + 1)
            ]

            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker thread failed: {e}")

        logger.debug(f"✅ Parallel coordination complete - processed {self.completed_count}/{total_tasks} tasks")
        logger.debug(f"❌ Failed workers: {len(self.failed_workers)}")

        return self.results

    # _setup_worker_fingerprint method removed - now handled by unified fingerprinting system


class AutoTraderAPIClient:
    def __init__(self, connection_pool_size=10, optimize_connection=True, proxy=None, proxy_manager=None, verify_ssl=False, worker_id=None):
        """Initialize the AutoTrader API client with unified fingerprinting system.

        Args:
            connection_pool_size: Size of the connection pool (default: 10)
            optimize_connection: Whether to use connection pooling and other optimizations (default: True)
            proxy: Optional proxy URL (e.g., "http://user:pass@host:port")
            proxy_manager: Optional ProxyManager instance for rotation
            verify_ssl: Whether to verify SSL certificates (set to False to ignore SSL errors)
            worker_id: Optional worker ID for consistent fingerprinting
        """
        self.base_url = "https://www.autotrader.co.uk/at-gateway"
        self.proxy_manager = proxy_manager
        self.proxy = proxy
        self.worker_id = worker_id
        self._first_request_made = False  # Track if initial request has been made
        self._session_start_time = time.time()  # Track session start for behavior

        # Generate unified fingerprint for this client instance
        self.fingerprint = FingerprintGenerator.create_http_fingerprint(
            proxy_url=proxy,
            worker_id=worker_id
        )

        # Create session with optimized connection parameters
        self.session = requests.Session()

        # Set SSL verification (can be disabled for testing)
        self.session.verify = verify_ssl

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
            adapter = CustomHTTPAdapter(
                pool_connections=connection_pool_size,
                pool_maxsize=connection_pool_size,
                max_retries=retry_strategy,  # Apply retry strategy
                pool_block=False,  # Don't block when pool is exhausted
                verify_ssl=verify_ssl  # Pass SSL verification setting
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
    
    # randomize_headers method removed - now handled by unified fingerprinting system
    
    def setup_headers(self):
        """Set up the headers with unified fingerprinting system"""
        # Use the CloudFlare-optimized headers from unified fingerprinting
        cloudflare_headers = self.fingerprint.get_cloudflare_headers()
        self.session.headers.update(cloudflare_headers)

        # Configure the custom TLS adapter for HTTPS connections (already done in __init__)
    
    def _evolve_session_headers(self):
        """Subtly evolve headers during session to simulate organic browsing"""
        # Regenerate headers with slight variations using unified fingerprinting
        evolved_headers = self.fingerprint.get_cloudflare_headers()
        self.session.headers.update(evolved_headers)
    
    def search_cars(self, make: str, model: str, postcode: str = "M15 4FN",
                   min_year: int = 2010, max_year: int = 2023,
                   min_mileage: int = 0, max_mileage: int = 100000, page: int = 1, private_only: bool = True,
                   max_retries: int = 2) -> Dict:
        """
        Search for cars using AutoTrader's GraphQL API

        Args:
            make: Car manufacturer
            model: Car model
            postcode: UK postcode for search location
            min_year: Minimum manufacturing year
            max_year: Maximum manufacturing year
            min_mileage: Minimum mileage
            max_mileage: Maximum mileage
            page: Results page number
            private_only: Whether to include only private seller listings
            max_retries: Maximum number of retries for CloudFlare errors

        Returns:
            Dict containing search results data
        """
        # Update headers with geo-consistent fingerprinting
        cloudflare_headers = self.fingerprint.get_cloudflare_headers()
        self.session.headers.update(cloudflare_headers)
        
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
                        {"filter": "min_mileage", "selected": [str(min_mileage)]},
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
                
                # Use geo-consistent headers for initial search request
                search_headers = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': self.fingerprint.accept_language
                }
                
                # Skip old cookie retry logic - now handled by IP rotation above
                
                # Make the initial search page request with custom headers
                initial_response = self.session.get(search_url, timeout=self.timeout, headers=search_headers)
                
                # Check for CloudFlare challenge page
                if "Just a moment" in initial_response.text and retry_count < max_retries:
                    # CloudFlare challenge detection - suppress verbose output
                    pass

                    # Blacklist current proxy if we have one
                    if self.proxy_manager and self.session.proxies.get('http'):
                        current_proxy_url = self.session.proxies['http']
                        # Extract IP from proxy URL (format: http://user:pass@ip:port)
                        import re
                        ip_match = re.search(r'@([^:]+):', current_proxy_url)
                        if ip_match:
                            current_ip = ip_match.group(1)
                            self.proxy_manager.blacklist_ip(current_ip, reason="CloudFlare challenge detected")

                    # Get new proxy and create fresh session
                    if self.proxy_manager:
                        new_proxy = self.proxy_manager.get_proxy()
                        if new_proxy:
                            self.session.proxies = {'http': new_proxy, 'https': new_proxy}
                            # Clear cookies and headers for fresh start
                            self.session.cookies.clear()
                            self.session.headers.clear()
                            # Regenerate fingerprint for new IP
                            self.fingerprint = FingerprintGenerator.create_http_fingerprint(
                                proxy_url=new_proxy,
                                worker_id=self.worker_id
                            )
                            self.setup_headers()

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
                
                # Get API-specific headers from unified fingerprinting
                api_headers = self.fingerprint.get_api_headers()
                api_headers.update({
                    'Origin': 'https://www.autotrader.co.uk',
                    'Referer': search_url
                })
                
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
                    # Request error - suppress verbose output
                    pass
                    retry_count += 1
                    continue
                else:
                    # Log the error and raise it
                    # Fatal request error - suppress verbose output
                    pass
                    raise
        
        # After the retry loop, process the response
        if 'response' in locals():  # Check if response variable exists
            # Log detailed response information for debugging
            if response.status_code != 200:
                # Response error - suppress verbose output
                pass
                
                # Try to extract and log detailed error information
                try:
                    error_content = response.text[:500]  # Get first 500 chars to avoid huge output
                    import logging
                    logging.info(f"Error response content: {error_content}")
                except:
                    pass
                
                # Check for CloudFlare challenge page in response
                if response.status_code == 403 and "Just a moment" in response.text:
                    # CloudFlare challenge in response - suppress verbose output
                    pass
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
                        # Empty JSON response - suppress verbose output
                        pass
                    elif 'errors' in response_json:
                        # Response contains errors - suppress verbose output
                        pass
                    
                    return response_json
                except Exception as e:
                    # JSON parsing error - suppress verbose output
                    pass
                    # Log the first part of the response content
                    try:
                        content = response.text[:500]
                        # Response content debugging - suppress verbose output
                        pass
                    except:
                        pass
                    return None
            
            response.raise_for_status()
            return None  # This will only be reached if raise_for_status doesn't raise an exception
        
        # If we couldn't even get a response object
        return None
    
    def get_all_cars(self, make: str, model: str, postcode: str = "M15 4FN",
                     min_year: int = 2010, max_year: int = 2023,
                     min_mileage: int = 0, max_mileage: int = 100000, max_pages: int = None) -> List[Dict]:
        """
        Get all cars from multiple pages

        Args:
            make: Car manufacturer
            model: Car model (use format like "3 series" not "3-series")
            postcode: UK postcode for search location
            min_year: Minimum manufacturing year
            max_year: Maximum manufacturing year
            min_mileage: Minimum mileage
            max_mileage: Maximum mileage
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of car listings data
        """
        all_cars = []
        page = 1
        
        # Scraping message already shown by caller
        
        # If max_pages is None, keep scraping until there are no more results
        # Otherwise, respect the max_pages limit
        while max_pages is None or page <= max_pages:
            # Print a dot for each page fetched (on same line)
            output_manager = get_output_manager()
            output_manager.api_dots_progress(page)
            
            # Use the search_cars method with retry mechanism
            try:
                data = self.search_cars(
                    make=make,
                    model=model,
                    postcode=postcode,
                    min_year=min_year,
                    max_year=max_year,
                    min_mileage=min_mileage,
                    max_mileage=max_mileage,
                    page=page,
                    max_retries=2  # Allow up to 2 retries for CloudFlare issues
                )
            except Exception as e:
                output_manager = get_output_manager()
                output_manager.error("AutoTrader API access blocked by Cloudflare")
                break
            
            if not data or len(data) == 0:
                output_manager = get_output_manager()
                output_manager.error(f"No data returned from API on page {page}")
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
        
        return all_cars

    def get_all_cars_with_mileage_splitting(self, make: str, model: str, postcode: str = "M15 4FN",
                                          min_year: int = 2010, max_year: int = 2023,
                                          max_mileage: int = 100000, max_pages: int = None) -> List[Dict]:
        """
        Get all cars using mileage range splitting to bypass AutoTrader's 2000 listing limit.

        This method splits the search into multiple mileage ranges and combines the results
        to ensure complete coverage of all available vehicles.

        Args:
            make: Car manufacturer
            model: Car model (use format like "3 series" not "3-series")
            postcode: UK postcode for search location
            min_year: Minimum manufacturing year
            max_year: Maximum manufacturing year
            max_mileage: Maximum mileage (used to determine range upper bound)
            max_pages: Maximum number of pages to fetch per range (None for all)

        Returns:
            List of car listings data (deduplicated)
        """
        # Import the mileage ranges from config
        try:
            from config.config import MILEAGE_RANGES_FOR_SPLITTING, ENABLE_MILEAGE_SPLITTING
        except ImportError:
            # Fallback to default ranges if config import fails
            MILEAGE_RANGES_FOR_SPLITTING = [
                (0, 20000),
                (20001, 40000),
                (40001, 60000),
                (60001, 80000),
                (80001, 100000)
            ]
            ENABLE_MILEAGE_SPLITTING = True

        # If mileage splitting is disabled, use the original method
        if not ENABLE_MILEAGE_SPLITTING:
            return self.get_all_cars(make, model, postcode, min_year, max_year, 0, max_mileage, max_pages)

        # Scraping message already shown by caller

        all_cars = []
        seen_ids = set()  # Track unique car IDs to avoid duplicates

        # Filter ranges to only include those within our max_mileage limit
        applicable_ranges = [
            (min_mile, min(max_mile, max_mileage))
            for min_mile, max_mile in MILEAGE_RANGES_FOR_SPLITTING
            if min_mile <= max_mileage
        ]

        for range_idx, (min_mile, max_mile) in enumerate(applicable_ranges):
            try:
                output_manager = get_output_manager()
                output_manager.api_dots_progress(page)  # Progress indicator

                # Get cars for this mileage range
                range_cars = self.get_all_cars(
                    make=make,
                    model=model,
                    postcode=postcode,
                    min_year=min_year,
                    max_year=max_year,
                    min_mileage=min_mile,
                    max_mileage=max_mile,
                    max_pages=max_pages
                )

                # Deduplicate based on deal_id or url
                for car in range_cars:
                    car_id = car.get('deal_id') or car.get('url', '')
                    if car_id and car_id not in seen_ids:
                        seen_ids.add(car_id)
                        all_cars.append(car)

            except Exception as e:
                # Mileage range error - suppress verbose output
                pass
                continue  # Continue with other ranges

        return all_cars

    def get_all_cars_parallel(self, make: str, model: str, postcode: str = "M15 4FN",
                            min_year: int = 2010, max_year: int = 2023,
                            max_mileage: int = 100000, max_pages: int = None,
                            progress_callback: Optional[callable] = None,
                            test_mode: bool = False, use_parallel: bool = True) -> List[Dict]:
        """
        Get all cars with optional parallel processing for maximum speed.

        Args:
            make: Car manufacturer
            model: Car model
            postcode: UK postcode for search location
            min_year: Minimum manufacturing year
            max_year: Maximum manufacturing year
            max_mileage: Maximum mileage
            max_pages: Maximum pages per range
            progress_callback: Optional progress callback
            test_mode: Skip delays in test mode
            use_parallel: Whether to use parallel processing (default: True)

        Returns:
            List of car listings data
        """
        # Use internal parallel processing when requested
        if use_parallel:
            try:
                # Import mileage ranges from config
                try:
                    from config.config import MILEAGE_RANGES_FOR_SPLITTING
                except ImportError:
                    MILEAGE_RANGES_FOR_SPLITTING = [
                        (0, 20000),
                        (20001, 40000),
                        (40001, 60000),
                        (60001, 80000),
                        (80001, 100000)
                    ]

                # Filter ranges to only include those within our max_mileage limit
                applicable_ranges = [
                    (min_mile, min(max_mile, max_mileage))
                    for min_mile, max_mile in MILEAGE_RANGES_FOR_SPLITTING
                    if min_mile <= max_mileage
                ]

                # Parallel processing messages removed for clean console output

                # Create tasks for each mileage range
                tasks = [
                    APITask(
                        make=make,
                        model=model,
                        postcode=postcode,
                        min_year=min_year,
                        max_year=max_year,
                        min_mileage=min_mile,
                        max_mileage=max_mile,
                        max_pages=max_pages
                    )
                    for min_mile, max_mile in applicable_ranges
                ]

                # Initialize coordinator and process tasks
                coordinator = ParallelCoordinator(max_workers=len(applicable_ranges))
                results = coordinator.process_batch(
                    tasks=tasks,
                    progress_callback=progress_callback,
                    test_mode=test_mode,
                    max_tasks_per_worker=1  # Each worker processes exactly one mileage range
                )

                # Combine and deduplicate results from all workers
                all_cars = []
                seen_ids = set()
                total_cars_found = 0

                for task_id, result in results.items():
                    if result.get('success', False):
                        cars = result.get('data', [])
                        total_cars_found += len(cars)

                        # Deduplicate based on deal_id or url
                        for car in cars:
                            car_id = car.get('deal_id') or car.get('url', '')
                            if car_id and car_id not in seen_ids:
                                seen_ids.add(car_id)
                                all_cars.append(car)

                logger.debug(f"✅ Parallel scraping complete for {make} {model}")
                logger.debug(f"📊 Results: {len(all_cars)} unique cars from {total_cars_found} total "
                           f"(deduplication rate: {((total_cars_found - len(all_cars)) / max(total_cars_found, 1) * 100):.1f}%)")

                return all_cars

            except Exception as e:
                # Parallel scraping failed - suppress verbose output
                pass

        # Fallback to sequential mileage splitting
        # Using sequential mileage splitting - suppress verbose output
        pass
        return self.get_all_cars_with_mileage_splitting(
            make=make,
            model=model,
            postcode=postcode,
            min_year=min_year,
            max_year=max_year,
            max_mileage=max_mileage,
            max_pages=max_pages
        )

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
            
            # Extract mileage from badges, description, or subtitle (moved from later in function)
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

            # Round up to nearest 10,000 for URL parameter
            if mileage > 0:
                max_mileage = ((mileage // 10000) + 1) * 10000
            else:
                max_mileage = 10000  # Default if no mileage found

            
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
        
        # Mileage extraction moved to earlier in function for URL building
        # subtitle and description already extracted above
        
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
        # Car data conversion error - suppress verbose output
        pass
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
        # Even with no text, apply petrol fallback
        specs['fuel_type'] = 'Petrol'
        specs['transmission'] = 'Manual'  # Also default transmission
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
        'petrol': ['petrol', 'tsi', 'tfsi', 'fsi', 'vvt', 'vtech', 'turbo petrol', '16v', 't-jet', 'multiair', 'gti'],
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

    # Last resort: default to petrol if no fuel type detected
    # This catches edge cases not covered by pattern matching
    # Petrol is the most common fuel type in UK, especially for performance cars
    if not specs['fuel_type']:
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA ADAPTER - Consolidated from data_adapter.py
# ══════════════════════════════════════════════════════════════════════════════

class NetworkDataAdapter:
    """
    Converts vehicle data from API client processed format to analyzer-compatible format.
    Consolidated from services/data_adapter.py for better architecture.
    Updated to work with the processed format from AutoTraderAPIClient.get_all_cars()
    """

    @staticmethod
    def _extract_fuel_type_from_title(vehicle_data: Dict) -> str:
        """
        Extract fuel type from vehicle title/subtitle when API field is missing.
        This is critical fallback logic since fuel type is important for analysis.
        Uses enhanced patterns from both original implementations.
        """
        # Build full title from available parts
        title_parts = []
        if vehicle_data.get('title'):
            title_parts.append(vehicle_data['title'])
        if vehicle_data.get('subtitle'):
            title_parts.append(vehicle_data['subtitle'])

        full_title = ' '.join(title_parts).lower()

        if not full_title:
            return 'Petrol'  # Default to petrol even with no text

        # Unified comprehensive fuel type keyword matching (merged from both implementations)
        fuel_keywords = {
            # Primary fuel types
            'petrol': 'Petrol',
            'diesel': 'Diesel',
            'hybrid': 'Hybrid',
            'electric': 'Electric',
            'plugin': 'Plug-in Hybrid',
            'plug-in': 'Plug-in Hybrid',

            # Diesel-specific patterns (enhanced from both files)
            'tdi': 'Diesel',
            'hdi': 'Diesel',
            'cdti': 'Diesel',
            'dci': 'Diesel',
            'dti': 'Diesel',
            'crdi': 'Diesel',
            'multijet': 'Diesel',
            'bluetec': 'Diesel',
            'bluemotion': 'Diesel',
            'biturbo diesel': 'Diesel',

            # Petrol-specific patterns (enhanced from both files)
            'tsi': 'Petrol',
            'tfsi': 'Petrol',
            'fsi': 'Petrol',
            'gti': 'Petrol',
            'vvt': 'Petrol',
            'vtech': 'Petrol',
            'vtec': 'Petrol',
            '16v': 'Petrol',
            'turbo': 'Petrol',
            'turbo petrol': 'Petrol',
            't-jet': 'Petrol',
            'multiair': 'Petrol',

            # Hybrid patterns (enhanced from both files)
            'hybrid': 'Hybrid',
            'plugin hybrid': 'Hybrid',
            'plug-in hybrid': 'Hybrid',
            'self-charging hybrid': 'Hybrid',
            'h-': 'Hybrid',
            'e-cvt': 'Hybrid',
            'synergy': 'Hybrid',

            # Electric patterns (enhanced from both files)
            'electric': 'Electric',
            'ev': 'Electric',
            'bev': 'Electric',
            'e-': 'Electric',
            'pure electric': 'Electric',
            'zero emission': 'Electric',
        }

        # Check for fuel type indicators
        for keyword, fuel_type in fuel_keywords.items():
            if keyword in full_title:
                return fuel_type

        # Model-specific fallback patterns (from both files)
        if any(pattern in full_title for pattern in ['116d', '118d', '120d', '318d', '320d', '520d']):
            return 'Diesel'

        # Additional petrol fallback patterns
        if any(x in full_title for x in ['16v', 'tsi', 'fsi', 'vvt']):
            return 'Petrol'

        # Last resort: default to petrol if no pattern matches
        # Most UK cars are petrol, especially performance/sports cars
        return 'Petrol'

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
                'seller_type': 'Dealer' if processed_vehicle.get('seller_type') == 'TRADE' else 'Private',
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


# Test function
def test_api_scraping():
    """Test the API scraping with SEAT Ibiza"""
    # Test function - suppress output
    pass
    
    client = AutoTraderAPIClient()
    
    # Test function - suppress output
    pass
    cars = client.get_all_cars("SEAT", "Ibiza", max_pages=3)
    
    # Test function - suppress output
    pass
    
    if len(cars) == 0:
        # Test function - suppress output
        pass
        # Test a single request to see what we get
        response = client.search_cars("SEAT", "Ibiza", page=1)
        # Test function - suppress output
        pass
        return []
    
    # Convert to our deal format
    deals = []
    for car in cars:
        deal = convert_api_car_to_deal(car)
        if deal:
            deals.append(deal)
    
    # Test function - suppress output
    pass
    
    # Show first few deals with all data
    for i, deal in enumerate(deals[:3]):
        if not deal or 'processed' not in deal:
            continue
            
        p = deal['processed']
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        
        # Basic info
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        
        # Vehicle specifications
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        
        # Images
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        # Test function - suppress output
        pass
        
    return deals

if __name__ == "__main__":
    test_api_scraping()