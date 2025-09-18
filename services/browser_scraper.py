"""
Core scraping functionality for AutoTrader.
Multiprocessing-safe version with retail price marker scraper.
"""
from playwright.sync_api import sync_playwright
from playwright_stealth.stealth import Stealth
import pyautogui
import random
import time
import re
import os
import logging
from datetime import datetime
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retail_price_scraper")


class WorkerCoordinator:
    """
    Enhanced worker coordinator for sequential URL distribution and anti-detection.
    Manages 6 workers with unique IPs and fingerprints, global rate limiting.
    """
    def __init__(self, target_rate_per_minute=300, max_workers=7):
        self.target_rate_per_minute = target_rate_per_minute
        self.max_workers = max_workers
        self.work_queue = queue.Queue()
        self.results = {}
        self.completed_count = 0
        self.failed_workers = set()
        self.results_lock = threading.Lock()
        self.rate_limiter = threading.Semaphore(1)  # Global rate coordination
        self.last_request_time = 0
        self.min_interval = 60.0 / target_rate_per_minute  # Seconds between requests
        self.last_result_status = "⚠️ Starting up"
        
    def add_work(self, urls):
        """Add URLs to work queue for sequential distribution."""
        for url in urls:
            self.work_queue.put(url)
            
    def get_next_work(self, worker_id, timeout=1):
        """Get next URL for worker, respecting rate limits and worker health."""
        if worker_id in self.failed_workers:
            return None
            
        try:
            # Use blocking get with timeout for proper parallel processing
            # Workers will wait briefly for new work, enabling true parallelism
            return self.work_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def report_result(self, url, result, worker_id):
        """Report result from worker, track failures for intelligent backoff."""
        with self.results_lock:
            self.results[url] = result
            self.completed_count += 1
            
            # Store result status for progress reporting
            self.last_result_status = self._get_result_status(result)
            
            # Check if this worker should be paused due to detection
            if self._is_worker_detected(result):
                self.failed_workers.add(worker_id)
                logger.debug(f"Worker {worker_id} detected as blocked, pausing")
        
        # Mark the task as done in the queue
        self.work_queue.task_done()
                
    def _get_result_status(self, result):
        """Get detailed status icon and text for result."""
        marker_text = result.get('marker_text', '')
        
        # Navigation errors
        if 'Navigation Error:' in marker_text:
            if 'CloudFlare' in marker_text:
                return "❌ CloudFlare challenge"
            elif '403' in marker_text:
                return "❌ 403 Forbidden (blocked)"
            elif 'Timeout' in marker_text:
                return "❌ Navigation timeout (15s)"
            elif 'aborted' in marker_text:
                return "❌ Request aborted"
            else:
                return f"❌ {marker_text.replace('Navigation Error: ', '')[:60]}"
        
        # Cookie errors
        elif 'Cookie Error:' in marker_text:
            return f"❌ {marker_text.replace('Cookie Error: ', '')[:35]}"
        
        # Element detection errors
        elif 'Element Error:' in marker_text:
            error_detail = marker_text.replace('Element Error: ', '')
            if 'Timeout Debug:' in error_detail:
                # Show detailed debug info for timeouts - extract and show URL prominently
                debug_part = error_detail.replace('Timeout Debug: ', '')
                
                return f"🐛 Debug: {debug_part[:60]}"
            elif 'timeout' in error_detail.lower():
                return "⚠️ Button search timeout"
            elif 'not found' in error_detail:
                return "⚠️ No market button found"
            elif 'not visible' in error_detail:
                return "⚠️ Button hidden/invisible"
            else:
                return f"⚠️ {error_detail[:35]}"
        
        # Skipped cases (imported vehicles, etc.)
        elif 'Skipped:' in marker_text:
            return f"⏭️ {marker_text.replace('Skipped: ', '')}"
        
        # Success cases - show actual marker text found
        elif ('market average' in marker_text.lower() and 
              'Error' not in marker_text):
            # Show the actual marker text found
            return f"✅ Found: \"{marker_text[:35]}\""
        
        else:
            return f"⚠️ Unknown: {marker_text[:40]}"
                
    def _is_worker_detected(self, result):
        """Check if result indicates worker was detected/blocked."""
        marker_text = result.get('marker_text', '')
        return ('Error' in marker_text and 
                ('403' in marker_text or 'blocked' in marker_text.lower() or 
                 'cloudflare' in marker_text.lower()))
                 
    def get_healthy_worker_count(self):
        """Get number of healthy (non-blocked) workers."""
        return max(1, self.max_workers - len(self.failed_workers))
        
    def is_work_complete(self):
        """Check if all work is complete."""
        return self.work_queue.empty()



# Geo-consistent configurations for different regions
GEO_PROFILES = {
    'us': {
        'locales': ['en-US'],
        'timezones': ['America/New_York', 'America/Chicago', 'America/Los_Angeles', 'America/Denver'],
        'user_agent_os': 'Windows NT 10.0; Win64; x64'
    },
    'uk': {
        'locales': ['en-GB'],
        'timezones': ['Europe/London'],
        'user_agent_os': 'Windows NT 10.0; Win64; x64'
    },
    'ca': {
        'locales': ['en-CA'],
        'timezones': ['America/Toronto', 'America/Vancouver', 'America/Edmonton'],
        'user_agent_os': 'Macintosh; Intel Mac OS X 10_15_7'
    },
    'au': {
        'locales': ['en-AU'],
        'timezones': ['Australia/Sydney', 'Australia/Melbourne'],
        'user_agent_os': 'Windows NT 10.0; Win64; x64'
    }
}

# Common realistic screen resolutions
REALISTIC_RESOLUTIONS = [
    (1920, 1080),  # Full HD - most common
    (1366, 768),   # Laptop standard
    (1440, 900),   # MacBook Air
    (1536, 864),   # Windows scaled
    (1280, 720),   # HD
    (1600, 900),   # 16:9 widescreen
    (2560, 1440),  # 1440p
    (1680, 1050),  # 16:10 widescreen
]

def detect_geo_profile_from_proxy(proxy_url):
    """Detect geographic profile from proxy URL or IP"""
    if not proxy_url:
        return 'uk'  # Default to UK for AutoTrader
    
    # Simple geo detection - in real implementation, you'd use IP geolocation
    proxy_lower = proxy_url.lower()
    if any(x in proxy_lower for x in ['us', 'usa', 'america', 'newyork', 'chicago', 'los']):
        return 'us'
    elif any(x in proxy_lower for x in ['uk', 'london', 'britain']):
        return 'uk'
    elif any(x in proxy_lower for x in ['ca', 'canada', 'toronto', 'vancouver']):
        return 'ca'
    elif any(x in proxy_lower for x in ['au', 'australia', 'sydney']):
        return 'au'
    else:
        return 'uk'  # Default for AutoTrader

def human_mouse_movement(width=None, height=None):
    """Perform human-like mouse movements using pyautogui"""
    try:
        # Get screen size if not provided
        if width is None or height is None:
            screen_width, screen_height = pyautogui.size()
        else:
            screen_width, screen_height = width, height
        
        # Random natural mouse movement
        target_x = random.randint(100, min(screen_width - 100, 800))
        target_y = random.randint(100, min(screen_height - 100, 600))
        
        # Move with natural easing and random duration
        duration = random.uniform(0.5, 1.5)
        pyautogui.moveTo(target_x, target_y, duration=duration, tween=pyautogui.easeInOutQuad)
        
        # Small pause like human would do
        time.sleep(random.uniform(0.1, 0.3))
        
    except Exception as e:
        # pyautogui might fail in headless environments - that's OK
        logger.debug(f"Mouse movement failed (expected in headless): {e}")

def human_delay():
    """Add human-like delays between actions"""
    time.sleep(random.uniform(0.2, 0.5))

class BrowserManager:
    """Manages a persistent browser instance for reuse across multiple scraping operations"""
    def __init__(self, playwright, headless=True, block_resources=True, proxy=None, shared_session_data=None):
        self.playwright = playwright
        self.headless = headless
        self.block_resources = block_resources
        self.proxy = proxy
        self.browser = None
        self.context = None
        self.shared_session_data = shared_session_data or {}  # For cross-worker session sharing
        self.cookies_handled = False  # Track if cookies have been handled for this session
        # Generate realistic, recent user agents dynamically
        self.user_agents = self._generate_realistic_user_agents()
        self._initialize_browser()
    
    def _generate_realistic_user_agents(self):
        """Generate realistic, current user agents with minor variations"""
        import random
        
        # Current Chrome versions (as of 2024)
        chrome_versions = ['120.0.0.0', '121.0.0.0', '122.0.0.0', '123.0.0.0']
        
        # Recent WebKit versions  
        webkit_versions = ['537.36']
        
        # OS variations with realistic versions
        os_variants = [
            'Windows NT 10.0; Win64; x64',
            'Windows NT 11.0; Win64; x64', 
            'Macintosh; Intel Mac OS X 10_15_7',
            'Macintosh; Intel Mac OS X 13_0_0',
            'Macintosh; Intel Mac OS X 14_0_0',
        ]
        
        user_agents = []
        
        for _ in range(8):  # Generate 8 different user agents
            chrome_ver = random.choice(chrome_versions)
            webkit_ver = random.choice(webkit_versions)
            os_variant = random.choice(os_variants)
            
            # Add minor version variations
            minor_ver = random.randint(0, 9)
            chrome_ver_full = f"{chrome_ver[:-1]}{minor_ver}"
            
            user_agent = f"Mozilla/5.0 ({os_variant}) AppleWebKit/{webkit_ver} (KHTML, like Gecko) Chrome/{chrome_ver_full} Safari/{webkit_ver}"
            user_agents.append(user_agent)
        
        return user_agents
    
    def _initialize_browser(self):
        """Initialize ultra-stealth browser with geo-consistent fingerprint"""
        
        # Initial mouse movement before browser launch (human-like)
        human_mouse_movement()
        
        # Detect geo profile from proxy for consistent fingerprinting
        geo_profile_key = detect_geo_profile_from_proxy(self.proxy)
        geo_profile = GEO_PROFILES[geo_profile_key]
        
        # Select realistic resolution (not random)
        viewport_width, viewport_height = random.choice(REALISTIC_RESOLUTIONS)
        
        # Generate geo-consistent user agent
        chrome_versions = ['120.0.0.0', '121.0.0.0', '122.0.0.0', '123.0.0.0']
        chrome_ver = random.choice(chrome_versions)
        selected_user_agent = f"Mozilla/5.0 ({geo_profile['user_agent_os']}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver} Safari/537.36"
        
        # Select geo-consistent locale and timezone
        locale = random.choice(geo_profile['locales'])
        timezone = random.choice(geo_profile['timezones'])
        
        # Using geo profile silently
        
        # Launch browser with ultra-stealth args
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',  # Critical: removes automation flags
                '--disable-dev-shm-usage',                        # Prevents crashes
                '--disable-web-security',                         # Allows cross-origin (sometimes needed)
                '--disable-features=IsolateOrigins,site-per-process',  # Makes fingerprint more normal
                '--no-sandbox'                                    # Needed in some environments
            ]
        )
        
        # Create geo-consistent context
        context_options = {
            'viewport': {'width': viewport_width, 'height': viewport_height},
            'user_agent': selected_user_agent,
            'locale': locale,
            'timezone_id': timezone,
            'device_scale_factor': random.choice([1, 1.25, 1.5, 2]),  # Different monitor DPI
            'java_script_enabled': True,
            'is_mobile': False,
            'has_touch': False  # Explicitly desktop
        }
        
        # Add proxy configuration if provided
        if self.proxy:
            # Parse proxy URL to extract components
            # Expected format: http://username:password@ip:port
            import re
            # Extract username:password if present
            auth_match = re.search(r'://(.*?)@', self.proxy)
            server = None
            
            if auth_match:
                # Get authentication credentials
                auth = auth_match.group(1)
                # Extract server part (after auth)
                server_match = re.search(r'@(.*)', self.proxy)
                if server_match:
                    server = server_match.group(1)
                
                # Split auth into username and password
                if ':' in auth:
                    username, password = auth.split(':', 1)
                    # Add proxy configuration with auth
                    context_options['proxy'] = {
                        'server': f'http://{server}',
                        'username': username,
                        'password': password
                    }
            else:
                # No auth, just extract server
                server_match = re.search(r'://(.*)', self.proxy)
                if server_match:
                    server = server_match.group(1)
                    # Add proxy configuration without auth
                    context_options['proxy'] = {
                        'server': f'http://{server}'
                    }
        
        # Create context with configured settings
        self.context = self.browser.new_context(**context_options)
        
        # Set default timeout - reduced for better performance
        self.context.set_default_timeout(2000)  # 2 seconds default timeout
        
        # Set default navigation waiting options
        self.context.set_default_navigation_timeout(4000)  # 4 seconds for navigation
        
        # Manual stealth JavaScript replaced by playwright-stealth
    
    def get_page(self):
        """Get a new ultra-stealth page with human-like behavior"""
        if not self.browser or not self.context:
            self._initialize_browser()
            
        # Create a new page
        page = self.context.new_page()
        
        # Apply playwright-stealth (20+ evasions)
        stealth_config = Stealth()
        stealth_config.apply_stealth_sync(page)
        
        # Strategic resource blocking (faster + less fingerprinting)
        if self.block_resources:
            # Block heavy media but allow essential resources
            page.route('**/*.{mp3,mp4,avi,mov,wmv,flv,ogg,webm,wav,mkv}', lambda route: route.abort())
            page.route('**/*.{png,jpg,jpeg,gif,svg}', 
                      lambda route: route.continue_() if any(x in route.request.url for x in ['logo', 'icon', 'sprite']) else route.abort())
            # Block tracking but not essential JS
            page.route('**/*{google-analytics,googletagmanager,facebook,twitter,doubleclick,hotjar,mixpanel}*', 
                      lambda route: route.abort())
        
        # Add human-like delay before first page action
        human_delay()
        
        # Perform mouse movement for this viewport
        try:
            viewport = page.viewport_size
            human_mouse_movement(viewport['width'], viewport['height'])
        except:
            human_mouse_movement()  # Fallback to screen size
        
        return page
    
    def handle_cookies_once(self, page):
        """Handle cookies once per browser session on the first real page load."""
        if not self.cookies_handled:
            # Navigate to AutoTrader homepage to handle cookies once
            try:
                page.goto('https://www.autotrader.co.uk/', wait_until='domcontentloaded', timeout=5000)
                handle_cookie_popup(page)
                self.cookies_handled = True
                logger.debug("Cookies handled once for this browser session")
            except Exception as e:
                logger.debug(f"Cookie handling failed: {e}")
                # Continue anyway, mark as handled to avoid retrying
                self.cookies_handled = True
    
    def close(self):
        """Close the browser and context"""
        if self.context:
            try:
                self.context.close()
            except Exception:
                pass
            self.context = None
        
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass
            self.browser = None
    
    def __del__(self):
        """Ensure browser resources are released on garbage collection"""
        self.close()


class EnhancedBrowserManager(BrowserManager):
    """
    Enhanced browser manager with unique fingerprints per worker.
    Each worker gets consistent fingerprint tied to their proxy IP.
    """
    def __init__(self, playwright, worker_id=0, proxy=None, **kwargs):
        self.worker_id = worker_id
        super().__init__(playwright, proxy=proxy, **kwargs)
        
    def _get_worker_fingerprint(self):
        """Generate consistent fingerprint for this worker."""
        # Use worker_id as seed for consistent fingerprinting
        random.seed(self.worker_id + 12345)
        
        # Different browser profiles per worker
        profiles = [
            {
                'os': 'Windows NT 10.0; Win64; x64',
                'chrome_version': '120.0.6099.129',
                'viewport': (1920, 1080),
                'locale': 'en-GB',
                'timezone': 'Europe/London'
            },
            {
                'os': 'Windows NT 11.0; Win64; x64', 
                'chrome_version': '121.0.6167.85',
                'viewport': (1366, 768),
                'locale': 'en-US',
                'timezone': 'America/New_York'
            },
            {
                'os': 'Macintosh; Intel Mac OS X 10_15_7',
                'chrome_version': '119.0.6045.199',
                'viewport': (1440, 900),
                'locale': 'en-GB',
                'timezone': 'Europe/London'
            },
            {
                'os': 'Windows NT 10.0; Win64; x64',
                'chrome_version': '122.0.6261.57',
                'viewport': (1536, 864),
                'locale': 'en-CA',
                'timezone': 'America/Toronto'
            },
            {
                'os': 'Macintosh; Intel Mac OS X 13_0_0',
                'chrome_version': '120.0.6099.71',
                'viewport': (1680, 1050),
                'locale': 'en-AU',
                'timezone': 'Australia/Sydney'
            },
            {
                'os': 'Windows NT 10.0; Win64; x64',
                'chrome_version': '121.0.6167.139',
                'viewport': (1280, 720),
                'locale': 'en-US',
                'timezone': 'America/Chicago'
            }
        ]
        
        # Reset random to normal state
        random.seed()
        
        # Return consistent profile for this worker
        return profiles[self.worker_id % len(profiles)]
        
    def _initialize_browser(self):
        """Initialize browser with worker-specific fingerprint."""
        fingerprint = self._get_worker_fingerprint()
        
        # Use the worker-specific fingerprint
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--no-sandbox'
            ]
        )
        
        context_options = {
            'viewport': {'width': fingerprint['viewport'][0], 'height': fingerprint['viewport'][1]},
            'user_agent': f"Mozilla/5.0 ({fingerprint['os']}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{fingerprint['chrome_version']} Safari/537.36",
            'locale': fingerprint['locale'],
            'timezone_id': fingerprint['timezone'],
            'device_scale_factor': random.choice([1, 1.25, 1.5, 2]),
            'java_script_enabled': True,
            'is_mobile': False,
            'has_touch': False
        }
        
        # Add proxy configuration if provided
        if self.proxy:
            import re
            auth_match = re.search(r'://(.*?)@', self.proxy)
            server = None
            
            if auth_match:
                auth = auth_match.group(1)
                server_match = re.search(r'@(.*)', self.proxy)
                if server_match:
                    server = server_match.group(1)
                
                if ':' in auth:
                    username, password = auth.split(':', 1)
                    context_options['proxy'] = {
                        'server': f'http://{server}',
                        'username': username,
                        'password': password
                    }
            else:
                server_match = re.search(r'://(.*)', self.proxy)
                if server_match:
                    server = server_match.group(1)
                    context_options['proxy'] = {
                        'server': f'http://{server}'
                    }
        
        self.context = self.browser.new_context(**context_options)
        self.context.set_default_timeout(2000)
        self.context.set_default_navigation_timeout(4000)


def setup_browser(playwright, headless=True):
    """Set up and return a browser instance with optimized settings (legacy function for backward compatibility)"""
    # Use the new dynamic user agent generation
    def generate_user_agent():
        chrome_versions = ['120.0.0.0', '121.0.0.0', '122.0.0.0', '123.0.0.0']
        os_variants = [
            'Windows NT 10.0; Win64; x64',
            'Windows NT 11.0; Win64; x64',
            'Macintosh; Intel Mac OS X 10_15_7',
            'Macintosh; Intel Mac OS X 13_0_0',
            'Macintosh; Intel Mac OS X 14_0_0',
        ]
        
        chrome_ver = random.choice(chrome_versions)
        os_variant = random.choice(os_variants)
        minor_ver = random.randint(0, 9)
        chrome_ver_full = f"{chrome_ver[:-1]}{minor_ver}"
        
        return f"Mozilla/5.0 ({os_variant}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver_full} Safari/537.36"
    
    # Launch browser with balanced optimizations
    browser = playwright.chromium.launch(
        headless=headless,
        args=[
            '--disable-dev-shm-usage',
            '--no-sandbox',
            '--disable-extensions',
            '--disable-default-apps',
            '--disable-component-extensions-with-background-pages',
            '--mute-audio',
            '--disable-background-networking',
            '--disable-sync',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding'
        ]
    )
    
    # Create context with realistic browser settings
    selected_user_agent = generate_user_agent()
    context = browser.new_context(
        user_agent=selected_user_agent,
        viewport={'width': 1280, 'height': 720},
        java_script_enabled=True,
        locale='en-GB',
        timezone_id='Europe/London',
        extra_http_headers={
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': f'"Not_A Brand";v="8", "Chromium";v="{selected_user_agent.split("Chrome/")[1].split(".")[0]}", "Google Chrome";v="{selected_user_agent.split("Chrome/")[1].split(".")[0]}"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"' if 'Windows' in selected_user_agent else '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }
    )
    
    # Set default timeouts - reduced for better performance
    context.set_default_timeout(2000)  # 2 seconds
    context.set_default_navigation_timeout(4000)  # 4 seconds
    
    return browser, context

def handle_cookie_popup(page):
    """Simple and fast cookie popup handling"""
    # Quick 2-second check for cookie popup
    try:
        # Look for cookie popup with a very short timeout
        cookie_element = page.wait_for_selector('#notice button.sp_choice_type_11', timeout=200)
        if cookie_element and cookie_element.is_visible():
            # Found cookie popup - try keyboard shortcut first
            try:
                # Press Tab 11 times to reach Accept All button (last-focusable-el)
                for _ in range(11):
                    page.keyboard.press('Tab')
                page.wait_for_timeout(50)
                page.keyboard.press('Enter') 
                page.wait_for_timeout(100)
                return True
            except:
                # Keyboard failed, try one simple click
                try:
                    cookie_element.click(timeout=1000)
                    page.wait_for_timeout(100)
                    return True
                except:
                    pass
    except:
        # No cookie popup found - continue without delay
        pass
    
    return False

def is_advertisement(vehicle, page):
    """Check if a vehicle listing is an advertisement"""
    # Check for ad indicators
    ad_span1 = vehicle.query_selector('span[data-testid="FEATURED_LISTING"]')
    if ad_span1 and ad_span1.inner_text().strip() == "Ad":
        return True

    ad_span2 = vehicle.query_selector('span[data-testid="YOU_MAY_ALSO_LIKE"]')
    if ad_span2 and ad_span2.inner_text().strip() == "You may also like":
        return True

    # Check URL for ad indicators
    url_elements = vehicle.query_selector_all('a')
    for url_element in url_elements:
        href = url_element.get_attribute('href') or ""
        if "journey=FEATURED_LISTING_JOURNEY" in href:
            return True

    return False

def extract_vehicle_details(vehicle, page):
    """Extract details from a vehicle listing"""
    # Extract data using the exact selectors
    make_model_element = vehicle.query_selector('a[data-testid="search-listing-title"]')
    subtitle_element = vehicle.query_selector('p[data-testid="search-listing-subtitle"]')
    price_element = vehicle.query_selector('span.at__sc-1n64n0d-8')
    mileage_element = vehicle.query_selector('li[data-testid="mileage"]')
    year_element = vehicle.query_selector('li[data-testid="registered_year"]')
    
    # Extract BOTH image URLs
    image_url = ""
    image_url_2 = ""
    
    # Try to get first image
    first_image_element = vehicle.query_selector('a[data-testid="carousel-image-1"] img.main-image')
    if first_image_element:
        image_url = first_image_element.get_attribute('src')
    
    # Try to get second image
    second_image_element = vehicle.query_selector('a[data-testid="carousel-image-2"] img.main-image')
    if second_image_element:
        image_url_2 = second_image_element.get_attribute('src')
    
    # Get seller type
    private_seller_element = vehicle.query_selector('p[data-testid="private-seller"]')
    seller_type = "Private" if private_seller_element else "Dealer"
    
    # Get location only (distance removed for user-specific calculation later)
    location_element = vehicle.query_selector('span[data-testid="search-listing-location"]')
    location = "Unknown"
    if location_element:
        location_text = location_element.inner_text()
        # Extract just the location name, ignoring distance in parentheses
        location_match = re.match(r'(.*?)\s*\(.*?\)', location_text)
        if location_match:
            location = location_match.group(1).strip()
        else:
            location = location_text.strip()
    
    # Get URL
    url_element = vehicle.query_selector('a[data-testid="search-listing-title"]')
    listing_url = ""
    if url_element:
        relative_url = url_element.get_attribute('href')
        listing_url = f"https://www.autotrader.co.uk{relative_url}" if relative_url.startswith('/') else relative_url
    
    # Process data
    if not make_model_element or not price_element:
        return None
        
    make_model_text = make_model_element.inner_text().strip()
    parts = make_model_text.split(' ', 1)
    make = parts[0] if len(parts) > 0 else "Unknown"
    
    # Extract model name
    if len(parts) > 1:
        model = parts[1].split('\n')[0]
    else:
        model = "Unknown"
    
    subtitle = subtitle_element.inner_text().strip() if subtitle_element else "N/A"
    
    price_text = price_element.inner_text().strip()
    price_numeric = float(''.join(c for c in price_text if c.isdigit() or c == '.'))
    
    mileage_text = mileage_element.inner_text().strip() if mileage_element else "N/A"
    mileage = int(''.join(c for c in mileage_text if c.isdigit()) or 0)
    
    year_text = year_element.inner_text().strip() if year_element else "N/A"
    year = int(year_text.split(' ')[0] or 0)
    
    # Return vehicle data with BOTH image URLs (distance removed)
    return {
        'make': make,
        'model': model,
        'spec': subtitle,
        'price_numeric': price_numeric,
        'mileage': mileage,
        'year': year,
        'seller_type': seller_type,
        'location': location,
        'url': listing_url,
        'image_url': image_url,
        'image_url_2': image_url_2  # Add second image URL
    }

def extract_vehicles_from_page(page, config_name=None):
    """Extract vehicle data from the current page"""
    vehicles_data = []
    vehicles = page.query_selector_all('div[data-testid="advertCard"]')
    non_ad_count = 0
    filtered_count = 0
    
    # Extract expected make/model if config_name is provided
    expected_make, expected_model = None, None
    if config_name:
        expected_make, expected_model = extract_make_and_model_from_config(config_name)
    
    for vehicle in vehicles:
        try:
            # Check if this is an ad and skip if it is
            if is_advertisement(vehicle, page):
                continue
            
            # Extract vehicle details
            vehicle_data = extract_vehicle_details(vehicle, page)
            if vehicle_data:
                non_ad_count += 1
                
                # Filter by make/model if specified
                if expected_make and expected_model:
                    matches, reason = check_make_model_match(
                        vehicle_data.get('make', ''), 
                        expected_make, 
                        expected_model
                    )
                    if not matches:
                        filtered_count += 1
                        continue
                
                vehicles_data.append(vehicle_data)
                
        except Exception as e:
            continue
    
    return vehicles_data, non_ad_count, filtered_count

def should_continue_to_next_page(page, current_page, non_ad_count, vehicles_count):
    """Determine whether to continue to the next page"""
    if non_ad_count == 0:
        return False
    
    # End scrape if we find 6 or fewer listings on a page (likely reaching the end)
    if vehicles_count <= 6:
        return False
    
    # Special case: if we're past page 20 and found fewer than 3 non-ad listings, likely reaching the end
    if current_page > 20 and non_ad_count < 3:
        return False
    
    return True

# Placeholder functions for compatibility
def scrape_vehicle_listings(playwright, config, worker_id=None):
    """Placeholder - function removed during cleanup"""
    return []

def extract_make_and_model_from_config(config_name):
    """Extract make and model from config name"""
    if not config_name:
        return None, None
    parts = config_name.split()
    if len(parts) >= 2:
        return parts[0].lower(), parts[1].lower()
    return None, None

def check_make_model_match(vehicle_title, expected_make, expected_model):
    """Check if vehicle matches expected make/model"""
    if not vehicle_title:
        return False, "No title"
    title_lower = vehicle_title.lower()
    if expected_make.lower() in title_lower and expected_model.lower() in title_lower:
        return True, "Match found"
    return False, "No match"

def filter_vehicles_by_make_model(vehicles_data, config_name):
    """Filter vehicles by make/model"""
    return vehicles_data

def scrape_price_marker(url, headless=True):
    """Scrape price marker from a URL"""
    try:
        with sync_playwright() as playwright:
            browser_manager = BrowserManager(playwright, headless=headless)
            page = browser_manager.get_page()
            
            # Navigate to the page
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            
            # Handle cookie popup
            handle_cookie_popup(page)
            
            # Look for price marker elements
            price_marker_selectors = [
                'button:text("market average")',
                '*:text("above market average")',
                '*:text("below market average")', 
                '*:text("Close to market average")',
                'button:has-text("market average")',
                'p:has-text("market average")',
            ]
            
            marker_found = False
            marker_text = None
            
            for selector in price_marker_selectors:
                try:
                    element = page.wait_for_selector(selector, timeout=3000, state='attached')
                    
                    if element and element.is_visible():
                        marker_text = element.inner_text()
                        # Clean the marker text by removing unwanted suffix
                        if marker_text and "Read more" in marker_text:
                            marker_text = marker_text.split("Read more")[0].strip()
                        marker_found = True
                        break
                        
                except Exception:
                    continue
            
            if not marker_found:
                # Try JavaScript approach
                js_result = page.evaluate("""
                    () => {
                        const marketPhrases = ['below market average', 'above market average', 'Close to market average', 'market average'];
                        const elementTypes = ['button', 'p', 'span', 'div'];
                        
                        for (const type of elementTypes) {
                            const elements = document.querySelectorAll(type);
                            for (const element of elements) {
                                const text = element.innerText || element.textContent;
                                if (!text) continue;
                                
                                for (const phrase of marketPhrases) {
                                    if (text.includes(phrase)) {
                                        return text.split("Read more")[0].trim();
                                    }
                                }
                            }
                        }
                        return null;
                    }
                """)
                
                if js_result:
                    marker_text = js_result
                    marker_found = True
            
            # Extract price information
            price = 0
            try:
                main_price_element = page.query_selector('span[data-testid="price"]')
                if main_price_element:
                    price_text = main_price_element.inner_text().strip()
                    # Extract numeric value from price text
                    import re
                    price_match = re.search(r'£?([\d,]+)', price_text)
                    if price_match:
                        price = int(price_match.group(1).replace(',', ''))
            except Exception:
                price = 0
            
            # Calculate market difference (placeholder for now)
            market_difference = 0
            if marker_text:
                if 'above market average' in marker_text.lower():
                    market_difference = 1
                elif 'below market average' in marker_text.lower():
                    market_difference = -1
            
            page.close()
            browser_manager.close()
            
            return {
                'price': price,
                'market_difference': market_difference,
                'marker_text': marker_text or 'No market marker found'
            }
            
    except Exception as e:
        return {
            'price': 0,
            'market_difference': 0,
            'marker_text': f'Error: {str(e)}'
        }

def batch_scrape_price_markers(urls, headless=True, progress_callback=None, test_mode=False, **kwargs):
    """Batch scrape price markers using optimized browser pool processing"""
    if progress_callback:
        progress_callback(0, len(urls))
    
    # Use optimized browser pool processing for better performance
    return batch_scrape_price_markers_optimized(
        urls=urls,
        headless=headless,
        progress_callback=progress_callback,
        test_mode=test_mode,
        **kwargs
    )

def batch_scrape_price_markers_threaded(*args, **kwargs):
    """Threaded batch scrape"""
    return {}

def batch_scrape_price_markers_sequential(urls, headless=True, test_mode=False, delay_between_requests=2):
    """Sequential batch scrape"""
    results = {}
    for url in urls:
        try:
            result = scrape_price_marker(url, headless=headless)
            results[url] = result
            if not test_mode:
                time.sleep(delay_between_requests)
        except Exception as e:
            results[url] = {
                'price': 0,
                'market_difference': 0,
                'marker_text': f'Error: {str(e)}'
            }
    return results

def batch_scrape_price_markers_staggered_sequential(urls, headless=True, test_mode=False, delay_between_requests=0.5):
    """Staggered sequential batch scrape"""
    return batch_scrape_price_markers_sequential(urls, headless, test_mode, delay_between_requests)

def batch_scrape_price_markers_queue_workers(*args, **kwargs):
    """Queue workers batch scrape"""
    return {}

def batch_scrape_price_markers_smart_scaling(*args, **kwargs):
    """Smart scaling batch scrape"""
    return {}

def _worker_wrapper(params):
    """Wrapper to put worker results into shared dictionary - must be at module level"""
    worker_id, url_batch, shared_dict, test_mode, headless = params
    try:
        result = _multiprocessing_worker((worker_id, url_batch, test_mode, headless))
        shared_dict.update(result)
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}")
        # Add error results
        for url in url_batch:
            shared_dict[url] = {
                'price': 0,
                'market_difference': 0,
                'marker_text': f'Worker {worker_id} error: {str(e)}'
            }

def _multiprocessing_worker(url_batch_with_params):
    """Multiprocessing worker function - must be at module level"""
    worker_id, url_batch, test_mode_flag, headless = url_batch_with_params
    
    try:
        # Each worker process gets its own Playwright instance
        with sync_playwright() as playwright:
            browser_manager = BrowserManager(playwright, headless=headless)
            page = browser_manager.get_page()
            
            batch_results = {}
            
            for i, url in enumerate(url_batch):
                try:
                    # Navigate to the page
                    page.goto(url, wait_until='domcontentloaded', timeout=15000)
                    
                    # Handle cookie popup
                    handle_cookie_popup(page)
                    
                    # Look for price marker elements
                    price_marker_selectors = [
                        'button:text("market average")',
                        '*:text("above market average")',
                        '*:text("below market average")', 
                        '*:text("Close to market average")',
                        'button:has-text("market average")',
                        'p:has-text("market average")',
                    ]
                    
                    marker_found = False
                    marker_text = None
                    
                    for selector in price_marker_selectors:
                        try:
                            element = page.wait_for_selector(selector, timeout=3000, state='attached')
                            
                            if element and element.is_visible():
                                marker_text = element.inner_text()
                                if marker_text and "Read more" in marker_text:
                                    marker_text = marker_text.split("Read more")[0].strip()
                                marker_found = True
                                break
                                
                        except Exception:
                            continue
                    
                    if not marker_found:
                        # Try JavaScript approach
                        js_result = page.evaluate("""
                            () => {
                                const marketPhrases = ['below market average', 'above market average', 'Close to market average', 'market average'];
                                const elementTypes = ['button', 'p', 'span', 'div'];
                                
                                for (const type of elementTypes) {
                                    const elements = document.querySelectorAll(type);
                                    for (const element of elements) {
                                        const text = element.innerText || element.textContent;
                                        if (!text) continue;
                                        
                                        for (const phrase of marketPhrases) {
                                            if (text.includes(phrase)) {
                                                return text.split("Read more")[0].trim();
                                            }
                                        }
                                    }
                                }
                                return null;
                            }
                        """)
                        
                        if js_result:
                            marker_text = js_result
                            marker_found = True
                    
                    # Extract price information
                    price = 0
                    try:
                        main_price_element = page.query_selector('span[data-testid="price"]')
                        if main_price_element:
                            price_text = main_price_element.inner_text().strip()
                            import re
                            price_match = re.search(r'£?([\d,]+)', price_text)
                            if price_match:
                                price = int(price_match.group(1).replace(',', ''))
                    except Exception:
                        price = 0
                    
                    # Calculate market difference
                    market_difference = 0
                    if marker_text:
                        if 'above market average' in marker_text.lower():
                            market_difference = 1
                        elif 'below market average' in marker_text.lower():
                            market_difference = -1
                    
                    batch_results[url] = {
                        'price': price,
                        'market_difference': market_difference,
                        'marker_text': marker_text or 'No market marker found'
                    }
                    
                    # Rate limiting handled by main worker thread, no additional delay needed here
                        
                except Exception as e:
                    batch_results[url] = {
                        'price': 0,
                        'market_difference': 0,
                        'marker_text': f'Error: {str(e)}'
                    }
            
            return batch_results
            
    except Exception as e:
        # Return error results for all URLs in this batch
        error_results = {}
        for url in url_batch:
            error_results[url] = {
                'price': 0,
                'market_difference': 0,
                'marker_text': f'Worker error: {str(e)}'
            }
        return error_results

def batch_scrape_price_markers_optimized(urls, headless=True, progress_callback=None, test_mode=False, 
                                       max_concurrent_browsers=6, urls_per_browser_cycle=500, **kwargs):
    """
    Enhanced batch scraping with orchestrated workers, dedicated IPs, and anti-detection.
    
    Args:
        urls: List of URLs to scrape  
        headless: Run browsers in headless mode
        progress_callback: Function to call with progress updates
        test_mode: Skip delays in test mode
        max_concurrent_browsers: Number of concurrent workers (fixed at 6)
        urls_per_browser_cycle: URLs per browser before restart (increased to 500)
    
    Returns:
        Dict mapping URLs to scraping results
    """
    total_urls = len(urls)
    max_workers = 7
    
    logger.info(f"🚀 Starting enhanced orchestration with 7 workers")
    logger.info(f"📊 Processing {total_urls} URLs with browser reuse every {urls_per_browser_cycle} URLs")
    
    # Initialize proxy manager for worker-specific IPs
    try:
        from services.stealth_orchestrator import ProxyManager
        proxy_manager = ProxyManager()
    except:
        proxy_manager = None
    
    # Initialize enhanced coordinator
    coordinator = WorkerCoordinator(target_rate_per_minute=300, max_workers=max_workers)
    coordinator.add_work(urls)
    
    def enhanced_worker_thread(worker_id):
        """Enhanced worker with dedicated IP and unique fingerprint."""
        try:
            # Get dedicated proxy for this worker
            worker_proxy = None
            if proxy_manager:
                worker_proxy = proxy_manager.get_proxy_for_worker(worker_id)
            
            # Initialize browser with worker-specific settings
            with sync_playwright() as playwright:
                browser_manager = EnhancedBrowserManager(
                    playwright, 
                    worker_id=worker_id,
                    proxy=worker_proxy,
                    headless=headless
                )
                page = browser_manager.get_page()
                
                # Handle cookies once for this browser session
                browser_manager.handle_cookies_once(page)
                
                # Process URLs in parallel from coordinator
                urls_processed = 0
                consecutive_empty_gets = 0
                
                while urls_processed < urls_per_browser_cycle:
                    url = coordinator.get_next_work(worker_id, timeout=0.2)
                    if url is None:
                        consecutive_empty_gets += 1
                        # If we've tried and gotten no work, probably done
                        if consecutive_empty_gets >= 1:
                            break
                        continue
                    
                    consecutive_empty_gets = 0  # Reset counter when we get work
                        
                    try:
                        # Use longer timeout for first URL (cold start), shorter for subsequent
                        is_first_url = (urls_processed == 0)
                        navigation_timeout = 10000 if is_first_url else 5000
                        
                        # Log worker activity for parallel verification
                        logger.debug(f"Worker {worker_id} starting URL {urls_processed+1}: {url[-30:]}")
                        
                        # Scrape price marker for this URL (skip cookies since handled once)
                        result = _scrape_single_url(page, url, test_mode, navigation_timeout, skip_cookies=True)
                        coordinator.report_result(url, result, worker_id)
                        
                        logger.debug(f"Worker {worker_id} completed URL {urls_processed+1} - Status: {coordinator.last_result_status}")
                        
                        # Update progress callback with status
                        if progress_callback:
                            progress_callback(coordinator.completed_count, total_urls, coordinator.last_result_status)
                            
                        urls_processed += 1
                        
                        # Per-worker rate limiting: aggressive pacing targeting 300 URLs/min (only if not in test mode)
                        if not test_mode:
                            time.sleep(random.uniform(0.05, 0.15))  # Aggressive pacing for high throughput
                        
                    except Exception as e:
                        error_result = {
                            'price': 0,
                            'market_difference': 0,
                            'marker_text': f'Error: {str(e)}'
                        }
                        coordinator.report_result(url, error_result, worker_id)
                        
                        # Check if this looks like detection - if so, worker will be paused
                        if progress_callback:
                            progress_callback(coordinator.completed_count, total_urls, coordinator.last_result_status)
                
                page.close()
                browser_manager.close()
                
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
    
    # Start all workers with minimal staggered initialization
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all workers immediately (no stagger needed)
        futures = []
        for worker_id in range(max_workers):
            future = executor.submit(enhanced_worker_thread, worker_id)
            futures.append(future)
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    logger.info(f"✅ Enhanced orchestration completed with {len(coordinator.results)} results")
    return coordinator.results


def _scrape_single_url(page, url, test_mode=False, navigation_timeout=5000, skip_cookies=True):
    """Helper function to scrape a single URL using existing page."""
    import time
    start_time = time.time()
    nav_time = 0
    element_time = 0
    
    try:
        # Navigate to the page with detailed error tracking
        nav_start = time.time()
        page.goto(url, wait_until='domcontentloaded', timeout=navigation_timeout)
        nav_time = time.time() - nav_start
        
        # Check for common blocking/error pages
        page_title = page.title().lower()
        page_content = page.content().lower()
        
        # Check for CloudFlare challenge
        if ("just a moment" in page_title or "checking your browser" in page_title or 
            "cloudflare" in page_content or "checking if the site connection is secure" in page_content):
            return {
                'price': 0,
                'market_difference': 0,
                'marker_text': 'Navigation Error: CloudFlare challenge detected'
            }
        
        # Check for 403/blocked pages
        if ("access denied" in page_title or "forbidden" in page_title or 
            "blocked" in page_content or "403" in page_title):
            return {
                'price': 0,
                'market_difference': 0,
                'marker_text': 'Navigation Error: 403 Forbidden (proxy blocked)'
            }
        
    except Exception as nav_error:
        nav_error_msg = str(nav_error).lower()
        if "net::err_aborted" in nav_error_msg:
            return {
                'price': 0,
                'market_difference': 0,
                'marker_text': 'Navigation Error: Request aborted'
            }
        elif "timeout" in nav_error_msg:
            return {
                'price': 0,
                'market_difference': 0,
                'marker_text': 'Navigation Error: Timeout (15s)'
            }
        else:
            return {
                'price': 0,
                'market_difference': 0,
                'marker_text': f'Navigation Error: {str(nav_error)[:30]}'
            }
    
    # Handle cookie popup with error tracking (skip if already handled)
    if not skip_cookies:
        try:
            cookie_result = handle_cookie_popup(page)
            # Note: handle_cookie_popup doesn't return meaningful status, so we'll assume success
        except Exception as cookie_error:
            return {
                'price': 0,
                'market_difference': 0,
                'marker_text': f'Cookie Error: {str(cookie_error)[:40]}'
            }
    
    # Simple, reliable price marker detection with detailed error tracking
    marker_found = False
    marker_text = None
    element_error = None
    
    try:
        # Single selector for all market average scenarios
        element_start = time.time()
        element = page.wait_for_selector('button:has-text("market average")', timeout=1000, state='attached')
        element_time = time.time() - element_start
        
        if element:
            if element.is_visible():
                marker_text = element.inner_text()
                if marker_text and "Read more" in marker_text:
                    marker_text = marker_text.split("Read more")[0].strip()
                marker_found = True
            else:
                element_error = "Button found but not visible"
        else:
            element_error = "Button element not found"
        element_time = time.time() - element_start    
    except Exception as e:
        element_time = time.time() - element_start
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            # Debug timeout cases - gather detailed page state info
            try:
                page_title = page.title()[:40] if page.title() else "No title"
                page_url = page.url[-50:] if page.url else "No URL"  # Last 50 chars
                
                # Check what buttons actually exist
                all_buttons = page.query_selector_all('button')
                button_count = len(all_buttons)
                
                # Get text from first few buttons for analysis
                button_texts = []
                for i, btn in enumerate(all_buttons[:3]):  # First 3 buttons
                    try:
                        btn_text = btn.inner_text()[:25] if btn.inner_text() else "No text"
                        button_texts.append(f"Btn{i+1}:{btn_text}")
                    except:
                        button_texts.append(f"Btn{i+1}:Error")
                
                # Check if "market average" text exists anywhere on page
                page_content = page.content().lower()
                has_market_text = "market average" in page_content
                has_cloudflare = "cloudflare" in page_content or "just a moment" in page_title.lower()
                
                debug_info = f"Title:{page_title}|Buttons:{button_count}|Market:{has_market_text}|CF:{has_cloudflare}"
                if button_texts:
                    debug_info += f"|{','.join(button_texts[:2])}"  # First 2 button texts
                
                # Add URL for manual inspection
                # URL removed to prevent navigation errors
                
                element_error = f"Timeout Debug: {debug_info}"
                
            except Exception:
                element_error = "Button search timeout (2s) - debug failed"
        else:
            element_error = f"Button search error: {str(e)[:30]}"
    
    if not marker_found:
        # First check if this is an imported vehicle (edge case)
        try:
            imported_element = page.wait_for_selector('h2.sc-1n64n0d-7.sc-6sdn0z-13', timeout=500)
            if imported_element and "imported vehicle" in imported_element.inner_text().lower():
                return {
                    'price': 0,
                    'market_difference': 0,
                    'marker_text': 'Skipped: Imported vehicle detected'
                }
        except Exception:
            pass  # Not imported, continue to JavaScript fallback
        
        # Try JavaScript approach
        try:
            js_result = page.evaluate("""
                () => {
                    const marketPhrases = ['below market average', 'above market average', 'Close to market average', 'market average'];
                    const elementTypes = ['button', 'p', 'span', 'div'];
                    
                    for (const type of elementTypes) {
                        const elements = document.querySelectorAll(type);
                        for (const element of elements) {
                            const text = element.innerText || element.textContent;
                            if (!text) continue;
                            
                            for (const phrase of marketPhrases) {
                                if (text.includes(phrase)) {
                                    return text.split("Read more")[0].trim();
                                }
                            }
                        }
                    }
                    return null;
                }
            """)
            
            if js_result:
                marker_text = js_result
                marker_found = True
        except Exception:
            pass
    
    # Extract price information
    price = 0
    try:
        main_price_element = page.query_selector('span[data-testid="price"]')
        if main_price_element:
            price_text = main_price_element.inner_text().strip()
            import re
            price_match = re.search(r'£?([\d,]+)', price_text)
            if price_match:
                price = int(price_match.group(1).replace(',', ''))
    except Exception:
        price = 0
    
    # Calculate market difference
    market_difference = 0
    if marker_text:
        if 'above market average' in marker_text.lower():
            market_difference = 1
        elif 'below market average' in marker_text.lower():
            market_difference = -1
    
    # Log timing breakdown (will be enhanced with worker ID by caller)
    total_time = time.time() - start_time
    logger.debug(f"URL timing - Total: {total_time:.2f}s | Nav: {nav_time:.2f}s | Element: {element_time:.2f}s | URL: {url[-30:]}")
    
    # Return with detailed status
    if marker_found and marker_text:
        return {
            'price': price,
            'market_difference': market_difference,
            'marker_text': marker_text
        }
    elif element_error:
        return {
            'price': price,
            'market_difference': 0,
            'marker_text': f'Element Error: {element_error}'
        }
    else:
        return {
            'price': price,
            'market_difference': 0,
            'marker_text': 'Element Error: Unknown issue'
        }

def format_market_difference(result):
    """Format market difference result"""
    marker_text = result.get('marker_text', '')
    if marker_text and marker_text != 'Test marker':
        return marker_text  # Return the actual marker text
    return f"Market difference: {result.get('market_difference', 0)}"

def print_result(result):
    """Print result"""
    print(result)

def print_batch_results(results):
    """Print batch results"""
    for result in results:
        print(result)

def run_test():
    """Run test"""
    print("Test completed")

def main():
    """Main function"""
    print("Main function executed")
