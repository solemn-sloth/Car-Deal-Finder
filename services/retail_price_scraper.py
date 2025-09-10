"""
Core scraping functionality for AutoTrader.
Multiprocessing-safe version with retail price marker scraper.
"""
from playwright.sync_api import sync_playwright
import random
import time
import re
import os
import logging
from datetime import datetime
import sys

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retail_price_scraper")

class BrowserManager:
    """Manages a persistent browser instance for reuse across multiple scraping operations"""
    def __init__(self, playwright, headless=True, block_resources=True, proxy=None):
        self.playwright = playwright
        self.headless = headless
        self.block_resources = block_resources
        self.proxy = proxy
        self.browser = None
        self.context = None
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        self._initialize_browser()
    
    def _initialize_browser(self):
        """Initialize the browser and context with optimized settings"""
        # Launch browser with optimized settings
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                # Base security settings
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--disable-setuid-sandbox',
                '--no-sandbox',
                
                # Rendering optimizations
                '--disable-accelerated-2d-canvas',
                '--disable-canvas-aa',  # Disable canvas anti-aliasing
                '--disable-2d-canvas-clip-aa',  # Disable canvas clip anti-aliasing
                '--disable-gl-drawing-for-tests',  # Disable GL drawing when possible
                
                # Memory optimizations
                '--disable-extensions',
                '--disable-component-extensions-with-background-pages',
                '--disable-default-apps',
                '--mute-audio',
                '--disable-speech-api',
                
                # CPU optimizations
                '--disable-backgrounding-occluded-windows',
                '--disable-ipc-flooding-protection',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                
                # Disable unnecessary features
                '--disable-features=IsolateOrigins,site-per-process,TranslateUI,BlinkGenPropertyTrees,AcceleratedSmallCanvases,TextBlobs,ReducedReferrerGranularity,Prerender2',
                '--disable-site-isolation-trials',
                '--disable-hang-monitor',
                '--disable-sync',
                
                # Network optimizations
                '--disable-background-networking',
                '--disable-breakpad',
                '--disable-domain-reliability',
                
                # JavaScript optimization flags
                '--js-flags=--lite-mode,--noanalyze-insane-loop-bounds,--max-heap-size=256,--stack-size=64,--use-strict,--no-always-opt,--no-opt-after-turbofan,--no-flush-bytecode,--jitless,--memory-reducer,--no-concurrent-recompilation,--no-incremental-marking,--optimize-for-size'
            ]
        )
        
        # Prepare context options
        context_options = {
            'user_agent': random.choice(self.user_agents),
            'viewport': {'width': 800, 'height': 600},  # Even smaller viewport for minimal resource usage
            'java_script_enabled': True,  # We need JS for price extraction
            # JavaScript execution settings - restrict JS execution to essentials only
            'extra_http_headers': {'Request-Priority': 'low'},  # Mark as low priority to reduce resource allocation
            'bypass_csp': True,  # Bypass Content-Security-Policy for maximum compatibility
            'ignore_https_errors': True,  # Ignore HTTPS errors to prevent failures
            'has_touch': False,  # Desktop mode only
            'is_mobile': False,  # Desktop mode only
            'locale': 'en-GB',
            'timezone_id': 'Europe/London',
            # Reduce memory usage with minimal permissions
            'permissions': ['geolocation'],  # Only allow essential permissions
            # Set reduced device scale factor
            'device_scale_factor': 1,
            # Reduce color depth for performance
            'color_scheme': 'light',  # 'light' uses less resources than 'dark'
            # Disable browser features we don't need
            'reduced_motion': 'reduce',  # Disable animations
            'forced_colors': 'none'
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
        self.context.set_default_timeout(5000)  # 5 seconds default timeout
        
        # Set default navigation waiting options - reduced for better performance
        self.context.set_default_navigation_timeout(8000)  # 8 seconds for navigation
    
    def get_page(self):
        """Get a new page from the existing browser context with resource blocking"""
        if not self.browser or not self.context:
            self._initialize_browser()
            
        # Create a new page
        page = self.context.new_page()
        
        if self.block_resources:
            # PHASE 1: Block specific resource types with more aggressive patterns
            
            # Block ALL images except critical UI sprites
            page.route('**/*.{png,jpg,jpeg,gif,webp,svg,ico,bmp,tiff}', 
                      lambda route: route.continue_() if 'sprite' in route.request.url else route.abort())
            
            # Block ALL fonts and stylesheets
            page.route('**/*.{css,woff,woff2,ttf,otf,eot}', lambda route: route.abort())
            
            # Block ALL tracking, analytics, ads, metrics, and social media resources
            page.route('**/*{analytics,tracking,advertisement,ads,ga,gtm,pixel,social,facebook,twitter,linkedin,collect}*', 
                      lambda route: route.abort())
            
            # Block media files
            page.route('**/*.{mp3,mp4,avi,mov,wmv,flv,ogg,webm,wav,mkv}', lambda route: route.abort())
            
            # Block unnecessary document types
            page.route('**/*.{pdf,doc,docx,xls,xlsx,ppt,pptx}', lambda route: route.abort())
            
            # PHASE 2: Set up a whitelist approach with highly targeted JavaScript control
            page.route('**/*', lambda route: route.continue_() if (
                # Allow essential network requests
                (route.request.resource_type in ['document', 'xhr', 'fetch']) or
                
                # Only allow these specific JavaScript resources:
                (route.request.resource_type == 'script' and (
                    # Core functionality scripts only
                    ('autotrader' in route.request.url and (
                        # Only essential script components
                        'price-indicator' in route.request.url or
                        'search-results' in route.request.url or
                        'vehicle-listing' in route.request.url or
                        'core' in route.request.url
                    )) or
                    # Main entry point scripts that are needed
                    'main' in route.request.url or
                    # Critical data chunks
                    ('chunk' in route.request.url and 'data' in route.request.url) or
                    # Any page loader scripts
                    'webpack' in route.request.url or
                    'polyfill' in route.request.url
                )) or
                
                # Allow specific API endpoints that contain essential data
                ('api' in route.request.url and (
                    'price' in route.request.url or
                    'search' in route.request.url or
                    'listing' in route.request.url or
                    'vehicle' in route.request.url or
                    'metadata' in route.request.url
                ))
            ) else route.abort())
        
        return page
    
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


def setup_browser(playwright, headless=True):
    """Set up and return a browser instance with optimized settings (legacy function for backward compatibility)"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    
    # Launch browser with highly optimized settings
    browser = playwright.chromium.launch(
        headless=headless,
        args=[
            # Base security settings
            '--disable-gpu',
            '--disable-dev-shm-usage',
            '--disable-setuid-sandbox',
            '--no-sandbox',
            
            # Rendering optimizations
            '--disable-accelerated-2d-canvas',
            '--disable-canvas-aa',  # Disable canvas anti-aliasing
            '--disable-2d-canvas-clip-aa',  # Disable canvas clip anti-aliasing
            '--disable-gl-drawing-for-tests',  # Disable GL drawing when possible
            
            # Memory optimizations
            '--disable-extensions',
            '--disable-component-extensions-with-background-pages',
            '--disable-default-apps',
            '--mute-audio',
            '--disable-speech-api',
            
            # CPU optimizations
            '--disable-backgrounding-occluded-windows',
            '--disable-ipc-flooding-protection',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            
            # Disable unnecessary features
            '--disable-features=IsolateOrigins,site-per-process,TranslateUI',
            '--disable-site-isolation-trials',
            '--disable-hang-monitor',
            '--disable-sync',
            
            # JavaScript optimization flags
            '--js-flags=--lite-mode,--max-heap-size=256,--stack-size=64,--use-strict,--no-always-opt,--jitless,--memory-reducer,--no-incremental-marking,--optimize-for-size'
        ]
    )
    
    # Create context with highly optimized settings
    context = browser.new_context(
        user_agent=random.choice(user_agents),
        viewport={'width': 800, 'height': 600},  # Much smaller viewport for minimal resource usage
        java_script_enabled=True,  # We need JS for price extraction
        # JavaScript execution settings - restrict JS execution to essentials only
        extra_http_headers={'Request-Priority': 'low'},  # Mark as low priority to reduce resource allocation
        bypass_csp=True,  # Bypass Content-Security-Policy
        ignore_https_errors=True,  # Ignore HTTPS errors
        has_touch=False,  # Desktop mode only
        is_mobile=False,  # Desktop mode only
        locale='en-GB',
        timezone_id='Europe/London',
        # Reduce memory usage with minimal permissions
        permissions=['geolocation'],  # Only allow essential permissions
        # Set reduced device scale factor
        device_scale_factor=1,
        # Reduce color depth for performance
        color_scheme='light',  # 'light' uses less resources than 'dark'
        # Disable browser features we don't need
        reduced_motion='reduce',  # Disable animations
        forced_colors='none'
    )
    
    # Set default timeouts - reduced for better performance
    context.set_default_timeout(5000)  # 5 seconds
    context.set_default_navigation_timeout(8000)  # 8 seconds
    
    return browser, context

def handle_cookie_popup(page):
    """Handle the cookie consent popup using a direct selector approach"""
    try:
        # Try the direct aria-label selector first - fastest approach
        try:
            # Look for button with aria-label="Accept All"
            accept_button = page.locator('button[aria-label="Accept All"]').first
            if accept_button:
                accept_button.click(timeout=1000)
                logger.info("Accepted cookies using aria-label selector")
                return
        except Exception:
            # If the aria-label selector fails, try other approaches
            pass
            
        # Try locating the button in an iframe as fallback
        try:
            # Find any cookie consent iframe
            frames = page.frame_locator('iframe[id*="sp_message"], iframe[id*="cookie"], iframe[title*="Cookie"], iframe[src*="consent"]')
            # Look for accept button in the frame
            accept_btn = frames.locator('button:has-text("Accept All"), button:has-text("Accept")').first
            
            if accept_btn:
                accept_btn.click(timeout=1000)
                logger.info("Accepted cookies using iframe selector")
        except Exception:
            logger.debug("Could not find or click cookie accept button")
            pass
        
    except Exception:
        # Silently continue on cookie errors - iframe might not be present
        pass

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

    # Check parent elements
    try:
        featured_parent = vehicle.evaluate('node => node.closest("[data-testid=\'ola-trader-seller-listing\']")')
        if featured_parent:
            featured_span = page.evaluate('node => node.querySelector("span[data-testid=\'FEATURED_LISTING\']")', featured_parent)
            if featured_span:
                return True
    except:
        pass
    
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
                # Apply make/model filtering if we have the expected values
                if expected_make and expected_model:
                    vehicle_title = f"{vehicle_data.get('make', '')} {vehicle_data.get('model', '')}".strip()
                    is_match, reason = check_make_model_match(vehicle_title, expected_make, expected_model)
                    
                    if is_match:
                        vehicles_data.append(vehicle_data)
                        non_ad_count += 1
                    else:
                        filtered_count += 1
                else:
                    # No filtering, add all non-ads
                    vehicles_data.append(vehicle_data)
                    non_ad_count += 1
                
        except Exception as e:
            print(f"Error processing a vehicle: {e}")
            continue
    
    return vehicles_data, non_ad_count

def should_continue_to_next_page(page, current_page, non_ad_count, vehicles_count):
    """Determine if we should continue to the next page"""
    # If we found no non-ad listings, stop
    if non_ad_count == 0:
        return False
    
    # End scrape if we find 6 or fewer listings on a page (likely reaching the end)
    if vehicles_count <= 6:
        return False
    
    # Special case: if we're past page 20 and found fewer than 3 non-ad listings, likely reaching the end
    if current_page > 20 and non_ad_count < 3:
        return False
    
    return True

def scrape_vehicle_listings(playwright, config, worker_id=None):
    """Scrape vehicle listings based on configuration"""
    search_url = config['base_url']
    browser, context = setup_browser(playwright)
    page = context.new_page()
    
    # Apply aggressive resource blocking for maximum performance
    
    # Block ALL images except critical UI sprites
    page.route('**/*.{png,jpg,jpeg,gif,webp,svg,ico,bmp,tiff}', 
              lambda route: route.continue_() if 'sprite' in route.request.url else route.abort())
    
    # Block ALL fonts and stylesheets
    page.route('**/*.{css,woff,woff2,ttf,otf,eot}', lambda route: route.abort())
    
    # Block ALL tracking, analytics, ads, metrics, and social media resources
    page.route('**/*{analytics,tracking,advertisement,ads,ga,gtm,pixel,social,facebook,twitter,linkedin,collect,tag,beacon}*', 
              lambda route: route.abort())
    
    # Block media files
    page.route('**/*.{mp3,mp4,avi,mov,wmv,flv,ogg,webm,wav,mkv}', lambda route: route.abort())
    
    # Block unnecessary document types
    page.route('**/*.{pdf,doc,docx,xls,xlsx,ppt,pptx}', lambda route: route.abort())
    
    # Set a whitelist approach with targeted JavaScript control
    page.route('**/*', lambda route: route.continue_() if (
        # Allow essential network requests
        (route.request.resource_type in ['document', 'xhr', 'fetch']) or
        
        # Only allow these specific JavaScript resources:
        (route.request.resource_type == 'script' and (
            # Core functionality scripts only
            ('autotrader' in route.request.url and (
                # Only essential script components
                'price-indicator' in route.request.url or
                'search-results' in route.request.url or
                'vehicle-listing' in route.request.url or
                'core' in route.request.url
            )) or
            # Main entry point scripts that are needed
            'main' in route.request.url or
            # Critical data chunks
            ('chunk' in route.request.url and 'data' in route.request.url) or
            # Any page loader scripts
            'webpack' in route.request.url or
            'polyfill' in route.request.url
        )) or
        
        # Allow specific API endpoints that contain essential data
        ('api' in route.request.url and (
            'price' in route.request.url or
            'search' in route.request.url or
            'listing' in route.request.url or
            'vehicle' in route.request.url
        ))
    ) else route.abort())
    
    # Get process ID for logging
    process_id = os.getpid()
    
    all_vehicles_data = []
    try:
        # No random delay - we'll implement proper rate limiting in the multiprocessing code later
        
        page.goto(search_url)
        handle_cookie_popup(page)
        try:
            page.wait_for_selector('div[data-testid="advertCard"]', timeout=6000)  # Further reduced timeout
        except Exception as e:
            # No artificial wait here either
            logger.error(f"Error waiting for advertCard: {e}")
            return []
        
        current_page = 1
        more_pages = True
        total_scraped = 0
        
        while more_pages:
            vehicles = page.query_selector_all('div[data-testid="advertCard"]')
            if len(vehicles) == 0:
                break
                
            if len(vehicles) == 6:
                all_ads = True
                for vehicle in vehicles:
                    ad_span1 = vehicle.query_selector('span[data-testid="FEATURED_LISTING"]')
                    ad_span2 = vehicle.query_selector('span[data-testid="YOU_MAY_ALSO_LIKE"]')
                    if not ad_span1 and not ad_span2:
                        all_ads = False
                        break
                if all_ads:
                    more_pages = False
                    break
            
            page_vehicles, non_ad_count = extract_vehicles_from_page(page, config['name'])
            all_vehicles_data.extend(page_vehicles)
            total_scraped = len(all_vehicles_data)
            
            # Updated progress printing
            if worker_id:
                print(f"\r[Worker {worker_id}] Scraping {total_scraped} {config['name']} vehicles", end="", flush=True)
            else:
                print(f"\r[Process {process_id}] Scraped {total_scraped} {config['name']} vehicles", end="", flush=True)
            
            more_pages = should_continue_to_next_page(page, current_page, non_ad_count, len(vehicles))
            
            if more_pages:
                current_page += 1
                next_page_url = f"{search_url}&page={current_page}"
                
                # No random delay between pages - rely on network timing
                page.goto(next_page_url)
                try:
                    page.wait_for_selector('div[data-testid="advertCard"]', timeout=6000)  # Reduced timeout
                except Exception as e:
                    more_pages = False
                    
    except Exception as e:
        if worker_id:
            print(f"\n[Worker {worker_id}] Error scraping {config['name']}: {e}")
        else:
            print(f"\n[Process {process_id}] Error scraping {config['name']}: {e}")
        raise
    finally:
        # Clean up
        context.close()
        browser.close()
    
    return all_vehicles_data

def extract_make_and_model_from_config(config_name):
    """Extract make and model from config name"""
    parts = config_name.split()
    if len(parts) >= 2:
        make = parts[0].lower()
        model = parts[1].lower()
        return make, model
    return None, None

def check_make_model_match(vehicle_title, expected_make, expected_model):
    """Check if the vehicle title contains the expected make and model"""
    if not vehicle_title or vehicle_title == "Unknown":
        return False, "Unable to extract title"
    
    title_lower = vehicle_title.lower()
    
    # Handle special cases for make names
    make_variations = {
        'kia': ['kia'],
        'vauxhall': ['vauxhall'],
        'volkswagen': ['volkswagen', 'vw'],
        'fiat': ['fiat'],
        'volvo': ['volvo'],
        'honda': ['honda'],
        'toyota': ['toyota'],
        'seat': ['seat'],
        'nissan': ['nissan'],
        'mercedes': ['mercedes', 'mercedes-benz'],
        'hyundai': ['hyundai'],
        'ford': ['ford'],
        'dacia': ['dacia'],
        'bmw': ['bmw'],
        'audi': ['audi']
    }
    
    # Handle special cases for model names
    model_variations = {
        'sportage': ['sportage'],
        'corsa': ['corsa'],
        'polo': ['polo'],
        'golf': ['golf'],
        '500': ['500', 'five hundred'],
        'v40': ['v40', 'v 40'],
        'civic': ['civic'],
        'yaris': ['yaris'],
        'ibiza': ['ibiza'],
        'leon': ['leon'],
        'qashqai': ['qashqai'],
        'a-class': ['a-class', 'a class', 'a200', 'a180', 'a250'],
        'tucson': ['tucson'],
        'i10': ['i10', 'i 10'],
        'i20': ['i20', 'i 20'],
        'fiesta': ['fiesta'],
        'sandero': ['sandero'],
        '1 series': ['1 series', '1series', '118', '120', '125'],
        '3 series': ['3 series', '3series', '318', '320', '325', '330'],
        'a1': ['a1', 'a 1'],
        'a3': ['a3', 'a 3'],
        'grandland': ['grandland'],
        'astra': ['astra'],
        'mokka': ['mokka']
    }
    
    # Get variations for the expected make and model
    make_vars = make_variations.get(expected_make, [expected_make])
    model_vars = model_variations.get(expected_model, [expected_model])
    
    # Check if any make variation is in the title
    make_found = False
    found_make = None
    for make_var in make_vars:
        if make_var in title_lower:
            make_found = True
            found_make = make_var
            break
    
    # Check if any model variation is in the title
    model_found = False
    found_model = None
    for model_var in model_vars:
        if model_var in title_lower:
            model_found = True
            found_model = model_var
            break
    
    # Both make and model must be found
    if make_found and model_found:
        return True, f"Found '{found_make}' and '{found_model}' in title"
    elif make_found and not model_found:
        return False, f"Found make '{found_make}' but model '{expected_model}' not found"
    elif not make_found and model_found:
        return False, f"Found model '{found_model}' but make '{expected_make}' not found"
    else:
        return False, f"Neither '{expected_make}' nor '{expected_model}' found in title"

def filter_vehicles_by_make_model(vehicles_data, config_name):
    """Filter vehicles to only include those matching the expected make/model"""
    expected_make, expected_model = extract_make_and_model_from_config(config_name)
    
    if not expected_make or not expected_model:
        print(f"Warning: Could not extract make/model from config name: {config_name}")
        return vehicles_data
    
    filtered_vehicles = []
    filtered_count = 0
    
    for vehicle in vehicles_data:
        # Create vehicle title from make and model
        vehicle_title = f"{vehicle.get('make', '')} {vehicle.get('model', '')}".strip()
        
        is_match, reason = check_make_model_match(vehicle_title, expected_make, expected_model)
        
        if is_match:
            filtered_vehicles.append(vehicle)
        else:
            filtered_count += 1
    
    return filtered_vehicles


def scrape_price_marker(url, browser_manager=None, headless=True):
    """Scrape the price marker information from a given URL.
    
    Args:
        url (str): The AutoTrader URL to scrape
        browser_manager (BrowserManager, optional): Shared browser manager instance for performance
        headless (bool): Whether to run in headless mode
        
    Returns:
        dict: A dictionary containing 'price' and 'market_difference'
              - price: The price as a float
              - market_difference: The difference from market value as a float (negative for below, positive for above, 0 for at market value)
    """
    # Track if we created our own browser manager and need to clean up
    created_browser_manager = False
    
    try:
        # Use provided browser manager or create a temporary one
        if browser_manager is None:
            created_browser_manager = True
            with sync_playwright() as playwright:
                temp_browser_manager = BrowserManager(playwright, headless=headless)
                browser_manager = temp_browser_manager
                return _scrape_price_marker_with_browser(url, browser_manager)
        else:
            # Use the shared browser manager
            return _scrape_price_marker_with_browser(url, browser_manager)
    
    finally:
        # Clean up if we created our own browser manager
        if created_browser_manager and 'temp_browser_manager' in locals():
            temp_browser_manager.close()


def _scrape_price_marker_with_browser(url, browser_manager):
    """Internal function to scrape price marker using a browser manager"""
    # Get a new page from the browser manager
    page = browser_manager.get_page()
    
    try:
        # Set a shorter timeout for better performance and to detect failing proxies faster
        page.set_default_timeout(5000)  # 5 seconds is sufficient
        
        # Navigate to the URL with a shorter timeout for failing faster with bad proxies
        page.goto(url, wait_until='domcontentloaded', timeout=10000)  # 10 second timeout for navigation
        
        # Handle cookie popup if it appears
        handle_cookie_popup(page)
        
        # Try to find the price marker with a more robust approach
        price_marker = None
        marker_text = None
        
        # Define multiple selectors to try in order of preference
        price_marker_selectors = [
            # Try button with market average text first (original selector)
            'button:has-text("market average")',
            # Try p tag with market average text
            'p:has-text("market average")',
            # Try button with below/above market
            'button:has-text("below market"), button:has-text("above market")',
            # Try p tag with below/above market
            'p:has-text("below market"), p:has-text("above market")',
            # Try any element with market text as last resort
            ':text("market average"), :text("below market"), :text("above market")'
        ]
        
        # Try each selector in turn
        marker_found = False
        for selector in price_marker_selectors:
            try:
                # Slightly longer timeout for stability
                price_marker = page.wait_for_selector(selector, timeout=4000, state='attached')
                if price_marker:
                    marker_text = price_marker.inner_text()
                    marker_found = True
                    logger.info(f"Found price marker using selector: {selector}")
                    break
            except Exception as selector_e:
                # Just try the next selector
                logger.debug(f"Selector '{selector}' failed: {selector_e}")
        
        if not marker_found:
            print(f"Error finding price marker with all selectors")
            
            # On any error, try clicking the cookie button again and retry
            
            # Force click the cookie button with JavaScript as a last resort
            page.evaluate("""
            () => {
                // Try clicking any button that looks like a cookie accept button
                const clickAllPossibleButtons = () => {
                    // Try to find Accept All buttons in any iframe
                    const frames = document.querySelectorAll('iframe');
                    for (const frame of frames) {
                        try {
                            if (frame.contentDocument) {
                                const buttons = frame.contentDocument.querySelectorAll('button');
                                for (const button of buttons) {
                                    if (button.textContent.includes('Accept') || 
                                        button.title === 'Accept All' ||
                                        button.getAttribute('aria-label') === 'Accept All') {
                                        button.click();
                                    }
                                }
                            }
                        } catch (e) {}
                    }
                    
                    // Also try clicking any button that might be a consent button on the main page
                    const mainButtons = document.querySelectorAll('button');
                    for (const button of mainButtons) {
                        if (button.textContent.includes('Accept') || 
                            button.textContent.includes('agree') || 
                            button.textContent.includes('Agree') ||
                            button.textContent.includes('consent')) {
                            button.click();
                        }
                    }
                };
                
                // Execute immediately and return
                clickAllPossibleButtons();
            }
            """)
            
            # No artificial wait - proceed immediately to check for price marker again
            marker_found = False
            for selector in price_marker_selectors:
                try:
                    price_marker = page.wait_for_selector(selector, timeout=4000, state='attached')
                    if price_marker:
                        marker_text = price_marker.inner_text()
                        marker_found = True
                        logger.info(f"Found price marker on retry using selector: {selector}")
                        break
                except Exception:
                    # Just try the next selector
                    pass
                    
            if not marker_found:
                # If still failing, try JavaScript to find the marker by content
                try:
                    js_result = page.evaluate("""
                        () => {
                            // More comprehensive search for price marker text in any element
                            const marketPhrases = ['below market average', 'above market average', 'Close to market average', 'market average'];
                            
                            // Check multiple element types that might contain market info
                            const elementTypes = ['button', 'p', 'span', 'div'];
                            
                            // Try each element type
                            for (const type of elementTypes) {
                                const elements = document.querySelectorAll(type);
                                for (const element of elements) {
                                    const text = element.innerText || element.textContent;
                                    if (!text) continue;
                                    
                                    // Check if element text contains any market phrase
                                    for (const phrase of marketPhrases) {
                                        if (text.includes(phrase)) {
                                            // Clean up the text by removing "Read more" part
                                            return text.split("Read more")[0].trim();
                                        }
                                    }
                                }
                            }
                            
                            // Last resort - try to find any element with price info
                            const allElements = document.querySelectorAll('*');
                            for (const element of allElements) {
                                const text = element.innerText || element.textContent;
                                if (text && (text.includes('£') && text.includes('market'))) {
                                    return text.split("Read more")[0].trim();
                                }
                            }
                            
                            return null;
                        }
                    """)
                    
                    if js_result:
                        marker_text = js_result
                except Exception:
                    # Error handling for failed selector
                    pass
            
            # Extract the numeric price - depends on where it appears on the page
            # First try to get main price
            price = 0.0
            try:
                main_price_element = page.query_selector('span[data-testid="price"]')
                if main_price_element:
                    price_text = main_price_element.inner_text().strip()
                    price = float(''.join(c for c in price_text if c.isdigit() or c == '.'))
            except Exception as e:
                print(f"Error extracting price: {e}")
            
            # Clean up the marker text by removing any "Read more" text
            if marker_text:
                marker_text = marker_text.split("Read more")[0].strip()
            
            # Process the marker text to get market difference
            market_difference = 0.0
            
            if marker_text and "below market average" in marker_text:
                # Extract the amount below market
                match = re.search(r'£([\d,]+)\s+below', marker_text)
                if match:
                    below_amount = float(match.group(1).replace(',', ''))
                    market_difference = -below_amount
            
            elif marker_text and "above market average" in marker_text:
                # Extract the amount above market
                match = re.search(r'£([\d,]+)\s+above', marker_text)
                if match:
                    above_amount = float(match.group(1).replace(',', ''))
                    market_difference = above_amount
            
            elif marker_text and "Close to market average" in marker_text:
                # At market value, difference is 0
                market_difference = 0.0
            
            return {
                'price': price,
                'market_difference': market_difference,
                'marker_text': marker_text
            }
    
    except Exception as e:
        error_msg = f"Error scraping price marker: {e}"
        print(error_msg)
        
        return {'price': 0.0, 'market_difference': 0.0, 'marker_text': f'Error: {str(e)}'}
    
    finally:
        # We only close the page, not the browser (browser manager handles browser lifecycle)
        try:
            page.close()
        except:
            pass


# Import multiprocessing for parallel processing
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from queue import Queue, Empty
import threading
import time


class BrowserPoolManager:
    """Manages a pool of browser instances for efficient URL processing
    
    This class creates and maintains a pool of browser instances that can be
    reused across multiple URL requests, significantly reducing overhead and
    improving scraping speed.
    """
    def __init__(self, pool_size=None, headless=True, block_resources=True, ttl_requests=50, max_browser_age=3600):
        """Initialize the browser pool manager
        
        Args:
            pool_size (int): Number of browser instances to maintain in the pool
                             (defaults to CPU count)
            headless (bool): Whether to run browsers in headless mode
            block_resources (bool): Whether to block unnecessary resources
            ttl_requests (int): Number of requests before recycling a browser
            max_browser_age (int): Maximum browser age in seconds before forced recycling
        """
        # Determine pool size based on available CPUs if not specified
        if pool_size is None:
            pool_size = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        
        self.pool_size = pool_size
        self.headless = headless
        self.block_resources = block_resources
        self.ttl_requests = ttl_requests  # Requests before recycling
        self.max_browser_age = max_browser_age  # Max browser lifetime
        
        # Initialize browser pool
        self._browser_pool = Queue(maxsize=pool_size)
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._active_browsers = {}  # Track active browsers and their request counts
        self._initialization_lock = threading.Lock()  # Lock for pool initialization
        self._initialized = False
        self._shutdown = False
        
        # Stats for monitoring
        self.total_requests = 0
        self.browser_creations = 0
        self.browser_recyclings = 0
        self.browser_errors = 0
        
        # Start health monitor thread
        self._stop_monitor = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_browser_health, daemon=True)
        self._monitor_thread.start()
        
    def initialize_pool(self):
        """Initialize the browser pool with the configured number of instances"""
        with self._initialization_lock:
            if self._initialized:
                return  # Already initialized
                
            # Launch a playwright instance for this process
            self._playwright = sync_playwright().start()
            
            # Create initial browser instances
            for _ in range(self.pool_size):
                self._add_browser_to_pool()
                
            self._initialized = True
            logger.info(f"Browser pool initialized with {self.pool_size} browsers")
    
    def _add_browser_to_pool(self):
        """Create a new browser instance and add it to the pool"""
        try:
            browser_manager = BrowserManager(self._playwright, 
                                        headless=self.headless, 
                                        block_resources=self.block_resources)
            browser_info = {
                'browser_manager': browser_manager,
                'request_count': 0,
                'created_at': time.time(),
                'last_used_at': time.time(),
                'health_status': 'healthy',  # Track browser health status
                'errors': 0  # Count errors for this browser
            }
            self._browser_pool.put(browser_info)
            self.browser_creations += 1
        except Exception as e:
            logger.error(f"Error creating browser: {e}")
            self.browser_errors += 1
            # Try again if we're not in shutdown
            if not self._shutdown and self._initialized:
                logger.info("Retrying browser creation after error")
                time.sleep(1)  # Brief delay before retry
                self._add_browser_to_pool()
    
    def _monitor_browser_health(self):
        """Monitor browser health in a background thread"""
        check_interval = 30  # Check every 30 seconds
        
        while not self._stop_monitor.is_set():
            try:
                # Sleep first to avoid immediate checking after creation
                for _ in range(check_interval):
                    if self._stop_monitor.is_set():
                        return
                    time.sleep(1)
                
                with self._lock:
                    # Skip if not initialized or shutting down
                    if not self._initialized or self._shutdown:
                        continue
                    
                    # Check active browsers
                    for browser_id, browser_info in list(self._active_browsers.items()):
                        # Check browser age
                        browser_age = time.time() - browser_info['created_at']
                        if browser_age > self.max_browser_age:
                            logger.info(f"Browser {browser_id} exceeded max age ({browser_age:.1f}s), marking for recycling")
                            browser_info['health_status'] = 'recycle'
                        
                        # Check if browser has been active for too long
                        last_used_duration = time.time() - browser_info['last_used_at']
                        if last_used_duration > 300:  # 5 minutes
                            logger.warning(f"Browser {browser_id} has been active for {last_used_duration:.1f}s, may be stuck")
                            browser_info['health_status'] = 'stuck'
                    
                    # Try to collect and check browsers in the pool
                    try:
                        # We'll remove all browsers, check them, and put healthy ones back
                        browsers_to_check = []
                        
                        # Empty the pool
                        while not self._browser_pool.empty():
                            browsers_to_check.append(self._browser_pool.get_nowait())
                        
                        # Check each browser
                        for browser_info in browsers_to_check:
                            # Check age
                            browser_age = time.time() - browser_info['created_at']
                            if browser_age > self.max_browser_age:
                                logger.info(f"Recycling aged browser ({browser_age:.1f}s)")
                                try:
                                    browser_info['browser_manager'].close()
                                except Exception as e:
                                    logger.warning(f"Error closing aged browser: {e}")
                                self._add_browser_to_pool()  # Add a replacement
                                self.browser_recyclings += 1
                            else:
                                # This browser is still good, put it back in the pool
                                self._browser_pool.put(browser_info)
                    except Empty:
                        pass  # Pool is empty
                
            except Exception as e:
                logger.error(f"Error in browser health monitoring: {e}")
            
        logger.info("Browser health monitoring thread stopped")
    
    def check_browser_health(self, browser_id):
        """Perform a health check on a specific browser
        
        Returns True if browser is healthy, False if it should be recycled
        """
        with self._lock:
            if browser_id not in self._active_browsers:
                return False  # Browser not found
                
            browser_info = self._active_browsers[browser_id]
            
            # Check if browser has been marked for recycling
            if browser_info['health_status'] != 'healthy':
                return False
                
            # Check browser age
            browser_age = time.time() - browser_info['created_at']
            if browser_age > self.max_browser_age:
                browser_info['health_status'] = 'recycle'
                return False
                
            # Check error count
            if browser_info['errors'] >= 3:  # 3 strikes rule
                browser_info['health_status'] = 'recycle'
                return False
                
            # Update last used timestamp
            browser_info['last_used_at'] = time.time()
            
            return True
        
    def get_browser(self, timeout=5):  # Reduced default timeout
        """Get a browser from the pool (initializing the pool if needed)
        
        Args:
            timeout (int): Timeout in seconds for waiting for an available browser
            
        Returns:
            tuple: (browser_id, browser_manager) or (None, None) if timed out
        """
        # Make sure the pool is initialized
        if not self._initialized:
            self.initialize_pool()
            
        # Try to get a browser from the pool
        start_time = time.time()
        end_time = start_time + timeout
        
        while time.time() < end_time:
            try:
                with self._lock:
                    if self._shutdown:
                        return None, None
                        
                    browser_info = self._browser_pool.get(block=True, timeout=1.0)  # Short timeout to allow retry
                    browser_id = id(browser_info['browser_manager'])
                    
                    # Check browser health before returning it
                    browser_age = time.time() - browser_info['created_at']
                    if browser_age > self.max_browser_age or browser_info['health_status'] != 'healthy':
                        # Browser is too old or unhealthy, recycle it
                        logger.info(f"Recycling unhealthy/old browser (age={browser_age:.1f}s, status={browser_info['health_status']})")
                        try:
                            browser_info['browser_manager'].close()
                        except Exception as e:
                            logger.warning(f"Error closing unhealthy browser: {e}")
                            
                        # Add a replacement browser
                        self._add_browser_to_pool()
                        self.browser_recyclings += 1
                        
                        # Try again if we have time left
                        continue
                    
                    # Update tracking info
                    browser_info['request_count'] += 1
                    browser_info['last_used_at'] = time.time()
                    self._active_browsers[browser_id] = browser_info
                    self.total_requests += 1
                    
                    return browser_id, browser_info['browser_manager']
                    
            except Empty:
                # No browser available right now, but we might still have time
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break
                # Small sleep to avoid hammering
                time.sleep(0.1)
        
        # If we get here, we timed out
        logger.warning(f"Timed out waiting for available browser after {timeout}s")
        return None, None
            
    def return_browser(self, browser_id, recycle=False, had_error=False):
        """Return a browser to the pool for reuse
        
        Args:
            browser_id: ID of the browser to return
            recycle (bool): Whether to force recycling this browser
            had_error (bool): Whether an error occurred while using this browser
        """
        with self._lock:
            if self._shutdown:
                return
                
            if browser_id not in self._active_browsers:
                logger.warning(f"Attempted to return unknown browser ID: {browser_id}")
                return
                
            browser_info = self._active_browsers.pop(browser_id)
            
            # Update error count if an error occurred
            if had_error:
                browser_info['errors'] += 1
                if browser_info['errors'] >= 3:  # 3 strikes rule
                    recycle = True
                    browser_info['health_status'] = 'unhealthy'
            
            # Check if we need to recycle this browser
            if (recycle or 
                browser_info['request_count'] >= self.ttl_requests or
                browser_info['health_status'] != 'healthy' or
                (time.time() - browser_info['created_at']) > self.max_browser_age):
                
                # Close old browser and create new one
                try:
                    browser_info['browser_manager'].close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
                    
                # Create a new browser to replace it
                self._add_browser_to_pool()
                self.browser_recyclings += 1
                
            else:
                # Update last used timestamp
                browser_info['last_used_at'] = time.time()
                # Return browser to the pool
                self._browser_pool.put(browser_info)
    
    def shutdown(self):
        """Shut down the browser pool and close all browsers"""
        with self._lock:
            self._shutdown = True
            
            # Stop the monitor thread
            self._stop_monitor.set()
            
            # Close all active browsers
            for browser_info in self._active_browsers.values():
                try:
                    browser_info['browser_manager'].close()
                except Exception as e:
                    logger.warning(f"Error closing browser during shutdown: {e}")
            self._active_browsers.clear()
            
            # Empty the browser pool, closing each browser
            try:
                while True:
                    browser_info = self._browser_pool.get_nowait()
                    try:
                        browser_info['browser_manager'].close()
                    except Exception as e:
                        logger.warning(f"Error closing browser during shutdown: {e}")
            except Empty:
                pass
                
            # Stop playwright
            try:
                if hasattr(self, '_playwright'):
                    self._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping playwright: {e}")
            
            # Wait for monitor thread to exit (with timeout)
            if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
                
            logger.info(f"Browser pool shutdown complete. Stats: created={self.browser_creations}, recycled={self.browser_recyclings}, errors={self.browser_errors}, total_requests={self.total_requests}")
                
    def __del__(self):
        """Clean up resources on garbage collection"""
        try:
            self.shutdown()
        except:
            pass

# This function is no longer needed with the new implementation
# def _process_url_wrapper(url, browser_pool=None, headless=True):
#     """This function has been replaced by _process_url_with_pool"""
#     pass

def _process_url_with_pool(url, pool_size=3, headless=True, proxy_manager=None):
    """Process a URL using a local browser pool within this process
    
    This function creates a browser pool inside the worker process,
    avoiding the need to pickle lock objects between processes.
    
    Args:
        url (str): The URL to process
        pool_size (int): Number of browsers in the pool
        headless (bool): Whether to run browsers in headless mode
        proxy_manager (ProxyManager, optional): Proxy manager for rotation
        
    Returns:
        tuple: (url, result)
    """
    # Get proxy if proxy manager is provided
    proxy_url = None
    if proxy_manager:
        proxy_url = proxy_manager.get_proxy()
    
    # Create a browser pool local to this process
    try:
        with sync_playwright() as playwright:
            # Create browser manager inside this process
            browser_manager = BrowserManager(playwright, headless=headless, proxy=proxy_url)
            # Add a very small random delay to avoid exact synchronization issues
            time.sleep(random.uniform(0.01, 0.05))
            result = scrape_price_marker(url, browser_manager=browser_manager)
            # Make sure to properly clean up browser resources
            browser_manager.close()
            return url, result
    except Exception as e:
        # Check if we should blacklist the proxy
        if proxy_manager and proxy_url and (
            str(e).find("403") != -1 or  # Look for 403 in exception message
            str(e).find("forbidden") != -1 or  # Look for forbidden in exception message
            str(e).find("blocked") != -1 or  # Look for blocked in exception message
            str(e).find("net::ERR_PROXY_CONNECTION_FAILED") != -1 or  # Proxy connection errors
            str(e).find("net::ERR_TUNNEL_CONNECTION_FAILED") != -1  # Proxy tunnel errors
        ):
            # Extract IP and blacklist it
            ip = proxy_manager.extract_ip(proxy_url)
            if ip:
                # Silent blacklisting with minimalist output following terminal style
                proxy_manager.blacklist_ip(ip, reason=f"Error: {str(e)[:100]}")
        
        return url, {
            'price': 0.0, 
            'market_difference': 0.0, 
            'marker_text': f'Error: {str(e)}'
        }

def batch_scrape_price_markers(urls, headless=True, max_workers=None, progress_callback=None, use_browser_pool=True, use_proxy_rotation=False, proxy_config_path="config/proxies.json", proxy_archive_path="archive", proxy_manager=None):
    """Efficiently scrape price markers for multiple URLs using multiprocessing with browser pooling
    
    Args:
        urls (list): List of URLs to scrape
        headless (bool): Whether to run in headless mode
        max_workers (int): Maximum number of worker processes (defaults to CPU count)
        progress_callback (function): Optional callback function to report progress
        use_browser_pool (bool): Whether to use browser pooling for better performance
        use_proxy_rotation (bool): Whether to use proxy rotation for better reliability
        proxy_config_path (str): Path to the proxy configuration file
        proxy_archive_path (str): Path to store blacklisted proxy information
        proxy_manager (ProxyManager, optional): An already initialized ProxyManager instance
            If not provided and use_proxy_rotation=True, a new ProxyManager will be created
        
    Returns:
        dict: Dictionary mapping URLs to their price marker results
    
    Note on proxy rotation:
        When use_proxy_rotation=True, the function will use proxies from config/proxies.json
        to avoid IP-based rate limiting and blocking. Failed proxies will be automatically
        blacklisted and stored in the archive folder. This can significantly improve the
        reliability of scraping, especially when dealing with websites that have strict
        anti-scraping measures.
    """
    # Use provided proxy manager or create one if proxy rotation is enabled
    if proxy_manager is None and use_proxy_rotation:
        try:
            from services.proxy_rotation import ProxyManager
            proxy_manager = ProxyManager(config_path=proxy_config_path, archive_path=proxy_archive_path)
            # Silent proxy rotation - no output except the IP Initialized line
        except Exception as e:
            print(f"Error initializing proxy rotation: {e}")
            use_proxy_rotation = False
    
    # Use CPU count if max_workers not specified - go back to original approach
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    results = {}
    
    # Print setup information
    print(f"\nProcessing {len(urls)} URLs with {max_workers} worker processes")
    
    # Use browser pooling if enabled (the new optimization)
    if use_browser_pool:
        # Each process will create its own browser pool
        print(f"Using in-process browser pooling mode")
        
        # Create a partial function with the pool size parameter and proxy manager if enabled
        process_url_with_pool = partial(_process_url_with_pool, 
                                      headless=headless,
                                      proxy_manager=proxy_manager if use_proxy_rotation else None)
        
        # Process all URLs at once with full parallelism
        completed = 0
        total = len(urls)
        
        # Print initial progress
        print(f"Progress: 0/{total} (0.0%)")
        
        # Use ProcessPoolExecutor to distribute work across multiple processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all URLs to the process pool and collect results
            future_to_url = {executor.submit(process_url_with_pool, url): url for url in urls}
            
            # Retrieve and store results as they complete
            for future in future_to_url:
                try:
                    url, result = future.result()
                    results[url] = result
                    
                    # Update progress
                    completed += 1
                    
                    # Always update progress on the same line for a cleaner output
                    if completed % 1 == 0 or completed == total:  # Report every item
                        progress_str = f"\rProgress: {completed}/{total} ({completed/total:.1%})"
                        print(progress_str, end="", flush=True)
                        
                        # If a progress callback was provided, call it
                        if progress_callback:
                            progress_callback(completed, total)
                    
                    # Print a newline at the end
                    if completed == total:
                        print("\nProcessing complete!")
                        
                except Exception as e:
                    url = future_to_url[future]
                    print(f"\nError processing {url}: {e}")
                    results[url] = {
                        'price': 0.0, 
                        'market_difference': 0.0, 
                        'marker_text': f'Error: {str(e)}'
                    }
    
    else:
        # Legacy approach - create a new browser for each URL
        print(f"Using legacy mode (no browser pooling)")
        
        # Define a simple legacy processor function that doesn't use pools
        def process_url_legacy(url):
            # Get proxy if proxy manager is provided
            proxy_url = None
            if proxy_manager:
                proxy_url = proxy_manager.get_proxy()
                
            try:
                with sync_playwright() as playwright:
                    # Add very small random delay to avoid exact synchronization
                    time.sleep(random.uniform(0.01, 0.05))
                    # Create browser manager with proxy if available
                    browser_manager = BrowserManager(playwright, headless=headless, proxy=proxy_url)
                    result = scrape_price_marker(url, browser_manager=browser_manager)
                    browser_manager.close()
                    return url, result
            except Exception as e:
                # Check if we should blacklist the proxy
                if proxy_manager and proxy_url and (
                    str(e).find("403") != -1 or
                    str(e).find("forbidden") != -1 or
                    str(e).find("blocked") != -1 or
                    str(e).find("net::ERR_PROXY_CONNECTION_FAILED") != -1 or
                    str(e).find("net::ERR_TUNNEL_CONNECTION_FAILED") != -1
                ):
                    # Extract IP and blacklist it
                    ip = proxy_manager.extract_ip(proxy_url)
                    if ip:
                        proxy_manager.blacklist_ip(ip, reason=f"Error: {str(e)[:100]}")
                
                return url, {
                    'price': 0.0, 
                    'market_difference': 0.0, 
                    'marker_text': f'Error: {str(e)}'
                }
        
        # Process all URLs at once with full parallelism
        completed = 0
        total = len(urls)
        
        # Print initial progress
        print(f"Progress: 0/{total} (0.0%)")
        
        # Use ProcessPoolExecutor to distribute work across multiple processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all URLs to the process pool at once
            future_to_url = {executor.submit(process_url_legacy, url): url for url in urls}
            
            # Retrieve and store results as they complete
            for future in future_to_url:
                try:
                    url, result = future.result()
                    results[url] = result
                    
                    # Update progress
                    completed += 1
                    
                    # Always update progress on the same line for a cleaner output
                    if completed % 1 == 0 or completed == total:  # Report every item
                        progress_str = f"\rProgress: {completed}/{total} ({completed/total:.1%})"
                        print(progress_str, end="", flush=True)
                        
                        # If a progress callback was provided, call it
                        if progress_callback:
                            progress_callback(completed, total)
                    
                    # Print a newline at the end
                    if completed == total:
                        print("\nProcessing complete!")
                        
                except Exception as e:
                    url = future_to_url[future]
                    print(f"\nError processing {url}: {e}")
                    results[url] = {
                        'price': 0.0, 
                        'market_difference': 0.0, 
                        'marker_text': f'Error: {str(e)}'
                    }
    
    return results


def format_market_difference(result):
    """Format the market difference value as a string"""
    if result['market_difference'] < 0:
        output = f"£{abs(result['market_difference']):.2f} below market average"
    elif result['market_difference'] > 0:
        output = f"£{result['market_difference']:.2f} above market average"
    else:
        output = "£0 above market average"
    
    # Return only the price marker without any additional text
    return output.split("Read more")[0].strip()


def print_result(result):
    """Helper function to print the price marker"""
    print(format_market_difference(result))


def print_batch_results(results):
    """Print only price markers for successful results, errors for failed ones"""
    success_count = sum(1 for r in results.values() if 'Error' not in r.get('marker_text', ''))
    
    # Print only price markers for successful results
    for url, result in results.items():
        if 'Error' not in result.get('marker_text', ''):
            print(format_market_difference(result))
    
    # Print errors for failed results
    for url, result in results.items():
        if 'Error' in result.get('marker_text', ''):
            print(f"Error: {url}")



def run_test():
    """Run a test using a predefined URL"""
    test_url = "https://www.autotrader.co.uk/car-details/202509046036529"
    
    result = scrape_price_marker(test_url)
    
    # Only print the market difference in the correct format
    if result['marker_text'] != 'Not found' and 'Error' not in result['marker_text']:
        print(format_market_difference(result))
        return True
    else:
        print(f"Error: Could not retrieve price marker")
        return False


def main():
    """Example usage of the price marker scraper"""
    # Check for --test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
        return
        
    # Check for --batch flag
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch processing mode
        if len(sys.argv) < 3:
            print("Error: Missing URLs file")
            sys.exit(1)
            
        urls_file = sys.argv[2]
        headless = "--no-headless" not in sys.argv
        use_proxy = "--proxy" in sys.argv
        
        try:
            # Read URLs from file
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                
            if not urls:
                print("Error: No valid URLs found")
                sys.exit(1)
            
            # Process URLs in batch
            results = batch_scrape_price_markers(urls, headless=headless, use_proxy_rotation=use_proxy)
            
            # Print only price markers for successful results and errors for failed ones
            print_batch_results(results)
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        return
    
    # Normal operation with a single URL
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print("Error: Missing URL")
        sys.exit(1)
    
    url = sys.argv[1]
    headless = "--no-headless" not in sys.argv
    
    result = scrape_price_marker(url)
    
    # Print only the market difference value
    print(format_market_difference(result))


if __name__ == "__main__":
    main()