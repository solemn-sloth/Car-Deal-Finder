"""
Core scraping functionality for AutoTrader.
Multiprocessing-safe version with retail price marker scraper.
"""
from playwright.sync_api import sync_playwright
import random
import time
import re
import os
from datetime import datetime
import sys

class BrowserManager:
    """Manages a persistent browser instance for reuse across multiple scraping operations"""
    def __init__(self, playwright, headless=True, block_resources=True):
        self.playwright = playwright
        self.headless = headless
        self.block_resources = block_resources
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
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--disable-setuid-sandbox',
                '--no-sandbox',
                '--disable-accelerated-2d-canvas',
                '--disable-extensions',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials'
            ]
        )
        
        # Create context with optimized settings
        self.context = self.browser.new_context(
            user_agent=random.choice(self.user_agents),
            viewport={'width': 1280, 'height': 800},  # Smaller viewport for less resource usage
            java_script_enabled=True,
            bypass_csp=True,  # Bypass Content-Security-Policy
            ignore_https_errors=True,
            has_touch=False,
            is_mobile=False,
            locale='en-GB',
            timezone_id='Europe/London'
        )
        
        # Set default navigation timeout
        self.context.set_default_timeout(10000)  # 10 seconds default timeout
        
        # Set default navigation waiting options
        self.context.set_default_navigation_timeout(15000)  # 15 seconds for navigation
    
    def get_page(self):
        """Get a new page from the existing browser context with resource blocking"""
        if not self.browser or not self.context:
            self._initialize_browser()
            
        # Create a new page
        page = self.context.new_page()
        
        if self.block_resources:
            # Block unnecessary resources to improve performance
            page.route('**/*.{png,jpg,jpeg,gif,webp,svg,ico}', lambda route: route.abort() if 'sprite' not in route.request.url else route.continue_())
            page.route('**/*.{css,woff,woff2,ttf,otf}', lambda route: route.abort())
            page.route('**/*{analytics,tracking,advertisement,ads,ga,gtm,pixel}*', lambda route: route.abort())
            
            # Allow only essential resources
            page.route('**/*', lambda route: route.continue_() if (
                route.request.resource_type in ['document', 'xhr', 'fetch', 'script'] or
                'autotrader' in route.request.url
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
    
    # Launch browser with optimized settings
    browser = playwright.chromium.launch(
        headless=headless,
        args=[
            '--disable-gpu',
            '--disable-dev-shm-usage',
            '--disable-setuid-sandbox',
            '--no-sandbox',
            '--disable-accelerated-2d-canvas',
            '--disable-extensions'
        ]
    )
    
    # Create context with optimized settings
    context = browser.new_context(
        user_agent=random.choice(user_agents),
        viewport={'width': 1280, 'height': 800},  # Smaller viewport for less resource usage
        java_script_enabled=True,
        bypass_csp=True,
        ignore_https_errors=True
    )
    
    # Set default timeouts
    context.set_default_timeout(10000)
    context.set_default_navigation_timeout(15000)
    
    return browser, context

# Screenshot functionality removed as per requirements


def handle_cookie_popup(page):
    """Handle the cookie consent popup using JavaScript for reliability"""
    # No artificial wait - use conditional wait instead
    try:
        # Look for cookie iframe
        page.wait_for_selector('iframe[id^="sp_message_iframe"]', timeout=2000, state='attached')
        
        # Simple, direct JavaScript approach to click Accept All in any iframe
        js_click_result = page.evaluate("""
            () => {
                // Try to find and click the Accept All button in any iframe
                const frames = document.querySelectorAll('iframe');
                for (const frame of frames) {
                    try {
                        if (frame.contentDocument) {
                            const buttons = frame.contentDocument.querySelectorAll('button');
                            for (const button of buttons) {
                                if (button.textContent.includes('Accept')) {
                                    button.click();
                                    return true;
                                }
                            }
                        }
                    } catch (e) {}
                }
                return false;
            }
        """)
        
        if not js_click_result:
            # Fallback to clicking cookie button
            try:
                cookie_iframe = page.frame_locator('#sp_message_iframe_1086457')
                cookie_button = cookie_iframe.locator('button:has-text("Accept All")')
                
                if cookie_button.count() > 0:
                    cookie_button.click(timeout=1000)
            except Exception:
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
    
    # Block unnecessary resources to improve performance
    page.route('**/*.{png,jpg,jpeg,gif,webp,svg,ico}', lambda route: route.abort() if 'sprite' not in route.request.url else route.continue_())
    page.route('**/*.{css,woff,woff2,ttf,otf}', lambda route: route.abort())
    page.route('**/*{analytics,tracking,advertisement,ads,ga,gtm,pixel}*', lambda route: route.abort())
    
    # Create process-specific screenshot directory
    process_id = os.getpid()
    screenshot_dir = f"screenshots/process_{process_id}"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    all_vehicles_data = []
    try:
        # No random delay - we'll implement proper rate limiting in the multiprocessing code later
        
        page.goto(search_url)
        handle_cookie_popup(page)
        try:
            page.wait_for_selector('div[data-testid="advertCard"]', timeout=10000)  # Reduced timeout
        except Exception as e:
            # No artificial wait here either
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            page.screenshot(path=f"{screenshot_dir}/{config['name'].replace(' ', '_')}_error_{timestamp}.png")
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
                    page.wait_for_selector('div[data-testid="advertCard"]', timeout=15000)
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
        # Set a short timeout for performance
        page.set_default_timeout(15000)  # 15 seconds to be safe
        
        # Navigate to the URL
        page.goto(url, wait_until='domcontentloaded')
        
        # Handle cookie popup if it appears
        handle_cookie_popup(page)
        
        # Try to find the price marker
        price_marker = None
        marker_text = None
        try:
            # Try standard selector
            price_marker = page.wait_for_selector('button.at__sc-141g4g0-0.at__sc-1tc704b-3', timeout=5000)
            if price_marker:
                marker_text = price_marker.inner_text()
        except Exception as e:
            print(f"Error finding price marker: {e}")
            
            # On any error, take screenshot and try clicking the cookie button again
            screenshot_path = take_error_screenshot(page, "before_retry")
            
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
            
            # No artificial wait - proceed immediately to check for price marker
            try:
                price_marker = page.wait_for_selector('button.at__sc-141g4g0-0.at__sc-1tc704b-3', timeout=5000)
                if price_marker:
                    marker_text = price_marker.inner_text()
            except Exception as retry_e:
                # If still failing, try JavaScript to find the marker by content
                try:
                    js_result = page.evaluate("""
                        () => {
                            // Look for any button containing price market text
                            const buttons = document.querySelectorAll('button');
                            for (const button of buttons) {
                                const text = button.innerText || button.textContent;
                                if (text && (
                                    text.includes('below market average') || 
                                    text.includes('above market average') ||
                                    text.includes('Close to market average')
                                )) {
                                    // Clean up the text by removing "Read more" part
                                    return text.split("Read more")[0].trim();
                                }
                            }
                            return null;
                        }
                    """)
                    
                    if js_result:
                        marker_text = js_result
                except Exception:
                    # Take final screenshot
                    take_error_screenshot(page, "after_retry")
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
            
            # Take a screenshot to help debug the issue
            screenshot_path = take_error_screenshot(page, "general_error")
            return {'price': 0.0, 'market_difference': 0.0, 'marker_text': f'Error: {str(e)} (screenshot: {screenshot_path})'}
        
        finally:
            # We only close the page, not the browser (browser manager handles browser lifecycle)
            try:
                page.close()
            except:
                pass


import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def _process_url_wrapper(url, headless=True):
    """Process a single URL with its own browser instance (for multiprocessing)"""
    try:
        with sync_playwright() as playwright:
            # Each process gets its own browser
            result = scrape_price_marker(url, headless=headless)
            return url, result
    except Exception as e:
        return url, {
            'price': 0.0, 
            'market_difference': 0.0, 
            'marker_text': f'Error: {str(e)}'
        }

def batch_scrape_price_markers(urls, headless=True, max_workers=None, progress_callback=None):
    """Efficiently scrape price markers for multiple URLs using multiprocessing
    
    Args:
        urls (list): List of URLs to scrape
        headless (bool): Whether to run in headless mode
        max_workers (int): Maximum number of worker processes (defaults to CPU count)
        progress_callback (function): Optional callback function to report progress
        
    Returns:
        dict: Dictionary mapping URLs to their price marker results
    """
    # Use CPU count if max_workers not specified
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        
    results = {}
    
    # Create a partial function with the headless parameter
    process_url = partial(_process_url_wrapper, headless=headless)
    
    # Use ProcessPoolExecutor to distribute work across multiple processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all URLs to the process pool and collect results
        future_to_url = {executor.submit(process_url, url): url for url in urls}
        
        # Retrieve and store results as they complete
        completed = 0
        total = len(urls)
        
        # Print initial progress
        print(f"\nProcessing {total} URLs with {max_workers} parallel workers")
        print(f"Progress: 0/{total} (0.0%)")
        
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
        
        try:
            # Read URLs from file
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                
            if not urls:
                print("Error: No valid URLs found")
                sys.exit(1)
            
            # Process URLs in batch
            results = batch_scrape_price_markers(urls, headless=headless)
            
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