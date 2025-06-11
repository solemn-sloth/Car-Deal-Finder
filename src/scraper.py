"""
Core scraping functionality for AutoTrader.
Multiprocessing-safe version.
"""
from playwright.sync_api import sync_playwright
import random
import time
import re
import os
from datetime import datetime

def setup_browser(playwright, headless=True):
    """Set up and return a browser instance"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    
    browser = playwright.chromium.launch(headless=headless)
    context = browser.new_context(
        user_agent=random.choice(user_agents),
        viewport={'width': 1920, 'height': 1080}
    )
    
    return browser, context

def handle_cookie_popup(page):
    """Handle the cookie consent popup"""
    time.sleep(2)
    try:
        cookie_iframe = page.frame_locator('#sp_message_iframe_1086457')
        cookie_button = cookie_iframe.locator('button:has-text("Accept All")')
        if cookie_button.count() > 0:
            cookie_button.click(timeout=5000)
    except Exception as e:
        print(f"Error handling cookie popup: {e}")

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
                        # Optional: uncomment to see what's being filtered
                        # print(f"Filtered: {vehicle_title} - {reason}")
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
    
    # Create process-specific screenshot directory
    process_id = os.getpid()
    screenshot_dir = f"screenshots/process_{process_id}"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    all_vehicles_data = []
    try:
        # Add random delay to avoid all processes hitting at once
        initial_delay = random.uniform(0, 2)
        time.sleep(initial_delay)
        
        page.goto(search_url)
        handle_cookie_popup(page)
        try:
            page.wait_for_selector('div[data-testid="advertCard"]', timeout=15000)
        except Exception as e:
            time.sleep(5)
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
                
                # Add random delay between pages
                delay = random.uniform(1, 3)
                time.sleep(delay)
                
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