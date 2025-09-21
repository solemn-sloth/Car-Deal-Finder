#!/usr/bin/env python3
"""
AutoTrader GraphQL API Client for Price Deviation Data
Follows the same patterns as AutoTraderAPIClient for consistency.
"""

import requests
import requests.adapters
import json
import time
import random
import logging
import sys
import os
import re
from typing import List, Dict, Optional, Callable, Any
from requests.packages.urllib3.util.retry import Retry
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import output manager for consistent logging
from src.output_manager import get_output_manager

try:
    from services.stealth_orchestrator import FingerprintGenerator, CustomHTTPAdapter, ProxyManager
except ImportError:
    # Fallback if stealth_orchestrator is not available
    FingerprintGenerator = None
    CustomHTTPAdapter = None
    ProxyManager = None

# Disable SSL warnings globally
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def log_price_marker_data(url, result, success=True, error=None, make=None, model=None):
    """Log price marker extraction data to JSON file for debugging"""
    import json
    import os
    from datetime import datetime

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Prepare log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'url': url,
        'success': success,
        'make': make,
        'model': model,
        'result': result
    }

    if error:
        log_entry['error'] = error

    # Append to JSON file
    log_file = os.path.join(logs_dir, 'price_markers_debug.json')

    try:
        # Read existing data or create empty list
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new entry
        data.append(log_entry)

        # Keep only last 1000 entries to prevent file from getting too large
        if len(data) > 1000:
            data = data[-1000:]

        # Write back to file
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        # If logging fails, don't crash the scraping process
        logger.error(f"Failed to log price marker data: {e}")


@dataclass
class GraphQLTask:
    """Represents a GraphQL API task for a specific advert ID."""
    advert_id: str
    original_url: str
    make: Optional[str] = None
    model: Optional[str] = None

    def __str__(self):
        return f"advert_{self.advert_id}"


class GraphQLCoordinator:
    """
    Coordinates parallel GraphQL API workers with anti-detection features.
    Mirrors the ParallelCoordinator pattern from network_requests.py.
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

    def add_work(self, tasks: List[GraphQLTask]):
        """Add tasks to work queue for parallel distribution."""
        for task in tasks:
            self.work_queue.put(task)

    def get_next_work(self, worker_id: int, timeout: float = 1) -> Optional[GraphQLTask]:
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
                logger.debug(f"GraphQL Worker {worker_id} detected as blocked, pausing")

        # Mark the task as done in the queue
        try:
            self.work_queue.task_done()
        except ValueError:
            logger.debug(f"task_done() called too many times - GraphQL worker {worker_id}")

    def _get_result_status(self, result: Dict[str, Any]) -> str:
        """Get GraphQL API-specific status indicator."""
        if not result.get('success', False):
            error = result.get('error', 'Unknown error')
            if 'cloudflare' in error.lower():
                return "❌ CF blocked"
            elif '403' in error or 'forbidden' in error.lower():
                return "❌ 403 Forbidden"
            elif 'timeout' in error.lower():
                return "❌ Timeout"
            elif '404' in error:
                return "❌ Not found"
            else:
                return f"❌ {error[:30]}"

        # Check if we got price deviation data
        if result.get('price_difference_found', False):
            market_diff = result.get('market_difference', 0)
            if market_diff != 0:
                sign = "↑" if market_diff > 0 else "↓"
                return f"✅ {sign}£{abs(market_diff)}"
            else:
                return "✅ ~market avg"
        else:
            return "⚠️ No price data"

    def _is_worker_detected(self, result: Dict[str, Any]) -> bool:
        """Check if GraphQL API worker was detected/blocked."""
        if result.get('success', True):
            return False

        error = result.get('error', '').lower()
        return any(keyword in error for keyword in [
            'cloudflare', '403', 'forbidden', 'blocked', 'challenge',
            'too many requests', 'rate limit'
        ])


class AutoTraderGraphQLClient:
    """
    AutoTrader GraphQL API client for price deviation data.
    Mirrors the structure and patterns of AutoTraderAPIClient.
    """

    def __init__(self, connection_pool_size=10, optimize_connection=True, proxy=None,
                 proxy_manager=None, verify_ssl=False, worker_id=None):
        """
        Initialize the GraphQL client.

        Args:
            connection_pool_size: Size of the connection pool (default: 10)
            optimize_connection: Whether to use connection pooling and other optimizations
            proxy: Optional proxy URL
            proxy_manager: Optional ProxyManager instance for rotation
            verify_ssl: Whether to verify SSL certificates
            worker_id: Optional worker ID for consistent fingerprinting
        """
        self.base_url = "https://www.autotrader.co.uk/product-page/v1/advert"
        self.postcode = "M15%204FN"  # Default postcode for API calls
        self.proxy_manager = proxy_manager
        self.proxy = proxy
        self.worker_id = worker_id
        self._first_request_made = False
        self._session_start_time = time.time()

        # Generate unified fingerprint for this client instance
        if FingerprintGenerator:
            self.fingerprint = FingerprintGenerator.create_http_fingerprint(
                worker_id=worker_id or 0
            )
        else:
            self.fingerprint = None

        # Set up session with optimizations
        self.session = requests.Session()

        if optimize_connection:
            self._setup_connection_pooling(connection_pool_size)

        self._setup_session_headers()
        self._setup_proxy()

    def _setup_connection_pooling(self, pool_size: int):
        """Set up HTTP connection pooling for better performance."""
        # Use standard HTTPAdapter to avoid socket_options issues
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=Retry(
                total=3,
                read=2,
                connect=2,
                backoff_factor=0.3,
                status_forcelist=(500, 502, 504)
            )
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _setup_session_headers(self):
        """Set up session headers using fingerprint if available."""
        if self.fingerprint and hasattr(self.fingerprint, 'headers'):
            headers = self.fingerprint.headers
        else:
            # Fallback headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-GB,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin'
            }

        self.session.headers.update(headers)

    def _setup_proxy(self):
        """Set up proxy configuration."""
        if self.proxy:
            self.session.proxies = {
                'http': self.proxy,
                'https': self.proxy
            }
        elif self.proxy_manager:
            proxy = self.proxy_manager.get_proxy_for_worker(self.worker_id or 0)
            if proxy:
                self.session.proxies = {
                    'http': proxy,
                    'https': proxy
                }

    def extract_advert_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract advert ID from AutoTrader URL.

        Args:
            url: AutoTrader URL containing advert ID

        Returns:
            Advert ID if found, None otherwise
        """
        # Look for pattern like 'car-details/202508295866886' or similar
        match = re.search(r'car-details/(\d+)', url)
        if match:
            return match.group(1)
        return None

    def get_price_deviation(self, advert_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Get price deviation data for a specific advert ID.

        Args:
            advert_id: AutoTrader advert ID
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary with price deviation data
        """
        url = f"{self.base_url}/{advert_id}?channel=cars&postcode={self.postcode}"

        for attempt in range(max_retries + 1):
            try:
                # Add delays for anti-detection (except on first request)
                if self._first_request_made and attempt == 0:
                    delay = random.uniform(0.1, 0.3)
                    time.sleep(delay)
                else:
                    self._first_request_made = True

                response = self.session.get(
                    url,
                    timeout=10,
                    verify=False  # Disable SSL verification like main API
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_price_data(data, advert_id)

                elif response.status_code == 404:
                    return {
                        'success': False,
                        'error': f'Advert {advert_id} not found (404)',
                        'price': 0,
                        'market_difference': 0,
                        'market_value': 0,
                        'price_difference_found': False,
                        'deviation_text': 'Advert not found'
                    }

                else:
                    error_msg = f"HTTP {response.status_code}"
                    if attempt < max_retries:
                        logger.debug(f"GraphQL API attempt {attempt + 1} failed: {error_msg}, retrying...")
                        time.sleep(random.uniform(1, 3))
                        continue
                    else:
                        return {
                            'success': False,
                            'error': error_msg,
                            'price': 0,
                            'market_difference': 0,
                            'market_value': 0,
                            'price_difference_found': False,
                            'deviation_text': f'HTTP Error: {response.status_code}'
                        }

            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if attempt < max_retries:
                    logger.debug(f"GraphQL API attempt {attempt + 1} failed: {error_msg}, retrying...")
                    time.sleep(random.uniform(1, 3))
                    continue
                else:
                    return {
                        'success': False,
                        'error': error_msg,
                        'price': 0,
                        'market_difference': 0,
                        'market_value': 0,
                        'price_difference_found': False,
                        'deviation_text': f'Request Error: {str(e)[:50]}'
                    }

        # Should not reach here, but just in case
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'price': 0,
            'market_difference': 0,
            'market_value': 0,
            'price_difference_found': False,
            'deviation_text': 'Max retries exceeded'
        }

    def _parse_price_data(self, data: Dict[str, Any], advert_id: str) -> Dict[str, Any]:
        """
        Parse price deviation data from GraphQL API response.
        Uses the same logic as browser scraper for consistency.

        Args:
            data: JSON response from GraphQL API
            advert_id: Advert ID for logging

        Returns:
            Dictionary with parsed price data
        """
        try:
            # Extract price information from response
            if 'heading' not in data:
                return {
                    'success': False,
                    'error': 'No heading data in response',
                    'price': 0,
                    'market_difference': 0,
                    'market_value': 0,
                    'price_difference_found': False,
                    'deviation_text': 'No heading data'
                }

            heading = data['heading']

            # Extract asking price
            asking_price = 0
            if 'priceBreakdown' in heading and 'price' in heading['priceBreakdown']:
                price_info = heading['priceBreakdown']['price']
                asking_price = price_info.get('price', 0)

            if asking_price <= 0:
                return {
                    'success': False,
                    'error': 'No valid price found',
                    'price': 0,
                    'market_difference': 0,
                    'market_value': 0,
                    'price_difference_found': False,
                    'deviation_text': 'No price data'
                }

            # Extract deviation text
            deviation_text = ''
            if 'priceBreakdown' in heading and 'price' in heading['priceBreakdown']:
                price_info = heading['priceBreakdown']['price']
                deviation_text = price_info.get('deviation', '')

            # Parse deviation using same logic as browser scraper
            market_difference = 0
            price_difference_found = False

            if deviation_text:
                deviation_lower = deviation_text.lower()

                # Look for price patterns like "£234 below market" or "£150 above market"
                price_pattern = r'£?([\d,]+).*?(below|above).*?market'
                match = re.search(price_pattern, deviation_lower)

                if match:
                    price_value = int(match.group(1).replace(',', ''))
                    direction = match.group(2)

                    if direction == 'below':
                        market_difference = -price_value  # Negative for below market
                        price_difference_found = True
                    elif direction == 'above':
                        market_difference = price_value   # Positive for above market
                        price_difference_found = True
                else:
                    # Fallback to simple above/below detection if no specific price found
                    if 'above market' in deviation_lower:
                        market_difference = 1
                        price_difference_found = True
                    elif 'below market' in deviation_lower:
                        market_difference = -1
                        price_difference_found = True
                    elif 'close to market' in deviation_lower:
                        market_difference = 0
                        price_difference_found = True

            # Calculate market value
            market_value = asking_price + market_difference if asking_price > 0 and price_difference_found else 0

            result = {
                'success': True,
                'price': asking_price,
                'market_difference': market_difference,
                'market_value': market_value,
                'price_difference_found': price_difference_found,
                'deviation_text': deviation_text or 'No deviation data'
            }

            # Log the successful extraction
            original_url = f"https://www.autotrader.co.uk/car-details/{advert_id}"
            log_price_marker_data(original_url, result, success=True)

            return result

        except Exception as e:
            error_result = {
                'success': False,
                'error': f'Parse error: {str(e)}',
                'price': 0,
                'market_difference': 0,
                'market_value': 0,
                'price_difference_found': False,
                'deviation_text': f'Parse error: {str(e)[:50]}'
            }

            # Log the error
            original_url = f"https://www.autotrader.co.uk/car-details/{advert_id}"
            log_price_marker_data(original_url, error_result, success=False, error=str(e))

            return error_result

    def close(self):
        """Close the session and clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()


# Helper function to maintain compatibility with existing code
def create_graphql_tasks_from_urls(urls: List[str], make: str = None, model: str = None) -> List[GraphQLTask]:
    """
    Create GraphQL tasks from a list of AutoTrader URLs.

    Args:
        urls: List of AutoTrader URLs
        make: Optional make for metadata
        model: Optional model for metadata

    Returns:
        List of GraphQL tasks
    """
    tasks = []
    client = AutoTraderGraphQLClient()

    for url in urls:
        advert_id = client.extract_advert_id_from_url(url)
        if advert_id:
            task = GraphQLTask(
                advert_id=advert_id,
                original_url=url,
                make=make,
                model=model
            )
            tasks.append(task)

    client.close()
    return tasks


# Main interface functions (for backward compatibility and cleaner imports)
def scrape_price_marker(url, headless=True):
    """
    Scrape price marker from a URL using GraphQL API.

    This is the main interface function that provides a clean API for single URL scraping.
    The headless parameter is ignored but kept for compatibility.

    Args:
        url: AutoTrader URL to scrape
        headless: Ignored (kept for compatibility)

    Returns:
        Dict with price marker data
    """
    try:
        # Create GraphQL client
        client = AutoTraderGraphQLClient(verify_ssl=False)

        # Extract advert ID from URL
        advert_id = client.extract_advert_id_from_url(url)

        if not advert_id:
            result = {
                'price': 0,
                'market_difference': 0,
                'market_value': 0,
                'price_difference_found': False,
                'marker_text': 'Error: Could not extract advert ID from URL'
            }
            log_price_marker_data(url, result, success=False, error='Could not extract advert ID')
            client.close()
            return result

        # Get price deviation data from GraphQL API
        api_result = client.get_price_deviation(advert_id)
        client.close()

        return api_result

    except Exception as e:
        result = {
            'price': 0,
            'market_difference': 0,
            'market_value': 0,
            'price_difference_found': False,
            'marker_text': f'Error: {str(e)}'
        }

        log_price_marker_data(url, result, success=False, error=str(e))
        return result


def batch_scrape_price_markers(urls, headless=True, progress_callback=None, test_mode=False, **kwargs):
    """
    Batch scrape price markers using GraphQL API.

    This is the main interface function for batch processing.
    The headless parameter is ignored but kept for compatibility.

    Args:
        urls: List of URLs to scrape
        headless: Ignored (kept for compatibility)
        progress_callback: Function to call with progress updates
        test_mode: Skip delays in test mode
        **kwargs: Additional arguments

    Returns:
        Dict mapping URLs to scraping results
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import logging
        import time
        import random

        logger = logging.getLogger(__name__)
        output_manager = get_output_manager()

        total_urls = len(urls)
        max_workers = min(kwargs.get('max_concurrent_browsers', 6), 10)  # Cap at 10 for HTTP requests

        # Use output manager for consistent progress reporting
        output_manager.progress_update(0, total_urls, "retail_prices", 0, "Starting...")

        # Track timing for rate calculation
        start_time = time.time()

        # Create GraphQL tasks from URLs
        tasks = create_graphql_tasks_from_urls(urls)

        if not tasks:
            output_manager.warning("No valid advert IDs found in URLs")
            logger.warning("No valid advert IDs found in URLs")
            return {}

        # Initialize proxy manager if available
        proxy_manager = None
        try:
            from services.stealth_orchestrator import ProxyManager
            proxy_manager = ProxyManager()
        except ImportError:
            logger.debug("ProxyManager not available, workers will use no proxy")

        # Initialize GraphQL coordinator
        coordinator = GraphQLCoordinator(target_rate_per_minute=300, max_workers=max_workers)
        coordinator.add_work(tasks)

        def graphql_worker_thread(worker_id):
            """GraphQL API worker thread."""
            try:
                # Get dedicated proxy for this worker
                worker_proxy = None
                if proxy_manager:
                    worker_proxy = proxy_manager.get_proxy_for_worker(worker_id)

                # Create GraphQL client for this worker
                client = AutoTraderGraphQLClient(
                    proxy=worker_proxy,
                    worker_id=worker_id,
                    verify_ssl=False
                )

                # Process tasks from coordinator
                tasks_processed = 0
                consecutive_empty_gets = 0

                while tasks_processed < 1000:  # Higher limit for API requests
                    task = coordinator.get_next_work(worker_id, timeout=0.2)
                    if task is None:
                        consecutive_empty_gets += 1
                        if consecutive_empty_gets >= 1:
                            break
                        continue

                    consecutive_empty_gets = 0

                    try:
                        logger.debug(f"GraphQL Worker {worker_id} processing advert {task.advert_id}")

                        # Get price deviation from GraphQL API
                        result = client.get_price_deviation(task.advert_id)

                        # Report result using original URL as key (for compatibility)
                        coordinator.report_result(task.original_url, result, worker_id)

                        logger.debug(f"GraphQL Worker {worker_id} completed advert {task.advert_id} - Status: {coordinator.last_result_status}")

                        # Update progress callback (both custom and output manager)
                        if progress_callback:
                            progress_callback(coordinator.completed_count, total_urls)

                        # Always update output manager for consistent UI
                        output_manager = get_output_manager()

                        # Calculate actual rate and ETA
                        elapsed_time = time.time() - start_time
                        rate = (coordinator.completed_count * 60 / elapsed_time) if elapsed_time > 0 else 0
                        remaining_items = total_urls - coordinator.completed_count
                        eta = ""
                        if rate > 0 and remaining_items > 0:
                            eta_minutes = remaining_items / rate
                            if eta_minutes < 1:
                                eta = f"{int(eta_minutes * 60)}s"
                            else:
                                eta = f"{int(eta_minutes)}m {int((eta_minutes % 1) * 60)}s"

                        output_manager.progress_update(
                            coordinator.completed_count,
                            total_urls,
                            "retail_prices",
                            rate,
                            eta
                        )

                        tasks_processed += 1

                        # Rate limiting for API requests (lighter than browser)
                        if not test_mode:
                            time.sleep(random.uniform(0.02, 0.1))  # Much faster than browser

                    except Exception as e:
                        error_result = {
                            'success': False,
                            'error': str(e),
                            'price': 0,
                            'market_difference': 0,
                            'market_value': 0,
                            'price_difference_found': False,
                            'marker_text': f'Worker Error: {str(e)}'
                        }
                        coordinator.report_result(task.original_url, error_result, worker_id)

                        # Update progress callback (both custom and output manager)
                        if progress_callback:
                            progress_callback(coordinator.completed_count, total_urls)

                        # Always update output manager for consistent UI
                        output_manager = get_output_manager()

                        # Calculate actual rate and ETA
                        elapsed_time = time.time() - start_time
                        rate = (coordinator.completed_count * 60 / elapsed_time) if elapsed_time > 0 else 0
                        remaining_items = total_urls - coordinator.completed_count
                        eta = ""
                        if rate > 0 and remaining_items > 0:
                            eta_minutes = remaining_items / rate
                            if eta_minutes < 1:
                                eta = f"{int(eta_minutes * 60)}s"
                            else:
                                eta = f"{int(eta_minutes)}m {int((eta_minutes % 1) * 60)}s"

                        output_manager.progress_update(
                            coordinator.completed_count,
                            total_urls,
                            "retail_prices",
                            rate,
                            eta
                        )

                client.close()

            except Exception as e:
                logger.error(f"GraphQL Worker {worker_id} fatal error: {e}")

        # Start all GraphQL workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for worker_id in range(max_workers):
                future = executor.submit(graphql_worker_thread, worker_id)
                futures.append(future)

            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"GraphQL Worker thread error: {e}")

        # Mark progress as complete using output manager
        output_manager = get_output_manager()

        # Calculate final statistics
        final_count = len(coordinator.results)
        elapsed_time = time.time() - start_time
        final_rate = (final_count * 60 / elapsed_time) if elapsed_time > 0 else 0

        output_manager.progress_complete("retail_prices", count=final_count, rate=final_rate, total_count=total_urls)

        logger.debug(f"✅ GraphQL API orchestration completed with {len(coordinator.results)} results")
        return coordinator.results

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)

        # Use output manager for error reporting
        output_manager = get_output_manager()
        output_manager.error(f"GraphQL batch processing failed: {e}")

        logger.error(f"Error in GraphQL batch processing: {e}")
        # Fallback to empty results
        return {}