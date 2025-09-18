#!/usr/bin/env python3
"""
Generic Stealth Orchestrator
Manages parallel workers with unique IPs and fingerprints for anti-detection.
Extracted from retail_price_scraper.py to support both browser and API scraping.
"""

import threading
import queue
import time
import random
import logging
import os
import json
import datetime
import re
import requests
import ssl
import secrets
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

# Try to import pyautogui for mouse movement (optional)
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BROWSER FINGERPRINTING CONSTANTS AND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Geographic profiles for consistent fingerprinting based on proxy location
GEO_PROFILES = {
    'us': {
        'locales': ['en-US'],
        'timezones': ['America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles'],
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
    if not PYAUTOGUI_AVAILABLE:
        return

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

def generate_realistic_user_agents(count=8):
    """Generate realistic, current user agents with minor variations"""
    # Current Chrome versions (as of 2024)
    chrome_versions = ['120.0.0.0', '121.0.0.0', '122.0.0.0', '123.0.0.0', '124.0.0.0']

    # Recent WebKit versions
    webkit_versions = ['537.36']

    # OS variations with realistic versions
    os_variants = [
        'Windows NT 10.0; Win64; x64',
        'Windows NT 11.0; Win64; x64',
        'Macintosh; Intel Mac OS X 10_15_7',
        'Macintosh; Intel Mac OS X 13_0_0',
        'Macintosh; Intel Mac OS X 14_0_0',
        'X11; Linux x86_64',
    ]

    user_agents = []

    for _ in range(count):
        chrome_ver = random.choice(chrome_versions)
        webkit_ver = random.choice(webkit_versions)
        os_variant = random.choice(os_variants)

        # Add minor version variations
        minor_ver = random.randint(0, 9)
        chrome_ver_full = f"{chrome_ver[:-1]}{minor_ver}"

        user_agent = f"Mozilla/5.0 ({os_variant}) AppleWebKit/{webkit_ver} (KHTML, like Gecko) Chrome/{chrome_ver_full} Safari/{webkit_ver}"
        user_agents.append(user_agent)

    return user_agents


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP/TLS FINGERPRINTING CLASSES AND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class CustomHTTPAdapter(requests.adapters.HTTPAdapter):
    """Custom adapter with specific cipher suites and TLS versions to mimic browser TLS fingerprints"""
    def __init__(self, *args, verify_ssl=False, **kwargs):
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


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED FINGERPRINT CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class BrowserFingerprint:
    """Complete browser fingerprint with geo-consistency"""

    def __init__(self, proxy_url: Optional[str] = None, worker_id: Optional[int] = None):
        self.proxy_url = proxy_url
        self.worker_id = worker_id
        self.geo_profile_key = detect_geo_profile_from_proxy(proxy_url)
        self.geo_profile = GEO_PROFILES[self.geo_profile_key]
        self._generate_fingerprint()

    def _generate_fingerprint(self):
        """Generate consistent fingerprint for this worker/proxy combination"""
        # Use worker_id as seed for consistency if provided
        if self.worker_id is not None:
            # Set seed based on worker_id for consistency
            local_random = random.Random(self.worker_id)
        else:
            local_random = random

        # Generate geo-consistent fingerprint components
        self.viewport = local_random.choice(REALISTIC_RESOLUTIONS)
        self.locale = local_random.choice(self.geo_profile['locales'])
        self.timezone = local_random.choice(self.geo_profile['timezones'])
        self.device_scale_factor = local_random.choice([1, 1.25, 1.5, 2])

        # Generate user agent with geo-consistent OS
        chrome_versions = ['120.0.0.0', '121.0.0.0', '122.0.0.0', '123.0.0.0', '124.0.0.0']
        chrome_ver = local_random.choice(chrome_versions)
        self.user_agent = f"Mozilla/5.0 ({self.geo_profile['user_agent_os']}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver} Safari/537.36"
        self.chrome_version = chrome_ver.split('.')[0]

    def get_browser_context_options(self):
        """Get browser context options for Playwright"""
        context_options = {
            'viewport': {'width': self.viewport[0], 'height': self.viewport[1]},
            'user_agent': self.user_agent,
            'locale': self.locale,
            'timezone_id': self.timezone,
            'device_scale_factor': self.device_scale_factor,
            'java_script_enabled': True,
            'is_mobile': False,
            'has_touch': False  # Explicitly desktop
        }

        # Add proxy configuration if provided
        if self.proxy_url:
            import re
            # Extract username:password if present
            auth_match = re.search(r'://(.*?)@', self.proxy_url)
            server = None

            if auth_match:
                # Get authentication credentials
                auth = auth_match.group(1)
                # Extract server part (after auth)
                server_match = re.search(r'@(.*)', self.proxy_url)
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

        return context_options


class HTTPFingerprint:
    """HTTP/TLS fingerprint for API requests with geo-consistency"""

    def __init__(self, proxy_url: Optional[str] = None, worker_id: Optional[int] = None):
        self.proxy_url = proxy_url
        self.worker_id = worker_id
        self.geo_profile_key = detect_geo_profile_from_proxy(proxy_url)
        self.geo_profile = GEO_PROFILES[self.geo_profile_key]
        self._generate_fingerprint()

    def _generate_fingerprint(self):
        """Generate consistent HTTP fingerprint for this worker/proxy combination"""
        # Use worker_id as seed for consistency if provided
        if self.worker_id is not None:
            local_random = random.Random(self.worker_id)
        else:
            local_random = random

        # Generate geo-consistent user agent
        chrome_versions = ['118.0.5993.88', '119.0.6045.123', '120.0.6099.129', '121.0.6167.85', '122.0.6261.94']
        self.chrome_version = local_random.choice(chrome_versions)
        self.browser_version = self.chrome_version.split(".")[0]

        # Use geo-profile for consistent OS
        self.user_agent = f"Mozilla/5.0 ({self.geo_profile['user_agent_os']}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{self.chrome_version} Safari/537.36"

        # Generate consistent client hints
        self.device_memory = local_random.choice(["4", "8", "16"])
        self.viewport_width = local_random.choice(["1280", "1366", "1440", "1536", "1920"])
        self.viewport_height = local_random.choice(["720", "768", "900", "864", "1080"])
        self.dpr = local_random.choice(["1", "1.25", "1.5", "2"])

        # Geo-consistent accept language
        if self.geo_profile_key == 'us':
            self.accept_language = local_random.choice([
                "en-US,en;q=0.9",
                "en-US,en-GB;q=0.9,en;q=0.8"
            ])
        elif self.geo_profile_key == 'uk':
            self.accept_language = local_random.choice([
                "en-GB,en;q=0.9",
                "en-GB,en-US;q=0.9,en;q=0.8"
            ])
        elif self.geo_profile_key == 'ca':
            self.accept_language = local_random.choice([
                "en-CA,en;q=0.9",
                "en-CA,en-US;q=0.9,en;q=0.8"
            ])
        elif self.geo_profile_key == 'au':
            self.accept_language = local_random.choice([
                "en-AU,en;q=0.9",
                "en-AU,en-US;q=0.9,en;q=0.8"
            ])
        else:
            self.accept_language = "en-GB,en;q=0.9"

    def get_cloudflare_headers(self):
        """Get CloudFlare-optimized headers with geo-consistency"""
        ua_brand = f'"Google Chrome";v="{self.browser_version}", "Chromium";v="{self.browser_version}", "Not=A?Brand";v="99"'

        # Extract platform from user agent for sec-ch-ua-platform
        if "Windows" in self.user_agent:
            platform = "Windows"
        elif "Macintosh" in self.user_agent:
            platform = "macOS"
        elif "Linux" in self.user_agent:
            platform = "Linux"
        else:
            platform = "Windows"  # Default

        headers = {
            # Essential headers CloudFlare always checks
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": self.accept_language,
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",

            # Security headers that CloudFlare checks
            "sec-ch-ua": ua_brand,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": f'"{platform}"',
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "navigate",
            "sec-fetch-user": "?1",
            "sec-fetch-dest": "document",

            # Enhanced client hints for better fingerprinting
            "sec-ch-ua-arch": random.choice(['x86', 'arm']),
            "sec-ch-ua-full-version": self.chrome_version,
            "sec-ch-ua-platform-version": random.choice(['10.0.0', '11.0.0', '12.0.0', '15.6.1']),
            "sec-ch-ua-model": "",  # Desktop browsers don't have a model
            "sec-ch-ua-bitness": random.choice(['64', '32']),
            "sec-ch-ua-wow64": "?0",  # Not Windows-on-Windows
            "sec-ch-device-memory": self.device_memory,

            # Additional browser capability indicators
            "device-memory": self.device_memory,
            "dpr": self.dpr,
            "viewport-width": self.viewport_width,
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

        return headers

    def get_api_headers(self):
        """Get API-specific headers for AJAX requests"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/plain, */*',
            'User-Agent': self.user_agent,
            'Accept-Language': self.accept_language,
            'Accept-Encoding': 'gzip, deflate, br',
            'X-Requested-With': 'XMLHttpRequest',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'DNT': str(random.choice([0, 1])),
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        return headers


class FingerprintGenerator:
    """Unified fingerprint generation for both browser and API"""

    @staticmethod
    def create_browser_fingerprint(proxy_url: Optional[str] = None, worker_id: Optional[int] = None) -> BrowserFingerprint:
        """Create a browser fingerprint with geo-consistency"""
        return BrowserFingerprint(proxy_url=proxy_url, worker_id=worker_id)

    @staticmethod
    def create_http_fingerprint(proxy_url: Optional[str] = None, worker_id: Optional[int] = None) -> HTTPFingerprint:
        """Create an HTTP fingerprint with geo-consistency"""
        return HTTPFingerprint(proxy_url=proxy_url, worker_id=worker_id)

    @staticmethod
    def create_worker_fingerprint(worker_id: int, proxy_url: Optional[str] = None) -> Dict[str, Any]:
        """Create a complete fingerprint set for a worker"""
        browser_fp = BrowserFingerprint(proxy_url=proxy_url, worker_id=worker_id)
        http_fp = HTTPFingerprint(proxy_url=proxy_url, worker_id=worker_id)

        return {
            'worker_id': worker_id,
            'proxy_url': proxy_url,
            'geo_profile': browser_fp.geo_profile_key,
            'browser': browser_fp,
            'http': http_fp,
            'consistent_user_agent': browser_fp.user_agent  # Same user agent for both
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROXY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class ProxyManager:
    """
    Manages proxy rotation, selection, and blacklisting.
    Uses a file-based blacklisting system where blacklisted IPs
    are stored as .txt files in the archive folder.
    """

    def __init__(self, config_path: str = "config/proxies.json", archive_path: str = "archive"):
        """
        Initialize the proxy manager.

        Args:
            config_path: Path to the proxy configuration file
            archive_path: Path to store blacklisted proxy information
        """
        self.proxies = []
        self.current_index = 0
        self.archive_path = archive_path
        self.config_path = config_path
        self.settings = {}

        # Webshare API cache variables
        self._api_proxies = []
        self._last_api_refresh = None

        self._load_config()

        # Create archive directory if it doesn't exist
        os.makedirs(self.archive_path, exist_ok=True)

    def _load_config(self) -> None:
        """Load proxy configuration, trying API first, then JSON fallback."""
        # Try to load from Webshare API first
        if self._should_refresh_api_cache():
            if self._fetch_webshare_proxies():
                # Successfully got API proxies, use them
                self.proxies = self._api_proxies
                self._load_proxy_settings()
                return
        elif self._api_proxies:
            # Use cached API proxies if available and not expired
            self.proxies = self._api_proxies
            self._load_proxy_settings()
            return

        # Fallback to JSON file
        self._load_from_json_file()

    def _should_refresh_api_cache(self) -> bool:
        """Check if API cache should be refreshed (>24h old or never fetched)."""
        if not self._last_api_refresh:
            return True

        # Check if more than 24 hours have passed
        try:
            from config.config import WEBSHARE_PROXY_CONFIG
            refresh_interval = WEBSHARE_PROXY_CONFIG.get('refresh_interval_hours', 24)
        except ImportError:
            refresh_interval = 24

        cache_age_hours = (time.time() - self._last_api_refresh) / 3600

        return cache_age_hours >= refresh_interval

    def _fetch_webshare_proxies(self) -> bool:
        """Fetch proxies from Webshare API and store in cache."""
        try:
            try:
                from config.config import WEBSHARE_API_TOKEN, WEBSHARE_API_BASE_URL, WEBSHARE_PROXY_CONFIG
            except ImportError:
                return False

            if not WEBSHARE_API_TOKEN:
                return False

            headers = {"Authorization": f"Token {WEBSHARE_API_TOKEN}"}
            timeout = WEBSHARE_PROXY_CONFIG.get('api_timeout', 30)
            max_proxies = WEBSHARE_PROXY_CONFIG.get('max_proxies', 100)

            response = requests.get(
                f"{WEBSHARE_API_BASE_URL}/proxy/list/",
                headers=headers,
                params={
                    "mode": "direct",
                    "page_size": max_proxies
                },
                timeout=timeout
            )

            if response.status_code == 200:
                api_data = response.json()
                api_proxies = api_data.get("results", [])

                # Convert API format to internal format
                self._api_proxies = []
                for proxy in api_proxies:
                    formatted_proxy = {
                        "ip": proxy.get("proxy_address"),
                        "port": str(proxy.get("port")),
                        "username": proxy.get("username"),
                        "password": proxy.get("password"),
                        "country": proxy.get("country_code", "Unknown")
                    }
                    self._api_proxies.append(formatted_proxy)

                self._last_api_refresh = time.time()
                print(f"✅ Proxies refreshed ({len(self._api_proxies)} available)")
                return True
            else:
                print(f"❌ API failed (HTTP {response.status_code}), using cached proxies")
                return False

        except Exception as e:
            print(f"❌ API failed ({str(e)}), using cached proxies")
            return False

    def _load_proxy_settings(self) -> None:
        """Load proxy settings from JSON file (for rotation strategy, etc.)."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.settings = config.get('proxy_settings', {})
        except Exception:
            self.settings = {}

        # Apply rotation strategy to current proxy list
        if self.settings.get('rotation_strategy') == 'random':
            random.shuffle(self.proxies)

    def _load_from_json_file(self) -> None:
        """Load proxy configuration from JSON file (fallback method)."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.proxies = config.get('proxies', [])
                self.settings = config.get('proxy_settings', {})

            # Shuffle the proxies if random strategy is selected
            if self.settings.get('rotation_strategy') == 'random':
                random.shuffle(self.proxies)

        except Exception as e:
            print(f"Error loading proxy config: {e}")
            self.proxies = []

    def is_blacklisted(self, ip: str) -> bool:
        """
        Check if an IP is blacklisted by looking for a matching filename.

        Args:
            ip: The IP address to check

        Returns:
            True if blacklisted, False otherwise
        """
        blacklist_file = os.path.join(self.archive_path, f"{ip}.txt")
        return os.path.exists(blacklist_file)

    def blacklist_ip(self, ip: str, reason: str = "HTTP 403 Forbidden") -> None:
        """
        Blacklist an IP by creating a .txt file in the archive folder.

        Args:
            ip: The IP address to blacklist
            reason: Reason for blacklisting
        """
        if not ip:
            return

        # Create blacklist file with timestamp and reason
        blacklist_file = os.path.join(self.archive_path, f"{ip}.txt")
        with open(blacklist_file, 'w') as f:
            f.write(f"Blacklisted at: {datetime.datetime.now()}\n")
            f.write(f"Reason: {reason}\n")

    def get_proxy(self) -> Optional[str]:
        """
        Get the next available non-blacklisted proxy.

        Returns:
            Formatted proxy URL or None if no proxies available
        """
        if not self.proxies:
            return None

        # Check all proxies to find a non-blacklisted one
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_index]

            # Update index based on rotation strategy
            if self.settings.get('rotation_strategy') == 'random':
                self.current_index = random.randint(0, len(self.proxies) - 1)
            else:  # Default to round-robin
                self.current_index = (self.current_index + 1) % len(self.proxies)

            ip = proxy['ip']
            if not self.is_blacklisted(ip):
                # Format and return the proxy
                auth = f"{proxy['username']}:{proxy['password']}" if 'username' in proxy and 'password' in proxy else ""
                proxy_url = f"http://{auth}@{ip}:{proxy['port']}" if auth else f"http://{ip}:{proxy['port']}"

                # Print the single line status message
                print(f"🌐 IP: {ip}:{proxy['port']} ({proxy.get('country', 'Unknown')}) ✅")
                print()

                return proxy_url

            attempts += 1

        print("All proxies are blacklisted. Proceeding without proxy.")
        return None

    def get_proxy_for_worker(self, worker_id: int) -> Optional[str]:
        """
        Get a dedicated proxy for a specific worker.
        Each worker gets a consistent IP to maintain session affinity.

        Args:
            worker_id: Unique worker identifier (0-5 for 6 workers)

        Returns:
            Formatted proxy URL or None if no proxies available
        """
        if not self.proxies:
            return None

        # Find all available (non-blacklisted) proxies
        available_proxies = [
            p for p in self.proxies
            if not self.is_blacklisted(p['ip'])
        ]

        if not available_proxies:
            print("All proxies are blacklisted. Proceeding without proxy.")
            return None

        # Assign proxy based on worker_id for consistency
        # This ensures each worker gets the same IP across restarts
        proxy_index = worker_id % len(available_proxies)
        proxy = available_proxies[proxy_index]

        ip = proxy['ip']
        auth = f"{proxy['username']}:{proxy['password']}" if 'username' in proxy and 'password' in proxy else ""
        proxy_url = f"http://{auth}@{ip}:{proxy['port']}" if auth else f"http://{ip}:{proxy['port']}"

        # Quiet assignment - no print to maintain existing console aesthetics
        return proxy_url

    def extract_ip(self, proxy_url: str) -> Optional[str]:
        """
        Extract IP address from a proxy URL.

        Args:
            proxy_url: The proxy URL to extract IP from

        Returns:
            The IP address or None if extraction failed
        """
        if not proxy_url:
            return None

        try:
            if '@' in proxy_url:
                # Format: http://user:pass@ip:port
                ip = proxy_url.split('@')[1].split(':')[0]
            else:
                # Format: http://ip:port
                ip = proxy_url.split('//')[1].split(':')[0]

            return ip
        except Exception:
            return None

    def get_proxy_for_country(self, country: str) -> Optional[str]:
        """
        Get a proxy from a specific country.

        Args:
            country: The country to get a proxy from

        Returns:
            Formatted proxy URL or None if no proxies available
        """
        # Filter proxies by country
        country_proxies = [
            p for p in self.proxies
            if p.get('country', '').lower() == country.lower() and not self.is_blacklisted(p['ip'])
        ]

        if not country_proxies:
            return self.get_proxy()  # Fall back to any proxy if no country match

        # Select a random proxy from the filtered list
        proxy = random.choice(country_proxies)
        ip = proxy['ip']

        # Format and return the proxy
        auth = f"{proxy['username']}:{proxy['password']}" if 'username' in proxy and 'password' in proxy else ""
        proxy_url = f"http://{auth}@{ip}:{proxy['port']}" if auth else f"http://{ip}:{proxy['port']}"

        # Print the single line status message
        print(f"IP Initialized: {ip}:{proxy['port']} ({proxy.get('country', 'Unknown')}) | Status: Working")

        return proxy_url


class WorkerInterface(ABC):
    """Abstract interface that all worker types must implement."""

    @abstractmethod
    def __init__(self, worker_id: int, proxy_url: Optional[str] = None, **kwargs):
        """Initialize worker with unique ID and proxy."""
        pass

    @abstractmethod
    def process_task(self, task_data: Any) -> Dict[str, Any]:
        """Process a single task and return standardized result."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up worker resources."""
        pass


class StealthOrchestrator:
    """
    Generic orchestrator for parallel workers with anti-detection features.
    Manages worker coordination, proxy distribution, rate limiting, and progress tracking.
    """

    def __init__(self, worker_class: type, target_rate_per_minute: int = 300, max_workers: int = 6):
        """
        Initialize orchestrator.

        Args:
            worker_class: Class implementing WorkerInterface
            target_rate_per_minute: Global rate limit for all workers combined
            max_workers: Maximum number of parallel workers
        """
        self.worker_class = worker_class
        self.target_rate_per_minute = target_rate_per_minute
        self.max_workers = max_workers
        self.work_queue = queue.Queue()
        self.results = {}
        self.completed_count = 0
        self.failed_workers = set()
        self.results_lock = threading.Lock()
        self.rate_limiter = threading.Semaphore(1)
        self.last_request_time = 0
        self.min_interval = 60.0 / target_rate_per_minute
        self.last_result_status = "⚠️ Starting up"

    def add_work(self, tasks: List[Any]):
        """Add tasks to work queue for parallel distribution."""
        for task in tasks:
            self.work_queue.put(task)

    def get_next_work(self, worker_id: int, timeout: float = 1) -> Optional[Any]:
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
        self.work_queue.task_done()

    def _get_result_status(self, result: Dict[str, Any]) -> str:
        """Get status indicator for result - can be overridden by subclasses."""
        if result.get('success', True):
            return "✅ Success"
        else:
            return f"❌ {result.get('error', 'Unknown error')}"

    def _is_worker_detected(self, result: Dict[str, Any]) -> bool:
        """Check if worker was detected/blocked - can be overridden by subclasses."""
        error_msg = result.get('error', '').lower()
        return any(keyword in error_msg for keyword in [
            'cloudflare', '403', 'forbidden', 'blocked', 'challenge'
        ])

    def process_batch(self, tasks: List[Any], progress_callback: Optional[Callable] = None,
                     test_mode: bool = False, worker_kwargs: Optional[Dict] = None,
                     max_tasks_per_worker: int = 500) -> Dict[str, Any]:
        """
        Process batch of tasks using parallel workers.

        Args:
            tasks: List of tasks to process
            progress_callback: Optional callback for progress updates
            test_mode: Skip delays in test mode
            worker_kwargs: Additional kwargs to pass to workers
            max_tasks_per_worker: Maximum tasks per worker before restart

        Returns:
            Dictionary mapping task IDs to results
        """
        total_tasks = len(tasks)
        worker_kwargs = worker_kwargs or {}

        logger.info(f"🚀 Starting orchestration with {self.max_workers} workers")
        logger.info(f"📊 Processing {total_tasks} tasks")

        # Initialize proxy manager if available
        proxy_manager = None
        try:
            proxy_manager = ProxyManager()
        except Exception as e:
            logger.warning(f"ProxyManager initialization failed: {e}, workers will use no proxy")

        # Add work to queue
        self.add_work(tasks)

        def worker_thread(worker_id: int):
            """Worker thread that processes tasks."""
            try:
                # Get dedicated proxy for this worker
                worker_proxy = None
                if proxy_manager:
                    worker_proxy = proxy_manager.get_proxy_for_worker(worker_id)

                # Initialize worker with specific configuration
                worker = self.worker_class(
                    worker_id=worker_id,
                    proxy_url=worker_proxy,
                    **worker_kwargs
                )

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
                        logger.debug(f"Worker {worker_id} processing task {tasks_processed+1}")

                        # Process the task
                        result = worker.process_task(task)

                        # Generate task ID (depends on task type)
                        task_id = str(task) if isinstance(task, (str, int)) else str(hash(str(task)))

                        self.report_result(task_id, result, worker_id)

                        logger.debug(f"Worker {worker_id} completed task {tasks_processed+1}")

                        # Update progress callback
                        if progress_callback:
                            progress_callback(self.completed_count, total_tasks, self.last_result_status)

                        tasks_processed += 1

                        # Rate limiting (skip in test mode)
                        if not test_mode:
                            time.sleep(random.uniform(0.05, 0.15))

                    except Exception as e:
                        error_result = {
                            'success': False,
                            'error': str(e),
                            'data': None
                        }
                        task_id = str(task) if isinstance(task, (str, int)) else str(hash(str(task)))
                        self.report_result(task_id, error_result, worker_id)
                        logger.error(f"Worker {worker_id} error processing task: {e}")

                # Cleanup worker resources
                worker.cleanup()
                logger.debug(f"Worker {worker_id} finished, processed {tasks_processed} tasks")

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

        logger.info(f"✅ Orchestration complete - processed {self.completed_count}/{total_tasks} tasks")
        logger.info(f"❌ Failed workers: {len(self.failed_workers)}")

        return self.results

    def get_completion_stats(self) -> Dict[str, Any]:
        """Get completion statistics."""
        return {
            'completed': self.completed_count,
            'failed_workers': len(self.failed_workers),
            'success_rate': self.completed_count / max(len(self.results), 1) * 100 if self.results else 0
        }