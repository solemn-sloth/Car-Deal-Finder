#!/usr/bin/env python
"""
ProxyManager for handling proxy rotation and blacklisting.
"""

import os
import json
import datetime
import random
import re
from typing import Optional, Dict, List, Any

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
        self._load_config()
        
        # Create archive directory if it doesn't exist
        os.makedirs(self.archive_path, exist_ok=True)
    
    def _load_config(self) -> None:
        """Load proxy configuration from file."""
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
                print(f"IP Initialized: {ip}:{proxy['port']} ({proxy.get('country', 'Unknown')}) | Status: Working")
                
                return proxy_url
                
            attempts += 1
            
        print("All proxies are blacklisted. Proceeding without proxy.")
        return None
    
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


# Example usage
if __name__ == "__main__":
    # Initialize the proxy manager
    proxy_manager = ProxyManager()
    
    # Get a proxy
    proxy_url = proxy_manager.get_proxy()
    print(f"Selected proxy: {proxy_url}")
    
    # Get country-specific proxy
    uk_proxy = proxy_manager.get_proxy_for_country("United Kingdom")
    print(f"UK proxy: {uk_proxy}")
    
    # Extract IP and blacklist for testing
    ip = proxy_manager.extract_ip(proxy_url)
    if ip:
        print(f"Extracted IP: {ip}")
        proxy_manager.blacklist_ip(ip, reason="Test blacklisting")
        print(f"Blacklisted {ip}")
        
    # Get another proxy (should be different due to blacklisting)
    proxy_url = proxy_manager.get_proxy()
    print(f"Next proxy: {proxy_url}")
    
    print("\nIntegration Example:")
    print("""
# To use with retail_price_scraper.py:

# Import the proxy manager
from services.proxy_rotation import ProxyManager

# Initialize the proxy manager
proxy_manager = ProxyManager()

# Use with batch_scrape_price_markers
results = batch_scrape_price_markers(
    urls=your_urls,
    use_proxy_rotation=True,  # Enable proxy rotation
    proxy_manager=proxy_manager  # Pass the proxy manager instance
)

# Alternatively, use with a direct browser:
with sync_playwright() as playwright:
    # Create browser manager with proxy
    proxy_url = proxy_manager.get_proxy()
    browser_manager = BrowserManager(playwright, proxy=proxy_url)
    
    # Use the browser manager
    # ...
    
    # If you encounter a 403 error, blacklist the IP
    ip = proxy_manager.extract_ip(proxy_url)
    if ip:
        proxy_manager.blacklist_ip(ip, reason="HTTP 403 Forbidden")
    """)
    