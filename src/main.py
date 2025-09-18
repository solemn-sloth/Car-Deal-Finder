#!/usr/bin/env python3
"""
Main entry point for the Car Dealer Bot.
Works from both project root and src directory.
"""
import sys
import os

# Always add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up from src/ to project root

# Make sure project root is first in path
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

# Change working directory to project root for consistent imports
original_cwd = os.getcwd()
os.chdir(project_root)

# Configure SSL bypass before any imports
import urllib3
import requests
import warnings

# Disable SSL verification globally for scraping reliability
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

# Patch requests to disable SSL verification by default
old_merge_environment_settings = requests.Session.merge_environment_settings

def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

requests.Session.merge_environment_settings = merge_environment_settings

# Patch Session class to default verify=False
orig_session = requests.Session

class PatchedSession(orig_session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify = False

requests.Session = PatchedSession

try:
    # Import from project root context
    from src.scraping import main
    
    if __name__ == "__main__":
        main()
finally:
    # Restore original working directory
    os.chdir(original_cwd)
