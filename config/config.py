"""
Configuration file for AutoTrader scraper.
Contains vehicle search configurations and database settings.
"""
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from pathlib import Path

# Load environment variables from config folder
config_path = Path(os.path.dirname(__file__)) / '.env'
load_dotenv(dotenv_path=config_path)

# Supabase Configuration (Primary Storage)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
CAR_DEALS_TABLE = os.getenv('CAR_DEALS_TABLE', 'car_deals')

# Application Configuration
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Webshare Proxy API Configuration
WEBSHARE_API_TOKEN = os.getenv('WEBSHARE_API_TOKEN') or os.getenv('WEBSHARE_API_KEY')
WEBSHARE_API_BASE_URL = "https://proxy.webshare.io/api/v2"
WEBSHARE_PROXY_CONFIG = {
    'refresh_interval_hours': int(os.getenv('PROXY_REFRESH_INTERVAL_HOURS', '24')),  # Hours between API refresh
    'api_timeout': int(os.getenv('WEBSHARE_API_TIMEOUT', '30')),  # API request timeout in seconds
    'max_proxies': int(os.getenv('MAX_PROXIES', '100'))  # Maximum number of proxies to fetch
}

# Weekly Retail Scraping Configuration
WEEKLY_RETAIL_SCRAPING = {
    'interval_days': int(os.getenv('RETAIL_SCRAPING_INTERVAL_DAYS', '7')),  # Days between retail scraping runs
    'schedule_file': os.getenv('RETAIL_SCHEDULE_FILE', 'config/scraping_scheduler.json'),  # Custom schedule file path
    'max_pages_per_model': int(os.getenv('MAX_PAGES_PER_MODEL', '0')) or None,  # 0 = unlimited
    'use_proxy': os.getenv('USE_PROXY_FOR_RETAIL', 'true').lower() == 'true',
    'verify_ssl': os.getenv('VERIFY_SSL', 'false').lower() == 'true'  # Disabled by default to avoid CloudFlare issues
}

# Universal ML Model Configuration  
# Note: Universal ML Model Configuration removed - now using model-specific training

# AutoTrader URL Generation Configuration
AUTOTRADER_BASE_URL = "https://www.autotrader.co.uk/car-search"
DEFAULT_SEARCH_PARAMS = {
    "advertising-location": "at_cars",
    "exclude-writeoff-categories": "on",
    "sort": "price-asc",
    "postcode": "MF15 4FN"
}

# Vehicle Search Parameters - Define what you want to search for
# The system will automatically generate URLs from these parameters
VEHICLE_SEARCH_CRITERIA = {
    "maximum_mileage": 100000,
    "year_from": 2010,
    "year_to": 2020
}

# Mileage range splitting configuration to bypass AutoTrader's 2000 listing limit
# Each search will be split into these ranges to ensure complete coverage
MILEAGE_RANGES_FOR_SPLITTING = [
    (0, 20000),
    (20001, 40000),
    (40001, 60000),
    (60001, 80000),
    (80001, 100000)
]

# Enable mileage splitting by default to bypass listing limits
ENABLE_MILEAGE_SPLITTING = True

# Parallel API Scraping Configuration
PARALLEL_API_SCRAPING = {
    'enabled': True,  # Enable parallel processing by default
    'max_workers': 5,  # One worker per mileage range
    'target_rate_per_minute': 300,  # Combined rate limit for all workers
    'fallback_to_sequential': True,  # Fallback if parallel fails
    'use_proxy_per_worker': True  # Each worker gets unique proxy
}

# Target vehicle makes/models - Organized by make for better maintainability
TARGET_VEHICLES_BY_MAKE = {
    "Abarth": ["595"],
    "Alfa Romeo": ["Giulietta"],
    "Audi": ["A1", "A3", "A4", "A4 Avant", "A5", "A6 Saloon", "Q2", "Q3", "Q5", "Q7", "S3", "TT"],
    "BMW": ["1 Series", "2 Series", "3 Series", "4 Series", "4 Series Gran Coupe", "5 Series", "7 Series", "M4", "X1", "X3"],
    "Citroen": ["C3", "C1", "C4 Cactus", "DS3"],
    "Dacia": ["Duster", "Sandero Stepway"],
    "DS Automobiles": ["DS 3"],
    "Fiat": ["500", "500X"],
    "Ford": ["B-Max", "C-Max", "EcoSport", "Fiesta", "Focus", "Ka", "Kuga", "Mondeo", "Puma", "Ranger"],
    "Honda": ["Jazz", "Civic", "CR-V", "HR-V"],
    "Hyundai": ["i10", "TUCSON", "KONA", "i20", "i30"],
    "Jaguar": ["F-PACE", "XE", "XF", "E-PACE", "F-Type"],
    "Kia": ["Sportage", "Picanto", "Ceed", "Niro", "Rio", "Venga"],
    "Lexus": ["CT"],
    "Mazda": ["CX-5", "Mazda2", "Mazda3", "CX-3", "MX-5"],
    "Mercedes-Benz": ["A Class", "C Class", "E Class", "GLC", "GLA", "CLA", "B Class", "GLE"],
    "Mini": ["Hatch", "Countryman", "Clubman", "Convertible"],
    "Mitsubishi": ["Outlander", "L200"],
    "Nissan": ["Qashqai", "Juke", "Micra", "Note", "X-Trail"],
    "Peugeot": ["208", "2008", "3008", "108", "308"],
    "Porsche": ["911", "Macan", "Cayenne"],
    "Renault": ["Captur", "Clio", "Kadjar", "Megane"],
    "SEAT": ["Leon", "Ibiza", "Arona", "Ateca"],
    "Skoda": ["Fabia", "Octavia", "Karoq"],
    "Suzuki": ["Swift", "Vitara"],
    "Tesla": ["Model 3"],
    "Toyota": ["AYGO", "C-HR", "Yaris", "RAV4", "Auris", "Corolla", "Prius"],
    "Vauxhall": ["ADAM", "Astra", "Corsa", "Crossland X", "Grandland X", "Insignia", "Meriva", "Mokka", "Mokka X", "Zafira Tourer"],
    "Volkswagen": ["Golf", "Polo", "Tiguan", "T-Roc", "Passat", "up!", "Scirocco"],
    "Volvo": ["XC60", "V40", "XC40", "XC90", "V60"]
}

# Generate TARGET_VEHICLES list from organized structure for backwards compatibility
TARGET_VEHICLES = []
for make, models in TARGET_VEHICLES_BY_MAKE.items():
    for model in models:
        TARGET_VEHICLES.append({"make": make, "model": model})

# Daily Training Model Groups - Balanced for similar training times per day
# Each day gets a mix of high-volume (popular) and low-volume (niche) models
# to ensure roughly equal scraping/training duration across all 14 days
DAILY_TRAINING_GROUPS = {
    1: [  # Day 1: Mix of popular + niche models
        ("BMW", "3 Series"),           # HIGH popularity
        ("Honda", "Jazz"),             # MEDIUM
        ("BMW", "5 Series"),           # MEDIUM
        ("Mini", "Hatch"),             # MEDIUM
        ("Skoda", "Octavia"),          # MEDIUM
        ("Porsche", "911"),            # LOW
        ("Tesla", "Model 3"),          # LOW
        ("Jaguar", "F-PACE"),          # LOW
        ("DS Automobiles", "DS 3"),    # LOW
        ("Abarth", "595")              # LOW
    ],
    2: [  # Day 2: Balance high-volume models across days
        ("Audi", "A3"),                # HIGH popularity
        ("Ford", "Fiesta"),            # HIGH popularity
        ("Honda", "Civic"),            # MEDIUM
        ("Toyota", "Auris"),           # MEDIUM
        ("SEAT", "Leon"),              # MEDIUM
        ("Porsche", "Macan"),          # LOW
        ("Lexus", "CT"),               # LOW
        ("Jaguar", "XE"),              # LOW
        ("Dacia", "Duster"),           # LOW
        ("Alfa Romeo", "Giulietta")    # LOW
    ],
    3: [  # Day 3
        ("Volkswagen", "Golf"),        # HIGH popularity
        ("Vauxhall", "Corsa"),         # HIGH popularity
        ("BMW", "1 Series"),           # MEDIUM
        ("Audi", "Q3"),                # MEDIUM
        ("Mercedes-Benz", "C Class"),  # MEDIUM
        ("Porsche", "Cayenne"),        # LOW
        ("Jaguar", "XF"),              # LOW
        ("Jaguar", "E-PACE"),          # LOW
        ("Dacia", "Sandero Stepway"), # LOW
        ("Citroen", "C1")              # LOW
    ],
    4: [  # Day 4
        ("Toyota", "Yaris"),           # HIGH popularity
        ("Nissan", "Qashqai"),         # HIGH popularity
        ("Audi", "A4"),                # MEDIUM
        ("BMW", "2 Series"),           # MEDIUM
        ("Mini", "Countryman"),        # MEDIUM
        ("Jaguar", "F-Type"),          # LOW
        ("Kia", "Venga"),              # LOW
        ("Citroen", "C3"),             # LOW
        ("Citroen", "C4 Cactus"),      # LOW
        ("Citroen", "DS3")             # LOW
    ],
    5: [  # Day 5
        ("Mercedes-Benz", "A Class"),  # HIGH popularity
        ("Ford", "Focus"),             # MEDIUM
        ("Volkswagen", "Polo"),        # MEDIUM
        ("Toyota", "C-HR"),            # MEDIUM
        ("Mazda", "CX-5"),             # MEDIUM
        ("Kia", "Sportage"),           # MEDIUM
        ("Fiat", "500"),               # LOW
        ("Fiat", "500X"),              # LOW
        ("Ford", "Ka"),                # LOW
        ("Volvo", "XC40")              # LOW (was missing)
    ],
    6: [  # Day 6
        ("Vauxhall", "Astra"),         # HIGH popularity
        ("BMW", "X1"),                 # MEDIUM
        ("Audi", "A1"),                # MEDIUM
        ("Mercedes-Benz", "GLC"),      # MEDIUM
        ("Skoda", "Fabia"),            # MEDIUM
        ("Peugeot", "208"),            # MEDIUM
        ("Ford", "B-Max"),             # LOW
        ("Ford", "C-Max"),             # LOW
        ("Ford", "EcoSport"),          # LOW
        ("Ford", "Kuga")               # LOW
    ],
    7: [  # Day 7
        ("SEAT", "Ibiza"),             # MEDIUM
        ("BMW", "4 Series"),           # MEDIUM
        ("Audi", "A5"),                # MEDIUM
        ("Mercedes-Benz", "E Class"),  # MEDIUM
        ("Toyota", "RAV4"),            # MEDIUM
        ("Nissan", "Juke"),            # MEDIUM
        ("Ford", "Mondeo"),            # LOW
        ("Ford", "Puma"),              # LOW
        ("Ford", "Ranger"),            # LOW
        ("Honda", "CR-V")              # LOW
    ],
    8: [  # Day 8
        ("BMW", "X3"),                 # MEDIUM
        ("Audi", "Q2"),                # MEDIUM
        ("Mercedes-Benz", "GLA"),      # MEDIUM
        ("Vauxhall", "ADAM"),          # MEDIUM
        ("Toyota", "Corolla"),         # MEDIUM
        ("Nissan", "Micra"),           # MEDIUM
        ("Honda", "HR-V"),             # LOW
        ("Hyundai", "i10"),            # LOW
        ("Hyundai", "TUCSON"),         # LOW
        ("Hyundai", "KONA")            # LOW
    ],
    9: [  # Day 9
        ("BMW", "7 Series"),           # MEDIUM
        ("Audi", "A4 Avant"),          # MEDIUM
        ("Mercedes-Benz", "CLA"),      # MEDIUM
        ("Vauxhall", "Crossland X"),   # MEDIUM
        ("Toyota", "Prius"),           # MEDIUM
        ("Nissan", "Note"),            # MEDIUM
        ("Hyundai", "i20"),            # LOW
        ("Hyundai", "i30"),            # LOW
        ("Kia", "Picanto"),            # LOW
        ("Kia", "Ceed")                # LOW
    ],
    10: [ # Day 10
        ("BMW", "4 Series Gran Coupe"), # MEDIUM
        ("Audi", "A6 Saloon"),         # MEDIUM
        ("Mercedes-Benz", "B Class"),  # MEDIUM
        ("Vauxhall", "Grandland X"),   # MEDIUM
        ("Toyota", "AYGO"),            # MEDIUM
        ("Nissan", "X-Trail"),         # MEDIUM
        ("Kia", "Niro"),               # LOW
        ("Kia", "Rio"),                # LOW
        ("Mazda", "Mazda2"),           # LOW
        ("Mazda", "Mazda3")            # LOW
    ],
    11: [ # Day 11
        ("BMW", "M4"),                 # MEDIUM (niche but BMW)
        ("Audi", "Q5"),                # MEDIUM
        ("Mercedes-Benz", "GLE"),      # MEDIUM
        ("Vauxhall", "Insignia"),      # MEDIUM
        ("SEAT", "Arona"),             # MEDIUM
        ("Peugeot", "2008"),           # MEDIUM
        ("Mazda", "CX-3"),             # LOW
        ("Mazda", "MX-5"),             # LOW
        ("Mini", "Clubman"),           # LOW
        ("Mini", "Convertible")        # LOW
    ],
    12: [ # Day 12
        ("Audi", "Q7"),                # MEDIUM
        ("Audi", "S3"),                # MEDIUM
        ("Vauxhall", "Meriva"),        # MEDIUM
        ("SEAT", "Ateca"),             # MEDIUM
        ("Peugeot", "3008"),           # MEDIUM
        ("Mitsubishi", "Outlander"),   # LOW
        ("Mitsubishi", "L200"),        # LOW
        ("Skoda", "Karoq"),            # LOW
        ("Suzuki", "Swift"),           # LOW
        ("Volvo", "V60")               # LOW (was missing)
    ],
    13: [ # Day 13
        ("Audi", "TT"),                # MEDIUM
        ("Vauxhall", "Mokka"),         # MEDIUM
        ("Vauxhall", "Mokka X"),       # MEDIUM
        ("Peugeot", "108"),            # MEDIUM
        ("Peugeot", "308"),            # MEDIUM
        ("Suzuki", "Vitara"),          # LOW
        ("Renault", "Captur"),         # LOW
        ("Renault", "Clio"),           # LOW
        ("Renault", "Kadjar"),         # LOW
        ("Volvo", "XC90")              # LOW (was missing)
    ],
    14: [ # Day 14 - Final day (9 remaining models)
        ("Vauxhall", "Zafira Tourer"), # MEDIUM
        ("Renault", "Megane"),         # MEDIUM
        ("Volkswagen", "Tiguan"),      # MEDIUM
        ("Volkswagen", "T-Roc"),       # MEDIUM
        ("Volkswagen", "Passat"),      # LOW
        ("Volkswagen", "up!"),         # LOW
        ("Volkswagen", "Scirocco"),    # LOW
        ("Volvo", "XC60"),             # LOW
        ("Volvo", "V40")               # LOW
    ]
}

def build_autotrader_url(make, model, search_criteria=None):
    """
    Build AutoTrader search URL from vehicle make/model and search criteria.
    
    Args:
        make (str): Vehicle make (e.g., "Toyota")
        model (str): Vehicle model (e.g., "Yaris")
        search_criteria (dict, optional): Override default search criteria
    
    Returns:
        str: Complete AutoTrader search URL
    """
    from urllib.parse import urlencode
    
    # Start with default parameters
    params = DEFAULT_SEARCH_PARAMS.copy()
    
    # Use provided criteria or default
    criteria = search_criteria if search_criteria else VEHICLE_SEARCH_CRITERIA
    
    # Add vehicle-specific parameters
    params["make"] = make
    params["model"] = model
    params["maximum-mileage"] = str(criteria["maximum_mileage"])
    params["year-from"] = str(criteria["year_from"])
    params["year-to"] = str(criteria["year_to"])
    
    return f"{AUTOTRADER_BASE_URL}?{urlencode(params)}"

def get_vehicle_configs():
    """
    Generate vehicle configurations dynamically from TARGET_VEHICLES and search criteria.
    
    Returns:
        list: List of vehicle configurations with 'base_url' field
    """
    configs = []
    for vehicle in TARGET_VEHICLES:
        config = {
            "name": f"{vehicle['make']} {vehicle['model']}",
            "make": vehicle["make"],
            "model": vehicle["model"],
            "base_url": build_autotrader_url(vehicle["make"], vehicle["model"])
        }
        # Add search criteria for reference
        config.update(VEHICLE_SEARCH_CRITERIA)
        configs.append(config)
    
    return configs

def get_weekly_retail_config():
    """
    Get weekly retail scraping configuration.
    
    Returns:
        dict: Weekly retail scraping configuration
    """
    return WEEKLY_RETAIL_SCRAPING.copy()

def get_universal_ml_config():
    """
    Get universal ML model configuration.
    
    Returns:
        dict: Universal ML model configuration
    """
    return UNIVERSAL_ML_CONFIG.copy()

def get_schedule_interval_days():
    """
    Get the retail scraping schedule interval in days.
    
    Returns:
        int: Days between retail scraping runs
    """
    return WEEKLY_RETAIL_SCRAPING['interval_days']

def should_use_proxy_for_retail():
    """
    Check if proxy should be used for retail price scraping.
    
    Returns:
        bool: True if proxy should be used
    """
    return WEEKLY_RETAIL_SCRAPING['use_proxy']

# Note: get_xgboost_params() and get_training_params() removed - universal model config no longer exists

def get_min_training_samples():
    """
    Get minimum number of samples required to train universal model.

    Returns:
        int: Minimum training samples
    """
    return UNIVERSAL_ML_CONFIG['min_training_samples']


# ──────────────────────────────────────────────────────────────────
# Schedule Management Functions (simplified from schedule_manager.py)
# ──────────────────────────────────────────────────────────────────

def _get_schedule_file_path():
    """Get the path to the schedule file."""
    return Path(__file__).parent / "scraping_scheduler.json"

def _load_schedule_data():
    """Load schedule data from file."""
    schedule_file = _get_schedule_file_path()

    default_data = {
        "last_retail_scraping": None,
        "last_scraping_success": True,
        "scraping_history": []
    }

    if not schedule_file.exists():
        return default_data

    try:
        with open(schedule_file, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return default_data

def _save_schedule_data(data):
    """Save schedule data to file."""
    schedule_file = _get_schedule_file_path()
    try:
        with open(schedule_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False

def is_retail_scraping_due():
    """Check if weekly retail scraping is due (7+ days since last run)."""
    data = _load_schedule_data()

    if not data.get("last_retail_scraping"):
        return True  # No previous scraping found

    try:
        from datetime import datetime, timedelta
        last_scraping = datetime.fromisoformat(data["last_retail_scraping"])
        days_since = (datetime.now() - last_scraping).days
        return days_since >= WEEKLY_RETAIL_SCRAPING['interval_days']
    except (ValueError, KeyError):
        return True  # Invalid data, assume due

def mark_retail_scraping_started():
    """Mark that retail scraping has started."""
    data = _load_schedule_data()
    data["scraping_started"] = datetime.now().isoformat()
    return _save_schedule_data(data)

def mark_retail_scraping_complete(success=True, total_vehicles=0, notes=None):
    """Mark retail scraping as complete."""
    data = _load_schedule_data()

    # Update main fields
    data["last_retail_scraping"] = datetime.now().isoformat()
    data["last_scraping_success"] = success

    # Add to history
    if "scraping_history" not in data:
        data["scraping_history"] = []

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "total_vehicles": total_vehicles,
        "notes": notes
    }
    data["scraping_history"].append(history_entry)

    # Keep only last 10 entries
    data["scraping_history"] = data["scraping_history"][-10:]

    return _save_schedule_data(data)

def get_schedule_status():
    """Get comprehensive schedule status information."""
    data = _load_schedule_data()

    status = {
        "is_due": is_retail_scraping_due(),
        "last_scraping": data.get("last_retail_scraping"),
        "last_success": data.get("last_scraping_success", True),
        "days_since_last": None,
        "next_scheduled": None
    }

    if data.get("last_retail_scraping"):
        try:
            from datetime import datetime, timedelta
            last_scraping = datetime.fromisoformat(data["last_retail_scraping"])
            days_since = (datetime.now() - last_scraping).days
            status["days_since_last"] = days_since

            next_scheduled = last_scraping + timedelta(days=WEEKLY_RETAIL_SCRAPING['interval_days'])
            status["next_scheduled"] = next_scheduled.strftime("%Y-%m-%d")
        except (ValueError, KeyError):
            pass

    return status