"""
Configuration file for AutoTrader scraper.
Contains vehicle search configurations and database settings.
"""
from dotenv import load_dotenv
import os
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

# Weekly Retail Scraping Configuration
WEEKLY_RETAIL_SCRAPING = {
    'interval_days': int(os.getenv('RETAIL_SCRAPING_INTERVAL_DAYS', '7')),  # Days between retail scraping runs
    'schedule_file': os.getenv('RETAIL_SCHEDULE_FILE', 'config/scraping_scheduler.json'),  # Custom schedule file path
    'max_pages_per_model': int(os.getenv('MAX_PAGES_PER_MODEL', '0')) or None,  # 0 = unlimited
    'use_proxy': os.getenv('USE_PROXY_FOR_RETAIL', 'true').lower() == 'true',
    'verify_ssl': os.getenv('VERIFY_SSL', 'false').lower() == 'true'
}

# Universal ML Model Configuration  
UNIVERSAL_ML_CONFIG = {
    'model_path': os.getenv('UNIVERSAL_MODEL_PATH', None),  # Custom model path (None = auto)
    'min_training_samples': int(os.getenv('MIN_TRAINING_SAMPLES', '50')),  # Minimum samples to train universal model
    'feature_columns': [  # Fixed feature set for universal model
        'asking_price', 'mileage', 'age', 'make_encoded', 'model_encoded',
        'fuel_type_numeric', 'transmission_numeric', 'engine_size', 'spec_numeric'
    ],
    'xgboost_params': {  # XGBoost parameters for universal model
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': int(os.getenv('XGBOOST_MAX_DEPTH', '6')),
        'eta': float(os.getenv('XGBOOST_ETA', '0.1')),
        'subsample': float(os.getenv('XGBOOST_SUBSAMPLE', '0.8')),
        'colsample_bytree': float(os.getenv('XGBOOST_COLSAMPLE_BYTREE', '0.8')),
        'min_child_weight': int(os.getenv('XGBOOST_MIN_CHILD_WEIGHT', '3')),
        'seed': 42
    },
    'training_params': {
        'num_boost_round': int(os.getenv('XGBOOST_ROUNDS', '150')),
        'early_stopping_rounds': int(os.getenv('XGBOOST_EARLY_STOPPING', '15')),
        'test_size': float(os.getenv('ML_TEST_SIZE', '0.2'))
    }
}

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

def get_xgboost_params():
    """
    Get XGBoost parameters for universal model training.
    
    Returns:
        dict: XGBoost parameters
    """
    return UNIVERSAL_ML_CONFIG['xgboost_params'].copy()

def get_training_params():
    """
    Get training parameters for universal model.
    
    Returns:
        dict: Training parameters
    """
    return UNIVERSAL_ML_CONFIG['training_params'].copy()

def get_min_training_samples():
    """
    Get minimum number of samples required to train universal model.
    
    Returns:
        int: Minimum training samples
    """
    return UNIVERSAL_ML_CONFIG['min_training_samples']