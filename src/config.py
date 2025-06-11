"""
Configuration file for AutoTrader scraper.
Contains vehicle search configurations and database settings.
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Supabase Configuration (Primary Storage)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
CAR_DEALS_TABLE = os.getenv('CAR_DEALS_TABLE', 'car_deals')

# Application Configuration
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Disabled Features (for future re-enablement if needed)
# PLATERECOGNIZER_API_KEY = os.getenv("PLATERECOGNIZER_API_KEY")  # ANPR disabled
# GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")  # Sheets disabled
# SHEET_NAME = os.getenv("SHEET_NAME", "All Car Deals")  # Sheets disabled

# AutoTrader URL Generation Configuration
AUTOTRADER_BASE_URL = "https://www.autotrader.co.uk/car-search"
DEFAULT_SEARCH_PARAMS = {
    "advertising-location": "at_cars",
    "exclude-writeoff-categories": "on",
    "fuel-type": "Petrol",
    "transmission": "Manual",
    "sort": "price-asc",
    "postcode": "HP13 7LW"
}

# Vehicle Search Parameters - Define what you want to search for
# The system will automatically generate URLs from these parameters
VEHICLE_SEARCH_CRITERIA = {
    "maximum_mileage": 100000,
    "year_from": 2012,
    "year_to": 2018
}

# Target vehicle makes/models - Organized by make for better maintainability
TARGET_VEHICLES_BY_MAKE = {
    "Abarth": ["595", "595C", "124 Spider", "500"],
    "Alfa Romeo": ["Giulietta", "Stelvio", "Giulia", "MiTo"],
    "Audi": ["A1", "A3", "Q3", "A5", "Q5", "A4", "Q7", "A6 Saloon", "A6 Avant", "Q2", "TT", "A7", "A8", "e-tron", "Q8", "R8", "RS3", "RS5", "S3", "SQ5"],
    "BMW": ["1 Series", "3 Series", "5 Series", "4 Series", "X5", "2 Series", "X3", "X1", "X6", "X4", "Z4", "i3", "M4", "M2", "M3"],
    "Citroen": ["C3", "C1", "C4 Cactus", "C3 Aircross", "C4 Picasso", "DS3", "Berlingo", "C5 Aircross", "Grand C4 Picasso", "C4", "C4 SpaceTourer"],
    "Dacia": ["Duster", "Sandero Stepway", "Sandero", "Logan MCV"],
    "DS Automobiles": ["DS 3", "DS 7 CROSSBACK", "DS 3 CROSSBACK"],
    "Fiat": ["500", "500X", "500L", "Panda", "Tipo", "124 Spider", "500C", "Punto"],
    "Ford": ["Fiesta", "Focus", "Kuga", "EcoSport", "Ranger", "C-Max", "S-Max", "Mondeo", "Puma", "Galaxy", "Ka", "Mustang", "Transit", "Tourneo Custom", "B-Max", "Edge", "Grand C-Max", "Transit Custom"],
    "Honda": ["Jazz", "Civic", "CR-V", "HR-V", "Fit", "Accord"],
    "Hyundai": ["i10", "TUCSON", "KONA", "i20", "i30", "IONIQ", "Santa Fe", "ix20", "ix35"],
    "Jaguar": ["F-PACE", "XE", "XF", "E-PACE", "F-Type", "I-PACE", "XJ", "XK"],
    "Kia": ["Sportage", "Picanto", "Ceed", "Niro", "Rio", "Stonic", "Sorento", "Venga", "Optima", "Carens"],
    "Lexus": ["NX", "CT", "RX", "IS", "UX"],
    "Mazda": ["CX-5", "Mazda2", "Mazda3", "CX-3", "Mazda6", "MX-5", "CX-30"],
    "Mercedes-Benz": ["A Class", "C Class", "E Class", "GLC", "GLA", "CLA", "B Class", "GLE", "G Class", "S Class", "V Class", "SLK", "SL"],
    "Mini": ["Hatch", "Countryman", "Clubman", "Convertible", "Paceman", "Coupe"],
    "Mitsubishi": ["Outlander", "L200", "ASX", "Eclipse Cross", "Mirage"],
    "Nissan": ["Qashqai", "Juke", "Micra", "X-Trail", "Leaf", "Navara", "Note", "GT-R"],
    "Peugeot": ["208", "2008", "3008", "108", "308", "5008", "107", "508", "308 SW", "508 SW"],
    "Porsche": ["911", "Macan", "Cayenne", "Panamera", "718 Cayman", "Taycan"],
    "Renault": ["Captur", "Clio", "Kadjar", "Megane", "Zoe", "Scenic", "Twingo"],
    "SEAT": ["Leon", "Ibiza", "Arona", "Ateca", "Alhambra", "Tarraco"],
    "Skoda": ["Fabia", "Octavia", "Karoq", "Kodiaq", "Superb", "Yeti", "Citigo"],
    "Suzuki": ["Swift", "Vitara", "Ignis", "Celerio", "Alto"],
    "Tesla": ["Model 3", "Model S", "Model X"],
    "Toyota": ["AYGO", "C-HR", "Yaris", "RAV4", "Auris", "Corolla", "Prius", "Hilux", "Estima"],
    "Vauxhall": ["Corsa", "Astra", "Grandland X", "Mokka X", "Crossland X", "Insignia", "Zafira Tourer", "Viva", "Meriva"],
    "Volkswagen": ["Golf", "Polo", "Tiguan", "T-Roc", "Passat", "up!", "Scirocco", "Sharan", "Touareg", "Touran"],
    "Volvo": ["XC60", "V40", "XC40", "XC90", "V60", "S90"]
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

