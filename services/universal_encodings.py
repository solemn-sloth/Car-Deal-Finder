#!/usr/bin/env python3
"""
Universal Encodings for ML Model
Fixed ordinal encodings for vehicle makes and models to ensure consistent
predictions across all car types in the universal ML model.
"""

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Fixed ordinal encoding for vehicle makes
# Alphabetical order for consistency and easy maintenance
MAKE_ENCODING = {
    "Abarth": 1,
    "Alfa Romeo": 2, 
    "Audi": 3,
    "BMW": 4,
    "Citroen": 5,
    "Dacia": 6,
    "DS Automobiles": 7,
    "Fiat": 8,
    "Ford": 9,
    "Honda": 10,
    "Hyundai": 11,
    "Jaguar": 12,
    "Kia": 13,
    "Lexus": 14,
    "Mazda": 15,
    "Mercedes-Benz": 16,
    "Mini": 17,
    "Mitsubishi": 18,
    "Nissan": 19,
    "Peugeot": 20,
    "Porsche": 21,
    "Renault": 22,
    "SEAT": 23,
    "Skoda": 24,
    "Suzuki": 25,
    "Tesla": 26,
    "Toyota": 27,
    "Vauxhall": 28,
    "Volkswagen": 29,
    "Volvo": 30
}

# Fixed ordinal encoding for vehicle models
# Grouped by make, then alphabetical within make for easy maintenance
MODEL_ENCODING = {
    # Abarth models
    "595": 1,
    
    # Alfa Romeo models  
    "Giulietta": 2,
    
    # Audi models
    "A1": 3,
    "A3": 4,
    "A4": 5,
    "A4 Avant": 6,
    "A5": 7,
    "A6 Saloon": 8,
    "Q2": 9,
    "Q3": 10,
    "Q5": 11,
    "Q7": 12,
    "S3": 13,
    "TT": 14,
    
    # BMW models
    "1 Series": 15,
    "2 Series": 16,
    "3 Series": 17,
    "4 Series": 18,
    "4 Series Gran Coupe": 19,
    "5 Series": 20,
    "7 Series": 21,
    "M4": 22,
    "X1": 23,
    "X3": 24,
    
    # Citroen models
    "C1": 25,
    "C3": 26,
    "C4 Cactus": 27,
    "DS3": 28,
    
    # Dacia models
    "Duster": 29,
    "Sandero Stepway": 30,
    
    # DS Automobiles models
    "DS 3": 31,
    
    # Fiat models
    "500": 32,
    "500X": 33,
    
    # Ford models
    "B-Max": 34,
    "C-Max": 35,
    "EcoSport": 36,
    "Fiesta": 37,
    "Focus": 38,
    "Ka": 39,
    "Kuga": 40,
    "Mondeo": 41,
    "Puma": 42,
    "Ranger": 43,
    
    # Honda models
    "Civic": 44,
    "CR-V": 45,
    "HR-V": 46,
    "Jazz": 47,
    
    # Hyundai models
    "i10": 48,
    "i20": 49,
    "i30": 50,
    "KONA": 51,
    "TUCSON": 52,
    
    # Jaguar models
    "E-PACE": 53,
    "F-PACE": 54,
    "F-Type": 55,
    "XE": 56,
    "XF": 57,
    
    # Kia models
    "Ceed": 58,
    "Niro": 59,
    "Picanto": 60,
    "Rio": 61,
    "Sportage": 62,
    "Venga": 63,
    
    # Lexus models
    "CT": 64,
    
    # Mazda models
    "CX-3": 65,
    "CX-5": 66,
    "Mazda2": 67,
    "Mazda3": 68,
    "MX-5": 69,
    
    # Mercedes-Benz models
    "A Class": 70,
    "B Class": 71,
    "C Class": 72,
    "CLA": 73,
    "E Class": 74,
    "GLA": 75,
    "GLC": 76,
    "GLE": 77,
    
    # Mini models
    "Clubman": 78,
    "Convertible": 79,
    "Countryman": 80,
    "Hatch": 81,
    
    # Mitsubishi models
    "L200": 82,
    "Outlander": 83,
    
    # Nissan models
    "Juke": 84,
    "Micra": 85,
    "Note": 86,
    "Qashqai": 87,
    "X-Trail": 88,
    
    # Peugeot models
    "108": 89,
    "208": 90,
    "2008": 91,
    "308": 92,
    "3008": 93,
    
    # Porsche models
    "911": 94,
    "Cayenne": 95,
    "Macan": 96,
    
    # Renault models
    "Captur": 97,
    "Clio": 98,
    "Kadjar": 99,
    "Megane": 100,
    
    # SEAT models
    "Arona": 101,
    "Ateca": 102,
    "Ibiza": 103,
    "Leon": 104,
    
    # Skoda models
    "Fabia": 105,
    "Karoq": 106,
    "Octavia": 107,
    
    # Suzuki models
    "Swift": 108,
    "Vitara": 109,
    
    # Tesla models
    "Model 3": 110,
    
    # Toyota models
    "Auris": 111,
    "AYGO": 112,
    "C-HR": 113,
    "Corolla": 114,
    "Prius": 115,
    "RAV4": 116,
    "Yaris": 117,
    
    # Vauxhall models
    "ADAM": 118,
    "Astra": 119,
    "Corsa": 120,
    "Crossland X": 121,
    "Grandland X": 122,
    "Insignia": 123,
    "Meriva": 124,
    "Mokka": 125,
    "Mokka X": 126,
    "Zafira Tourer": 127,
    
    # Volkswagen models
    "Golf": 128,
    "Passat": 129,
    "Polo": 130,
    "Scirocco": 131,
    "T-Roc": 132,
    "Tiguan": 133,
    "up!": 134,
    
    # Volvo models
    "V40": 135,
    "V60": 136,
    "XC40": 137,
    "XC60": 138,
    "XC90": 139
}

# Reverse mappings for decoding
MAKE_DECODING = {v: k for k, v in MAKE_ENCODING.items()}
MODEL_DECODING = {v: k for k, v in MODEL_ENCODING.items()}


class UniversalEncoder:
    """
    Handles encoding and decoding of vehicle makes and models for the universal ML model.
    
    Features:
    - Fixed ordinal encodings for production stability
    - Handles unknown makes/models gracefully
    - Provides validation and logging
    """
    
    def __init__(self):
        self.make_encoder = MAKE_ENCODING.copy()
        self.model_encoder = MODEL_ENCODING.copy()
        self.make_decoder = MAKE_DECODING.copy()
        self.model_decoder = MODEL_DECODING.copy()
        
        # Track unknown values for monitoring
        self.unknown_makes = set()
        self.unknown_models = set()
    
    def encode_make(self, make: str) -> int:
        """
        Encode vehicle make to numeric value.
        
        Args:
            make: Vehicle make (e.g., "BMW", "Audi")
            
        Returns:
            int: Encoded make (1-30), or 0 for unknown makes
        """
        if not make or not isinstance(make, str):
            return 0
            
        # Try exact match first
        if make in self.make_encoder:
            return self.make_encoder[make]
            
        # Try case-insensitive match
        make_lower = make.lower()
        for encoded_make, code in self.make_encoder.items():
            if encoded_make.lower() == make_lower:
                return code
                
        # Unknown make
        if make not in self.unknown_makes:
            logger.warning(f"Unknown make encountered: '{make}' - encoding as 0")
            self.unknown_makes.add(make)
        return 0
    
    def encode_model(self, model: str) -> int:
        """
        Encode vehicle model to numeric value.
        
        Args:
            model: Vehicle model (e.g., "3 Series", "A3")
            
        Returns:
            int: Encoded model (1-139), or 0 for unknown models
        """
        if not model or not isinstance(model, str):
            return 0
            
        # Try exact match first
        if model in self.model_encoder:
            return self.model_encoder[model]
            
        # Try case-insensitive match
        model_lower = model.lower()
        for encoded_model, code in self.model_encoder.items():
            if encoded_model.lower() == model_lower:
                return code
                
        # Unknown model
        if model not in self.unknown_models:
            logger.warning(f"Unknown model encountered: '{model}' - encoding as 0")
            self.unknown_models.add(model)
        return 0
    
    def decode_make(self, code: int) -> str:
        """Decode numeric make back to string."""
        return self.make_decoder.get(code, "Unknown")
    
    def decode_model(self, code: int) -> str:
        """Decode numeric model back to string."""
        return self.model_decoder.get(code, "Unknown")
    
    def encode_listing(self, listing: Dict) -> Dict:
        """
        Add encoded make/model fields to a listing dictionary.
        
        Args:
            listing: Car listing dictionary
            
        Returns:
            Dict: Updated listing with 'make_encoded' and 'model_encoded' fields
        """
        updated_listing = listing.copy()
        
        # Add encoded fields
        updated_listing['make_encoded'] = self.encode_make(listing.get('make', ''))
        updated_listing['model_encoded'] = self.encode_model(listing.get('model', ''))
        
        return updated_listing
    
    def encode_listings(self, listings: List[Dict]) -> List[Dict]:
        """
        Add encoded make/model fields to a list of listings.
        
        Args:
            listings: List of car listing dictionaries
            
        Returns:
            List[Dict]: Updated listings with encoded fields
        """
        return [self.encode_listing(listing) for listing in listings]
    
    def get_encoding_stats(self) -> Dict:
        """Get statistics about encoding operations."""
        return {
            'total_makes': len(self.make_encoder),
            'total_models': len(self.model_encoder),
            'unknown_makes_encountered': len(self.unknown_makes),
            'unknown_models_encountered': len(self.unknown_models),
            'unknown_makes': list(self.unknown_makes),
            'unknown_models': list(self.unknown_models)
        }
    
    def validate_encodings(self) -> Tuple[bool, List[str]]:
        """
        Validate the encoding mappings for consistency.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for duplicate codes in makes
        make_codes = list(self.make_encoder.values())
        if len(make_codes) != len(set(make_codes)):
            errors.append("Duplicate codes found in MAKE_ENCODING")
            
        # Check for duplicate codes in models  
        model_codes = list(self.model_encoder.values())
        if len(model_codes) != len(set(model_codes)):
            errors.append("Duplicate codes found in MODEL_ENCODING")
            
        # Check for gaps in encoding sequences
        make_codes_sorted = sorted(make_codes)
        expected_make_codes = list(range(1, len(make_codes) + 1))
        if make_codes_sorted != expected_make_codes:
            errors.append("Gaps or non-sequential codes in MAKE_ENCODING")
            
        model_codes_sorted = sorted(model_codes)
        expected_model_codes = list(range(1, len(model_codes) + 1))
        if model_codes_sorted != expected_model_codes:
            errors.append("Gaps or non-sequential codes in MODEL_ENCODING")
        
        return len(errors) == 0, errors


# Global encoder instance for easy access
universal_encoder = UniversalEncoder()

# Convenience functions for direct access
def encode_make(make: str) -> int:
    """Encode vehicle make to numeric value."""
    return universal_encoder.encode_make(make)

def encode_model(model: str) -> int:
    """Encode vehicle model to numeric value."""
    return universal_encoder.encode_model(model)

def encode_listing(listing: Dict) -> Dict:
    """Add encoded make/model fields to a listing dictionary."""
    return universal_encoder.encode_listing(listing)

def encode_listings(listings: List[Dict]) -> List[Dict]:
    """Add encoded make/model fields to a list of listings."""
    return universal_encoder.encode_listings(listings)

def get_encoding_stats() -> Dict:
    """Get statistics about encoding operations."""
    return universal_encoder.get_encoding_stats()

def validate_encodings() -> Tuple[bool, List[str]]:
    """Validate the encoding mappings for consistency."""
    return universal_encoder.validate_encodings()


if __name__ == "__main__":
    # Test the encodings
    print("Testing Universal Encodings...")
    
    # Validate encodings
    is_valid, errors = validate_encodings()
    if is_valid:
        print("✅ Encoding validation passed")
    else:
        print("❌ Encoding validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    # Test some encodings
    test_cases = [
        ("BMW", "3 Series"),
        ("Audi", "A3"),
        ("Ford", "Fiesta"),
        ("Toyota", "Yaris"),
        ("Unknown Make", "Unknown Model")
    ]
    
    for make, model in test_cases:
        make_code = encode_make(make)
        model_code = encode_model(model)
        print(f"{make} {model} → Make: {make_code}, Model: {model_code}")
    
    print(f"\nEncoding Statistics:")
    stats = get_encoding_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")