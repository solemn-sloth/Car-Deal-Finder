# Car Deal Finder

Vehicle analysis system that scrapes used car listings using GraphQL API, then employs machine learning to identify underpriced vehicles and predict profit margins with statistical confidence.

## 🔍 How It Works

```
                                   ┌─────────────────┐
                                   │   GraphQL API   │
                                   │  + Anti-Detect  │
                                   └────────┬────────┘
                                            │
                                            ▼
┌─────────────────────┐          ┌─────────────────────┐         ┌───────────────────┐
│  Retail Markers     │◄─────────┤   Universal ML      ├────────►│   Supabase DB     │
│  6-Worker Parallel  │          │   XGBoost System    │         │  + Notifications  │
│  Playwright Engine  │          │  All Makes/Models   │         │   (Email Alerts)  │
└─────────────────────┘          └─────────────────────┘         └───────────────────┘
```

### Core Process Flow

1. **Data Collection**: Scrape vehicle listings from GraphQL API with anti-detection
2. **Price Analysis**: Extract retail price markers from dealer listings using parallel workers
3. **ML Prediction**: Universal XGBoost model predicts market values for private listings
4. **Deal Identification**: Calculate profit margins and filter for profitable opportunities
5. **Storage & Alerts**: Save results to database and send email notifications

## 🔧 Scraping System

### GraphQL API
- **Direct API Access**: Bypasses HTML parsing with GraphQL queries to `/at-gateway`
- **Complete Data**: Returns JSON with all listing details, images, seller info, specifications
- **Automatic Limit Bypass**: Intelligently splits searches into 5 mileage ranges (0-20k, 20-40k, 40-60k, 60-80k, 80-100k miles) to bypass ~2,000 listing pagination limit
- **Smart Deduplication**: Removes duplicate listings across mileage ranges to ensure clean results
- **Rate Limiting**: Intelligent delays and retry mechanisms to avoid detection

### Anti-Detection & Stealth

#### Webshare API Integration
- **Automatic Daily Refresh**: Fetches fresh proxies every 24 hours
- **In-Memory Caching**: Stores proxies in memory with timestamp-based refresh logic
- **Worker Assignment**: Each of 6 workers gets a consistent proxy IP for session affinity
- **Graceful Fallback**: Uses cached proxies if API fails, JSON backup if both fail

```python
# Proxy system automatically handles rotation and blacklisting
proxy_manager = ProxyManager()
proxy = proxy_manager.get_proxy()  # Gets fresh proxy with rotation
```

#### CloudFlare Bypass Techniques
- **Dynamic TLS Fingerprinting**: Randomized cipher suites and curve orders per request
- **Browser Simulation**: Chrome user agents with realistic client hints and headers
- **Cookie Generation**: Plausible CloudFlare challenge and bot management cookies
- **Request Patterns**: Human-like delays and realistic browsing behavior

### Multi-Worker Parallel Processing
- **6-Worker System**: Playwright browsers run in parallel for retail price scraping
- **Session Affinity**: Each worker maintains consistent proxy IP for reliable access
- **Anti-Fingerprinting**: Randomized viewport sizes, device memory, connection speeds
- **Fault Tolerance**: Workers continue if others fail, graceful degradation

### Complete Coverage System
- **Automatic Range Division**: Every search is transparently split into 5 mileage ranges to bypass pagination limits
- **Configurable Ranges**: Default ranges (0-20k, 20-40k, 40-60k, 60-80k, 80-100k miles) ensure comprehensive coverage and can be customized
- **Silent Operation**: Fully transparent to users - existing code continues to work without any changes
- **Smart Deduplication**: Duplicate detection removes overlapping listings using unique advertiser IDs
- **Fault Tolerance**: If one mileage range fails, others continue processing to maximize data capture
- **Performance Optimized**: Sequential processing prevents rate limiting while maintaining high throughput

## 🤖 Analytics System

### Universal ML Model
The system uses a single XGBoost model trained on data from all vehicle makes and models:

- **Cross-Vehicle Learning**: Model learns patterns across BMW, Audi, Ford, etc.
- **Standardized Features**: Universal encoding system for makes, models, fuel types, specs
- **Enhanced Accuracy**: More training data leads to better predictions than individual models
- **Efficient Processing**: Single model deployment reduces complexity and memory usage

### Feature Engineering Pipeline
```python
# Universal feature set used across all vehicles
FEATURE_COLUMNS = [
    'asking_price', 'mileage', 'age',           # Core metrics
    'make_encoded', 'model_encoded',            # Vehicle identity
    'fuel_type_numeric', 'transmission_numeric', # Drivetrain
    'engine_size', 'spec_numeric'               # Performance & trim
]
```

### XGBoost Configuration
- **Objective**: Regression with squared error loss
- **Hyperparameters**: Optimized depth (6), learning rate (0.1), subsample (0.8)
- **Regularization**: Column sampling and minimum child weight to prevent overfitting
- **Early Stopping**: 15 rounds to prevent overtraining

### ML Model Configuration
```python
# XGBoost hyperparameters optimized for vehicle data
UNIVERSAL_ML_CONFIG = {
    'min_training_samples': 50,      # Minimum samples to train
    'xgboost_params': {
        'max_depth': 6,              # Tree depth
        'eta': 0.1,                  # Learning rate
        'subsample': 0.8,            # Row sampling
        'colsample_bytree': 0.8,     # Feature sampling
    },
    'training_params': {
        'num_boost_round': 150,      # Training iterations
        'early_stopping_rounds': 15,  # Overfitting prevention
    }
}
```


## ⚙️ Setup & Usage

### Installation & Requirements
```bash
# Install Python dependencies
pip install -r config/requirements.txt

# Install Playwright browsers
playwright install chromium

# Set up environment variables
cp config/.env.example config/.env
# Edit config/.env with your actual API keys
```

### Required Environment Variables
```bash
# Database Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Proxy Service
WEBSHARE_API_TOKEN=your_webshare_api_token_here

# Notifications
RESEND_API_KEY=your_resend_api_key_here
```

### Configuration

#### Target Vehicle Configuration
```python
# Excerpt from config/config.py
TARGET_VEHICLES_BY_MAKE = {
    "Audi": ["A1", "A3", "A4", "A4 Avant", "A5", "Q2", "Q3", "Q5", "Q7", "S3", "TT"],
    "BMW": ["1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "X1", "X3"],
    "Mercedes-Benz": ["A Class", "C Class", "E Class", "GLC", "GLA", "CLA"],
    "Ford": ["Fiesta", "Focus", "Kuga", "Mondeo", "Puma", "EcoSport"],
    "Volkswagen": ["Golf", "Polo", "Tiguan", "T-Roc", "Passat", "up!"],
    # ... 20+ more manufacturers with 200+ models total
}
```

#### Search Parameters
```python
# Default search criteria applied to all vehicles
VEHICLE_SEARCH_CRITERIA = {
    "maximum_mileage": 100000,    # 100k miles max
    "year_from": 2010,           # 2010 onwards
    "year_to": 2023,             # Up to 2023 models
    "postcode": "M15 4FN"        # Manchester city center
}
```

#### Proxy Configuration
```python
# Webshare API settings
WEBSHARE_PROXY_CONFIG = {
    'refresh_interval_hours': 24,    # Daily refresh
    'api_timeout': 30,               # 30 second timeout
    'max_proxies': 100               # Up to 100 proxies
}
```

### Running Commands

#### Basic Operations
```bash
# Run complete analysis for all configured vehicles
python src/analyser.py

# Analyze specific make/model
python src/analyser.py --model "3 Series"

# Run smart grouped scraping with notifications
python src/scraping.py

# Run weekly model training
python services/ML_trainer.py
```

#### Testing & Development
```bash
# Test mode with specific model
python src/analyser.py --test --model "Focus"

# Scraping with filters
python src/scraping.py --make Ford --model Fiesta

# Test weekly training
python services/ML_trainer.py --test

# Validate API connections
python services/network_requests.py
```

### Core System Components
- **`services/network_requests.py`** - GraphQL API client with mileage splitting and anti-detection
- **`services/stealth_orchestrator.py`** - Proxy management with Webshare API integration and anti-fingerprinting
- **`services/browser_scraper.py`** - 6-worker Playwright system for parallel price marker extraction
- **`services/ML_model.py`** - XGBoost training and prediction system for all vehicle types
- **`src/analyser.py`** - Main analysis pipeline coordinating scraping and ML prediction
- **`services/ML_trainer.py`** - Automated weekly model training and performance monitoring
- **`src/scraping.py`** - Smart grouped scraping orchestration with deal notification
- **`config/config.py`** - Centralized configuration with environment variables and scheduling functions
- **`config/encodings.py`** - Standardized encoding mappings for makes, models, and specifications

This system provides automated vehicle deal finding, combining web scraping techniques with machine learning to identify profitable opportunities in the used car market.