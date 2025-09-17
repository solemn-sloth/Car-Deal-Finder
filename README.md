# Car Deal Finder

Advanced vehicle analysis system that scrapes used car listings using AutoTrader's GraphQL API, then employs machine learning to identify underpriced vehicles and predict profit margins with statistical confidence.

## 🔍 How It Works

```
                                   ┌─────────────────┐
                                   │   AutoTrader    │
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

1. **Daily Proxy Refresh**: Automatically fetches fresh proxies from Webshare API with 24-hour caching
2. **GraphQL Scraping**: High-performance AutoTrader API client with CloudFlare bypass and anti-fingerprinting
3. **Parallel Processing**: 6-worker Playwright system scrapes retail price markers from dealer listings  
4. **Universal ML Analysis**: Single XGBoost model trained on all vehicle data predicts market values
5. **Profit Calculation**: Statistical algorithms identify underpriced vehicles with confidence scores

## 🌐 Proxy & Anti-Detection System

### Webshare API Integration
- **Automatic Daily Refresh**: Fetches up to 100 fresh proxies every 24 hours
- **In-Memory Caching**: Stores proxies in memory with timestamp-based refresh logic
- **Worker Assignment**: Each of 6 workers gets a consistent proxy IP for session affinity
- **Graceful Fallback**: Uses cached proxies if API fails, JSON backup if both fail

```python
# Proxy system automatically handles rotation and blacklisting
proxy_manager = ProxyManager()
proxy = proxy_manager.get_proxy()  # Gets fresh proxy with rotation
```

### CloudFlare Bypass Techniques
- **Dynamic TLS Fingerprinting**: Randomized cipher suites and curve orders per request
- **Browser Simulation**: Chrome user agents with realistic client hints and headers
- **Cookie Generation**: Plausible CloudFlare challenge and bot management cookies
- **Request Patterns**: Human-like delays and realistic browsing behavior

## 📊 Universal ML System

### Unified Model Architecture
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

## 🚀 Enhanced Scraping Architecture

### AutoTrader GraphQL API
- **Direct API Access**: Bypasses HTML parsing with GraphQL queries to `/at-gateway`
- **Complete Data**: Returns JSON with all listing details, images, seller info, specifications
- **Automatic Limit Bypass**: Intelligently splits searches into 5 mileage ranges (0-20k, 20-40k, 40-60k, 60-80k, 80-100k miles) to completely bypass AutoTrader's ~2,000 listing pagination limit
- **Smart Deduplication**: Removes duplicate listings across mileage ranges to ensure clean results
- **Rate Limiting**: Intelligent delays and retry mechanisms to avoid detection

### Multi-Worker Parallel Processing  
- **6-Worker System**: Playwright browsers run in parallel for retail price scraping
- **Session Affinity**: Each worker maintains consistent proxy IP for reliable access
- **Anti-Fingerprinting**: Randomized viewport sizes, device memory, connection speeds
- **Fault Tolerance**: Workers continue if others fail, graceful degradation

### Performance Optimizations
- **Connection Pooling**: Reuses HTTP connections with configurable pool sizes
- **Intelligent Caching**: API responses cached to avoid duplicate requests
- **Batch Processing**: Groups similar requests to maximize throughput
- **Memory Management**: Efficient data structures and garbage collection

### Complete Listing Coverage System
- **Automatic Range Division**: Every search is transparently split into 5 mileage ranges to completely bypass AutoTrader's pagination limits
- **Configurable Ranges**: Default ranges (0-20k, 20-40k, 40-60k, 60-80k, 80-100k miles) ensure comprehensive coverage and can be customized
- **Silent Operation**: Fully transparent to users - existing code continues to work without any changes
- **Smart Deduplication**: Advanced duplicate detection removes overlapping listings using unique advertiser IDs
- **Fault Tolerance**: If one mileage range fails, others continue processing to maximize data capture
- **Performance Optimized**: Sequential processing prevents rate limiting while maintaining high throughput

## 📁 Core System Components

### Primary Scraping & Analysis
- **`network_requests.py`** - AutoTrader GraphQL API client with mileage splitting and anti-detection
- **`proxy_rotation.py`** - Webshare API integration with automatic refresh and blacklisting
- **`retail_price_scraper.py`** - 6-worker Playwright system for parallel price marker extraction
- **`universal_ml_model.py`** - XGBoost training and prediction system for all vehicle types

### ML & Data Processing
- **`ML_analyser.py`** - Main analysis pipeline coordinating scraping and ML prediction
- **`ML_trainer.py`** - Automated weekly model training and performance monitoring
- **`data_adapter.py`** - Data format standardization and validation

### Configuration & Infrastructure
- **`config/config.py`** - Centralized configuration with environment variables and scheduling functions
- **`config/encodings.py`** - Standardized encoding mappings for makes, models, and specifications
- **`services/notifications.py`** - Email alerts and result reporting
- **`src/storage.py`** - Supabase database operations and caching

## 🔧 Environment Setup

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

### Installation & Dependencies
```bash
# Install Python dependencies
pip install -r config/requirements.txt

# Install Playwright browsers
playwright install chromium

# Set up environment variables
cp config/.env.example config/.env
# Edit config/.env with your actual API keys
```

## 🚀 Running the System

### Basic Operations
```bash
# Run complete analysis for all configured vehicles
python src/ML_analyser.py

# Analyze specific make/model
python src/ML_analyser.py --make BMW --model "3 Series"

# Test scraping without ML analysis
python src/scraper.py --test

# Run weekly model training
python services/weekly_trainer.py
```

### Advanced Configuration
```bash
# Custom search parameters
python src/ML_analyser.py --make Ford --model Focus --max-year 2020

```

## ⚙️ System Configuration

### Target Vehicle Configuration
The system is configured to analyze 100+ popular vehicle models across all major manufacturers:

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

### Search Parameters  
```python
# Default search criteria applied to all vehicles
VEHICLE_SEARCH_CRITERIA = {
    "maximum_mileage": 100000,    # 100k miles max
    "year_from": 2010,           # 2010 onwards
    "year_to": 2023,             # Up to 2023 models
    "postcode": "M15 4FN"        # Manchester city center
}
```

### Proxy Configuration
```python
# Webshare API settings  
WEBSHARE_PROXY_CONFIG = {
    'refresh_interval_hours': 24,    # Daily refresh
    'api_timeout': 30,               # 30 second timeout
    'max_proxies': 100               # Up to 100 proxies
}
```

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

## 📊 Performance Characteristics

### Throughput Metrics
- **API Scraping**: Complete vehicle coverage with mileage splitting - bypasses 2,000 listing limits
- **Comprehensive Coverage**: 5x more listings captured through automatic mileage range splitting
- **Retail Processing**: 6 parallel workers handle 100+ dealer listings per minute
- **ML Predictions**: Universal model processes 1,000+ listings in seconds
- **Daily Capacity**: Can analyze 50+ vehicle models in automated daily runs

### Resource Usage
- **Memory**: ~50MB for proxy cache, ~200MB for ML model, ~100MB per worker
- **CPU**: Optimized for multi-core systems, scales with available cores
- **Network**: Intelligent rate limiting prevents IP blacklisting  
- **Storage**: Compressed data in Supabase, minimal local storage requirements

## 🔮 Architecture Benefits

### Scalability
- **Horizontal Scaling**: Add more workers or proxy IPs for increased throughput
- **Modular Design**: Components can be deployed independently
- **Caching Strategy**: Reduces API calls and improves response times

### Reliability  
- **Fault Tolerance**: System continues operation if individual components fail
- **Automatic Recovery**: Proxy rotation and retry mechanisms handle temporary blocks
- **Data Persistence**: All results stored in reliable cloud database

### Maintainability
- **Configuration-Driven**: Easy to add new vehicle models or adjust parameters  
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Environment Separation**: Clear separation between development and production configs

## 🔧 Development & Testing

### Code Quality
- **Type Hints**: Comprehensive type annotations throughout codebase
- **Error Handling**: Graceful degradation and informative error messages
- **Documentation**: Detailed docstrings and inline comments
- **Modular Architecture**: Clean separation of concerns

### Testing Infrastructure  
```bash
# Run system tests
python tests/test_proxies.py
python tests/performance_test.py

# Test ML model performance
python src/ML_analyser.py --test-mode

# Validate API connections
python services/network_requests.py
```

This system represents a sophisticated approach to automated vehicle deal finding, combining advanced web scraping techniques with modern machine learning to identify profitable opportunities in the used car market.