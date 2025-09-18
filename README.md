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

### Model-Specific ML Architecture
The system uses **139 individual XGBoost models** (one per car model) with a **distributed daily training system**:

- **Specialized Learning**: Each car model gets its own ML model trained only on relevant data
- **Daily Training**: 10 models retrained per day in a 14-day rotating cycle
- **Balanced Workload**: Popular models distributed evenly across days for consistent training times
- **Clean Features**: 8 focused features per model (no complex one-hot encoding needed)

### Daily Training System
```
Day 1: BMW 3-Series, Honda Jazz, Porsche 911, Tesla Model 3... (10 models)
Day 2: Audi A3, Ford Fiesta, Honda Civic, Lexus CT... (10 models)
...
Day 14: VW Tiguan, Volvo XC60, Renault Megane... (9 models)
```

**Benefits:**
- **🎯 Specialized**: Each model learns its specific market patterns
- **⚡ Fast Training**: 10 models/day vs 139 models/week
- **🔄 Fresh Data**: Models updated every 14 days
- **🛡️ Isolated**: Model failures don't affect others

### Model Storage Structure
```
archive/ml_models/
├── audi/a3/model.xgb + scaler.pkl
├── bmw/3_series/model.xgb + scaler.pkl
├── ford/fiesta/model.xgb + scaler.pkl
└── ... (139 model pairs in 30 make folders)
```

### Feature Engineering Pipeline
```python
# Model-specific feature set (8 focused features)
FEATURE_COLUMNS = [
    'asking_price', 'mileage', 'age', 'market_value',    # Core metrics
    'fuel_type_numeric', 'transmission_numeric',         # Drivetrain
    'engine_size', 'spec_numeric'                        # Performance & trim
]
# No make/model encoding needed - each model is specialized!
```

### XGBoost Configuration
- **Objective**: Regression with squared error loss
- **Hyperparameters**: Optimized depth (6), learning rate (0.1), subsample (0.8)
- **Regularization**: Column sampling and minimum child weight to prevent overfitting
- **Early Stopping**: 10 rounds to prevent overtraining
- **Minimum Data**: 50 samples required before training

### Daily Training Management
```python
# Hardcoded balanced groups in config/config.py
DAILY_TRAINING_GROUPS = {
    1: [("BMW", "3 Series"), ("Honda", "Jazz"), ...],  # Mix of popular + niche
    2: [("Audi", "A3"), ("Ford", "Fiesta"), ...],      # Different balanced mix
    # ... up to day 14
}
```


## 🤖 Daily Automation Workflow

The system provides a unified entry point (`main.py`) for complete daily automation:

### Training-First Architecture
```
Daily Automation Workflow:

1. 🧠 PHASE 1: Daily Model Training
   ├── Train 10 models from today's balanced group
   ├── Models refreshed every 14 days in rotating cycle
   ├── Clean model storage in organized sub-folders
   └── Automatic cycle day advancement

2. 🔍 PHASE 2: Deal Finding & Analysis
   ├── Load existing trained models (no on-demand training)
   ├── Scrape all vehicle makes/models with GraphQL API
   ├── Predict market values using model-specific features
   ├── Filter for profitable deals with statistical confidence
   ├── Send email notifications via Resend API
   └── Store results in Supabase database

3. 📊 PHASE 3: Reporting & Logging
   ├── Unified logging across both phases
   ├── Success/failure tracking with error handling
   ├── Performance metrics and duration tracking
   └── Email summaries for unattended operation
```

### Key Benefits
- **🎯 Single Command**: `python main.py` handles everything
- **⚡ Training-First**: Fresh models available before analysis
- **🔄 Automated Cycles**: 14-day model refresh schedule
- **🛡️ Fault Tolerant**: Continues on individual model failures
- **📈 Production Ready**: Designed for unattended cron operation

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

#### 🚀 Daily Automation (Main Entry Point)
```bash
# Full daily automation - training + deal finding
python src/main.py

# Run only daily model training (10 models per day)
python src/main.py --training-only

# Run only deal finding & analysis
python src/main.py --scraping-only

# Show what would be done without executing
python src/main.py --dry-run

# Enable detailed logging for debugging
python src/main.py --verbose

# Test single model in full production mode
python src/main.py --model "3 Series"

# Force retrain specific model + run deal finding
python src/main.py --model "3 Series" --force-retrain

# Force retrain only (no deal finding)
python src/main.py --training-only --model "3 Series" --force-retrain

# Cron job example for daily automation
0 2 * * * cd /path/to/car-deal-finder && python src/main.py >> logs/daily.log 2>&1
```

#### 🧠 ML Training & Management
```bash
# Standalone daily model training (10 models per day)
python ml/daily_training.py

# See which models would be trained today
python ml/daily_training.py --dry-run

# Validate training group assignments
python ml/validate_groups.py

# Show specific day breakdown
python ml/validate_groups.py --day 5

# Training with options
python ml/daily_training.py --max-pages 2 --verbose
```

#### 🧪 Testing & Development
```bash
# Test single model workflow (production mode with DB/notifications)
python src/main.py --model "Focus" --dry-run          # Preview what would happen
python src/main.py --model "Focus"                    # Run full production test

# Force retrain and test specific model
python src/main.py --model "Focus" --force-retrain --dry-run  # Preview forced retrain
python src/main.py --model "Focus" --force-retrain            # Execute forced retrain + test

# Test training only for specific model
python src/main.py --training-only --model "Focus" --force-retrain

# Legacy individual operations
python src/analyser.py --model "3 Series"             # Direct analysis
python src/scraping.py --make Ford --model Fiesta     # Direct scraping
python services/network_requests.py                   # API validation
```

### Core System Components

#### Data Collection & Processing
- **`services/network_requests.py`** - GraphQL API client with mileage splitting and anti-detection
- **`services/stealth_orchestrator.py`** - Proxy management with Webshare API integration and anti-fingerprinting
- **`services/browser_scraper.py`** - 6-worker Playwright system for parallel price marker extraction
- **`src/analyser.py`** - Main analysis pipeline coordinating scraping and ML prediction
- **`src/scraping.py`** - Smart grouped scraping orchestration with deal notification

#### ML Training & Prediction
- **`services/daily_trainer.py`** - Orchestrates daily training of 10 models in 14-day cycles
- **`services/model_specific_trainer.py`** - Individual model training with specialized features
- **`ml/daily_training.py`** - CLI tool for running and monitoring daily training
- **`ml/validate_groups.py`** - Validation utility for training group assignments

#### Automation & Orchestration
- **`src/main.py`** - Unified daily automation entry point with training-first workflow

#### Configuration & Data
- **`config/config.py`** - Centralized configuration with hardcoded balanced training groups
- **`config/daily_training_config.json`** - Daily training cycle state and configuration
- **`config/encodings.py`** - Legacy encoding mappings (superseded by model-specific approach)
- **`archive/ml_models/`** - 139 trained models organized in make/model subfolders

This system provides automated vehicle deal finding, combining web scraping techniques with specialized machine learning models to identify profitable opportunities in the used car market.