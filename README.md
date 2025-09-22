# Car Deal Finder

Vehicle analysis system that scrapes car listings via GraphQL API and uses machine learning to identify potentially underpriced vehicles.

## 🔍 How It Works

```
                    ┌─────────────────┐
                    │   GraphQL API   │
                    │    (AutoTrader) │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────┐         ┌───────────────────┐
│   Model-Specific    ├────────►│   Supabase DB     │
│   XGBoost Models    │         │  + Notifications  │
│    (139 models)     │         │   (Email Alerts)  │
└─────────────────────┘         └───────────────────┘
```

### Core Process Flow

1. **Data Collection**: Scrape vehicle listings from AutoTrader GraphQL API
2. **ML Prediction**: Model-specific XGBoost models predict market values
3. **Deal Identification**: Compare predictions to asking prices to identify deals
4. **Storage & Alerts**: Save results to database and send email notifications

## 🔧 Scraping System

### GraphQL API
- **API Access**: Makes GraphQL queries to AutoTrader's `/at-gateway` endpoint
- **Data Extraction**: Returns JSON with listing details, images, seller info, specifications
- **Mileage Splitting**: Splits searches into 5 mileage ranges to bypass pagination limits
- **Deduplication**: Removes duplicate listings across mileage ranges
- **Rate Limiting**: Uses delays and retry mechanisms

### Anti-Detection Features

#### Proxy Integration
- **Webshare API**: Fetches proxies for IP rotation
- **Proxy Rotation**: Automatic IP switching when blocked
- **Session Management**: Maintains separate sessions per proxy

#### Request Patterns
- **User Agent Rotation**: Uses different browser fingerprints
- **Header Randomization**: Varies request headers to appear human
- **Timing Delays**: Random delays between requests
- **Cookie Handling**: Manages CloudFlare and session cookies

## 🤖 Machine Learning System

### Model-Specific Architecture
The system uses **139 individual XGBoost models** (one per car model) with **optimized anti-overfitting approach**:

- **Specialized Models**: Each car model has its own trained ML model
- **Daily Training**: 10 models retrained per day in a 14-day cycle
- **Optimized Performance**: 20%+ accuracy improvement through A/B/C+ testing
- **Anti-Overfitting**: 7.6% overfitting gap (down from 15-20%)
- **Feature Set**: 6 core features per model (mileage, age, engine_size, fuel_type, transmission, spec)
- **Model Storage**: Models stored in organized make/model directories

### Daily Training System
```
Day 1: BMW 3-Series, Honda Jazz, Porsche 911, Tesla Model 3... (10 models)
Day 2: Audi A3, Ford Fiesta, Honda Civic, Lexus CT... (10 models)
...
Day 14: VW Tiguan, Volvo XC60, Renault Megane... (9 models)
```

**Features:**
- **Rotating Schedule**: 10 models trained per day over 14-day cycle
- **Balanced Groups**: Mix of popular and niche models per day
- **Fresh Models**: Each model updated every 14 days
- **Independent Training**: Models trained separately

### Model Storage Structure
```
archive/ml_models/
├── audi/a3/model.json + scaler.pkl
├── bmw/3_series/model.json + scaler.pkl
├── ford/fiesta/model.json + scaler.pkl
└── ... (139 model pairs in 30 make folders)
```

### Feature Set
```python
# 6 core features used for each model (optimized for anti-overfitting)
FEATURE_COLUMNS = [
    'mileage', 'age', 'engine_size',              # Numeric features
    'fuel_type_extracted', 'transmission_extracted', 'spec_normalized'  # Categorical features
]
```

### Optimized Balanced Approach (2024)
Based on comprehensive A/B/C+ testing, the system uses **optimized parameters** proven to minimize overfitting:

| Dataset Size | Max Depth | Learning Rate | Trees | L2 Reg | Early Stop | Validation | Transform |
|--------------|-----------|---------------|-------|--------|------------|------------|-----------|
| **TINY** (<100) | 4 | 0.03 | 150 | 3.0 | 20 | 25% test | Log |
| **SMALL** (100-500) | 4 | 0.04 | 250 | 2.5 | 25 | 20% test | Log |
| **MEDIUM** (500+) | 4 | 0.05 | 300 | 2.0 | 30 | 20% test | Log |

**Key Optimizations:**
- **Balanced Approach**: Conservative depth (4) with optimized learning rate (0.05)
- **Anti-Overfitting**: L2 regularization (2.0) + early stopping (30 rounds)
- **Log Transformation**: Target variable log transformation for 20%+ accuracy improvement
- **Categorical Support**: Native XGBoost categorical feature handling
- **A/B/C+ Tested**: Parameters validated across BMW 3 Series, Audi A3, Ford Fiesta

### Performance Improvements
Recent optimizations achieved significant performance gains:

- **Accuracy**: MAPE improved from 24.9% to 14.8% (40% reduction in error)
- **Overfitting**: Gap reduced from 15-20% to 7.6% (62% improvement)
- **Deal Precision**: False positives reduced from 247 to 1-15 quality deals
- **R² Score**: Expected improvement from ~0.45 to ~0.58 (20%+ gain)

### A/B/C+ Testing Framework
The system includes comprehensive ML optimization testing via `model_diagnostics.py`:

```bash
# Test specific model with A/B/C+ approaches
python model_diagnostics.py --models "bmw_3_series,audi_a3"

# Quick test with reduced parameter grid
python model_diagnostics.py --quick

# Test with sample data instead of live scraping
python model_diagnostics.py --test-mode
```

**Testing Framework Features:**
- **10 Systematic Approaches**: Control, Lower Learning, Higher Reg, Deeper Trees, Feature Sampling, etc.
- **Cross-Validation**: 5-fold CV for stability assessment
- **Sample Predictions**: Saves 10 sample predictions per approach for analysis
- **Comprehensive Logging**: Detailed results saved to `archive/outputs/` for analysis
- **Target Distribution Analysis**: Metrics for future dynamic parameter tuning

### Daily Training Groups
Training groups are defined in `config/config.py` with balanced model distribution across 14 days.

## 🤖 Daily Automation

The main automation runs through `src/main.py`:
```
Workflow:
1. Scrape vehicle listings from AutoTrader API
2. Load appropriate trained ML models
3. Predict market values and identify deals
4. Store results in database and send notifications
```

## ⚙️ Setup & Usage

### Installation & Requirements
```bash
# Install Python dependencies
pip install -r config/requirements.txt

# Set up environment variables
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

### Required Environment Variables
```bash
# Database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Proxy Service (optional)
WEBSHARE_API_TOKEN=your_webshare_token

# Notifications
RESEND_API_KEY=your_resend_key
```

### Configuration

Vehicle targets and search parameters are defined in `config/config.py`:
- 30+ car manufacturers with 139+ models total
- Search criteria: 2010-2020 vehicles, up to 100k miles
- Proxy settings for optional IP rotation

### Running Commands

#### Main Usage
```bash
# Run full automation
python src/main.py

# Show what would be done without executing
python src/main.py --dry-run

# Test specific model
python src/main.py --model "3 Series"

# Force retrain specific model
python src/main.py --model "3 Series" --force-retrain

# Export predictions to JSON
python src/main.py --export-predictions
```

#### Training Commands
```bash
# Train today's models (10 models per day)
python services/daily_trainer.py

# Validate training groups
python ml/validate_groups.py
```

#### ML Optimization & Diagnostics
```bash
# Run A/B/C+ testing framework on specific models
python model_diagnostics.py --models "bmw_3_series,audi_a3,ford_fiesta"

# Quick parameter testing with reduced grid
python model_diagnostics.py --quick

# Test with sample data (no live scraping)
python model_diagnostics.py --test-mode

# Full diagnostic on all test models
python model_diagnostics.py
```

### Core System Components

#### Main Files
- **`src/main.py`** - Main entry point for daily automation
- **`services/network_requests.py`** - AutoTrader GraphQL API client
- **`src/scraping.py`** - Vehicle scraping orchestration
- **`src/analyser.py`** - ML analysis pipeline
- **`services/daily_trainer.py`** - Daily model training orchestrator
- **`services/model_specific_trainer.py`** - Individual model training with optimized parameters
- **`model_diagnostics.py`** - A/B/C+ testing framework for ML optimization
- **`services/stealth_orchestrator.py`** - Proxy management and anti-detection
- **`config/config.py`** - System configuration and vehicle targets

#### Data Storage
- **`archive/ml_models/`** - 139 trained XGBoost models organized by make/model
- **Supabase database** - Stores found deals and analysis results