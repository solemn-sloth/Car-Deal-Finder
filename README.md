# Car Dealer Bot - Backend

Automated car deal scraping and analysis system for AutoTrader UK. Uses ML-powered market analysis to identify profitable deals and sends targeted notifications.

## 🏗️ Architecture

```
AutoTrader API → Network Scraper → Data Adapter → ML Analyzer → Database Sync
                      ↓
               Smart Grouping (242 models) → Quality Deals → Email Notifications
                      ↓
              JSON Archives (quarterly analytics)
```

## 📁 Core Scripts

### Main Components
- **`scraper.py`** - Main entry point with intelligent deal synchronization
- **`network_scraper.py`** - High-speed AutoTrader API client 
- **`analyser.py`** - ML-powered market value estimation and deal rating
- **`supabase_storage.py`** - Database operations with intelligent sync
- **`scrape_grouping.py`** - Smart batching for 242 car models
- **`json_data_adapter.py`** - Data format conversion between scraper and analyzer

### Supporting Services
- **`api.py`** - REST API for frontend integration
- **`deal_notifications.py`** - Email notification pipeline
- **`config.py`** - Vehicle targets and search criteria

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add: SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY, RESEND_API_KEY

# Run main scraper
python src/scraper.py

# Test with limited scope
python src/scraper.py --max-groups 5 --test-mode

# Start API server
python src/api.py  # Runs on http://localhost:8000
```

## 🔧 Key Features

- **Fast Network Scraping**: Direct API calls (4x faster than Playwright)
- **Intelligent Deal Sync**: Only removes truly sold deals, never time-based cleanup
- **ML Market Analysis**: Regression models with similarity scoring
- **Smart Profit Calculation**: Conservative estimates with recon costs
- **JSON Archiving**: Complete deal data for quarterly analytics (~0.2MB per run)
- **Email Notifications**: Targeted alerts for quality deals

## ⚙️ Configuration

### Vehicle Targets (`src/config.py`)
```python
TARGET_VEHICLES_BY_MAKE = {
    "Ford": ["Fiesta", "Focus", "Kuga"],
    "BMW": ["1 Series", "3 Series", "X1"],
    # ... 22 brands, 242 models total
}

VEHICLE_SEARCH_CRITERIA = {
    "fuel_type": "Petrol",
    "transmission": "Manual", 
    "distance": 25,  # miles from postcode
    "min_year": 2018
}
```

### Deal Analysis Thresholds (`analyser.py`)
```python
CONFIG = {
    'recon_cost': 300,                 # Fixed reconditioning cost
    'excellent_margin_pct': 0.25,      # 25%+ = excellent deal
    'good_margin_pct': 0.20,           # 20%+ = good deal  
    'negotiation_margin_pct': 0.15,    # 15%+ = negotiation opportunity
    'min_cash_margin': 800,            # Minimum £800 profit
}
```

## 🌐 API Endpoints

```bash
# Get filtered deals
GET /deals?make=Ford&model=Fiesta&rating=excellent&limit=50

# Get statistics
GET /stats

# Trigger manual scraping
POST /scrape {"model": "Swift", "max_results": 100}

# Health check
GET /health
```

## 📊 JSON Archives

Each run creates a comprehensive archive in `archive/`:
- **Storage**: ~0.2MB per archive
- **Contents**: Complete deal data + analytics + session metadata
- **Management**: Use `python archive/manage_archives.py --summary`

## 🔍 How It Works

1. **Smart Grouping**: Organizes 242 car models into efficient batches
2. **Network Scraping**: Fast API calls to AutoTrader (no browser needed)
3. **ML Analysis**: Regression models calculate market value and profit margins
4. **Intelligent Sync**: Compares with database, adds new deals, removes sold ones
5. **Quality Rating**: Classifies deals (Excellent/Good/Negotiation/Reject)
6. **Notifications**: Sends targeted emails for quality deals
7. **Archiving**: Saves complete datasets for quarterly analysis

### Key Algorithms
- **Deal Synchronization**: Compares current scrape vs database, only removes truly missing deals
- **Market Value Estimation**: Weighted regression with similarity scoring and outlier detection
- **Production Safety**: Uses service role key to bypass RLS for reliable operations

---

**🚗 Car Dealer Bot - Automated Intelligence for Profitable Car Trading**