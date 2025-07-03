# Car Dealer Bot - Backend

Python backend service for automated car deal scraping and analysis from AutoTrader. Deployed on Hetzner cloud infrastructure with intelligent batch scheduling for 24/7 autonomous operation.

## 📁 Backend Scripts Overview

### Core Components
- **`scraper.py`** - Main entry point with intelligent deal synchronization + JSON archiving
- **`scrape_grouping.py`** - Smart grouping orchestrator for efficient batch processing (242 models)
- **`network_scraper.py`** - High-speed network request scraping engine for AutoTrader API
- **`json_data_adapter.py`** - Data format conversion between scraper and analyzer
- **`analyser.py`** - ML-powered deal analysis with regression-based market value estimation
- **`supabase_storage.py`** - Database operations with intelligent deal synchronization (bypasses RLS)

### Supporting Services
- **`api.py`** - REST API endpoints for frontend integration
- **`deal_notifications.py`** - Complete email notification pipeline for quality deals
- **`notification_service.py`** - Core email sending utilities and HTML templates
- **`config.py`** - Vehicle targets and search criteria configuration

### Archive Management
- **`manage_archives.py`** - JSON archive management and cleanup tools (in archive/ folder)
- **`estimate_storage.py`** - Storage requirement estimation for quarterly archives

## 🚗 Key Features

- **Fast Network Scraping**: Direct API calls (4x faster than legacy Playwright)
- **Intelligent Deal Sync**: Only deletes truly sold deals, never time-based cleanup
- **ML Market Analysis**: Regression models with similarity scoring for accurate valuations
- **Smart Profit Calculation**: Conservative estimates with recon costs and market premiums
- **JSON Archiving**: Complete deal data saved for quarterly analytics (~0.2MB per run)
- **Email Notifications**: Targeted alerts for quality deals with user preferences
- **Production Safety**: Service role key bypasses RLS for reliable database operations

## 🛠️ Quick Start

### Prerequisites
- Python 3.12+
- Supabase project with service role key

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add: SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY

# Run main scraper
python src/scraper.py

# Test with limited scope
python src/scraper.py --max-groups 5 --test-mode
```

### API Server
```bash
python src/api.py
# Runs on http://localhost:8000
```

## 🚀 How It Works

### System Architecture
```
AutoTrader API → Network Scraper → Data Adapter → ML Analyzer → Database Sync
                      ↓
               Smart Grouping (242 models) → Quality Deals → Email Notifications
                      ↓
              JSON Archives (quarterly analytics)
```

### Core Workflow
1. **Smart Grouping**: Organizes 242 car models into efficient batches
2. **Network Scraping**: Fast API calls to AutoTrader (no browser needed)
3. **Data Conversion**: Transforms raw API data into analyzer format
4. **ML Analysis**: Regression models calculate market value and profit margins
5. **Intelligent Sync**: Compares with database, adds new deals, removes sold ones
6. **Quality Rating**: Classifies deals (Excellent/Good/Negotiation/Reject)
7. **Notifications**: Sends targeted emails for quality deals
8. **Archiving**: Saves complete datasets for quarterly business analysis

### Key Algorithms

#### Deal Synchronization
- **Problem Solved**: Previous system deleted all deals after 24h (dangerous for production)
- **Solution**: Compare current scrape vs database, only delete truly missing deals
- **Safety**: Uses service role key to bypass RLS, ensures reliability

#### Market Value Estimation
- **Similarity Scoring**: Multi-factor comparison (year, mileage, spec level)
- **Weighted Regression**: Higher weight for more similar vehicles
- **Conservative Estimates**: Uses lower confidence bounds for profit calculations
- **Outlier Detection**: Removes price anomalies using 2σ threshold

## ⚙️ Configuration

### Vehicle Targets
Edit `src/config.py` to modify search scope:

```python
# Target vehicles (22 brands, 242 models)
TARGET_VEHICLES_BY_MAKE = {
    "Ford": ["Fiesta", "Focus", "Kuga", "EcoSport", "Ranger"],
    "BMW": ["1 Series", "3 Series", "5 Series", "X1", "X3"],
    # ... complete list in config.py
}

# Search criteria
VEHICLE_SEARCH_CRITERIA = {
    "fuel_type": "Petrol",
    "transmission": "Manual", 
    "distance": 25,  # miles from postcode
    "min_year": 2018
}
```

### Analysis Thresholds
```python
# Profit analysis (in analyser.py)
CONFIG = {
    'recon_cost': 300,                 # Fixed reconditioning cost
    'excellent_margin_pct': 0.25,      # 25%+ = excellent deal
    'good_margin_pct': 0.20,           # 20%+ = good deal  
    'negotiation_margin_pct': 0.15,    # 15%+ = negotiation opportunity
    'min_cash_margin': 800,            # Minimum £800 profit
}
```

### Environment Variables
```bash
# Required for operation
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key  
SUPABASE_SERVICE_ROLE_KEY=your_service_key
RESEND_API_KEY=your_resend_key
```

## 🌐 API Endpoints

### Start Server
```bash
python src/api.py
# Runs on http://localhost:8000
```

### Key Endpoints
```bash
# Get deals with filtering
GET /deals?make=Ford&model=Fiesta&rating=excellent&limit=50

# Get deal statistics  
GET /stats

# Trigger manual scraping
POST /scrape {"model": "Swift", "max_results": 100}

# Health check
GET /health
```

## 📊 JSON Archives

Each scraping run creates a comprehensive archive in `archive/`:

```json
{
  "scrape_metadata": {
    "timestamp": "20250622_143052",
    "groups_processed": 50,
    "total_vehicles_scraped": 3420,
    "quality_deals_found": 150,
    "runtime_minutes": 25.3
  },
  "deals": [...],  // Complete deal data + analytics
  "session_summary": {...}
}
```

**Storage**: ~0.2MB per archive • Perfect for quarterly analytics • Auto-cleanup tools included

## 🔧 Archive Management

```bash
# View archive summary
python archive/manage_archives.py --summary

# Clean up old files (30+ days)
python archive/manage_archives.py --cleanup 30
```

## 🚀 Production Notes

- **Service Role Key**: Essential for bypassing RLS during sync operations
- **Intelligent Sync**: Never deletes all deals due to downtime (production-safe)
- **Error Handling**: Robust retry logic and comprehensive logging
- **Scalability**: Current system handles 242 models, expandable to 968 (4x)

---

**🚗 Car Dealer Bot - Automated Intelligence for Profitable Car Trading**
