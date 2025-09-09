# Car Deal Finder

Advanced vehicle analysis system that scrapes used car listings, then uses machine learning to find underpriced vehicles and predict profit margins.

## 🔍 How It Works

```
                                   ┌─────────────────┐
                                   │     Scraper     │
                                   │  High-Speed API │
                                   └────────┬────────┘
                                            │
                                            ▼
┌─────────────────────┐          ┌─────────────────────┐         ┌───────────────────┐
│  Price Markers      │◄─────────┤   ML Processing     ├────────►│  Database Storage │
│  Parallel Scraping  │          │   & Deal Analysis   │         │  & Notifications  │
└─────────────────────┘          └─────────────────────┘         └───────────────────┘
```

### Core Process Flow

1. **Scraping**: Multi-process AutoTrader API client retrieves vehicle listings with minimal overhead
2. **Retail Price Analysis**: Sub-process scrapes retail price markers from dealer listings to improve prediction accurary
3. **ML Deal Analysis**: Machine learning models calculate market values and identify profitable opportunities
4. **Profit Calculation**: Algorithms determine profit margins with statistical confidence

## 📊 Machine Learning System

The integrated ML system offers major advantages over typical rule-based deal finders:

- **Accurate Market Value Prediction**: Uses actual dealer price markers to train models
- **Multi-Factor Analysis**: Considers year, mileage, fuel type, transmission, spec, and engine size
- **Parallel Processing**: High-throughput design processes hundreds of listings in minutes

### ML Architecture

```python
# ML processor with parallel price marker scraping
def process_car_model(make, model):
    # 1. Scrape all listings (dealers and private)
    all_listings = scrape_listings(make, model)
    
    # 2. Separate dealer and private listings
    dealer_listings = filter_by_seller_type(all_listings, "Dealer")
    private_listings = filter_by_seller_type(all_listings, "Private")
    
    # 3. Enrich dealer listings with price markers (parallel processing)
    enriched_dealers = enrich_with_price_markers(dealer_listings)
    
    # 4. Train ML model on dealer data
    model, scaler = train_model(make, model, enriched_dealers)
    
    # 5. Predict profit on private listings
    predictions = predict_profit_margins(private_listings, model, scaler)
    
    # 6. Filter for profitable deals and notify
    profitable_deals = filter_profitable_deals(predictions)
```

## 🚀 Features

- **Optimized Scraping Engine**: Super fast processing, optimised as much as possible
- **Parallel Processing**: Multi-core architecture for price marker extraction
- **Real Market Data**: Uses actual dealer price markers rather than estimates
- **Spec-Based Analysis**: Considers premium trims (GTI, R-line, etc.) in valuations
- **Profit Prediction**: Calculates potential profit with statistical confidence
- **Adaptive Filtering**: Intelligent deal quality classification

## 📁 Core Components

### Core Scraping & Analysis
- **`retail_price_scraper.py`** - High-speed parallel price marker extraction
- **`ML_analyser.py`** - Machine learning system for profit prediction
- **`scraper.py`** - High-performance AutoTrader API client

### Supporting Components
- **`data_adapter.py`** - Data format standardization for ML processing
- **`network_requests.py`** - Low-level network operations
- **`config.py`** - Application configuration

## 🔧 Running the System

```bash
# Install dependencies
pip install -r config/requirements.txt

# Basic run (all vehicles)
python src/scraper.py

# Test mode (limited makes/models)
python src/scraper.py --test

# ML analysis with price markers
python tests/ML_analyser.py --make BMW --model "3 Series"
```

## ⚙️ Configuration

### Target Vehicles
```python
# Vehicle makes and models to scrape and analyze
TARGET_VEHICLES = {
    "Ford": ["Fiesta", "Focus", "Kuga"],
    "BMW": ["1 Series", "3 Series", "X1"],
    "Audi": ["A1", "A3", "A4", "Q3"],
    # Plus many more popular models
}
```

### ML Configuration
```python
# ML analysis thresholds
ML_CONFIG = {
    'profit_threshold': 0.15,      # Minimum 15% profit margin
    'high_profit_threshold': 0.25, # 25%+ is a great deal
    'min_profit_amount': 800,      # Minimum £800 profit
    'engine_size_weight': 1.2,     # Weighting factor for engine size
    'spec_weight': 1.5             # Weighting factor for vehicle spec
}
```

## 🔮 Future Enhancements

- **AI Spec Analysis**: Advanced text processing of vehicle specifications
- **ANPR Integration**: Extract numberplate information from listing images
- **Cross-Platform Integration**: Expand to other used car marketplaces