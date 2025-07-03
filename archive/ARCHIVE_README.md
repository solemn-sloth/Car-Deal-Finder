# Car Deal Archive System

The scraper automatically saves comprehensive JSON archives of all scraped car deals for quarterly analytics and historical tracking.

## 📁 Archive Structure

Archives are saved to `backend/archive/` with the filename format:
```
car_deals_archive_YYYYMMDD_HHMMSS.json
```

Each archive contains:
```json
{
  "scrape_metadata": {
    "timestamp": "20250622_143052",
    "datetime": "2025-06-22T14:30:52",
    "groups_processed": 50,
    "total_api_calls": 1250,
    "api_call_savings": 850,
    "total_vehicles_scraped": 3420,
    "quality_deals_found": 150,
    "runtime_minutes": 25.3
  },
  "deals": [...],  // Complete deal data with analytics
  "session_summary": {...}
}
```

## 💾 Storage Requirements

Based on typical scraping results:

| Period | Storage | Archives |
|--------|---------|----------|
| Daily | ~0.2 MB | 1 file |
| Weekly | ~1.3 MB | 7 files |
| Monthly | ~5.7 MB | 30 files |
| Quarterly | ~17 MB | 90 files |
| Yearly | ~70 MB | 365 files |

**Note:** These are conservative estimates. Actual storage may vary based on deal volume.

## 🛠️ Archive Management

Use the `manage_archives.py` script to manage archive files:

### View Archives
```bash
python3 manage_archives.py --summary
```

### List All Archives  
```bash
python3 manage_archives.py --list
```

### Clean Up Old Archives (Dry Run)
```bash
python3 manage_archives.py --cleanup 30 --dry-run
```

### Actually Delete Old Archives
```bash
python3 manage_archives.py --cleanup 30
```

## 📊 Analytics Benefits

Each archive contains complete deal data including:

- **Raw Deal Data**: Make, model, price, mileage, location, etc.
- **Enhanced Analytics**: Profit margins, market value estimates, deal ratings
- **Market Intelligence**: Spec premiums, fuel type trends, geographic patterns
- **Performance Metrics**: API efficiency, scraping speed, quality ratios

## 🗂️ Recommended Cleanup Strategy

1. **Daily Archives**: Keep for 1 month (automatic cleanup)
2. **Monthly Summaries**: Extract and keep for 1 year
3. **Quarterly Reports**: Keep indefinitely for trend analysis
4. **Annual Summaries**: Compress and archive long-term

## 🚀 Using Archives for Analytics

Archives can be processed with any JSON-compatible analytics tool:

- **Python**: pandas, numpy, matplotlib
- **R**: jsonlite, dplyr, ggplot2  
- **Excel**: Power Query JSON import
- **Tableau**: JSON connector
- **Jupyter Notebooks**: Direct JSON loading

Example Python usage:
```python
import json
import pandas as pd

# Load archive
with open('archive/car_deals_archive_20250622_143052.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['deals'])

# Analyze trends
quarterly_trends = df.groupby('make')['enhanced_gross_margin_pct'].mean()
```

## 🔄 Automatic Archiving

Archives are created automatically on every scraping run. No manual intervention required. The system:

1. ✅ Saves complete deal data before database cleaning
2. ✅ Includes all analytics and metadata
3. ✅ Uses timestamped filenames for easy sorting
4. ✅ Calculates and reports file sizes
5. ✅ Works with both test and production modes
