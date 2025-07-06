# Database Column Detection Test Script

## Overview

This script (`test_saving_deals.py`) provides comprehensive debugging and analysis for database column detection issues when saving car deals to the database. It helps identify why the script might not be correctly identifying all database columns.

## Features

### 1. Database Connection Testing
- Tests Supabase connection establishment
- Verifies table accessibility 
- Shows current record count

### 2. Column Detection Analysis (`analyze_database_columns()`)
- Compares detected columns with expected columns
- Shows missing, extra, and matching columns
- Calculates detection success rate
- Provides detailed breakdown of column differences

### 3. Case Sensitivity & Whitespace Detection
- Identifies case sensitivity issues between expected and detected columns
- Detects whitespace issues in column names
- Provides recommendations for fixing naming conventions

### 4. Enhanced `get_valid_columns()` Method Debugging
- Provides verbose output about column retrieval process
- Shows raw database response data
- Displays query execution details
- Helps identify where the column detection is failing

### 5. Comprehensive Reporting
- Generates detailed debug reports
- Provides specific recommendations based on findings
- Supports both live database testing and mock mode

## Expected Columns

The script validates against these expected columns:
```
id, deal_id, created_at, updated_at, make, model, year, spec, registration, 
engine_size, fuel_type, transmission, body_type, doors, mileage, price_numeric, 
enhanced_retail_estimate, enhanced_net_sale_price, enhanced_gross_cash_profit, 
enhanced_gross_margin_pct, profit_potential_pct, absolute_profit, deal_rating, 
enhanced_rating, spec_analysis, location, distance, seller_type, title, 
subtitle, full_title, url, comparison_url, image_url, image_url_2, 
date_added, test_record, price, analysis_method
```

## Usage

### With Database Connection
```bash
# Set environment variables
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"

# Run the test
python test_saving_deals.py
```

### Mock Mode (No Database Connection)
```bash
# Just run without environment variables
python test_saving_deals.py
```

The script will automatically detect missing credentials and run in mock mode, showing example scenarios that might occur in real testing.

## Sample Output

### Mock Mode Output
```
🚀 Enhanced Database Column Detection Analysis
============================================================

📊 Mock Column Analysis Results:
   Expected columns: 39
   Mock detected columns: 23
   Matching columns: 23
   Missing columns:  16
   Extra columns:    0

❌ Missing Columns (16):
   • absolute_profit
   • analysis_method
   • comparison_url
   • deal_rating
   • enhanced_gross_cash_profit
   ...

📈 Mock Column Detection Success Rate: 59.0%
```

### Debugging Output
```
🛠️  Debugging get_valid_columns() Method
============================================================
🔄 Clearing column cache and performing fresh query...
📋 Querying information_schema.columns for table: 'car_deals'
📡 Query executed successfully
📊 Response type: <class 'postgrest.APIResponse'>
📊 Has data attribute: True
📊 Data length: 23
```

## Recommendations Generated

The script provides specific recommendations based on findings:

1. **Missing Columns**: Suggests checking table schema matches expectations
2. **Case Sensitivity Issues**: Recommends reviewing column naming conventions  
3. **Low Success Rate**: Suggests verifying table name and schema
4. **Query Failures**: Recommends checking database permissions

## Integration

This script can be integrated into CI/CD pipelines or run manually for debugging column detection issues. It provides actionable insights for fixing database schema mismatches.

## Files Modified

- `test_saving_deals.py` - New comprehensive test script
- `src/supabase_storage.py` - Fixed structural issues:
  - Fixed indentation in `get_valid_columns()` method
  - Fixed `filter_deal_fields()` method return statement
  - Fixed `save_deal()` method indentation

## Technical Details

The script:
- Uses the existing `SupabaseStorage` class to test column detection
- Queries `information_schema.columns` to retrieve actual database columns
- Compares against a predefined list of expected columns
- Provides detailed debugging output for troubleshooting
- Handles both live database connections and mock testing scenarios