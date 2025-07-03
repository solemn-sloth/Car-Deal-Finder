# Car Dealer Bot - Server Monitoring & Management Guide

## 🚀 System Overview

---

## 🔐 Server Access

### SSH Connection
```bash
# Connect to your server
ssh -i id_rsa_backup root@49.13.192.23

# Set correct permissions for SSH key (if needed)
chmod 600 id_rsa_backup
```

### Server Details
- **IP Address**: `49.13.192.23`
- **Username**: `root`
- **SSH Key**: `id_rsa_backup` (in your project root)
- **Backend Location**: `/root/Car/src/`

---

## 📁 Current Directory Structure

```
/root/Car/
├── src/                                    # Main application code
│   ├── scraper.py                         # NEW: Main entry point 
│   ├── scrape_grouping.py                 # NEW: Smart grouping orchestrator
│   ├── json_data_adapter.py               # NEW: Data conversion 
│   ├── deal_notifications.py              # NEW: Email pipeline 
│   ├── network_scraper.py                 # High-speed AutoTrader API client
│   ├── analyser.py                        # ML-powered market analysis
│   ├── supabase_storage.py                # Database operations with intelligent sync
│   ├── notification_service.py            # Email sending utilities
│   ├── api.py                             # REST API endpoints
│   ├── config.py                          # Vehicle configurations
│   └── [DELETED: main.py, old_scraper.py, main_smart_grouped.py]
├── archive/                                # NEW: JSON archiving system
│   ├── manage_archives.py                 # Archive cleanup tools
│   ├── car_deals_archive_YYYYMMDD_HHMMSS.json  # Daily archives
│   └── ARCHIVE_README.md                  # Archive documentation
├── requirements.txt                        # Python dependencies
└── .env                                   # Environment variables
```

---

## 🚀 Deployment Commands

### 1. Connect to Server
```bash
ssh -i id_rsa_backup root@49.13.192.23
cd /root/Car/
```

### 2. Update Code from Git
```bash
# Pull latest changes
git pull origin main

# Check what files changed
git log --oneline -5

# Verify new file structure
ls -la src/
```

### 3. Update Dependencies (if needed)
```bash
# Install any new dependencies
pip install -r requirements.txt

# Verify Python can import new modules
python3 -c "from src.scraper import *; print('✅ Import successful')"
```

### 4. Test New System
```bash
cd src/

# Test with very limited scope
python3 scraper.py --max-groups 1 --test-mode

# Test with normal scope
python3 scraper.py --max-groups 5 --test-mode

# Check if archives are being created
ls -la ../archive/
```

### 5. Update Cron Job (if needed)
```bash
# Edit cron job to use new entry point
crontab -e

# Change from: python3 main.py
# Change to:   python3 scraper.py

# View updated cron job
crontab -l
```
```

---

## 🔍 System Monitoring Commands

### 1. Check System Status

```bash
# Navigate to application directory
cd /root/Car/src

# Quick system health check (test new entry point)
python3 scraper.py --max-groups 1 --test-mode

# Test run with limited scope
python3 scraper.py --max-groups 3 --test-mode

# Full production test (careful - this runs the full system)
python3 scraper.py --test-mode
```

### 2. Monitor Logs & Archives

```bash
# View live logs (press Ctrl+C to exit)
tail -f scraper.log

# Check JSON archives
ls -la ../archive/car_deals_archive_*.json

# View latest archive metadata
python3 -c "
import json, glob
files = glob.glob('../archive/car_deals_archive_*.json')
if files:
    with open(max(files), 'r') as f:
        data = json.load(f)
        print(f'Last run: {data[\"scrape_metadata\"][\"datetime\"]}')
        print(f'Deals found: {data[\"scrape_metadata\"][\"quality_deals_found\"]}')
        print(f'Runtime: {data[\"scrape_metadata\"][\"runtime_minutes\"]:.1f} min')
"

# View last 50 log entries
tail -50 scraper.log

# Search for intelligent sync messages
grep -i "intelligent\|sync" scraper.log

# Search for errors
grep -i "error\|failed" scraper.log
```

### 3. Check Database Synchronization

```bash
# Test database connection with service role key
python3 -c "
from supabase_storage import SupabaseStorage
storage = SupabaseStorage()
print('✅ Database connection successful')
"

# Verify intelligent sync is working
grep -i "sync completed\|added.*deals\|removed.*deals" scraper.log | tail -5
```

### 4. Archive Management

```bash
# Check archive storage usage
cd /root/Car/archive
du -sh .

# View archive summary
python3 manage_archives.py --summary

# Clean up old archives (30+ days)
python3 manage_archives.py --cleanup 30 --dry-run
```

### 3. Check Cron Job Status

```bash
# View current cron jobs
crontab -l

# Check if cron service is running
systemctl status cron

# View cron logs
grep CRON /var/log/syslog | tail -20

# Check last cron execution
ls -la scraper.log
```

### 4. Performance Monitoring

```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top

# Check Python processes
ps aux | grep python
```

---

## 🎯 Production Operations

### Manual Execution

```bash
# Navigate to app directory
cd /root/Car/src

# Test run (safe - no database writes)
python3 scraper.py --max-groups 5 --test-mode

# Limited production run (5 groups)
python3 scraper.py --max-groups 5

# Full production run
python3 scraper.py

# Check latest archive created
ls -la ../archive/car_deals_archive_*.json | tail -1
```

### Cron Job Management

```bash
# Current cron job should use new entry point
crontab -l

# Edit cron job if needed (change main.py to scraper.py)
crontab -e

# Example updated cron job:
# 0 2 * * * cd /root/Car/src && python3 scraper.py >> scraper.log 2>&1
```

### Emergency Procedures

```bash
# If intelligent sync fails, check service role key
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
print('✅ Service role key configured' if key else '❌ Service role key missing')
"

# Rollback to previous version (if needed)
git log --oneline -5
git checkout <previous_commit_hash>

# Test rollback
python3 scraper.py --max-groups 1 --test-mode
```
crontab -l
# Output: 0 0 * * * /root/Car/src/run_production_scraper.sh

# Edit cron job (if needed)
crontab -e

# Disable cron job temporarily
crontab -l > cron_backup.txt
crontab -r

# Restore cron job
crontab cron_backup.txt
```

---

## 📊 Understanding the Output

### Successful Run Indicators
```
🚀 Smart Grouped Car Deal Scraper
📊 Created 242 configuration groups
💡 Smart grouping will save 402 API calls
✅ API call successful: X vehicles found
📊 Conversion rate: 100.0%
🎯 Orchestration Complete!
🎉 Smart grouped scraping complete!
```

### Warning Signs
```
❌ API call failed
❌ Conversion failed
❌ Smart grouped scraping failed
❌ Missing modules
❌ Test run failed
```

### Performance Metrics to Monitor
- **Groups processed**: Should be 242 for full run
- **API calls made**: Should be ~242 (vs old system's 644)
- **Conversion rate**: Should be 100% (excluding empty objects)
- **Quality deals found**: Varies based on market conditions
- **Runtime**: Should be 60-80 minutes (vs 4+ hours before)

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors
```bash
# Check Python dependencies
python3 -c "import requests, numpy, supabase; print('All imports OK')"

# Reinstall if needed
pip install requests numpy supabase python-dotenv --break-system-packages
```

#### 2. Permission Errors
```bash
# Fix script permissions
chmod +x /root/Car/src/run_production_scraper.sh
chmod +x /root/Car/src/deploy_production.sh
```

#### 3. API Connection Issues
```bash
# Test basic connectivity
curl -I https://www.autotrader.co.uk

# Check DNS resolution
nslookup www.autotrader.co.uk

# Test with verbose output
python3 main.py --max-groups 1 --test-mode
```

#### 4. Database Connection Issues
```bash
# Check environment variables
grep -v "^#" /root/Car/.env

# Test Supabase connection
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('SUPABASE_URL:', os.getenv('SUPABASE_URL'))
print('SUPABASE_KEY:', 'SET' if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else 'MISSING')
"
```

### Emergency Rollback
```bash
# If new system fails, rollback to old system
cd /root/Car/src
cp main_old.py main.py
---

## 📈 Performance Comparison

| Metric | Old System | New System (Intelligent Sync) | Improvement |
|--------|------------|-------------------------------|-------------|
| **Synchronization** | Time-based deletion (24h) | Intelligent comparison | Production-safe |
| **Data Loss Risk** | High (deletes all if down) | Zero (only removes sold deals) | Critical fix |
| **RLS Handling** | User-level permissions | Service role bypass | Reliable |
| **Archive Storage** | None | JSON archives (~0.2MB/run) | Analytics ready |
| **File Structure** | Complex legacy names | Clean, descriptive names | Maintainable |

---

## 🚨 Alert Thresholds

Monitor these metrics and investigate if:
- **No archives created** for 1+ days
- **Sync failure** messages in logs
- **All deals deleted** (should never happen with intelligent sync)
- **Service role key errors**
- **Import errors** after deployment

---

## 📞 Quick Reference Commands

```bash
# Essential commands for daily monitoring
ssh -i id_rsa_backup root@49.13.192.23
cd /root/Car/src
tail -20 scraper.log                         # Check recent activity
python3 scraper.py --max-groups 3 --test-mode  # Quick test
ls -la ../archive/ | tail -5                # Check archives
crontab -l                                   # Verify cron job
```

---

## 🎉 Success Indicators

Your system is working perfectly when you see:
1. ✅ Daily cron job execution with `scraper.py` 
2. ✅ JSON archives being created in `/root/Car/archive/`
3. ✅ "Intelligent sync completed" messages in logs
4. ✅ Service role key working (no RLS errors)
5. ✅ Quality deals being found and synced
6. ✅ No "deleted all deals" errors

The new Intelligent Sync Car Dealer Bot is now production-safe and will never lose all deals due to downtime! 🚀
