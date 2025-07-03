#!/usr/bin/env python3
"""
Archive management script for car deal JSON files.
Provides utilities to clean up old archives and generate summaries.
"""

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path

class ArchiveManager:
    def __init__(self, archive_dir="archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
    
    def list_archives(self):
        """List all archive files with metadata."""
        archives = []
        pattern = str(self.archive_dir / "car_deals_archive_*.json")
        
        for file_path in glob.glob(pattern):
            file_path = Path(file_path)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Extract timestamp from filename
            filename = file_path.name
            timestamp_str = filename.replace('car_deals_archive_', '').replace('.json', '')
            
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                age_days = (datetime.now() - timestamp).days
                
                archives.append({
                    'file_path': file_path,
                    'timestamp': timestamp,
                    'age_days': age_days,
                    'size_mb': size_mb,
                    'filename': filename
                })
            except ValueError:
                print(f"⚠️  Skipping file with invalid timestamp: {filename}")
        
        return sorted(archives, key=lambda x: x['timestamp'], reverse=True)
    
    def get_total_storage(self):
        """Calculate total storage used by archives."""
        archives = self.list_archives()
        total_mb = sum(a['size_mb'] for a in archives)
        return total_mb, len(archives)
    
    def cleanup_old_archives(self, keep_days=30, dry_run=True):
        """Remove archives older than specified days."""
        archives = self.list_archives()
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        files_to_delete = [a for a in archives if a['timestamp'] < cutoff_date]
        total_size_to_free = sum(a['size_mb'] for a in files_to_delete)
        
        print(f"🗑️  Cleanup analysis (keep last {keep_days} days):")
        print(f"   Files to delete: {len(files_to_delete)}")
        print(f"   Storage to free: {total_size_to_free:.2f} MB")
        
        if files_to_delete:
            for file_info in files_to_delete:
                if dry_run:
                    print(f"   [DRY RUN] Would delete: {file_info['filename']} ({file_info['size_mb']:.2f} MB)")
                else:
                    try:
                        file_info['file_path'].unlink()
                        print(f"   ✅ Deleted: {file_info['filename']}")
                    except Exception as e:
                        print(f"   ❌ Failed to delete {file_info['filename']}: {e}")
        else:
            print("   No files to delete.")
        
        return len(files_to_delete), total_size_to_free
    
    def generate_summary_report(self):
        """Generate a summary of all archives."""
        archives = self.list_archives()
        total_mb, total_count = self.get_total_storage()
        
        print("📊 Archive Summary Report")
        print("=" * 40)
        print(f"Total archives: {total_count}")
        print(f"Total storage: {total_mb:.2f} MB ({total_mb/1024:.3f} GB)")
        
        if archives:
            oldest = archives[-1]
            newest = archives[0]
            print(f"Date range: {oldest['timestamp'].strftime('%Y-%m-%d')} to {newest['timestamp'].strftime('%Y-%m-%d')}")
            print(f"Average size: {total_mb/total_count:.2f} MB per archive")
        
        print("\n📅 Recent archives:")
        for archive in archives[:5]:  # Show last 5
            deals_count = "unknown"
            try:
                with open(archive['file_path'], 'r') as f:
                    data = json.load(f)
                    deals_count = len(data.get('deals', []))
            except:
                pass
            
            print(f"   {archive['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                  f"{archive['size_mb']:.2f} MB | {deals_count} deals | "
                  f"{archive['age_days']} days old")
    
    def extract_quarterly_summary(self, quarter_start_date):
        """Extract summary data for a specific quarter."""
        quarter_end = quarter_start_date + timedelta(days=90)
        archives = self.list_archives()
        
        quarter_archives = [
            a for a in archives 
            if quarter_start_date <= a['timestamp'] < quarter_end
        ]
        
        if not quarter_archives:
            print(f"No archives found for quarter starting {quarter_start_date.strftime('%Y-%m-%d')}")
            return None
        
        # Aggregate data
        total_deals = 0
        all_makes = set()
        price_data = []
        
        for archive_info in quarter_archives:
            try:
                with open(archive_info['file_path'], 'r') as f:
                    data = json.load(f)
                    deals = data.get('deals', [])
                    total_deals += len(deals)
                    
                    for deal in deals:
                        all_makes.add(deal.get('make', 'Unknown'))
                        if deal.get('price_numeric'):
                            price_data.append(deal['price_numeric'])
            except Exception as e:
                print(f"⚠️  Error reading {archive_info['filename']}: {e}")
        
        summary = {
            'quarter_start': quarter_start_date.strftime('%Y-%m-%d'),
            'archives_count': len(quarter_archives),
            'total_deals': total_deals,
            'unique_makes': len(all_makes),
            'avg_price': sum(price_data) / len(price_data) if price_data else 0,
            'price_range': [min(price_data), max(price_data)] if price_data else [0, 0]
        }
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Car Deal Archive Manager')
    parser.add_argument('--list', action='store_true', help='List all archives')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', help='Clean up archives older than DAYS')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted (use with --cleanup)')
    parser.add_argument('--summary', action='store_true', help='Show archive summary report')
    
    args = parser.parse_args()
    
    manager = ArchiveManager()
    
    if args.list:
        manager.generate_summary_report()
    elif args.cleanup:
        manager.cleanup_old_archives(keep_days=args.cleanup, dry_run=args.dry_run)
    elif args.summary:
        manager.generate_summary_report()
    else:
        print("Use --help to see available options")

if __name__ == "__main__":
    main()
