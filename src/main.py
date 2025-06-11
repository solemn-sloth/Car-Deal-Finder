"""
Main script for AutoTrader scraper with concurrent processing using multiprocessing.
Orchestrates the scraping, analysis, and storage of vehicle data.
"""
from playwright.sync_api import sync_playwright
import random
import time
import os
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from typing import List, Dict, Tuple
import signal
import sys

# Import modules
from config import get_vehicle_configs
from scraper import scrape_vehicle_listings
from analyzer import analyse_listings, keep_listing, comparison_url

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AutoTrader Scraper')
    parser.add_argument('--model', help='Specific model to scrape (leave empty for all)')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    parser.add_argument('--concurrent', type=int, default=3, 
                        help='Number of concurrent scrapers (default: 3)')
    return parser.parse_args()

def init_worker():
    """Initialize worker process - ignore SIGINT to allow clean shutdown"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def scrape_config_worker_with_status(args: Tuple[int, Dict, dict]) -> Tuple[int, str, List[Dict]]:
    """
    Worker function with status tracking
    """
    worker_id, config, status_dict = args
    try:
        # Update status
        status_dict[worker_id] = f"Starting {config['name']}..."
        
        with sync_playwright() as playwright:
            # Create a callback to update status during scraping
            class StatusUpdater:
                def __init__(self, worker_id, config_name, status_dict):
                    self.worker_id = worker_id
                    self.config_name = config_name
                    self.status_dict = status_dict
                
                def update(self, count):
                    self.status_dict[self.worker_id] = f"Scraping {count} {self.config_name} vehicles"
            
            updater = StatusUpdater(worker_id, config['name'], status_dict)
            
            # Modified scrape function call - we'll update the scraper to accept a callback
            vehicles_data = scrape_vehicle_listings(playwright, config, worker_id)
            
            # Clear status when done
            status_dict[worker_id] = "Idle"
            return worker_id, config['name'], vehicles_data
            
    except Exception as e:
        status_dict[worker_id] = "Idle"
        with open("error_log.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Error processing {config['name']}: {str(e)}\n")
        return worker_id, config['name'], []

def print_worker_status(status_dict, completed_tasks, total_configs, start_time):
    """Print current worker status"""
    # Clear screen
    print("\033[H\033[J", end="")
    
    # Header
    print(f"AutoTrader Scraper - Processing {total_configs} vehicle configurations")
    
    # Calculate elapsed time in minutes
    elapsed_seconds = time.time() - start_time
    elapsed_minutes = elapsed_seconds / 60
    print(f"Time elapsed: {elapsed_minutes:.1f} minutes")
    
    print("\n" + "="*60 + "\n")
    
    # Active Workers
    print("Active Workers:")
    print("-" * 60)
    active_count = 0
    for worker_id in sorted(status_dict.keys()):
        status = status_dict[worker_id]
        if status and status != "Idle":
            print(f"[Worker {worker_id}] {status}")
            active_count += 1
        elif status == "Idle":
            print(f"[Worker {worker_id}] Idle")
    
    # Completed Tasks
    if completed_tasks:
        print(f"\nCompleted Tasks ({len(completed_tasks)} of {total_configs}):")
        print("-" * 60)
        # Show last 10 completed tasks
        for task in completed_tasks[-10:]:
            print(task)
    
    print("\n" + "="*60)
    sys.stdout.flush()

def process_vehicles_pipeline(all_vehicles: List[Tuple[str, List[Dict]]], args):
    """
    Process all scraped vehicles through analysis and storage pipeline.
    """
    # Clear screen for pipeline processing
    print("\033[H\033[J", end="")
    
    # Flatten all vehicles with their config names
    vehicles_to_process = []
    
    for config_name, vehicles in all_vehicles:
        if vehicles:
            # Analyze vehicles silently (no print statements)
            analyse_listings(vehicles)
            
            # Filter for deals using the original function signature
            filtered = [v for v in vehicles if keep_listing(v)]
            
            # Add config name to each vehicle for tracking
            for v in filtered:
                v['config_name'] = config_name
                vehicles_to_process.append(v)
    
    if not vehicles_to_process:
        print("\nNo deals found across all configurations")
        return
    
    print(f"{'='*50}")
    print(f"Total deals found: {len(vehicles_to_process)}")
    print(f"{'='*50}")
    
    # Save to Supabase only (no ANPR processing)
    if vehicles_to_process:
        print(f"\nSaving {len(vehicles_to_process)} deals to Supabase...")
        
        # Clean up vehicles for Supabase storage (remove analysis-only fields)
        cleaned_vehicles = []
        for v in vehicles_to_process:
            cleaned_v = v.copy()
            # Remove fields not in the Supabase schema - these are analysis-only fields
            # Note: bucket_key is also removed since it's not in the current Supabase schema
            analysis_only_fields = [
                'config_name',           # Just for tracking
                'bucket_key',            # Not in current Supabase schema
                'retail_estimate',       # Added by analyzer
                'net_sale_price',        # Added by analyzer
                'gross_cash_profit',     # Added by analyzer
                'gross_margin_pct',      # Added by analyzer
                'rating',                # Added by analyzer (use deal_rating instead)
                'value_index',           # Added by analyzer
                'deal_percentage',       # Added by analyzer
                'dealer_avg_vi',         # Added by analyzer
                'price_per_10k_miles',   # Added by analyzer
                'deal_score',            # Added by analyzer
                'pricing_method',        # Added by enhanced analyzer
                'comparison_sample_size', # Added by enhanced analyzer
                'avg_similarity',        # Added by enhanced analyzer
                'effective_sample',      # Added by enhanced analyzer
                'year_range',            # Added by enhanced analyzer
                'similarity_weights',    # Added by enhanced analyzer
                'r_squared'              # Added by enhanced analyzer
            ]
            
            for field in analysis_only_fields:
                cleaned_v.pop(field, None)
            
            cleaned_vehicles.append(cleaned_v)
        
        try:
            from supabase_storage import SupabaseStorage
            supabase_storage = SupabaseStorage()
            result = supabase_storage.save_deals(cleaned_vehicles)
            print(f"✅ Successfully saved {result.get('saved', 0)}/{len(cleaned_vehicles)} deals to Supabase")
            
            if result.get('failed', 0) > 0:
                print(f"⚠️  {result.get('failed', 0)} deals failed to save")
                
        except Exception as e:
            print(f"❌ Error saving to Supabase: {e}")
            print("Note: Make sure Supabase credentials are configured")

def main():
    """Main function to run the AutoTrader scraper with concurrent processing"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create screenshots directory if it doesn't exist
    os.makedirs("screenshots", exist_ok=True)
    
    # Filter configurations based on command line arguments
    configs_to_process = get_vehicle_configs()
    if args.model:
        configs_to_process = [c for c in get_vehicle_configs() if args.model.lower() in c['name'].lower()]
        if not configs_to_process:
            print(f"No configuration found for model: {args.model}")
            return
    
    # Determine number of workers - removed cpu_count() limitation
    max_workers = min(args.concurrent, len(configs_to_process))
    
    # Print start message
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting AutoTrader scraper at {start_time_str}")
    print(f"Processing {len(configs_to_process)} vehicle configurations")
    print(f"Using {max_workers} concurrent scrapers")
    time.sleep(2)  # Brief pause before starting
    
    # Concurrent scraping using multiprocessing
    all_results = []
    start_time = time.time()
    completed_tasks = []
    
    # Create manager for shared status
    manager = Manager()
    status_dict = manager.dict()
    
    # Initialize status dict
    for i in range(1, max_workers + 1):
        status_dict[i] = "Idle"
    
    # Create a list of (worker_id, config, status_dict) tuples
    worker_configs = [(i % max_workers + 1, config, status_dict) 
                      for i, config in enumerate(configs_to_process)]
    
    try:
        # Create pool of workers
        with Pool(processes=max_workers, initializer=init_worker) as pool:
            try:
                # Start async processing
                result_objects = []
                for worker_config in worker_configs:
                    result = pool.apply_async(scrape_config_worker_with_status, (worker_config,))
                    result_objects.append(result)
                
                # Monitor progress
                while any(not r.ready() for r in result_objects):
                    print_worker_status(status_dict, completed_tasks, len(configs_to_process), start_time)
                    time.sleep(0.5)  # Update every 0.5 seconds
                    
                    # Check for completed tasks
                    for i, result in enumerate(result_objects):
                        if result.ready() and i < len(worker_configs):
                            try:
                                worker_id, config_name, vehicles = result.get(timeout=0.1)
                                if config_name not in [task.split(']')[0][1:] for task in completed_tasks]:
                                    if vehicles:
                                        completed_tasks.append(f"[{config_name}] Completed - found {len(vehicles)} vehicles")
                                    else:
                                        completed_tasks.append(f"[{config_name}] Completed - no vehicles found")
                            except:
                                pass
                
                # Collect all results
                for result in result_objects:
                    worker_id, config_name, vehicles = result.get()
                    all_results.append((config_name, vehicles))
                
                # Final status update
                print_worker_status(status_dict, completed_tasks, len(configs_to_process), start_time)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user, terminating workers...")
                pool.terminate()
                pool.join()
                return
                
    except Exception as e:
        print(f"\nError in multiprocessing: {e}")
        with open("error_log.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Multiprocessing error: {str(e)}\n")
    
    # Calculate scraping time
    scraping_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Scraping completed in {scraping_time:.1f} seconds")
    print(f"{'='*50}")
    time.sleep(2)  # Brief pause before processing pipeline
    
    # Process all vehicles through the pipeline
    if all_results:
        process_vehicles_pipeline(all_results, args)
    
    # Print final summary with timing
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"AutoTrader scraper completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"{'='*50}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        # Save error to log file
        with open("error_log.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Error: {str(e)}\n")
        print("Error has been logged to error_log.txt")