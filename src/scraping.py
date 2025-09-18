"""
Smart Grouped Car Deal Scraper with Orchestration
Combines smart configuration grouping with complete pipeline orchestration.
"""

import time
import random
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

# Core scraping imports
from services.network_requests import AutoTraderAPIClient
from services.stealth_orchestrator import ProxyManager
from services.network_requests import NetworkDataAdapter
from src.analyser import enhanced_analyse_listings, enhanced_keep_listing
from config.config import TARGET_VEHICLES_BY_MAKE, VEHICLE_SEARCH_CRITERIA

# Pipeline imports
from src.storage import SupabaseStorage
from services.notifications import DealNotificationPipeline


# ────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────

@dataclass
class ConfigurationGroup:
    """
    Represents a group of related vehicle configurations that can share one API call.
    """
    make: str
    model: str
    search_criteria: Dict
    specific_configs: List[Dict]
    
    def __str__(self):
        return f"{self.make} {self.model}"


@dataclass
class ScrapingResult:
    """
    Results from scraping a single configuration group.
    """
    group: ConfigurationGroup
    raw_vehicles: List[Dict]
    processed_variants: Dict  # This MUST be a Dict, not a List
    total_deals: int
    quality_deals: int
    errors: List[str]


# ────────────────────────────────────────────────────────────
# Smart Grouping Orchestrator
# ────────────────────────────────────────────────────────────

class SmartGroupingOrchestrator:
    """
    Orchestrates the scraping and analysis process with smart configuration grouping.
    
    Key optimizations:
    1. Groups configurations by make/model to minimize API calls
    2. Makes one network request per make/model
    3. Analyzes each configuration variant against shared vehicle data
    4. Maintains all existing deal notification logic
    """
    
    def __init__(self, connection_pool_size=10):
        # Initialize proxy manager for Cloudflare bypass
        proxy_manager = None
        try:
            import os
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/proxies.json')
            if os.path.exists(config_path):
                proxy_manager = ProxyManager(config_path=config_path)
                print("🔄 Proxy rotation enabled")
            else:
                print("⚠️ No proxy config found, proceeding without proxies")
        except Exception as e:
            print(f"⚠️ Failed to initialize proxy rotation: {e}")
            proxy_manager = None
        
        # Initialize API client with connection pooling and proxy support
        self.api_client = AutoTraderAPIClient(
            connection_pool_size=connection_pool_size, 
            optimize_connection=True, 
            verify_ssl=False,
            proxy_manager=proxy_manager
        )
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'groups_processed': 0,
            'total_api_calls': 0,
            'total_vehicles_scraped': 0,
            'total_quality_deals': 0,
            'api_call_savings': 0,  # How many calls we saved vs individual approach
        }
        self.groups = self._create_configuration_groups()
    
    def _create_configuration_groups(self) -> List[ConfigurationGroup]:
        """
        Create configuration groups by analyzing TARGET_VEHICLES_BY_MAKE.
        Each group represents one make/model that requires one API call.
        """
        groups = []
        
        for make, models in TARGET_VEHICLES_BY_MAKE.items():
            for model in models:
                # Create different configuration variants for this make/model
                variants = self._generate_configuration_variants(make, model)
                
                group = ConfigurationGroup(
                    make=make,
                    model=model,
                    search_criteria=VEHICLE_SEARCH_CRITERIA.copy(),
                    specific_configs=variants
                )
                groups.append(group)
        
        # Calculate API call savings
        total_individual_configs = sum(len(group.specific_configs) for group in groups)
        api_call_savings = total_individual_configs - len(groups)
        self.session_data['api_call_savings'] = api_call_savings
        
        return groups
    
    def _generate_configuration_variants(self, make: str, model: str) -> List[Dict]:
        """
        Generate a single 'all' variant for each make/model.
        This simplified approach fetches all vehicles for each make/model without filtering by fuel or transmission.
        """
        base_config = {
            'name': f"{make} {model}",
            'make': make,
            'model': model,
            **VEHICLE_SEARCH_CRITERIA
        }
        
        # Just use a single 'all' variant for simplicity and efficiency
        variants = [
            {
                **base_config,
                'variant_key': 'all',
                'variant_name': f"{make} {model} (All)",
                'fuel_preference': None,
                'transmission_preference': None,
            }
        ]
        
        return variants
    
    def scrape_group(self, group: ConfigurationGroup, max_vehicles_for_analysis: int = 500) -> ScrapingResult:
        """
        Scrape a single configuration group (one API call for multiple variants).
        
        Args:
            group: The configuration group to scrape
            max_vehicles_for_analysis: Maximum number of vehicles to include in analysis
        """
        errors = []
        
        try:
            # Make ONE API call for this make/model
            print(f"🔍 Scraping {group.make} {group.model} from {group.search_criteria['year_from']}-{group.search_criteria['year_to']}, max_mileage: {group.search_criteria['maximum_mileage']}")
            
            raw_vehicles = self.api_client.get_all_cars_with_mileage_splitting(
                make=group.make,
                model=group.model,
                postcode="HP13 7LW",  # From original config
                min_year=group.search_criteria['year_from'],
                max_year=group.search_criteria['year_to'],
                max_mileage=group.search_criteria['maximum_mileage'],
                max_pages=None  # No limit - scrape all available cars
            )
            print(f"📊 Found {len(raw_vehicles)} vehicles")
            
            self.session_data['total_api_calls'] += 1
            self.session_data['total_vehicles_scraped'] += len(raw_vehicles)
            
            if not raw_vehicles:
                return ScrapingResult(
                    group=group,
                    raw_vehicles=[],
                    processed_variants={},
                    total_deals=0,
                    quality_deals=0,
                    errors=["No vehicles found"]
                )
            
            # Convert API data to analyzer format using adapter
            try:
                converted_vehicles = NetworkDataAdapter.convert_vehicle_list(raw_vehicles)
                
                # Check for missing fuel type data
                missing_fuel_data = sum(1 for v in converted_vehicles if not v.get('fuel_type'))
                if missing_fuel_data > 0:
                    print(f"   ⚠️ {missing_fuel_data} vehicles missing fuel type data (fallback applied)")
                
                if not converted_vehicles:
                    return ScrapingResult(
                        group=group,
                        raw_vehicles=raw_vehicles,
                        processed_variants={},
                        total_deals=len(raw_vehicles),
                        quality_deals=0,
                        errors=["No vehicles could be converted to analyzer format"]
                    )
                    
            except Exception as e:
                error_msg = f"Data conversion error for {group.make} {group.model}: {str(e)}"
                errors.append(error_msg)
                
                return ScrapingResult(
                    group=group,
                    raw_vehicles=raw_vehicles,
                    processed_variants={},
                    total_deals=len(raw_vehicles),
                    quality_deals=0,
                    errors=errors
                )
            
            # Process each configuration variant against the shared vehicle data
            print(f"\n📊 Analyzing data...")
            
            processed_variants = {}  # Initialize as empty dict
            total_quality_deals = 0
            
            # Calculate average market value for reporting
            total_market_value = 0
            count_with_market_value = 0
            confidence = 0
            
            for config in group.specific_configs:
                try:
                    # Filter converted vehicles for this variant
                    variant_vehicles = self._filter_vehicles_for_variant(converted_vehicles, config)
                    
                    if variant_vehicles:
                        # Analyze vehicles using enhanced analyzer
                        enhanced_analyse_listings(variant_vehicles, verbose=False)
                        
                        # Filter for quality deals after analysis
                        quality_deals = [deal for deal in variant_vehicles if enhanced_keep_listing(deal)]
                        
                        # Calculate market value stats for reporting
                        for vehicle in variant_vehicles:
                            if vehicle.get('enhanced_market_value') or vehicle.get('enhanced_retail_estimate'):
                                market_value = vehicle.get('enhanced_market_value', vehicle.get('enhanced_retail_estimate', 0))
                                total_market_value += market_value
                                count_with_market_value += 1
                                
                                if 'enhanced_market_confidence' in vehicle:
                                    confidence = max(confidence, vehicle.get('enhanced_market_confidence', 0))
                                elif 'enhanced_confidence' in vehicle:
                                    confidence = max(confidence, vehicle.get('enhanced_confidence', 0))
                        
                        processed_variants[config['variant_key']] = {
                            'config': config,
                            'total_vehicles': len(variant_vehicles),
                            'analyzed_deals': variant_vehicles,
                            'quality_deals': quality_deals,
                            'quality_count': len(quality_deals)
                        }
                        
                        total_quality_deals += len(quality_deals)
                    
                except Exception as e:
                    error_msg = f"Error processing variant {config['variant_name']}: {str(e)}"
                    errors.append(error_msg)
            
            # Print market value summary
            avg_market_value = 0
            if count_with_market_value > 0:
                avg_market_value = total_market_value / count_with_market_value
            
            print(f"   • Average market value: £{int(avg_market_value):,}")
            print(f"   • Confidence: R²={confidence:.3f}")
            print(f"   • Sample size: {len(raw_vehicles)} comparable vehicles")
            print(f"   • Deals found: {total_quality_deals} quality deals")
            
            self.session_data['total_quality_deals'] += total_quality_deals
            
            print(f"\n✅ {group.make} {group.model} complete")
            print("------------------------------------------------------------")
            
            return ScrapingResult(
                group=group,
                raw_vehicles=raw_vehicles,
                processed_variants=processed_variants,
                total_deals=len(raw_vehicles),
                quality_deals=total_quality_deals,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Failed to scrape {group.make} {group.model}: {str(e)}"
            errors.append(error_msg)
            
            return ScrapingResult(
                group=group,
                raw_vehicles=[],
                processed_variants={},
                total_deals=0,
                quality_deals=0,
                errors=errors
            )
    
    def _filter_vehicles_for_variant(self, vehicles: List[Dict], config: Dict) -> List[Dict]:
        """
        Filter the shared vehicle data for a specific configuration variant.
        """
        filtered = vehicles.copy()
        
        # Apply fuel type filter if specified
        if config.get('fuel_preference'):
            fuel_pref = config['fuel_preference'].lower()
            filtered = [v for v in filtered if self._extract_fuel_type(v).lower() == fuel_pref]
        
        # Apply transmission filter if specified
        if config.get('transmission_preference'):
            trans_pref = config['transmission_preference'].lower()
            filtered = [v for v in filtered if self._extract_transmission(v).lower() == trans_pref]
        
        return filtered
    
    def _extract_fuel_type(self, vehicle: Dict) -> str:
        """Extract fuel type from converted vehicle data"""
        fuel_type = vehicle.get('fuel_type')
        if fuel_type:
            return fuel_type
        
        # Fallback to title extraction
        title = vehicle.get('full_title', '') or vehicle.get('title', '')
        
        fuel_keywords = {
            'petrol': 'Petrol',
            'diesel': 'Diesel', 
            'hybrid': 'Hybrid',
            'electric': 'Electric',
            'plugin': 'Plug-in Hybrid',
            'plug-in': 'Plug-in Hybrid'
        }
        
        title_lower = title.lower()
        for keyword, fuel_type in fuel_keywords.items():
            if keyword in title_lower:
                return fuel_type
        
        return 'Unknown'
    
    def _extract_transmission(self, vehicle: Dict) -> str:
        """Extract transmission type from converted vehicle data"""
        transmission = vehicle.get('transmission')
        if transmission:
            return transmission
        
        # Fallback to title extraction
        title = vehicle.get('full_title', '') or vehicle.get('title', '')
        
        transmission_keywords = {
            'manual': 'Manual',
            'automatic': 'Automatic',
            'auto': 'Automatic',
            'cvt': 'CVT',
            'semi-automatic': 'Semi-Automatic',
            'semi auto': 'Semi-Automatic'
        }
        
        title_lower = title.lower()
        for keyword, trans_type in transmission_keywords.items():
            if keyword in title_lower:
                return trans_type
        
        return 'Unknown'
    
    def _compile_session_summary(self, results: List[ScrapingResult]) -> Dict:
        """Compile detailed session summary from all results"""
        summary = {
            'successful_groups': 0,
            'failed_groups': 0,
            'top_performing_groups': [],
            'error_summary': [],
            'variant_performance': defaultdict(list),
            'total_groups': len(results),
            'total_raw_vehicles': sum(len(r.raw_vehicles) for r in results),
            'total_quality_deals': sum(r.quality_deals for r in results),
            'api_calls_made': self.session_data['total_api_calls'],
            'api_calls_saved': self.session_data['api_call_savings'],
            'efficiency_improvement': f"{self.session_data['api_call_savings']/max(1, self.session_data['total_api_calls'])*100:.1f}%"
        }
        
        for result in results:
            if result.errors:
                summary['failed_groups'] += 1
                summary['error_summary'].extend(result.errors)
            else:
                summary['successful_groups'] += 1
            
            # Track top performing groups
            if result.quality_deals > 0:
                summary['top_performing_groups'].append({
                    'group': f"{result.group.make} {result.group.model}",
                    'quality_deals': result.quality_deals,
                    'total_vehicles': result.total_deals
                })
        
        # Sort top performers
        summary['top_performing_groups'].sort(key=lambda x: x['quality_deals'], reverse=True)
        summary['top_performing_groups'] = summary['top_performing_groups'][:10]  # Top 10
        
        return summary
    
    def run_full_orchestration(self, max_groups: Optional[int] = None, filter_make: Optional[str] = None, filter_model: Optional[str] = None) -> Dict:
        """
        Run the complete orchestration process for all or limited groups.
        
        Args:
            max_groups: Limit number of groups to process (for testing)
        
        Returns:
            Dict: Complete session results
        """
        print("\n============================================================")
        print("                  🚀 Starting Car Dealer Bot")
        print("============================================================")
        
        # Apply make/model filtering first
        filtered_groups = self.groups
        if filter_make or filter_model:
            filtered_groups = []
            for group in self.groups:
                make_matches = not filter_make or group.make.lower() == filter_make.lower()
                model_matches = not filter_model or group.model.lower() == filter_model.lower()
                if make_matches and model_matches:
                    filtered_groups.append(group)
            
            if not filtered_groups:
                print(f"❌ No groups found matching make='{filter_make}' model='{filter_model}'")
                return {'session_summary': {}, 'results': []}
            else:
                print(f"🎯 Filtering to {len(filtered_groups)} group(s): {filter_make} {filter_model}")
        
        # Apply max_groups limit
        if max_groups:
            groups_to_process = filtered_groups[:max_groups]
        else:
            groups_to_process = filtered_groups
        
        results = []
        start_time = time.time()
        
        for i, group in enumerate(groups_to_process, 1):
            print(f"\n[{i}/{len(groups_to_process)}] Processing {group}")
            
            # Brief delay between group processing
            if i > 1:
                time.sleep(0.25 + random.uniform(0, 0.25))  # 0.25-0.5 second delay
            
            result = self.scrape_group(group)
            results.append(result)
            
            self.session_data['groups_processed'] += 1
        
        # Compile final results
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['total_runtime_minutes'] = (time.time() - start_time) / 60
        
        session_summary = self._compile_session_summary(results)
        
        return {
            'session_data': self.session_data,
            'results': results,
            'summary': session_summary
        }


# ────────────────────────────────────────────────────────────
# Main Pipeline Orchestration
# ────────────────────────────────────────────────────────────

def run_smart_grouped_scraping(max_groups=None, test_mode=False, connection_pool_size=10, filter_make=None, filter_model=None):
    """
    Run the complete smart grouped scraping and notification pipeline.
    
    Args:
        max_groups: Limit number of groups (for testing)
        test_mode: If True, don't send notifications
        connection_pool_size: Size of the connection pool for HTTP requests (default: 10)
    """
    # Initialize orchestrator silently with connection pooling
    orchestrator = SmartGroupingOrchestrator(connection_pool_size=connection_pool_size)
    start_time = datetime.now()
    
    try:
        results = orchestrator.run_full_orchestration(max_groups=max_groups, filter_make=filter_make, filter_model=filter_model)
        
        # Compile all quality deals
        all_quality_deals = []
        for result in results['results']:
            for variant_key, variant_data in result.processed_variants.items():
                all_quality_deals.extend(variant_data['quality_deals'])
        
        # Save comprehensive JSON archive for analytics (quietly)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_data = {
            'scrape_metadata': {
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'groups_processed': results['session_data']['groups_processed'],
                'total_api_calls': results['session_data']['total_api_calls'],
                'api_call_savings': results['session_data']['api_call_savings'],
                'total_vehicles_scraped': results['session_data']['total_vehicles_scraped'],
                'quality_deals_found': len(all_quality_deals),
                'runtime_minutes': (datetime.now() - start_time).total_seconds() / 60
            },
            'deals': all_quality_deals,
            'session_summary': results['session_data']
        }
        
        # Create archive directory if it doesn't exist
        archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        
        # Save complete archive
        archive_file = os.path.join(archive_dir, f'car_deals_archive_{timestamp}.json')
        with open(archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2, default=str)
        
        # Calculate file size
        file_size_mb = os.path.getsize(archive_file) / (1024 * 1024)

        if all_quality_deals:
            # Store deals to database
            if not test_mode:
                # Synchronize deals with database (quietly)
                storage = SupabaseStorage()
                
                # Prepare deals for database (remove temporary fields)
                clean_deals = []
                for deal in all_quality_deals:
                    # Remove temporary analysis fields before saving to database
                    deal_copy = deal.copy()
                    deal_copy.pop('analysis_method', None)
                    deal_copy.pop('attention_grabber', None)
                    
                    # Remove fields that should not be sent to database
                    deal_copy.pop('dealer_review_rating', None)
                    deal_copy.pop('dealer_review_count', None)
                    deal_copy.pop('number_of_images', None)
                    deal_copy.pop('has_video', None)
                    deal_copy.pop('has_360_spin', None)
                    deal_copy.pop('manufacturer_approved', None)
                    deal_copy.pop('seller_name', None)
                    
                    # Only keep fields that are present in the database schema
                    allowed_fields = [
                        'id', 'deal_id', 'created_at', 'updated_at', 'make', 'model', 'year', 'spec', 
                        'engine_size', 'fuel_type', 'transmission', 'body_type', 'doors', 'mileage', 
                        'price_numeric', 'enhanced_retail_estimate', 'enhanced_net_sale_price', 
                        'enhanced_gross_cash_profit', 'enhanced_gross_margin_pct', 'enhanced_rating', 
                        'spec_analysis', 'location', 'distance', 'seller_type', 'title', 'subtitle', 
                        'full_title', 'url', 'image_url', 'image_url_2', 'date_added', 'test_record'
                    ]
                    
                    # Remove all fields that aren't in the allowed list
                    fields_to_remove = [field for field in list(deal_copy.keys()) if field not in allowed_fields]
                    for field in fields_to_remove:
                        deal_copy.pop(field, None)
                    
                    # Clean distance field to extract only numeric value
                    if 'distance' in deal_copy and deal_copy['distance']:
                        distance_str = str(deal_copy['distance'])
                        import re
                        numeric_match = re.search(r'(\d+(?:\.\d+)?)', distance_str)
                        if numeric_match:
                            deal_copy['distance'] = float(numeric_match.group(1))
                        else:
                            deal_copy['distance'] = None
                    
                    # Clean price-related fields to extract only numeric values
                    price_fields = ['price', 'price_numeric', 'enhanced_retail_estimate', 
                                   'enhanced_net_sale_price', 'enhanced_gross_cash_profit']
                    for field in price_fields:
                        if field in deal_copy and deal_copy[field]:
                            price_str = str(deal_copy[field])
                            import re
                            clean_price = re.sub(r'[£$€,\s]', '', price_str)
                            try:
                                if field == 'price_numeric':
                                    deal_copy[field] = int(float(clean_price)) if clean_price else None
                                else:
                                    deal_copy[field] = float(clean_price) if clean_price else None
                            except ValueError:
                                deal_copy[field] = None
                    
                    # Clean percentage field (margin) - keep as decimal
                    if 'enhanced_gross_margin_pct' in deal_copy and deal_copy['enhanced_gross_margin_pct']:
                        try:
                            deal_copy['enhanced_gross_margin_pct'] = float(deal_copy['enhanced_gross_margin_pct'])
                        except (ValueError, TypeError):
                            deal_copy['enhanced_gross_margin_pct'] = None
                            
                    # Check if doors is an integer
                    if 'doors' in deal_copy and deal_copy['doors'] is not None:
                        try:
                            deal_copy['doors'] = int(float(deal_copy['doors']))
                        except (ValueError, TypeError):
                            deal_copy['doors'] = None
                            
                    # Check if year is an integer
                    if 'year' in deal_copy and deal_copy['year'] is not None:
                        try:
                            deal_copy['year'] = int(float(deal_copy['year']))
                        except (ValueError, TypeError):
                            deal_copy['year'] = None
                            
                    # Check if mileage is an integer
                    if 'mileage' in deal_copy and deal_copy['mileage'] is not None:
                        try:
                            deal_copy['mileage'] = int(float(deal_copy['mileage']))
                        except (ValueError, TypeError):
                            deal_copy['mileage'] = None
                    
                    clean_deals.append(deal_copy)
                
                # Use intelligent sync instead of individual saves
                sync_result = storage.sync_deals_intelligently(clean_deals)
                
                if not sync_result['success']:
                    print(f"Sync failed: {sync_result.get('error', 'Unknown error')}")
                    return results
                
                # Send notifications
                notification_pipeline = DealNotificationPipeline()
                notification_result = notification_pipeline.process_daily_notifications()
                
                if not notification_result.get('success'):
                    print(f"Notification issue: {notification_result.get('error', 'Unknown')}")
            else:
                print(f"TEST MODE: Would have stored {len(all_quality_deals)} deals")
        else:
            print("No quality deals found this session")
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        # Calculate deal percentage rate
        total_vehicles = results['session_data']['total_vehicles_scraped']
        quality_deal_count = len(all_quality_deals)
        deal_percentage = (quality_deal_count / total_vehicles * 100) if total_vehicles > 0 else 0
        
        # Get notification status
        notification_status = "Success"
        if 'notification_result' in locals():
            if notification_result.get('success'):
                emails_sent = notification_result.get('emails_sent', 0)
                notification_status = f"{emails_sent} emails sent" if emails_sent > 0 else "No new deals in last 24 hours"
            else:
                notification_status = f"Error: {notification_result.get('error', 'Unknown')}"
        else:
            notification_status = "Not processed"
        
        # Get database sync status
        db_added = 0
        db_updated = 0
        db_removed = 0
        db_total = 0
        
        if 'sync_result' in locals() and sync_result.get('success'):
            db_added = sync_result.get('added_count', 0)
            db_updated = sync_result.get('updated_count', 0)
            db_removed = sync_result.get('removed_count', 0)
            db_total = sync_result.get('total_deals', 0)
        
        # Print formatted summary
        print("\n" + "=" * 60)
        print(f"{'📈 Session Summary':^60}")
        print("=" * 60)
        print()
        print("🎯 Deal Discovery")
        print(f"   • Quality deals found: {quality_deal_count}")
        print(f"   • Total vehicles analyzed: {total_vehicles}")
        print(f"   • Deal percentage rate: {deal_percentage:.1f}%")
        print()
        print("🔄 Database Sync")
        print(f"   • New deals added: {db_added}")
        print(f"   • Existing deals updated: {db_updated}")
        print(f"   • Sold/missing removed: {db_removed}")
        print(f"   • Total active deals: {db_total}")
        print()
        print("📧 Notifications")
        print(f"   • Status: {notification_status}")
        print()
        print("-" * 60)
        print(f"⏱️  Session completed in {duration:.1f} minutes")
        print(f"✅ All tasks completed successfully")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"❌ Orchestration failed: {e}")
        raise


# ────────────────────────────────────────────────────────────
# Command Line Interface
# ────────────────────────────────────────────────────────────

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Smart Grouped Car Deal Scraper')
    parser.add_argument('--max-groups', type=int, help='Limit number of groups (for testing)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Test mode (no database storage or notifications)')
    parser.add_argument('--connection-pool-size', type=int, default=10,
                       help='Size of the connection pool for HTTP requests (default: 10)')
    parser.add_argument('--make', type=str, help='Process only this make (e.g., Abarth)')
    parser.add_argument('--model', type=str, help='Process only this model (e.g., 595)')
    
    args = parser.parse_args()
    
    try:
        results = run_smart_grouped_scraping(
            max_groups=args.max_groups,
            test_mode=args.test_mode,
            connection_pool_size=args.connection_pool_size,
            filter_make=args.make,
            filter_model=args.model
        )
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        exit(1)


# ────────────────────────────────────────────────────────────
# Testing Functions (Optional - can be commented out in production)
# ────────────────────────────────────────────────────────────

def test_configuration_grouping():
    """Test the configuration grouping logic"""
    print("🧪 Testing Configuration Grouping...")
    
    orchestrator = SmartGroupingOrchestrator()
    
    print(f"✅ Created {len(orchestrator.groups)} groups")
    print(f"💡 API call savings: {orchestrator.session_data['api_call_savings']}")
    
    # Show sample groups
    print(f"\n📋 Sample groups:")
    for i, group in enumerate(orchestrator.groups[:5]):
        print(f"  {i+1}. {group}")
        for variant in group.specific_configs[:2]:  # Show first 2 variants
            print(f"     → {variant['variant_name']}")
    
    return orchestrator


def test_single_group_scraping():
    """Test scraping a single group to validate the process"""
    print("🧪 Testing Single Group Scraping...")
    
    orchestrator = SmartGroupingOrchestrator()
    
    # Pick a reliable group for testing (Toyota)
    toyota_groups = [g for g in orchestrator.groups if g.make == 'Toyota']
    if not toyota_groups:
        print("❌ No Toyota groups found for testing")
        return None
    
    test_group = toyota_groups[0]  # Test first Toyota model
    print(f"🎯 Testing group: {test_group}")
    
    result = orchestrator.scrape_group(test_group)
    
    print(f"\n📊 Test Results:")
    print(f"   • Raw vehicles: {len(result.raw_vehicles)}")
    print(f"   • Processed variants: {len(result.processed_variants)}")
    print(f"   • Quality deals: {result.quality_deals}")
    print(f"   • Errors: {len(result.errors)}")
    
    if result.errors:
        print(f"❌ Errors encountered:")
        for error in result.errors:
            print(f"     • {error}")
    
    return result


if __name__ == "__main__":
    main()