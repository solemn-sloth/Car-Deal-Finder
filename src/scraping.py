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
from src.output_manager import get_output_manager

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
    
    def __init__(self, connection_pool_size=10, export_predictions=False, force_retrain=False):
        # Initialize proxy manager for Cloudflare bypass (API-only mode)
        self.proxy_manager = None
        self._proxy_status_message = ""
        try:
            self.proxy_manager = ProxyManager()
            if self.proxy_manager.proxies:
                self._proxy_status_message = ""  # Silent proxy operations
            else:
                self._proxy_status_message = ""  # Silent proxy operations
        except Exception as e:
            self._proxy_status_message = ""  # Silent proxy operations
            self.proxy_manager = None

        # Store connection pool size for worker initialization
        self.connection_pool_size = connection_pool_size
        self.export_predictions = export_predictions
        self.force_retrain = force_retrain
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
            # Get output manager for final result display
            output_manager = get_output_manager()

            # Use multi-worker API scraping with proper anti-detection
            import time

            # Track scraping time
            scraping_start_time = time.time()

            # Create temporary API client to use the multi-worker scraping method
            from services.network_requests import AutoTraderAPIClient
            temp_client = AutoTraderAPIClient(
                connection_pool_size=self.connection_pool_size,
                optimize_connection=True,
                verify_ssl=False,
                proxy_manager=self.proxy_manager
            )

            # Create progress callback for API scraping
            def api_progress_callback(completed, total_tasks, status=None):
                if completed == 0:
                    output_manager.progress_update(0, total_tasks, "api_scraping", 0, "Starting...")
                else:
                    elapsed_time = time.time() - scraping_start_time
                    speed = (completed * 60 / elapsed_time) if elapsed_time > 0 else 0
                    # Note: completed is now total listings scraped, not tasks completed
                    # We can't reliably calculate ETA since we don't know total expected listings
                    # Just show current progress with rate
                    output_manager.progress_update(completed, total_tasks, "api_scraping", speed, "")

            # Use parallel multi-worker scraping for better anti-detection
            raw_vehicles = temp_client.get_all_cars_parallel(
                make=group.make,
                model=group.model,
                postcode="HP13 7LW",  # From original config
                min_year=group.search_criteria['year_from'],
                max_year=group.search_criteria['year_to'],
                max_mileage=group.search_criteria['maximum_mileage'],
                max_pages=None,  # No limit - scrape all available cars
                use_parallel=True,  # Force parallel mode
                test_mode=False,  # Enable proper delays
                progress_callback=api_progress_callback
            )

            # Calculate elapsed time
            scraping_elapsed = time.time() - scraping_start_time

            # Show final scraping result (replaces progress)
            output_manager = get_output_manager()
            output_manager.scraping_result(len(raw_vehicles), elapsed_time=scraping_elapsed)
            
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
                    output_manager = get_output_manager()
                    output_manager.scraping_result(len(raw_vehicles), missing_fuel_data)
                
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
            output_manager = get_output_manager()

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
                        try:
                            enhanced_analyse_listings(variant_vehicles, verbose=False, training_mode=False, export_predictions=self.export_predictions, force_retrain=self.force_retrain)
                        except Exception as ml_error:
                            error_msg = f"ML analysis failed for {config['variant_name']}: {str(ml_error)}"
                            errors.append(error_msg)
                            output_manager.ml_model_error(f"Analysis failed for {config['variant_name']}: {str(ml_error)}")
                            # Log the error but continue processing
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"Enhanced analysis failed: {ml_error}")
                            # Continue with unanalyzed vehicles (they won't have ML predictions but can still be processed)
                        
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
            
            # Complete ML processing section
            output_manager.ml_model_complete()

            # Print market value summary
            avg_market_value = 0
            if count_with_market_value > 0:
                avg_market_value = total_market_value / count_with_market_value

            # Analysis results will be displayed by the analyser, not here
            
            self.session_data['total_quality_deals'] += total_quality_deals
            
            output_manager = get_output_manager()
            output_manager.group_complete(group.make, group.model)
            # Separator handled by group_complete method
            
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
        # Startup banner is now handled by run_smart_grouped_scraping function
        
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
                output_manager = get_output_manager()
                output_manager.error(f"No groups found matching make='{filter_make}' model='{filter_model}'")
                return {'session_summary': {}, 'results': []}
            else:
                output_manager = get_output_manager()
                output_manager.group_filtering(len(filtered_groups), filter_make, filter_model)
        
        # Apply max_groups limit
        if max_groups:
            groups_to_process = filtered_groups[:max_groups]
        else:
            groups_to_process = filtered_groups
        
        results = []
        start_time = time.time()
        
        for i, group in enumerate(groups_to_process, 1):
            output_manager = get_output_manager()
            output_manager.group_start(i, len(groups_to_process), group.make, group.model)
            
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

def run_smart_grouped_scraping(max_groups=None, test_mode=False, connection_pool_size=10, filter_make=None, filter_model=None, export_predictions=False, force_retrain=False):
    """
    Run the complete smart grouped scraping and notification pipeline.

    Args:
        max_groups: Limit number of groups (for testing)
        test_mode: If True, don't send notifications
        connection_pool_size: Size of the connection pool for HTTP requests (default: 10)
        filter_make: Filter to specific make (e.g., "BMW")
        filter_model: Filter to specific model (e.g., "3 Series")
        export_predictions: Export ML predictions to JSON with features
        force_retrain: Force ML model retraining regardless of model age
    """
    # Startup banner is now handled by main.py automation orchestrator
    # Initialize orchestrator silently with connection pooling
    orchestrator = SmartGroupingOrchestrator(connection_pool_size=connection_pool_size, export_predictions=export_predictions, force_retrain=force_retrain)
    start_time = datetime.now()

    try:
        results = orchestrator.run_full_orchestration(max_groups=max_groups, filter_make=filter_make, filter_model=filter_model)
        
        # Compile all quality deals
        all_quality_deals = []
        for result in results['results']:
            for variant_key, variant_data in result.processed_variants.items():
                all_quality_deals.extend(variant_data['quality_deals'])

        # Clean deals for archive storage
        def clean_deals_for_archive(deals):
            """Clean and format deals for archive storage with proper field naming and formatting."""
            clean_deals = []
            for deal in deals:
                # Only keep essential fields with proper naming and formatting
                clean_deal = {
                    'deal_id': deal.get('deal_id', ''),
                    'url': deal.get('url', ''),
                    'price': deal.get('price', ''),
                    'year': deal.get('year', ''),
                    'mileage': deal.get('mileage', ''),
                    'make': deal.get('make', ''),
                    'model': deal.get('model', ''),
                    'seller_type': deal.get('seller_type', ''),
                    'location': deal.get('location', ''),
                    'engine_size': deal.get('engine_size', ''),
                    'fuel_type': deal.get('fuel_type', ''),
                    'transmission': deal.get('transmission', ''),
                    'doors': deal.get('doors', ''),
                    'image_url': deal.get('image_url', ''),
                    'image_url_2': deal.get('image_url_2', ''),
                    # Renamed and formatted prediction fields
                    'predicted_market_value': round(deal.get('enhanced_retail_estimate', 0), 2),
                    'predicted_margin_pct': round(deal.get('enhanced_gross_margin_pct', 0), 3),
                    'predicted_profit': round(deal.get('enhanced_gross_cash_profit', 0), 0),
                    'enhanced_rating': deal.get('enhanced_rating', '')
                }
                clean_deals.append(clean_deal)
            return clean_deals

        # Only save archive if there are actual quality deals found
        if all_quality_deals:
            # Clean deals for archive storage
            clean_archive_deals = clean_deals_for_archive(all_quality_deals)

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
                'deals': clean_archive_deals,
                'session_summary': results['session_data']
            }

            # Create archive/outputs directory if it doesn't exist
            archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive', 'outputs')
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
                        'engine_size', 'fuel_type', 'transmission', 'doors', 'mileage', 'price_numeric',
                        'predicted_market_value', 'predicted_margin_pct', 'predicted_profit',
                        'enhanced_rating', 'location', 'seller_type', 'url', 'image_url', 'image_url_2',
                        'test_record'
                    ]
                    
                    # Remove all fields that aren't in the allowed list
                    fields_to_remove = [field for field in list(deal_copy.keys()) if field not in allowed_fields]
                    for field in fields_to_remove:
                        deal_copy.pop(field, None)
                    
                    
                    # Clean price-related fields to extract only numeric values
                    price_fields = ['price', 'price_numeric', 'predicted_market_value', 'predicted_profit']
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
                    if 'predicted_margin_pct' in deal_copy and deal_copy['predicted_margin_pct']:
                        try:
                            deal_copy['predicted_margin_pct'] = float(deal_copy['predicted_margin_pct'])
                        except (ValueError, TypeError):
                            deal_copy['predicted_margin_pct'] = None
                            
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
                    output_manager = get_output_manager()
                    output_manager.error(f"Sync failed: {sync_result.get('error', 'Unknown error')}")
                    return results
                
                # Send notifications
                notification_pipeline = DealNotificationPipeline()
                notification_result = notification_pipeline.process_daily_notifications()
                
                if not notification_result.get('success'):
                    output_manager = get_output_manager()
                    output_manager.warning(f"Notification issue: {notification_result.get('error', 'Unknown')}")
            else:
                output_manager = get_output_manager()
                output_manager.info(f"TEST MODE: Would have stored {len(all_quality_deals)} deals")
        else:
            output_manager = get_output_manager()
        
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
        
        # Print formatted summary using centralized output
        output_manager = get_output_manager()
        summary_data = {
            'quality_deals': quality_deal_count,
            'total_vehicles': total_vehicles,
            'deal_percentage': deal_percentage,
            'db_added': db_added,
            'db_updated': db_updated,
            'db_removed': db_removed,
            'db_total': db_total,
            'notification_status': notification_status,
            'duration_minutes': duration
        }
        output_manager.session_summary(summary_data)
        
        return results
        
    except Exception as e:
        output_manager = get_output_manager()
        output_manager.error(f"Orchestration failed: {e}")
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
        output_manager = get_output_manager()
        output_manager.error(f"Failed: {e}")
        exit(1)


# ────────────────────────────────────────────────────────────
# Testing Functions (Optional - can be commented out in production)
# ────────────────────────────────────────────────────────────

def test_configuration_grouping():
    """Test the configuration grouping logic"""
    # Test function - suppress output
    pass
    
    orchestrator = SmartGroupingOrchestrator()
    
    # Test function - suppress output
    pass
    # Test function - suppress output
    pass
    
    # Show sample groups
    # Test function - suppress output
    pass
    for i, group in enumerate(orchestrator.groups[:5]):
        # Test function - suppress output
        pass
        for variant in group.specific_configs[:2]:  # Show first 2 variants
            # Test function - suppress output
            pass
    
    return orchestrator


def test_single_group_scraping():
    """Test scraping a single group to validate the process"""
    # Test function - suppress output
    pass
    
    orchestrator = SmartGroupingOrchestrator()
    
    # Pick a reliable group for testing (Toyota)
    toyota_groups = [g for g in orchestrator.groups if g.make == 'Toyota']
    if not toyota_groups:
        # Test function - suppress output
        pass
        return None
    
    test_group = toyota_groups[0]  # Test first Toyota model
    # Test function - suppress output
    pass
    
    result = orchestrator.scrape_group(test_group)
    
    # Test function - suppress output
    pass
    # Test function - suppress output
    pass
    # Test function - suppress output
    pass
    # Test function - suppress output
    pass
    # Test function - suppress output
    pass
    
    if result.errors:
        # Test function - suppress output
        pass
        for error in result.errors:
            # Test function - suppress output
            pass
    
    return result


if __name__ == "__main__":
    main()