"""
Smart Configuration Grouping Orchestrator for AutoTrader scraper.
Groups vehicle configurations by make/model to minimize API calls and optimize analysis.
"""
import time
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import json
from datetime import datetime

from config import TARGET_VEHICLES_BY_MAKE, VEHICLE_SEARCH_CRITERIA
from network_scraper import AutoTraderAPIClient
from json_data_adapter import NetworkDataAdapter
from analyser import enhanced_analyse_listings, enhanced_keep_listing


@dataclass
class ConfigurationGroup:
    """Represents a group of configurations for the same make/model"""
    make: str
    model: str
    search_criteria: Dict
    specific_configs: List[Dict]  # Different fuel/transmission combinations for this make/model
    
    def __repr__(self):
        return f"ConfigurationGroup({self.make} {self.model}, {len(self.specific_configs)} variants)"


@dataclass
class ScrapingResult:
    """Results from scraping a make/model group"""
    group: ConfigurationGroup
    raw_vehicles: List[Dict]
    processed_variants: Dict[str, List[Dict]]  # variant_key -> analyzed vehicles
    total_deals: int
    quality_deals: int
    errors: List[str]


class SmartGroupingOrchestrator:
    """
    Orchestrates the scraping and analysis process with smart configuration grouping.
    
    Key optimizations:
    1. Groups configurations by make/model to minimize API calls
    2. Makes one network request per make/model
    3. Analyzes each configuration variant against shared vehicle data
    4. Maintains all existing deal notification logic
    """
    
    def __init__(self):
        self.api_client = AutoTraderAPIClient()
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
                # In this implementation, we'll create variants based on fuel/transmission preferences
                variants = self._generate_configuration_variants(make, model)
                
                group = ConfigurationGroup(
                    make=make,
                    model=model,
                    search_criteria=VEHICLE_SEARCH_CRITERIA.copy(),
                    specific_configs=variants
                )
                groups.append(group)
        
        print(f"📊 Created {len(groups)} configuration groups from {sum(len(models) for models in TARGET_VEHICLES_BY_MAKE.values())} make/model combinations")
        
        # Calculate API call savings
        total_individual_configs = sum(len(group.specific_configs) for group in groups)
        api_call_savings = total_individual_configs - len(groups)
        self.session_data['api_call_savings'] = api_call_savings
        
        print(f"💡 Smart grouping will save {api_call_savings} API calls ({len(groups)} calls vs {total_individual_configs} individual calls)")
        
        return groups
    
    def _generate_configuration_variants(self, make: str, model: str) -> List[Dict]:
        """
        Generate different configuration variants for a make/model.
        Each variant represents a different combination of preferences (fuel, transmission, etc.).
        """
        # For now, we'll create variants based on fuel/transmission preferences
        # This could be expanded to include other criteria in the future
        
        base_config = {
            'name': f"{make} {model}",
            'make': make,
            'model': model,
            **VEHICLE_SEARCH_CRITERIA
        }
        
        # Primary configuration (all vehicles)
        variants = [
            {
                **base_config,
                'variant_key': 'all',
                'variant_name': f"{make} {model} (All)",
                'fuel_preference': None,
                'transmission_preference': None,
            }
        ]
        
        # Additional variants for specific preferences (could be user-configurable)
        fuel_preferences = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
        transmission_preferences = ['Manual', 'Automatic']
        
        # For high-value makes, create more specific variants
        high_value_makes = ['Audi', 'BMW', 'Mercedes-Benz', 'Porsche', 'Jaguar', 'Lexus']
        
        if make in high_value_makes:
            for fuel in fuel_preferences:
                variants.append({
                    **base_config,
                    'variant_key': f'fuel_{fuel.lower()}',
                    'variant_name': f"{make} {model} ({fuel})",
                    'fuel_preference': fuel,
                    'transmission_preference': None,
                })
            
            for transmission in transmission_preferences:
                variants.append({
                    **base_config,
                    'variant_key': f'trans_{transmission.lower()}',
                    'variant_name': f"{make} {model} ({transmission})",
                    'fuel_preference': None,
                    'transmission_preference': transmission,
                })
        
        return variants
    
    def scrape_group(self, group: ConfigurationGroup) -> ScrapingResult:
        """
        Scrape a single configuration group (one API call for multiple variants).
        """
        print(f"\n🔍 Scraping {group.make} {group.model} ({len(group.specific_configs)} variants)...")
        
        errors = []
        
        try:
            # Make ONE API call for this make/model
            raw_vehicles = self.api_client.get_all_cars(
                make=group.make,
                model=group.model,
                postcode="HP13 7LW",  # From original config
                min_year=group.search_criteria['year_from'],
                max_year=group.search_criteria['year_to'],
                max_mileage=group.search_criteria['maximum_mileage'],
                max_pages=10
            )
            
            self.session_data['total_api_calls'] += 1
            self.session_data['total_vehicles_scraped'] += len(raw_vehicles)
            
            print(f"✅ API call successful: {len(raw_vehicles)} vehicles found")
            
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
                print(f"🔄 Converted {len(converted_vehicles)} of {len(raw_vehicles)} vehicles to analyzer format")
                
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
                print(f"❌ {error_msg}")
                
                return ScrapingResult(
                    group=group,
                    raw_vehicles=raw_vehicles,
                    processed_variants={},
                    total_deals=len(raw_vehicles),
                    quality_deals=0,
                    errors=errors
                )
            
            # Process each configuration variant against the shared vehicle data
            processed_variants = {}
            total_quality_deals = 0
            
            for config in group.specific_configs:
                try:
                    # Filter converted vehicles for this variant
                    variant_vehicles = self._filter_vehicles_for_variant(converted_vehicles, config)
                    
                    if variant_vehicles:
                        # Analyze vehicles using enhanced analyzer (it mutates the list)
                        enhanced_analyse_listings(variant_vehicles)
                        
                        # Filter for quality deals after analysis
                        quality_deals = [deal for deal in variant_vehicles if enhanced_keep_listing(deal)]
                        
                        processed_variants[config['variant_key']] = {
                            'config': config,
                            'total_vehicles': len(variant_vehicles),
                            'analyzed_deals': variant_vehicles,  # All vehicles are now analyzed
                            'quality_deals': quality_deals,
                            'quality_count': len(quality_deals)
                        }
                        
                        total_quality_deals += len(quality_deals)
                        
                        print(f"  📊 {config['variant_name']}: {len(variant_vehicles)} vehicles → {len(quality_deals)} quality deals")
                    
                except Exception as e:
                    error_msg = f"Error processing variant {config['variant_name']}: {str(e)}"
                    errors.append(error_msg)
                    print(f"  ❌ {error_msg}")
            
            self.session_data['total_quality_deals'] += total_quality_deals
            
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
            print(f"❌ {error_msg}")
            
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
        # First try the converted fuel_type field
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
        # First try the converted transmission field
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
    
    def run_full_orchestration(self, max_groups: Optional[int] = None) -> Dict:
        """
        Run the complete orchestration process for all or limited groups.
        
        Args:
            max_groups: Limit number of groups to process (for testing)
        
        Returns:
            Dict: Complete session results
        """
        print(f"\n🚀 Starting Smart Grouping Orchestration")
        print(f"📋 Processing {len(self.groups)} configuration groups")
        
        if max_groups:
            groups_to_process = self.groups[:max_groups]
            print(f"🔬 Testing mode: Limited to {max_groups} groups")
        else:
            groups_to_process = self.groups
        
        print(f"💾 API call optimization: {self.session_data['api_call_savings']} calls saved")
        print("="*60)
        
        results = []
        start_time = time.time()
        
        for i, group in enumerate(groups_to_process, 1):
            print(f"\n[{i}/{len(groups_to_process)}] Processing {group}")
            
            # Add some delay between requests to be respectful
            if i > 1:
                delay = 2 + random.uniform(0, 2)  # 2-4 second delay
                print(f"⏱️  Waiting {delay:.1f}s before next request...")
                time.sleep(delay)
            
            result = self.scrape_group(group)
            results.append(result)
            
            self.session_data['groups_processed'] += 1
            
            # Print progress summary
            elapsed = time.time() - start_time
            rate = i / elapsed * 60  # groups per minute
            eta = (len(groups_to_process) - i) / rate if rate > 0 else 0
            
            print(f"⏱️  Progress: {i}/{len(groups_to_process)} groups | {rate:.1f} groups/min | ETA: {eta:.1f}min")
        
        # Compile final results
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['total_runtime_minutes'] = (time.time() - start_time) / 60
        
        session_summary = self._compile_session_summary(results)
        
        print(f"\n🎯 Orchestration Complete!")
        print(f"📊 Session Summary:")
        print(f"   • Groups processed: {self.session_data['groups_processed']}")
        print(f"   • Total API calls: {self.session_data['total_api_calls']}")
        print(f"   • API calls saved: {self.session_data['api_call_savings']}")
        print(f"   • Total vehicles: {self.session_data['total_vehicles_scraped']}")
        print(f"   • Quality deals: {self.session_data['total_quality_deals']}")
        print(f"   • Runtime: {self.session_data['total_runtime_minutes']:.1f} minutes")
        
        return {
            'session_data': self.session_data,
            'results': results,
            'summary': session_summary
        }
    
    def _compile_session_summary(self, results: List[ScrapingResult]) -> Dict:
        """Compile detailed session summary from all results"""
        summary = {
            'successful_groups': 0,
            'failed_groups': 0,
            'top_performing_groups': [],
            'error_summary': [],
            'variant_performance': defaultdict(list)
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
            
            # Track variant performance
            for variant_key, variant_data in result.processed_variants.items():
                summary['variant_performance'][variant_key].append({
                    'group': f"{result.group.make} {result.group.model}",
                    'quality_deals': variant_data['quality_count'],
                    'total_vehicles': variant_data['total_vehicles']
                })
        
        # Sort top performers
        summary['top_performing_groups'].sort(key=lambda x: x['quality_deals'], reverse=True)
        summary['top_performing_groups'] = summary['top_performing_groups'][:10]  # Top 10
        
        return summary


# ────────────────────────────────────────────────────────────
# Testing and Validation Functions
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
    import random
    
    print("🔧 Smart Configuration Grouping - Orchestrator")
    print("="*60)
    
    # Test 1: Configuration grouping
    orchestrator = test_configuration_grouping()
    
    print("\n" + "="*60)
    
    # Test 2: Single group scraping
    test_result = test_single_group_scraping()
    
    print("\n" + "="*60)
    
    # Test 3: Limited orchestration (first 3 groups)
    if test_result and not test_result.errors:
        print("🚀 Running limited orchestration test (3 groups)...")
        full_results = orchestrator.run_full_orchestration(max_groups=3)
        
        print(f"\n🎯 Limited Test Complete!")
        print(f"📄 Results saved for {len(full_results['results'])} groups")
    else:
        print("⚠️  Skipping full orchestration due to single group test issues")
