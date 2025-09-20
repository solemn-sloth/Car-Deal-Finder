#!/usr/bin/env python3
"""
Weekly Training Orchestrator
Manages the weekly retail price scraping and universal model training process.
Processes each make/model sequentially to prevent failures from affecting the entire run.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TARGET_VEHICLES_BY_MAKE
from config.config import mark_retail_scraping_started, mark_retail_scraping_complete
from services.ML_model import train_universal_model
from src.analyser import scrape_listings, enrich_with_price_markers, filter_listings_by_seller_type

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from training a specific make/model group."""
    make: str
    model: str
    success: bool
    dealer_count: int
    enriched_count: int
    error_message: str = ""
    processing_time: float = 0.0


class WeeklyTrainingOrchestrator:
    """
    Orchestrates weekly retail price scraping and universal model training.
    
    Features:
    - Sequential processing of make/model groups
    - Failure isolation (one group failure doesn't break others)
    - Accumulates dealer data across all successful groups
    - Trains universal model on combined dataset
    - Comprehensive logging and error handling
    """
    
    def __init__(self, max_pages_per_model: int = None, verify_ssl: bool = False, 
                 use_proxy: bool = True, test_mode: bool = False):
        """
        Initialize weekly training orchestrator.
        
        Args:
            max_pages_per_model: Maximum pages to scrape per make/model (None for all)
            verify_ssl: Whether to verify SSL certificates
            use_proxy: Whether to use proxy rotation
            test_mode: Run in test mode (affects caching and delays)
        """
        self.max_pages_per_model = max_pages_per_model
        self.verify_ssl = verify_ssl
        self.use_proxy = use_proxy
        self.test_mode = test_mode
        
        # Training state
        self.accumulated_dealer_data: List[Dict[str, Any]] = []
        self.training_results: List[TrainingResult] = []
        self.start_time = None
        self.total_groups = 0
        self.successful_groups = 0
        
    def run_weekly_training(self) -> bool:
        """
        Run the complete weekly retail price scraping and model training process.
        
        Returns:
            bool: True if training completed successfully, False otherwise
        """
        self.start_time = datetime.now()
        
        try:
            logger.info("🚀 Starting weekly retail price scraping and universal model training")
            
            # Mark scraping as started in schedule
            mark_retail_scraping_started()
            
            # Process each make/model group sequentially
            self._process_all_vehicle_groups()
            
            # Train universal model on accumulated data
            training_success = self._train_universal_model()
            
            # Generate summary report
            self._generate_training_report()
            
            # Mark training as complete
            total_vehicles = len(self.accumulated_dealer_data)
            success_rate = self.successful_groups / max(1, self.total_groups)
            
            notes = f"Processed {self.successful_groups}/{self.total_groups} groups ({success_rate:.1%} success rate)"
            mark_retail_scraping_complete(
                success=training_success and success_rate > 0.5,  # Success if model trained and >50% groups processed
                total_vehicles=total_vehicles,
                notes=notes
            )
            
            return training_success and success_rate > 0.5
            
        except Exception as e:
            logger.error(f"Fatal error in weekly training orchestrator: {e}")
            mark_retail_scraping_complete(success=False, total_vehicles=0, notes=f"Fatal error: {str(e)}")
            return False
    
    def _process_all_vehicle_groups(self):
        """Process all make/model groups sequentially."""
        self.total_groups = sum(len(models) for models in TARGET_VEHICLES_BY_MAKE.values())
        current_group = 0
        
        logger.info(f"📊 Processing {self.total_groups} make/model groups for retail price scraping")
        
        for make, models in TARGET_VEHICLES_BY_MAKE.items():
            for model in models:
                current_group += 1
                
                logger.info(f"\n{'='*60}")
                logger.info(f"GROUP {current_group}/{self.total_groups}: {make} {model}")
                logger.info(f"{'='*60}")
                
                try:
                    result = self._process_single_group(make, model)
                    self.training_results.append(result)
                    
                    if result.success:
                        self.successful_groups += 1
                        logger.info(f"✅ {make} {model}: {result.enriched_count} enriched dealers added to training set")
                    else:
                        logger.warning(f"❌ {make} {model}: {result.error_message}")
                    
                    # Progress update
                    progress = current_group / self.total_groups
                    logger.info(f"📈 Progress: {current_group}/{self.total_groups} ({progress:.1%}) - "
                               f"Successful: {self.successful_groups} - "
                               f"Total dealers: {len(self.accumulated_dealer_data)}")
                    
                except Exception as e:
                    error_result = TrainingResult(
                        make=make, model=model, success=False,
                        dealer_count=0, enriched_count=0,
                        error_message=f"Unexpected error: {str(e)}"
                    )
                    self.training_results.append(error_result)
                    logger.error(f"❌ {make} {model}: Unexpected error: {e}")
                    # Continue with next group
                    continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SCRAPING COMPLETE: {self.successful_groups}/{self.total_groups} groups successful")
        logger.info(f"Total accumulated dealer listings: {len(self.accumulated_dealer_data)}")
        logger.info(f"{'='*60}\n")
    
    def _process_single_group(self, make: str, model: str) -> TrainingResult:
        """
        Process a single make/model group for retail price scraping.
        
        Args:
            make: Vehicle make (e.g., "BMW")
            model: Vehicle model (e.g., "3 Series")
            
        Returns:
            TrainingResult: Results of processing this group
        """
        group_start_time = datetime.now()
        
        try:
            # Step 1: Scrape all listings for this make/model
            logger.info(f"🔍 Scraping all listings for {make} {model}...")
            all_listings = scrape_listings(
                make=make.lower(),
                model=model,
                max_pages=self.max_pages_per_model,
                use_cache=True,  # Use cache to reduce data usage where possible
                verify_ssl=self.verify_ssl,
                use_proxy=self.use_proxy
            )
            
            if not all_listings:
                return TrainingResult(
                    make=make, model=model, success=False,
                    dealer_count=0, enriched_count=0,
                    error_message="No listings found"
                )
            
            # Step 2: Filter for dealer listings only
            dealer_listings = filter_listings_by_seller_type(all_listings, "Dealer")
            
            if len(dealer_listings) < 5:
                return TrainingResult(
                    make=make, model=model, success=False,
                    dealer_count=len(dealer_listings), enriched_count=0,
                    error_message=f"Insufficient dealer listings: {len(dealer_listings)} (need at least 5)"
                )
            
            logger.info(f"📋 Found {len(dealer_listings)} dealer listings for {make} {model}")
            
            # Step 3: Enrich dealer listings with retail price markers (uses proxy data)
            logger.info(f"💰 Scraping retail price markers for {make} {model}...")
            enriched_dealers = enrich_with_price_markers(
                dealer_listings,
                make=make.lower(),
                model=model,
                use_cache=True,  # Use cache where available
                use_proxy=self.use_proxy
            )
            
            # Step 4: Filter for successfully enriched dealers (those with price markers)
            successfully_enriched = [
                listing for listing in enriched_dealers 
                if listing.get('price_vs_market', 0) != 0
            ]
            
            if len(successfully_enriched) < 3:
                return TrainingResult(
                    make=make, model=model, success=False,
                    dealer_count=len(dealer_listings), enriched_count=len(successfully_enriched),
                    error_message=f"Too few enriched dealers: {len(successfully_enriched)} (need at least 3)"
                )
            
            # Step 5: Add to accumulated training data
            self.accumulated_dealer_data.extend(successfully_enriched)
            
            processing_time = (datetime.now() - group_start_time).total_seconds()
            
            return TrainingResult(
                make=make, model=model, success=True,
                dealer_count=len(dealer_listings), enriched_count=len(successfully_enriched),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - group_start_time).total_seconds()
            return TrainingResult(
                make=make, model=model, success=False,
                dealer_count=0, enriched_count=0,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _train_universal_model(self) -> bool:
        """Train the universal model on accumulated dealer data."""
        if len(self.accumulated_dealer_data) < 50:
            logger.error(f"Insufficient training data for universal model: {len(self.accumulated_dealer_data)} samples (need at least 50)")
            return False
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🤖 TRAINING UNIVERSAL MODEL")
            logger.info(f"{'='*60}")
            logger.info(f"Training data: {len(self.accumulated_dealer_data)} dealer listings")
            
            # Get representation stats
            make_counts = {}
            model_counts = {}
            for listing in self.accumulated_dealer_data:
                make = listing.get('make', 'Unknown')
                model = listing.get('model', 'Unknown')
                make_counts[make] = make_counts.get(make, 0) + 1
                model_counts[model] = model_counts.get(model, 0) + 1
            
            logger.info(f"Represented makes: {len(make_counts)}")
            logger.info(f"Represented models: {len(model_counts)}")
            logger.info(f"Top makes: {sorted(make_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
            # Train the universal model
            success = train_universal_model(self.accumulated_dealer_data)
            
            if success:
                logger.info("✅ Universal model training completed successfully")
            else:
                logger.error("❌ Universal model training failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training universal model: {e}")
            return False
    
    def _generate_training_report(self):
        """Generate comprehensive training report."""
        if not self.start_time:
            return
        
        total_time = datetime.now() - self.start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"WEEKLY TRAINING REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {total_time}")
        logger.info(f"Groups processed: {self.total_groups}")
        logger.info(f"Groups successful: {self.successful_groups}")
        logger.info(f"Success rate: {self.successful_groups/max(1, self.total_groups):.1%}")
        logger.info(f"Total dealer listings: {len(self.accumulated_dealer_data)}")
        
        # Success breakdown by make
        make_success = {}
        for result in self.training_results:
            if result.make not in make_success:
                make_success[result.make] = {'success': 0, 'total': 0}
            make_success[result.make]['total'] += 1
            if result.success:
                make_success[result.make]['success'] += 1
        
        logger.info(f"\nSuccess by make:")
        for make, stats in sorted(make_success.items()):
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            logger.info(f"  {make}: {stats['success']}/{stats['total']} ({rate:.1%})")
        
        # Failed groups
        failed_results = [r for r in self.training_results if not r.success]
        if failed_results:
            logger.info(f"\nFailed groups ({len(failed_results)}):")
            for result in failed_results[:10]:  # Show first 10 failures
                logger.info(f"  {result.make} {result.model}: {result.error_message}")
            if len(failed_results) > 10:
                logger.info(f"  ... and {len(failed_results) - 10} more")
        
        logger.info(f"{'='*60}\n")
    
    def get_training_summary(self) -> Dict:
        """Get summary of training results."""
        if not self.training_results:
            return {}
        
        total_time = None
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_time_seconds': total_time,
            'total_groups': self.total_groups,
            'successful_groups': self.successful_groups,
            'success_rate': self.successful_groups / max(1, self.total_groups),
            'total_dealer_listings': len(self.accumulated_dealer_data),
            'failed_groups': [
                {'make': r.make, 'model': r.model, 'error': r.error_message}
                for r in self.training_results if not r.success
            ]
        }


def run_weekly_training(max_pages_per_model: int = None, verify_ssl: bool = False, 
                       use_proxy: bool = True, test_mode: bool = False) -> bool:
    """
    Convenience function to run weekly training.
    
    Args:
        max_pages_per_model: Maximum pages to scrape per make/model
        verify_ssl: Whether to verify SSL certificates  
        use_proxy: Whether to use proxy rotation
        test_mode: Run in test mode
        
    Returns:
        bool: True if training completed successfully
    """
    orchestrator = WeeklyTrainingOrchestrator(
        max_pages_per_model=max_pages_per_model,
        verify_ssl=verify_ssl,
        use_proxy=use_proxy,
        test_mode=test_mode
    )
    
    return orchestrator.run_weekly_training()


if __name__ == "__main__":
    # Test the weekly training orchestrator
    import argparse
    
    parser = argparse.ArgumentParser(description="Run weekly retail price scraping and universal model training")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages per make/model")
    parser.add_argument("--no-proxy", action="store_true", help="Disable proxy rotation")
    
    args = parser.parse_args()

    from src.output_manager import get_output_manager
    output_manager = get_output_manager()
    output_manager.info("Starting Weekly Training Orchestrator...")
    
    success = run_weekly_training(
        max_pages_per_model=args.max_pages,
        use_proxy=not args.no_proxy,
        test_mode=args.test
    )
    
    if success:
        output_manager.success("Weekly training completed successfully")
    else:
        output_manager.error("Weekly training failed or completed with errors")
        sys.exit(1)