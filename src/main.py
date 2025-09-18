#!/usr/bin/env python3
"""
Unified Daily Automation Entry Point for Car Deal Finder

This is the single entry point for full daily automation:
1. Run daily model training (10 models per day in 14-day cycle)
2. Run complete deal finding across all models
3. Send notifications and store results

Usage:
    python main.py                    # Full automation (training + scraping)
    python main.py --training-only    # Run only daily training
    python main.py --scraping-only    # Run only deal finding
    python main.py --dry-run          # Show what would be done
"""

import sys
import os
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# Always add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up from src/ to project root

# Make sure project root is first in path
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

# Change working directory to project root for consistent imports
original_cwd = os.getcwd()
os.chdir(project_root)

# Configure SSL bypass before any imports
import urllib3
import requests
import warnings

# Disable SSL verification globally for scraping reliability
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

# Patch requests to disable SSL verification by default
old_merge_environment_settings = requests.Session.merge_environment_settings

def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

requests.Session.merge_environment_settings = merge_environment_settings

# Patch Session class to default verify=False
orig_session = requests.Session

class PatchedSession(orig_session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify = False

requests.Session = PatchedSession

# Import after SSL configuration
from services.daily_trainer import run_daily_training
from src.scraping import run_smart_grouped_scraping

logger = logging.getLogger(__name__)


class DailyAutomationOrchestrator:
    """
    Orchestrates the complete daily automation workflow:
    1. Daily model training (10 models per day)
    2. Complete deal finding across all models
    3. Notifications and reporting
    """

    def __init__(self, training_only: bool = False, scraping_only: bool = False,
                 dry_run: bool = False, verbose: bool = False,
                 target_model: str = None, force_retrain: bool = False):
        """
        Initialize the daily automation orchestrator.

        Args:
            training_only: Run only the daily training phase
            scraping_only: Run only the deal finding phase
            dry_run: Show what would be done without executing
            verbose: Enable detailed logging
            target_model: Target specific model for testing (e.g., "3 Series")
            force_retrain: Force model retraining regardless of daily schedule
        """
        self.training_only = training_only
        self.scraping_only = scraping_only
        self.dry_run = dry_run
        self.verbose = verbose
        self.target_model = target_model
        self.force_retrain = force_retrain

        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/daily_automation.log', mode='a') if not dry_run else logging.NullHandler()
            ]
        )

        # Create logs directory if it doesn't exist
        if not dry_run:
            os.makedirs('logs', exist_ok=True)

        logger.info("=" * 80)
        logger.info("🚀 Daily Automation Orchestrator Started")
        logger.info(f"   Mode: {'Training Only' if training_only else 'Scraping Only' if scraping_only else 'Full Automation'}")
        logger.info(f"   Dry Run: {dry_run}")
        if target_model:
            logger.info(f"   Target Model: {target_model}")
        if force_retrain:
            logger.info(f"   Force Retrain: {force_retrain}")
        logger.info(f"   Start Time: {datetime.now().isoformat()}")
        logger.info("=" * 80)

    def run_daily_training_phase(self) -> Dict[str, Any]:
        """
        Run the daily model training phase.

        Returns:
            Dict with training results
        """
        logger.info("🧠 PHASE 1: Daily Model Training")
        logger.info("-" * 50)

        if self.dry_run:
            from services.daily_trainer import DailyModelTrainingOrchestrator

            trainer = DailyModelTrainingOrchestrator()

            # Get models based on target_model and force_retrain parameters
            if self.target_model:
                # Filter for specific model across all makes
                from config.config import TARGET_VEHICLES_BY_MAKE
                todays_models = []
                for make, models in TARGET_VEHICLES_BY_MAKE.items():
                    if self.target_model in models:
                        todays_models.append({
                            'make': make,
                            'model': self.target_model,
                            'key': f"{make.lower()}_{self.target_model.lower().replace(' ', '_').replace('-', '_')}"
                        })

                if self.force_retrain:
                    logger.info(f"🔄 Would force retrain model: {self.target_model} (across all makes)")
                else:
                    logger.info(f"🎯 Would train model if scheduled: {self.target_model} (across all makes)")
            else:
                # Get today's scheduled models
                todays_models = trainer.get_todays_models()
                if self.force_retrain:
                    logger.info(f"🔄 Would force retrain Day {trainer.config['current_cycle_day']} models")
                else:
                    logger.info(f"🗓️  Would train Day {trainer.config['current_cycle_day']} of {trainer.cycle_days}")

            logger.info(f"📊 Would train {len(todays_models)} models:")

            for i, model_info in enumerate(todays_models, 1):
                make = model_info['make']
                model = model_info['model']
                model_dir = trainer.get_model_path(make, model)[0].parent
                exists = "✅" if model_dir.joinpath("model.xgb").exists() else "🆕"
                logger.info(f"  {i:2d}. {exists} {make} {model}")

            return {
                'enabled': True,
                'dry_run': True,
                'cycle_day': trainer.config.get('current_cycle_day', 1),
                'models_scheduled': len(todays_models),
                'models_trained': 0,
                'models_failed': 0,
                'target_model': self.target_model,
                'force_retrain': self.force_retrain
            }

        else:
            # Run actual training
            try:
                results = run_daily_training(
                    max_pages_per_model=None,  # No page limit for production
                    verify_ssl=False,
                    target_model=self.target_model,
                    force_retrain=self.force_retrain
                )

                if results.get('enabled', True):
                    logger.info("✅ Daily training completed successfully")
                    logger.info(f"   Day: {results['cycle_day']} of 14")
                    logger.info(f"   Models trained: {results['models_trained']}/{results['models_scheduled']}")

                    if results['failed_models']:
                        logger.warning(f"   Failed models: {', '.join(results['failed_models'])}")
                else:
                    logger.info("⏸️  Daily training is disabled in config")

                return results

            except Exception as e:
                logger.error(f"❌ Daily training failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return {
                    'enabled': True,
                    'error': str(e),
                    'models_trained': 0,
                    'models_failed': 999  # Indicate complete failure
                }

    def run_deal_finding_phase(self) -> Dict[str, Any]:
        """
        Run the complete deal finding phase.

        Returns:
            Dict with scraping results
        """
        logger.info("🔍 PHASE 2: Deal Finding & Analysis")
        logger.info("-" * 50)

        if self.dry_run:
            if self.target_model:
                logger.info(f"🎯 Would run scraping for model: {self.target_model} (across all makes)")
            else:
                logger.info("🎯 Would run smart grouped scraping across all vehicle models")
            logger.info("📧 Would send deal notifications if profitable deals found")
            logger.info("💾 Would store results in Supabase database")

            return {
                'dry_run': True,
                'groups_processed': 0,
                'total_deals': 0,
                'quality_deals': 0,
                'target_model': self.target_model
            }

        else:
            try:
                if self.target_model:
                    # Run scraping for the target model across all makes
                    # Since the scraping function filters by make+model, we need to run it for each make
                    from config.config import TARGET_VEHICLES_BY_MAKE

                    all_results = {
                        'groups_processed': 0,
                        'total_deals': 0,
                        'quality_deals': 0,
                        'target_model': self.target_model
                    }

                    makes_with_model = [make for make, models in TARGET_VEHICLES_BY_MAKE.items()
                                       if self.target_model in models]

                    logger.info(f"🎯 Running scraping for {self.target_model} across {len(makes_with_model)} makes")

                    for make in makes_with_model:
                        logger.info(f"  Processing {make} {self.target_model}")
                        try:
                            results = run_smart_grouped_scraping(
                                max_groups=None,
                                test_mode=False,  # Enable database storage and notifications
                                connection_pool_size=10,
                                filter_make=make,
                                filter_model=self.target_model
                            )

                            # Aggregate results
                            all_results['groups_processed'] += results.get('groups_processed', 0)
                            all_results['total_deals'] += results.get('total_deals', 0)
                            all_results['quality_deals'] += results.get('quality_deals', 0)

                        except Exception as e:
                            logger.warning(f"  Failed to process {make} {self.target_model}: {e}")

                    logger.info("✅ Deal finding completed successfully")
                    logger.info(f"   Groups processed: {all_results['groups_processed']}")
                    logger.info(f"   Total deals found: {all_results['total_deals']}")
                    logger.info(f"   Quality deals: {all_results['quality_deals']}")

                    return all_results
                else:
                    # Run the complete deal finding pipeline
                    results = run_smart_grouped_scraping(
                        max_groups=None,  # Process all groups
                        test_mode=False,  # Enable database storage and notifications
                        connection_pool_size=10,
                        filter_make=None,
                        filter_model=None
                    )

                    logger.info("✅ Deal finding completed successfully")
                    logger.info(f"   Groups processed: {results.get('groups_processed', 0)}")
                    logger.info(f"   Total deals found: {results.get('total_deals', 0)}")
                    logger.info(f"   Quality deals: {results.get('quality_deals', 0)}")

                    return results

            except Exception as e:
                logger.error(f"❌ Deal finding failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return {
                    'error': str(e),
                    'groups_processed': 0,
                    'total_deals': 0,
                    'quality_deals': 0
                }

    def run_full_automation(self) -> Dict[str, Any]:
        """
        Run the complete daily automation workflow.

        Returns:
            Dict with combined results from both phases
        """
        start_time = datetime.now()

        # Initialize results
        automation_results = {
            'start_time': start_time.isoformat(),
            'training_results': None,
            'scraping_results': None,
            'total_duration_minutes': 0,
            'success': False
        }

        try:
            # Phase 1: Daily Model Training (unless scraping-only mode)
            if not self.scraping_only:
                training_results = self.run_daily_training_phase()
                automation_results['training_results'] = training_results

                # Check if training had critical failures
                if (training_results.get('models_failed', 0) > 5 and
                    not training_results.get('dry_run', False)):
                    logger.warning("⚠️  Many training failures detected, but continuing with scraping")

            logger.info("")  # Blank line between phases

            # Phase 2: Deal Finding & Analysis (unless training-only mode)
            if not self.training_only:
                scraping_results = self.run_deal_finding_phase()
                automation_results['scraping_results'] = scraping_results

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            automation_results['total_duration_minutes'] = round(duration, 2)
            automation_results['end_time'] = end_time.isoformat()
            automation_results['success'] = True

            # Summary logging
            logger.info("")
            logger.info("=" * 80)
            logger.info("📊 DAILY AUTOMATION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"⏱️  Total Duration: {duration:.2f} minutes")

            if automation_results['training_results']:
                tr = automation_results['training_results']
                if not tr.get('dry_run', False) and tr.get('enabled', True):
                    logger.info(f"🧠 Training: {tr.get('models_trained', 0)}/{tr.get('models_scheduled', 0)} models successful")
                elif tr.get('enabled', True):
                    logger.info(f"🧠 Training: {tr.get('models_scheduled', 0)} models scheduled (dry run)")
                else:
                    logger.info("🧠 Training: Disabled")

            if automation_results['scraping_results']:
                sr = automation_results['scraping_results']
                if not sr.get('dry_run', False):
                    logger.info(f"🔍 Scraping: {sr.get('quality_deals', 0)} quality deals from {sr.get('groups_processed', 0)} groups")
                else:
                    logger.info("🔍 Scraping: Would process all vehicle groups (dry run)")

            logger.info("✅ Daily automation completed successfully!")
            logger.info("=" * 80)

            return automation_results

        except Exception as e:
            logger.error(f"❌ Daily automation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

            automation_results['error'] = str(e)
            automation_results['success'] = False
            automation_results['total_duration_minutes'] = (datetime.now() - start_time).total_seconds() / 60

            return automation_results


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Daily Automation for Car Deal Finder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full automation (training + scraping)
  python main.py --training-only              # Run only daily training
  python main.py --scraping-only              # Run only deal finding
  python main.py --dry-run                    # Show what would be done
  python main.py --verbose                    # Enable detailed logging
  python main.py --model "3 Series"           # Test single model in production mode
  python main.py --model "3 Series" --force-retrain  # Force retrain + test model
        """
    )

    parser.add_argument(
        '--training-only',
        action='store_true',
        help='Run only the daily model training phase'
    )

    parser.add_argument(
        '--scraping-only',
        action='store_true',
        help='Run only the deal finding phase'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually executing'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Target specific model for testing (e.g., "3 Series")'
    )

    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force model retraining regardless of daily schedule'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.training_only and args.scraping_only:
        print("❌ Error: Cannot specify both --training-only and --scraping-only")
        sys.exit(1)

    try:
        # Create and run orchestrator
        orchestrator = DailyAutomationOrchestrator(
            training_only=args.training_only,
            scraping_only=args.scraping_only,
            dry_run=args.dry_run,
            verbose=args.verbose,
            target_model=args.model,
            force_retrain=args.force_retrain
        )

        results = orchestrator.run_full_automation()

        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Daily automation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Daily automation failed: {e}")
        sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()