#!/usr/bin/env python3
"""
Unified Car Deal Finder Automation

Single entry point for intelligent automation:
1. Scrape vehicle listings first
2. Intelligently determine if ML training is needed for those listings
3. If needed, trigger retail price scraping and model training seamlessly
4. Continue with deal analysis and notifications

Usage:
    python main.py                    # Run unified automation
    python main.py --dry-run          # Show what would be done
    python main.py --model "595"      # Focus on specific model
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
from src.output_manager import get_output_manager

logger = logging.getLogger(__name__)


class DailyAutomationOrchestrator:
    """
    Orchestrates the unified automation workflow:
    1. Scrape vehicle listings first
    2. Intelligently determine if ML training is needed
    3. If needed, trigger retail price scraping and training seamlessly
    4. Continue with deal analysis and notifications
    """

    def __init__(self, dry_run: bool = False, verbose: bool = False,
                 target_model: str = None, force_retrain: bool = False,
                 export_predictions: bool = False):
        """
        Initialize the unified automation orchestrator.

        Args:
            dry_run: Show what would be done without executing
            verbose: Enable detailed logging
            target_model: Target specific model for testing (e.g., "3 Series")
            force_retrain: Force model retraining regardless of daily schedule
            export_predictions: Export ML predictions to JSON with features
        """
        self.dry_run = dry_run
        self.verbose = verbose
        self.target_model = target_model
        self.force_retrain = force_retrain
        self.export_predictions = export_predictions

        # Create logs directory if it doesn't exist
        if not dry_run:
            os.makedirs('logs', exist_ok=True)

        # Configure logging - suppress verbose INFO logs, only show errors
        log_level = logging.DEBUG if verbose else logging.ERROR
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/daily_automation.log', mode='a') if not dry_run else logging.NullHandler()
            ]
        )

        # No startup logging - handled by main print statements


    def run_unified_automation_phase(self) -> Dict[str, Any]:
        """
        Run the unified automation phase: scraping with intelligent ML training.

        This combines scraping and ML training into a single seamless flow:
        1. Scrape listings first
        2. Intelligently determine if ML training is needed
        3. If needed, trigger retail price scraping + training
        4. Continue with analysis and deal finding

        Returns:
            Dict with automation results
        """
        # Deal finding phase - no logging

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

                    for make in makes_with_model:
                        try:
                            results = run_smart_grouped_scraping(
                                max_groups=None,
                                test_mode=False,  # Enable database storage and notifications
                                connection_pool_size=10,
                                filter_make=make,
                                filter_model=self.target_model,
                                export_predictions=self.export_predictions
                            )

                            # Aggregate results
                            all_results['groups_processed'] += results.get('groups_processed', 0)
                            all_results['total_deals'] += results.get('total_deals', 0)
                            all_results['quality_deals'] += results.get('quality_deals', 0)

                        except Exception as e:
                            print(f"Error processing {make} {self.target_model}: {e}")
                            if self.verbose:
                                import traceback
                                traceback.print_exc()

                    # Results handled by main print statements
                    pass

                    return all_results
                else:
                    # Run the complete deal finding pipeline
                    results = run_smart_grouped_scraping(
                        max_groups=None,  # Process all groups
                        test_mode=False,  # Enable database storage and notifications
                        connection_pool_size=10,
                        filter_make=None,
                        filter_model=None,
                        export_predictions=self.export_predictions
                    )

                    # Results handled by main print statements
                    pass

                    return results

            except Exception as e:
                print(f"Error in deal finding automation: {e}")
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
        # Show startup banner at the very beginning
        output_manager = get_output_manager()
        output_manager.startup_banner()

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
            # Unified Phase: Scraping with Intelligent ML Training
            # This replaces the dual-phase architecture with a single seamless flow
            scraping_results = self.run_unified_automation_phase()
            automation_results['scraping_results'] = scraping_results

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            automation_results['total_duration_minutes'] = round(duration, 2)
            automation_results['end_time'] = end_time.isoformat()
            automation_results['success'] = True

            # Summary handled by main print statements
            pass

            return automation_results

        except Exception as e:
            print(f"Error in main automation: {e}")
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
  python main.py                              # Run unified automation
  python main.py --dry-run                    # Show what would be done
  python main.py --verbose                    # Enable detailed logging
  python main.py --model "595"                # Focus on Abarth 595 across all makes
  python main.py --model "595" --force-retrain # Force retrain 595 models + analyze
        """
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

    parser.add_argument(
        '--export-predictions',
        action='store_true',
        help='Export ML model predictions for private seller cars to JSON with URLs and features for verification'
    )


    args = parser.parse_args()


    try:
        # Create and run orchestrator
        orchestrator = DailyAutomationOrchestrator(
            dry_run=args.dry_run,
            verbose=args.verbose,
            target_model=args.model,
            force_retrain=args.force_retrain,
            export_predictions=args.export_predictions
        )

        results = orchestrator.run_full_automation()

        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()