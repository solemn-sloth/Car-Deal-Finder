#!/usr/bin/env python3
"""
Daily Training CLI
Simple command-line interface to run daily model training.
"""

import argparse
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.daily_trainer import run_daily_training

def main():
    parser = argparse.ArgumentParser(
        description="Run daily model training for car deal finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml/daily_training.py                    # Run with default settings
  python ml/daily_training.py --max-pages 2     # Limit to 2 pages per model
  python ml/daily_training.py --verify-ssl      # Enable SSL verification
  python ml/daily_training.py --dry-run         # Show what would be trained
        """
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum pages to scrape per model (default: unlimited)"
    )

    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Verify SSL certificates during scraping"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which models would be trained without actually training"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    try:
        if args.dry_run:
            # Show what would be trained
            from services.daily_trainer import DailyModelTrainingOrchestrator

            trainer = DailyModelTrainingOrchestrator()
            todays_models = trainer.get_todays_models()

            print(f"🗓️  Daily Training Plan (Day {trainer.config['current_cycle_day']} of {trainer.cycle_days})")
            print(f"📊 Would train {len(todays_models)} models:")
            print()

            for i, model_info in enumerate(todays_models, 1):
                make = model_info['make']
                model = model_info['model']
                model_dir = trainer.get_model_path(make, model)[0].parent
                exists = "✅" if model_dir.joinpath("model.xgb").exists() else "🆕"
                print(f"  {i:2d}. {exists} {make} {model}")

            print()
            print("Legend: ✅ = Model exists (will retrain), 🆕 = New model")

        else:
            # Run actual training
            print("🚀 Starting daily model training...")

            results = run_daily_training(
                max_pages_per_model=args.max_pages,
                verify_ssl=args.verify_ssl
            )

            if results.get('enabled', True):
                print()
                print("📈 Training Results:")
                print(f"   Day: {results['cycle_day']} of 14")
                print(f"   Models scheduled: {results['models_scheduled']}")
                print(f"   Models trained: {results['models_trained']}")
                print(f"   Models failed: {results['models_failed']}")

                if results['trained_models']:
                    print()
                    print("✅ Successfully trained:")
                    for model in results['trained_models']:
                        print(f"   • {model}")

                if results['failed_models']:
                    print()
                    print("❌ Failed to train:")
                    for model in results['failed_models']:
                        print(f"   • {model}")

                success_rate = results['models_trained'] / results['models_scheduled'] * 100 if results['models_scheduled'] > 0 else 0
                print()
                print(f"📊 Success rate: {success_rate:.1f}%")

            else:
                print("⏸️  Daily training is disabled in config")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()