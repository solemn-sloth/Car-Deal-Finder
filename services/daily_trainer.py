#!/usr/bin/env python3
"""
Daily Model Training Orchestrator
Manages daily training of 10 specific car models out of 139 total models.
Each model gets refreshed every 14 days in a rotating cycle.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TARGET_VEHICLES_BY_MAKE
from src.analyser import scrape_listings, enrich_with_price_markers, filter_listings_by_seller_type

logger = logging.getLogger(__name__)


class DailyModelTrainingOrchestrator:
    """
    Orchestrates daily training of 10 specific car models in a 14-day rotating cycle.

    Features:
    - 139 models total, 10 models trained per day
    - 14-day cycle (139 ÷ 10 ≈ 14 days)
    - Each model gets fresh training every 14 days
    - Clean sub-folder organization by make/model
    """

    def __init__(self, max_pages_per_model: int = None, verify_ssl: bool = False):
        """
        Initialize daily training orchestrator.

        Args:
            max_pages_per_model: Maximum pages to scrape per model
            verify_ssl: Whether to verify SSL certificates
        """
        self.max_pages_per_model = max_pages_per_model
        self.verify_ssl = verify_ssl

        # Configuration
        self.models_per_day = 10
        self.cycle_days = 14
        self.min_samples_per_model = 50

        # Paths
        project_root = Path(__file__).parent.parent
        self.config_file = project_root / "config" / "daily_training_config.json"
        self.model_base_path = project_root / "archive" / "ml_models"

        # Initialize config
        self.config = self._load_or_create_config()

        logger.info(f"Initialized daily trainer: 139 models in hardcoded groups, {self.models_per_day} per day")

    def _load_or_create_config(self) -> Dict:
        """Load existing config or create new one."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {
                'enabled': True,
                'models_per_day': self.models_per_day,
                'cycle_days': self.cycle_days,
                'min_samples_per_model': self.min_samples_per_model,
                'current_cycle_day': 1,
                'last_run_date': None,
                'total_models': 139
            }
            self._save_config(config)

        return config

    def _save_config(self, config: Dict):
        """Save configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def get_todays_models(self) -> List[Dict[str, str]]:
        """Get the models to train today from hardcoded balanced groups."""
        from config.config import DAILY_TRAINING_GROUPS

        cycle_day = self.config['current_cycle_day']
        model_tuples = DAILY_TRAINING_GROUPS.get(cycle_day, [])

        todays_models = []
        for make, model in model_tuples:
            todays_models.append({
                'make': make,
                'model': model,
                'key': f"{make.lower()}_{model.lower().replace(' ', '_').replace('-', '_')}"
            })

        logger.info(f"Day {cycle_day}: Training {len(todays_models)} models from hardcoded groups")

        return todays_models

    def advance_cycle(self):
        """Advance to next day in the cycle."""
        current_day = self.config['current_cycle_day']
        next_day = current_day + 1 if current_day < self.cycle_days else 1

        self.config['current_cycle_day'] = next_day
        self.config['last_run_date'] = datetime.now().isoformat()
        self._save_config(self.config)

        logger.info(f"Advanced cycle: Day {current_day} → Day {next_day}")

    def get_model_path(self, make: str, model: str) -> Tuple[Path, Path]:
        """Get file paths for model and scaler."""
        make_clean = make.lower().replace(' ', '_').replace('-', '_')
        model_clean = model.lower().replace(' ', '_').replace('-', '_')

        model_dir = self.model_base_path / make_clean / model_clean
        model_file = model_dir / "model.xgb"
        scaler_file = model_dir / "scaler.pkl"

        return model_file, scaler_file

    def train_single_model(self, make: str, model: str) -> bool:
        """
        Train a single car model with fresh data.

        Args:
            make: Car make (e.g., "BMW")
            model: Car model (e.g., "3 Series")

        Returns:
            bool: True if training succeeded, False otherwise
        """
        try:
            logger.info(f"Training model: {make} {model}")

            # Step 1: Scrape fresh dealer data
            dealer_listings = scrape_listings(
                make=make,
                model=model,
                seller_type="dealer",
                max_pages=self.max_pages_per_model,
                verify_ssl=self.verify_ssl
            )

            if not dealer_listings:
                logger.warning(f"No dealer listings found for {make} {model}")
                return False

            logger.info(f"Scraped {len(dealer_listings)} dealer listings for {make} {model}")

            # Step 2: Enrich with price markers
            enriched_listings = enrich_with_price_markers(dealer_listings, make, model)

            # Filter to only dealer listings with valid price markers
            enriched_dealers = filter_listings_by_seller_type(enriched_listings, "dealer")
            enriched_dealers = [l for l in enriched_dealers if l.get('price_vs_market', 0) != 0]

            if len(enriched_dealers) < self.min_samples_per_model:
                logger.warning(f"Insufficient training data for {make} {model}: {len(enriched_dealers)} samples (need {self.min_samples_per_model})")
                return False

            logger.info(f"Training with {len(enriched_dealers)} enriched dealer listings")

            # Step 3: Train model-specific XGBoost
            from services.model_specific_trainer import train_model_specific
            success = train_model_specific(make, model, enriched_dealers)

            if success:
                logger.info(f"✅ Successfully trained {make} {model}")
            else:
                logger.error(f"❌ Failed to train {make} {model}")

            return success

        except Exception as e:
            logger.error(f"Error training {make} {model}: {e}")
            return False

    def run_daily_training(self, target_model: str = None, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Run today's batch of model training.

        Args:
            target_model: Target specific model for training (e.g., "3 Series")
            force_retrain: Force retraining regardless of daily schedule

        Returns:
            Dict with training results summary
        """
        if not self.config['enabled']:
            logger.info("Daily training is disabled")
            return {'enabled': False}

        logger.info("🚀 Starting daily model training")

        # Get models to train based on parameters
        if target_model:
            # Filter for specific model across all makes
            from config.config import TARGET_VEHICLES_BY_MAKE
            todays_models = []
            for make, models in TARGET_VEHICLES_BY_MAKE.items():
                if target_model in models:
                    todays_models.append({
                        'make': make,
                        'model': target_model,
                        'key': f"{make.lower()}_{target_model.lower().replace(' ', '_').replace('-', '_')}"
                    })

            if force_retrain:
                logger.info(f"🔄 Force retraining model: {target_model} (across {len(todays_models)} makes)")
            else:
                logger.info(f"🎯 Training model: {target_model} (across {len(todays_models)} makes)")
        else:
            # Get today's scheduled models
            todays_models = self.get_todays_models()
            if force_retrain:
                logger.info(f"🔄 Force retraining Day {self.config['current_cycle_day']} models ({len(todays_models)} models)")
            else:
                logger.info(f"📅 Training Day {self.config['current_cycle_day']} models ({len(todays_models)} models)")

        results = {
            'date': datetime.now().isoformat(),
            'cycle_day': self.config['current_cycle_day'],
            'models_scheduled': len(todays_models),
            'models_trained': 0,
            'models_failed': 0,
            'trained_models': [],
            'failed_models': []
        }

        # Train each model
        for model_info in todays_models:
            make = model_info['make']
            model = model_info['model']

            try:
                success = self.train_single_model(make, model)

                if success:
                    results['models_trained'] += 1
                    results['trained_models'].append(f"{make} {model}")
                else:
                    results['models_failed'] += 1
                    results['failed_models'].append(f"{make} {model}")

            except Exception as e:
                logger.error(f"Unexpected error training {make} {model}: {e}")
                results['models_failed'] += 1
                results['failed_models'].append(f"{make} {model}")

        # Advance to next day
        self.advance_cycle()

        logger.info(f"✅ Daily training complete: {results['models_trained']} trained, {results['models_failed']} failed")

        return results


def run_daily_training(max_pages_per_model: int = None, verify_ssl: bool = False,
                      target_model: str = None, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Main entry point for daily training.

    Args:
        max_pages_per_model: Maximum pages to scrape per model
        verify_ssl: Whether to verify SSL certificates
        target_model: Target specific model for training (e.g., "3 Series")
        force_retrain: Force retraining regardless of daily schedule

    Returns:
        Dict with training results
    """
    orchestrator = DailyModelTrainingOrchestrator(
        max_pages_per_model=max_pages_per_model,
        verify_ssl=verify_ssl
    )

    return orchestrator.run_daily_training(target_model=target_model, force_retrain=force_retrain)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily Model Training")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to scrape per model")
    parser.add_argument("--verify-ssl", action="store_true", help="Verify SSL certificates")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results = run_daily_training(
        max_pages_per_model=args.max_pages,
        verify_ssl=args.verify_ssl
    )

    print(f"Training complete: {results}")