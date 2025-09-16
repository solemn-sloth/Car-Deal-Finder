#!/usr/bin/env python3
"""
Weekly Schedule Manager
Tracks when weekly retail price scraping is due and manages scheduling state.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class WeeklyScheduleManager:
    """
    Manages weekly retail price scraping schedule.
    
    Features:
    - Tracks last retail scraping timestamp
    - Determines if weekly scraping is due (7+ days)
    - Persistent state storage
    - Configurable schedule interval
    """
    
    def __init__(self, schedule_file: Optional[str] = None, interval_days: int = 7):
        """
        Initialize schedule manager.
        
        Args:
            schedule_file: Path to schedule state file (defaults to project archive)
            interval_days: Days between retail scraping runs (default 7)
        """
        self.interval_days = interval_days
        
        # Default schedule file location
        if schedule_file is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config"
            config_path.mkdir(exist_ok=True)
            schedule_file = config_path / "scraping_scheduler.json"
        
        self.schedule_file = Path(schedule_file)
        self.schedule_data = self._load_schedule_data()
    
    def _load_schedule_data(self) -> Dict:
        """Load schedule data from persistent storage."""
        default_data = {
            "last_retail_scraping": None,
            "last_scraping_success": True,
            "scraping_history": [],
            "created_at": datetime.now().isoformat()
        }
        
        if not self.schedule_file.exists():
            logger.info(f"Creating new schedule file: {self.schedule_file}")
            self._save_schedule_data(default_data)
            return default_data
        
        try:
            with open(self.schedule_file, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            for key in default_data.keys():
                if key not in data:
                    data[key] = default_data[key]
            
            logger.debug(f"Loaded schedule data from {self.schedule_file}")
            return data
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading schedule file {self.schedule_file}: {e}")
            logger.info("Creating new schedule file with default data")
            self._save_schedule_data(default_data)
            return default_data
    
    def _save_schedule_data(self, data: Dict) -> bool:
        """Save schedule data to persistent storage."""
        try:
            # Ensure directory exists
            self.schedule_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data with pretty formatting
            with open(self.schedule_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Schedule data saved to {self.schedule_file}")
            return True
            
        except IOError as e:
            logger.error(f"Error saving schedule file {self.schedule_file}: {e}")
            return False
    
    def is_retail_scraping_due(self) -> bool:
        """
        Check if weekly retail scraping is due.
        
        Returns:
            bool: True if scraping should run, False if not due yet
        """
        last_scraping = self.schedule_data.get("last_retail_scraping")
        
        # If never scraped, it's due
        if last_scraping is None:
            logger.info("No previous retail scraping found - scraping is due")
            return True
        
        try:
            # Parse last scraping timestamp
            if isinstance(last_scraping, str):
                last_scraping_dt = datetime.fromisoformat(last_scraping.replace('Z', '+00:00'))
            else:
                last_scraping_dt = last_scraping
                
            # Calculate days since last scraping
            now = datetime.now()
            days_since = (now - last_scraping_dt).days
            
            is_due = days_since >= self.interval_days
            
            if is_due:
                logger.info(f"Retail scraping is due: {days_since} days since last run (interval: {self.interval_days} days)")
            else:
                logger.info(f"Retail scraping not due yet: {days_since} days since last run (interval: {self.interval_days} days)")
            
            return is_due
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing last scraping timestamp '{last_scraping}': {e}")
            logger.info("Treating as due for safety")
            return True
    
    def get_days_since_last_scraping(self) -> Optional[int]:
        """
        Get number of days since last retail scraping.
        
        Returns:
            int: Days since last scraping, or None if never scraped
        """
        last_scraping = self.schedule_data.get("last_retail_scraping")
        
        if last_scraping is None:
            return None
        
        try:
            if isinstance(last_scraping, str):
                last_scraping_dt = datetime.fromisoformat(last_scraping.replace('Z', '+00:00'))
            else:
                last_scraping_dt = last_scraping
                
            now = datetime.now()
            return (now - last_scraping_dt).days
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating days since last scraping: {e}")
            return None
    
    def mark_retail_scraping_started(self) -> bool:
        """
        Mark that retail scraping has started.
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        now = datetime.now()
        
        # Add to scraping history
        history_entry = {
            "started_at": now.isoformat(),
            "status": "started"
        }
        
        self.schedule_data["scraping_history"].append(history_entry)
        
        # Keep only last 10 history entries
        if len(self.schedule_data["scraping_history"]) > 10:
            self.schedule_data["scraping_history"] = self.schedule_data["scraping_history"][-10:]
        
        logger.info(f"Marked retail scraping as started at {now.isoformat()}")
        return self._save_schedule_data(self.schedule_data)
    
    def mark_retail_scraping_complete(self, success: bool = True, 
                                   total_vehicles: int = 0, 
                                   notes: Optional[str] = None) -> bool:
        """
        Mark retail scraping as complete.
        
        Args:
            success: Whether scraping completed successfully
            total_vehicles: Total number of vehicles processed
            notes: Optional notes about the scraping run
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        now = datetime.now()
        
        # Update main fields
        if success:
            self.schedule_data["last_retail_scraping"] = now.isoformat()
        
        self.schedule_data["last_scraping_success"] = success
        
        # Update last history entry or create new one
        if (self.schedule_data["scraping_history"] and 
            self.schedule_data["scraping_history"][-1].get("status") == "started"):
            # Update existing entry
            self.schedule_data["scraping_history"][-1].update({
                "completed_at": now.isoformat(),
                "status": "completed" if success else "failed",
                "success": success,
                "total_vehicles": total_vehicles,
                "notes": notes
            })
        else:
            # Create new entry
            history_entry = {
                "completed_at": now.isoformat(),
                "status": "completed" if success else "failed",
                "success": success,
                "total_vehicles": total_vehicles,
                "notes": notes
            }
            self.schedule_data["scraping_history"].append(history_entry)
        
        # Keep only last 10 history entries
        if len(self.schedule_data["scraping_history"]) > 10:
            self.schedule_data["scraping_history"] = self.schedule_data["scraping_history"][-10:]
        
        status_msg = "successfully" if success else "with errors"
        logger.info(f"Marked retail scraping as completed {status_msg} at {now.isoformat()}")
        
        if total_vehicles > 0:
            logger.info(f"Processed {total_vehicles} total vehicles")
        
        if notes:
            logger.info(f"Notes: {notes}")
        
        return self._save_schedule_data(self.schedule_data)
    
    def get_next_scraping_date(self) -> Optional[datetime]:
        """
        Get the next scheduled retail scraping date.
        
        Returns:
            datetime: Next scraping date, or None if never scraped
        """
        last_scraping = self.schedule_data.get("last_retail_scraping")
        
        if last_scraping is None:
            return None
        
        try:
            if isinstance(last_scraping, str):
                last_scraping_dt = datetime.fromisoformat(last_scraping.replace('Z', '+00:00'))
            else:
                last_scraping_dt = last_scraping
                
            return last_scraping_dt + timedelta(days=self.interval_days)
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating next scraping date: {e}")
            return None
    
    def get_schedule_status(self) -> Dict:
        """
        Get comprehensive schedule status information.
        
        Returns:
            Dict: Schedule status including timing, history, etc.
        """
        last_scraping = self.schedule_data.get("last_retail_scraping")
        days_since = self.get_days_since_last_scraping()
        next_date = self.get_next_scraping_date()
        is_due = self.is_retail_scraping_due()
        
        status = {
            "is_due": is_due,
            "last_scraping": last_scraping,
            "days_since_last": days_since,
            "interval_days": self.interval_days,
            "next_scheduled": next_date.isoformat() if next_date else None,
            "last_success": self.schedule_data.get("last_scraping_success", None),
            "total_history_entries": len(self.schedule_data.get("scraping_history", [])),
            "schedule_file": str(self.schedule_file)
        }
        
        return status
    
    def reset_schedule(self) -> bool:
        """
        Reset the schedule (mainly for testing purposes).
        
        Returns:
            bool: True if successfully reset, False otherwise
        """
        logger.warning("Resetting retail scraping schedule")
        
        reset_data = {
            "last_retail_scraping": None,
            "last_scraping_success": True,
            "scraping_history": [],
            "reset_at": datetime.now().isoformat()
        }
        
        self.schedule_data = reset_data
        return self._save_schedule_data(reset_data)


# Global schedule manager instance
schedule_manager = WeeklyScheduleManager()

# Convenience functions for easy access
def is_retail_scraping_due() -> bool:
    """Check if weekly retail scraping is due."""
    return schedule_manager.is_retail_scraping_due()

def mark_retail_scraping_started() -> bool:
    """Mark that retail scraping has started."""
    return schedule_manager.mark_retail_scraping_started()

def mark_retail_scraping_complete(success: bool = True, total_vehicles: int = 0, notes: Optional[str] = None) -> bool:
    """Mark retail scraping as complete."""
    return schedule_manager.mark_retail_scraping_complete(success, total_vehicles, notes)

def get_schedule_status() -> Dict:
    """Get comprehensive schedule status information."""
    return schedule_manager.get_schedule_status()

def get_days_since_last_scraping() -> Optional[int]:
    """Get number of days since last retail scraping."""
    return schedule_manager.get_days_since_last_scraping()


if __name__ == "__main__":
    # Test the schedule manager
    print("Testing Weekly Schedule Manager...")
    
    # Get status
    status = get_schedule_status()
    print(f"\nSchedule Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test due check
    is_due = is_retail_scraping_due()
    print(f"\nIs retail scraping due? {is_due}")
    
    if status["days_since_last"]:
        print(f"Days since last scraping: {status['days_since_last']}")
    else:
        print("No previous scraping found")
    
    # Test marking operations (commented out to avoid changing state during testing)
    # print("\nTesting mark operations...")
    # mark_retail_scraping_started()
    # mark_retail_scraping_complete(success=True, total_vehicles=150, notes="Test run")