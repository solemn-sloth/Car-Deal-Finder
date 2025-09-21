#!/usr/bin/env python3
"""
Centralized Output Manager
Handles all user-facing console output to prevent scattered print statements.
"""

import sys
import threading
from datetime import datetime
from typing import Optional, Dict, Any


class OutputManager:
    """
    Centralized output manager for clean, coordinated console output.
    Thread-safe singleton that prevents output conflicts.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, quiet_mode: bool = False):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, quiet_mode: bool = False):
        if self._initialized:
            return

        self.quiet_mode = quiet_mode
        self._output_lock = threading.Lock()
        self._current_progress = None
        self._initialized = True

    def _print(self, message: str, end: str = "\n", flush: bool = True):
        """Thread-safe print with optional quiet mode."""
        if self.quiet_mode:
            return

        with self._output_lock:
            print(message, end=end, flush=flush)

    def startup_banner(self):
        """Display startup banner."""
        self._print("============================================================")
        self._print("                  🚀 Starting Car Dealer Bot")
        self._print("============================================================")

    def training_phase_start(self, mode: str, details: str = ""):
        """Display training phase start."""
        self._print(f"\n🎯 Training Phase: {mode}")
        if details:
            self._print(f"   {details}")

    def training_phase_skip(self, reason: str):
        """Display training phase skip."""
        self._print(f"\n🎯 Training Phase: SKIPPED ({reason})")

    def scraping_phase_start(self, target: str):
        """Display scraping phase start."""
        self._print(f"\n🎯 Deal Finding Phase: {target}")
        self._print("------------------------------------------------------------")

    def group_filtering(self, count: int, make: str = None, model: str = None):
        """Display group filtering information."""
        if make or model:
            filter_text = f"{make or 'All'} {model or 'All Models'}"
            self._print(f"🎯 Filtering to {count} group(s): {filter_text}")

    def group_start(self, current: int, total: int, make: str, model: str):
        """Display group processing start."""
        self._print(f"\n[{current}/{total}] Processing {make} {model}")

    def scraping_start(self, make: str, model: str):
        """Display scraping start."""
        # Reset progress state to ensure clean sequencing
        self._current_progress = None
        self._print(f"🔍 Scraping {make} {model}")

    def scraping_result(self, count: int, missing_fuel: int = 0, elapsed_time: float = 0):
        """Display scraping results as single line, replacing initial scraping message."""
        with self._output_lock:
            # Calculate rate
            if elapsed_time > 0:
                rate = (count * 60) / elapsed_time  # listings per minute
                dots = "." * 50  # Full progress bar since we're done
                # Clear line and show complete result with operation prefix
                print(f"\r{' ' * 150}\r🔍 Scraping Listings {dots} {count} listings | {rate:.1f} listings/min | Complete!{' ' * 20}")
            else:
                # Fallback if no timing info
                dots = "." * 50
                print(f"\r{' ' * 150}\r🔍 Scraping Listings {dots} {count} listings | Complete!{' ' * 20}")

            if missing_fuel > 0:
                self._print(f"   ⚠️ {missing_fuel} vehicles missing fuel type data (fallback applied)")

    def ml_model_processing_start(self):
        """Display ML model processing start."""
        self._print("\n🤖 Processing ML Model:")

    def ml_model_initializing(self, make: str, model: str):
        """Display ML model initialization."""
        self._print(f"   • Initializing ML Model for {make} {model}")

    def ml_model_training_start(self):
        """Display ML model training start."""
        self._print("   • Training ML Model", end="", flush=True)

    def ml_model_training_complete(self):
        """Display ML model training completion."""
        dots = "." * 50
        self._print(f"{dots} Complete!")

    def ml_model_error(self, error_msg: str):
        """Display ML model error."""
        self._print(f"   • ❌ ML Model Error: {error_msg}")

    def ml_model_status(self, message: str):
        """Display ML model status with indented bullet point."""
        self._print(f"   • {message}")

    def ml_model_complete(self):
        """Complete ML model processing section."""
        self._print("")

    def progress_update(self, current: int, total: int, operation: str, rate: float = 0, eta: str = ""):
        """Display progress update with coordination."""
        if self.quiet_mode:
            return

        with self._output_lock:
            # Format progress message
            percentage = (current / total * 100) if total > 0 else 0

            if operation == "api_scraping":
                # Use 1 dot per 10 pages for API scraping (no total known)
                progress_bar = "." * min(50, current // 10)
                if rate > 0:
                    message = f"🔍 Scraping Listings {progress_bar} {current} listings | {rate:.1f} listings/min"
                else:
                    message = f"🔍 Scraping Listings {progress_bar} {current} listings"
            elif operation == "retail_prices":
                # Use percentage-based dots for operations with known totals
                progress_bar = "." * min(50, int(percentage / 2))
                rate_text = f" | {rate:.1f} URLs/min" if rate > 0 else ""
                eta_text = f" | ETA: {eta}" if eta else ""
                message = f"🔍 Scraping Retail Prices {progress_bar} {current}/{total} ({percentage:.1f}%){rate_text}{eta_text}"
            else:
                # Use percentage-based dots for other operations with known totals
                progress_bar = "." * min(50, int(percentage / 2))
                message = f"⏳ {operation}: {progress_bar} {current}/{total} ({percentage:.1f}%)"

            # Only start clearing lines if we already have a current progress (prevents overwriting context)
            if self._current_progress is not None:
                # Clear the entire line first, then write the new message
                print(f"\r{' ' * 150}\r{message}", end="", flush=True)
            else:
                # First progress update - don't clear, just print on new line to preserve context
                print(f"\n{message}", end="", flush=True)

            self._current_progress = message

    def progress_complete(self, operation: str = "", count: int = None, rate: float = None, total_count: int = None):
        """Complete progress display with optional final statistics."""
        with self._output_lock:
            if self._current_progress:
                # Replace with completion message with full dots
                dots = "." * 50
                if operation == "retail_prices" and count is not None and rate is not None:
                    # Show full statistics like API scraping
                    rate_text = f" | {rate:.1f} URLs/min" if rate > 0 else ""
                    print(f"\r{' ' * 150}\r🔍 Scraping Retail Prices {dots} {count} URLs{rate_text} | Complete!{' ' * 20}")
                elif operation == "retail_prices":
                    print(f"\r{' ' * 150}\r🔍 Scraping Retail Prices {dots} Complete!{' ' * 20}")
                else:
                    print(f"\r{' ' * 150}\r✅ {operation}: {dots} Complete!{' ' * 20}")
                print()  # Add newline after completion
                self._current_progress = None
            else:
                print()  # Just add newline

    def api_dots_progress(self, count: int):
        """Handle API scraping dots (legacy support) - now uses coordinated progress."""
        # Legacy method - now coordinates through proper progress system
        # This prevents competing direct print() calls
        pass

    def model_training_result(self, make: str, model: str, success: bool, details: str = ""):
        """Display model training result."""
        status = "✅" if success else "❌"
        self._print(f"{status} {make} {model} training {'succeeded' if success else 'failed'}")
        if details:
            self._print(f"   {details}")

    def analysis_start(self):
        """Display analysis start header."""
        self._print("📊 Analysis Results:")

    def analysis_results(self, avg_market_value: float, confidence: float, sample_size: int, deals_found: int):
        """Display analysis results."""
        self._print(f"   • Average market value: £{int(avg_market_value):,}")
        self._print(f"   • Confidence: R²={confidence:.3f}")
        self._print(f"   • Sample size: {sample_size} comparable vehicles")
        self._print(f"   • Deals found: {deals_found} quality deals")

    def group_complete(self, make: str, model: str):
        """Display group completion."""
        self._print(f"✅ {make} {model} complete")
        self._print("------------------------------------------------------------")

    def no_quality_deals(self):
        """Display no quality deals message."""
        self._print("\nNo quality deals found this session")

    def session_summary(self, summary_data: Dict[str, Any]):
        """Display final session summary."""
        self._print("\n" + "=" * 60)
        self._print(f"{'📈 Session Summary':^60}")
        self._print("=" * 60)
        self._print("")

        # Deal Discovery
        self._print("🎯 Deal Discovery")
        self._print(f"   • Quality deals found: {summary_data.get('quality_deals', 0)}")
        self._print(f"   • Total vehicles analyzed: {summary_data.get('total_vehicles', 0)}")
        deal_rate = summary_data.get('deal_percentage', 0)
        self._print(f"   • Deal percentage rate: {deal_rate:.1f}%")
        self._print("")

        # Database Sync
        self._print("🔄 Database Sync")
        self._print(f"   • New deals added: {summary_data.get('db_added', 0)}")
        self._print(f"   • Existing deals updated: {summary_data.get('db_updated', 0)}")
        self._print(f"   • Sold/missing removed: {summary_data.get('db_removed', 0)}")
        self._print(f"   • Total active deals: {summary_data.get('db_total', 0)}")
        self._print("")

        # Notifications
        self._print("📧 Notifications")
        self._print(f"   • Status: {summary_data.get('notification_status', 'Not processed')}")
        self._print("")

        # Timing
        self._print("-" * 60)
        duration = summary_data.get('duration_minutes', 0)
        self._print(f"⏱️  Session completed in {duration:.1f} minutes")
        self._print("✅ All tasks completed successfully")
        self._print("=" * 60)

    def error(self, message: str):
        """Display error message."""
        self._print(f"❌ {message}")

    def warning(self, message: str):
        """Display warning message."""
        self._print(f"⚠️ {message}")

    def success(self, message: str):
        """Display success message."""
        self._print(f"✅ {message}")

    def info(self, message: str):
        """Display info message."""
        self._print(f"ℹ️ {message}")

    def debug(self, message: str):
        """Display debug message (only if not in quiet mode)."""
        if not self.quiet_mode:
            self._print(f"🐛 {message}")


# Global instance
_output_manager = None

def get_output_manager(quiet_mode: bool = False) -> OutputManager:
    """Get the global OutputManager instance."""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager(quiet_mode)
    return _output_manager