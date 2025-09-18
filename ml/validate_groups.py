#!/usr/bin/env python3
"""
Training Groups Validation Utility
Validates that all 139 models are assigned exactly once across 14 days
and shows the balanced distribution.
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TARGET_VEHICLES_BY_MAKE, DAILY_TRAINING_GROUPS

def validate_training_groups():
    """Validate training groups for completeness and balance."""

    print("🔍 Validating Daily Training Groups...")
    print()

    # Get all target models
    all_target_models = set()
    for make, models in TARGET_VEHICLES_BY_MAKE.items():
        for model in models:
            all_target_models.add((make, model))

    print(f"📊 Total target models: {len(all_target_models)}")

    # Analyze groups
    all_grouped_models = set()
    duplicates = []
    day_counts = {}

    for day, models in DAILY_TRAINING_GROUPS.items():
        day_counts[day] = len(models)
        print(f"   Day {day:2d}: {len(models):2d} models")

        for make, model in models:
            if (make, model) in all_grouped_models:
                duplicates.append((day, make, model))
            all_grouped_models.add((make, model))

    print()

    # Check for issues
    missing = all_target_models - all_grouped_models
    extra = all_grouped_models - all_target_models

    # Results
    print("✅ Validation Results:")
    print(f"   Total grouped models: {len(all_grouped_models)}")
    print(f"   Duplicates: {len(duplicates)}")
    print(f"   Missing: {len(missing)}")
    print(f"   Extra: {len(extra)}")

    if duplicates:
        print()
        print("❌ Duplicate assignments:")
        for day, make, model in duplicates:
            print(f"   Day {day}: {make} {model}")

    if missing:
        print()
        print("❌ Missing models:")
        for make, model in sorted(missing):
            print(f"   {make} {model}")

    if extra:
        print()
        print("❌ Extra models (not in TARGET_VEHICLES):")
        for make, model in sorted(extra):
            print(f"   {make} {model}")

    print()

    # Balance analysis
    print("⚖️  Balance Analysis:")
    total_models = sum(day_counts.values())
    avg_per_day = total_models / len(day_counts)
    print(f"   Average models per day: {avg_per_day:.1f}")

    min_day = min(day_counts.values())
    max_day = max(day_counts.values())
    print(f"   Range: {min_day} - {max_day} models per day")

    if max_day - min_day <= 1:
        print("   ✅ Well balanced (difference ≤ 1)")
    else:
        print(f"   ⚠️  Imbalanced (difference = {max_day - min_day})")

    print()

    # Final verdict
    is_perfect = (len(all_grouped_models) == 139 and
                  len(duplicates) == 0 and
                  len(missing) == 0 and
                  len(extra) == 0)

    if is_perfect:
        print("🎉 PERFECT! All 139 models assigned exactly once across 14 balanced days!")
        return True
    else:
        print("❌ Issues found - please fix the group assignments")
        return False

def show_day_details(day: int):
    """Show detailed breakdown for a specific day."""

    if day not in DAILY_TRAINING_GROUPS:
        print(f"❌ Day {day} not found (valid range: 1-14)")
        return

    models = DAILY_TRAINING_GROUPS[day]

    print(f"📅 Day {day} Training Schedule ({len(models)} models):")
    print()

    # Group by popularity level based on comments
    high_pop = []
    medium_pop = []
    low_pop = []

    for make, model in models:
        # This is a simple heuristic - you can refine based on actual data
        if make in ['BMW', 'Audi', 'Mercedes-Benz'] and model in ['3 Series', 'A3', 'A Class']:
            high_pop.append(f"{make} {model}")
        elif make in ['Porsche', 'Tesla', 'Jaguar', 'DS Automobiles', 'Abarth', 'Lexus']:
            low_pop.append(f"{make} {model}")
        else:
            medium_pop.append(f"{make} {model}")

    if high_pop:
        print(f"🔥 High Volume ({len(high_pop)}):")
        for model in high_pop:
            print(f"   • {model}")
        print()

    if medium_pop:
        print(f"📊 Medium Volume ({len(medium_pop)}):")
        for model in medium_pop:
            print(f"   • {model}")
        print()

    if low_pop:
        print(f"⚡ Low Volume ({len(low_pop)}):")
        for model in low_pop:
            print(f"   • {model}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate daily training groups")
    parser.add_argument("--day", type=int, help="Show details for specific day (1-14)")

    args = parser.parse_args()

    if args.day:
        show_day_details(args.day)
    else:
        is_valid = validate_training_groups()
        sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()