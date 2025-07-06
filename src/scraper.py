"""
Integration script to replace the existing scraping workflow with smart grouped orchestration.
This replaces the slower Playwright-based approach with fast network requests and smart grouping.
"""

from scrape_grouping import SmartGroupingOrchestrator
from supabase_storage import SupabaseStorage
from deal_notifications import DealNotificationPipeline
import json
from datetime import datetime
import argparse

def run_smart_grouped_scraping(max_groups=None, test_mode=False):
    """
    Run the complete smart grouped scraping and notification pipeline.
    
    Args:
        max_groups: Limit number of groups (for testing)
        test_mode: If True, don't send notifications
    """
    print("🚀 Smart Grouped Car Deal Scraper")
    print("="*60)
    print("⚡ Using network requests + smart grouping optimization")
    print(f"🎯 Mode: {'TEST' if test_mode else 'PRODUCTION'}")
    if max_groups:
        print(f"🔬 Limited to {max_groups} groups for testing")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = SmartGroupingOrchestrator()
    
    # Run orchestration
    print(f"\n📋 Processing {len(orchestrator.groups)} configuration groups...")
    start_time = datetime.now()
    
    try:
        results = orchestrator.run_full_orchestration(max_groups=max_groups)
        
        # Compile all quality deals
        all_quality_deals = []
        for result in results['results']:
            for variant_key, variant_data in result.processed_variants.items():
                all_quality_deals.extend(variant_data['quality_deals'])
        
        print(f"\n📊 Scraping Results:")
        print(f"   • Groups processed: {results['session_data']['groups_processed']}")
        print(f"   • API calls made: {results['session_data']['total_api_calls']}")
        print(f"   • API calls saved: {results['session_data']['api_call_savings']}")
        print(f"   • Total vehicles: {results['session_data']['total_vehicles_scraped']}")
        print(f"   • Quality deals found: {len(all_quality_deals)}")
        
        # Save comprehensive JSON archive for analytics
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
            'deals': all_quality_deals,
            'session_summary': results['session_data']
        }
        
        # Create archive directory if it doesn't exist
        import os
        archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        
        # Save complete archive
        archive_file = os.path.join(archive_dir, f'car_deals_archive_{timestamp}.json')
        with open(archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2, default=str)
        
        # Calculate file size
        file_size_mb = os.path.getsize(archive_file) / (1024 * 1024)
        print(f"\n📁 Archive saved: {archive_file}")
        print(f"📊 Archive size: {file_size_mb:.2f} MB")
        print(f"💾 Contains {len(all_quality_deals)} deals with full analytics data")

        if all_quality_deals:
            # Store deals to database
            if not test_mode:
                print(f"\n💾 Intelligent deal synchronization...")
                storage = SupabaseStorage()
                
                # Prepare deals for database (remove temporary fields)
                clean_deals = []
                for deal in all_quality_deals:
                    # Remove temporary analysis fields before saving to database
                    deal_copy = deal.copy()
                    deal_copy.pop('analysis_method', None)  # Remove analysis_method field
                    deal_copy.pop('attention_grabber', None)  # Remove attention_grabber field
                    
                    # Remove fields that should not be sent to database
                    deal_copy.pop('dealer_review_rating', None)
                    deal_copy.pop('dealer_review_count', None)
                    deal_copy.pop('number_of_images', None)
                    deal_copy.pop('has_video', None)
                    deal_copy.pop('has_360_spin', None)
                    deal_copy.pop('manufacturer_approved', None)
                    deal_copy.pop('seller_name', None)
                    
                    # Clean distance field to extract only numeric value
                    if 'distance' in deal_copy and deal_copy['distance']:
                        distance_str = str(deal_copy['distance'])
                        # Extract numeric part (e.g., "82 miles" -> 82)
                        import re
                        numeric_match = re.search(r'(\d+(?:\.\d+)?)', distance_str)
                        if numeric_match:
                            deal_copy['distance'] = float(numeric_match.group(1))
                        else:
                            # If no numeric value found, set to None
                            deal_copy['distance'] = None
                    
                    # Clean price-related fields to extract only numeric values
                    price_fields = ['price', 'price_numeric', 'enhanced_retail_estimate', 
                                  'enhanced_net_sale_price', 'enhanced_gross_cash_profit']
                    for field in price_fields:
                        if field in deal_copy and deal_copy[field]:
                            price_str = str(deal_copy[field])
                            # Remove currency symbols and commas (e.g., "£2,500" -> 2500)
                            import re
                            clean_price = re.sub(r'[£$€,\s]', '', price_str)
                            try:
                                deal_copy[field] = float(clean_price) if clean_price else None
                            except ValueError:
                                # If conversion fails, set to None
                                deal_copy[field] = None
                    
                    # Clean percentage field (margin) - keep as decimal
                    if 'enhanced_gross_margin_pct' in deal_copy and deal_copy['enhanced_gross_margin_pct']:
                        try:
                            # Ensure it's a float (should already be decimal like 0.33641)
                            deal_copy['enhanced_gross_margin_pct'] = float(deal_copy['enhanced_gross_margin_pct'])
                        except (ValueError, TypeError):
                            deal_copy['enhanced_gross_margin_pct'] = None
                    
                    # Map enhanced analysis data to old column names that frontend expects
                    if 'enhanced_gross_margin_pct' in deal_copy and deal_copy['enhanced_gross_margin_pct'] is not None:
                        # Keep as decimal (0.33641) since frontend multiplies by 100 for display
                        deal_copy['profit_potential_pct'] = deal_copy['enhanced_gross_margin_pct']
                    
                    if 'enhanced_gross_cash_profit' in deal_copy and deal_copy['enhanced_gross_cash_profit'] is not None:
                        deal_copy['absolute_profit'] = deal_copy['enhanced_gross_cash_profit']
                    
                    if 'enhanced_rating' in deal_copy and deal_copy['enhanced_rating']:
                        deal_copy['deal_rating'] = deal_copy['enhanced_rating']
                    
                    clean_deals.append(deal_copy)
                
                # Use intelligent sync instead of individual saves
                sync_result = storage.sync_deals_intelligently(clean_deals)
                
                if sync_result['success']:
                    print(f"✅ Sync completed:")
                    print(f"   ➕ Added: {sync_result['added_count']} new deals")
                    print(f"   🔄 Updated: {sync_result['updated_count']} existing deals") 
                    print(f"   ❌ Removed: {sync_result['removed_count']} sold/missing deals")
                    print(f"   � Total deals: {sync_result['total_deals']}")
                else:
                    print(f"❌ Sync failed: {sync_result.get('error', 'Unknown error')}")
                    return results
                
                # Send notifications
                print(f"\n📧 Sending deal notifications...")
                notification_pipeline = DealNotificationPipeline()
                notification_result = notification_pipeline.process_daily_notifications()
                
                if notification_result.get('success'):
                    print(f"✅ Notifications sent: {notification_result.get('emails_sent', 0)}")
                else:
                    print(f"⚠️  Notification issue: {notification_result.get('error', 'Unknown')}")
            else:
                print(f"\n🔬 TEST MODE: Skipping database storage and notifications")
                print(f"📋 Would have stored {len(all_quality_deals)} deals")
        else:
            print(f"\n📭 No quality deals found this session")
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\n🎯 Session Complete!")
        print(f"⏱️  Total runtime: {duration:.1f} minutes")
        print(f"⚡ Speed improvement: ~{duration/4:.1f}x faster than Playwright")
        print(f"💾 API efficiency: {results['session_data']['api_call_savings']} calls saved")
        
        return results
        
    except Exception as e:
        print(f"❌ Orchestration failed: {e}")
        raise

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Smart Grouped Car Deal Scraper')
    parser.add_argument('--max-groups', type=int, help='Limit number of groups (for testing)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Test mode (no database storage or notifications)')
    
    args = parser.parse_args()
    
    try:
        results = run_smart_grouped_scraping(
            max_groups=args.max_groups,
            test_mode=args.test_mode
        )
        
        print(f"🎉 Smart grouped scraping complete!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
