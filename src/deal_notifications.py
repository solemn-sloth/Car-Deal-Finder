"""
Deal notification pipeline integration
Handles sending email notifications to users when new deals are found
"""
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from supabase_storage import SupabaseStorage
from notification_service import NotificationService

class DealNotificationPipeline:
    def __init__(self):
        self.storage = SupabaseStorage()
        self.notification_service = NotificationService()
        
    def get_notification_eligible_users(self) -> List[Dict]:
        """
        Get all users who are eligible for email notifications
        Returns list of users with their notification preferences
        """
        try:
            # Query stripe_customers for users who have email notifications enabled
            response = self.storage.supabase.table('stripe_customers').select(
                'id, user_id, name, email, postcode, notification_preferences'
            ).not_.is_('email', 'null').execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"⚠️  Error fetching notification users: {response.error}")
                return []
                
            users = response.data or []
            print(f"📋 Found {len(users)} total users in stripe_customers")
            
            # Filter users who have email notifications enabled
            eligible_users = []
            for user in users:
                print(f"🔍 Checking user: {user.get('name')} ({user.get('email')})")
                prefs = user.get('notification_preferences', {})
                print(f"   Notification preferences: {prefs}")
                
                # More flexible checking for email preferences
                email_enabled = False
                if isinstance(prefs, dict):
                    email_enabled = prefs.get('email', False)
                elif isinstance(prefs, str):
                    # Handle case where JSON might be stored as string
                    try:
                        import json
                        prefs_dict = json.loads(prefs)
                        email_enabled = prefs_dict.get('email', False)
                    except:
                        email_enabled = False
                
                print(f"   Email enabled: {email_enabled}")
                
                if email_enabled and user.get('email'):
                    eligible_users.append(user)
                    print(f"   ✅ Added to eligible users")
                else:
                    print(f"   ❌ Not eligible")
            
            print(f"📧 Found {len(eligible_users)} users eligible for email notifications")
            return eligible_users
            
        except Exception as e:
            print(f"❌ Error getting notification eligible users: {e}")
            return []
    
    def get_user_email(self, user_id: str) -> Optional[str]:
        """
        Get user's email address from stripe_customers table
        """
        try:
            response = self.storage.supabase.table('stripe_customers').select('email').eq('id', user_id).single().execute()
            
            if hasattr(response, 'data') and response.data and response.data.get('email'):
                return response.data['email']
            
            print(f"⚠️  No email found for user {user_id}")
            return None
            
        except Exception as e:
            print(f"❌ Error getting user email for {user_id}: {e}")
            return None
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """Get user's deal preferences for filtering"""
        try:
            response = self.storage.supabase.table('user_preferences').select('*').eq('user_id', user_id).execute()
            
            if hasattr(response, 'error') and response.error:
                if 'PGRST116' in str(response.error.get('code', '')):  # No data found
                    print(f"ℹ️  No user preferences found for {user_id} (using defaults)")
                    return None
                print(f"⚠️  Error fetching preferences for user {user_id}: {response.error}")
                return None
            
            if response.data and len(response.data) > 0:
                return response.data[0]  # Get first record
            else:
                print(f"ℹ️  No user preferences found for {user_id} (using defaults)")
                return None
                
        except Exception as e:
            print(f"ℹ️  No user preferences table or error getting preferences for {user_id}: {e}")
            return None
    
    def filter_deals_for_user(self, deals: List[Dict], user: Dict, user_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Filter deals based on user's notification preferences and general preferences
        """
        if not deals:
            return []
        
        filtered_deals = deals.copy()
        notification_prefs = user.get('notification_preferences', {})
        
        print(f"🎯 Filtering {len(deals)} deals for user {user.get('email')}")
        print(f"   Notification preferences: {notification_prefs}")
        print(f"   User preferences: {user_preferences is not None}")
        
        # Filter by deal types (excellent, good, etc.)
        deal_types = notification_prefs.get('deal_types', ['excellent', 'good'])
        if deal_types:
            deal_type_map = {
                'excellent': ['Excellent Deal'],
                'good': ['Good Deal', 'Great Deal'],  # Include both Good Deal and Great Deal
                'negotiation': ['Negotiation Target']
            }
            allowed_ratings = []
            for dt in deal_types:
                if dt in deal_type_map:
                    allowed_ratings.extend(deal_type_map[dt])
            
            if allowed_ratings:
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('deal_rating') in allowed_ratings]
                print(f"   After deal type filter (looking for {allowed_ratings}): {len(filtered_deals)} (was {before_count})")
                if before_count > 0 and len(filtered_deals) == 0:
                    print(f"   Available ratings in deals: {set(d.get('deal_rating') for d in deals)}")
        else:
            print(f"   No deal type filter specified - including all deals")
        
        # Filter by price range if specified
        price_range = notification_prefs.get('price_range', {})
        if price_range.get('min'):
            before_count = len(filtered_deals)
            filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) >= price_range['min']]
            print(f"   After min price filter (>= £{price_range['min']}): {len(filtered_deals)} (was {before_count})")
        if price_range.get('max'):
            before_count = len(filtered_deals)
            filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) <= price_range['max']]
            print(f"   After max price filter (<= £{price_range['max']}): {len(filtered_deals)} (was {before_count})")
        
        # Filter by maximum distance if specified
        max_distance = notification_prefs.get('max_distance')
        if max_distance and user.get('postcode'):
            # For now, we'll skip distance filtering as it requires geolocation logic
            # This can be implemented later with postcode distance calculation
            print(f"   Distance filtering not implemented yet (max: {max_distance} miles)")
        
        # Apply user's general deal preferences if available (but don't be too restrictive)
        if user_preferences:
            print(f"   Applying additional user preferences...")
            # Min/max price (override notification prefs if more restrictive)
            if user_preferences.get('min_price_numeric'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) >= user_preferences['min_price_numeric']]
                print(f"   After user min price: {len(filtered_deals)} (was {before_count})")
            if user_preferences.get('max_price_numeric'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) <= user_preferences['max_price_numeric']]
                print(f"   After user max price: {len(filtered_deals)} (was {before_count})")
            
            # Max mileage
            if user_preferences.get('max_mileage'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('mileage', 0) <= user_preferences['max_mileage']]
                print(f"   After mileage filter: {len(filtered_deals)} (was {before_count})")
            
            # Min year
            if user_preferences.get('min_year'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('year', 0) >= user_preferences['min_year']]
                print(f"   After min year filter: {len(filtered_deals)} (was {before_count})")
            
            # Max year  
            if user_preferences.get('max_year'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('year', 0) <= user_preferences['max_year']]
                print(f"   After max year filter: {len(filtered_deals)} (was {before_count})")
            
            # Preferred makes
            preferred_makes = user_preferences.get('preferred_makes', [])
            if preferred_makes:
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('make') in preferred_makes]
                print(f"   After preferred makes filter: {len(filtered_deals)} (was {before_count})")
            
            # Preferred models
            preferred_models = user_preferences.get('preferred_models', [])
            if preferred_models:
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('model') in preferred_models]
                print(f"   After preferred models filter: {len(filtered_deals)} (was {before_count})")
        
        print(f"   Final filtered deals: {len(filtered_deals)}")
        return filtered_deals
    
    def get_new_deals_for_notification(self, hours_back: int = 24) -> List[Dict]:
        """
        Get deals that were added in the last N hours (new deals worth notifying about)
        """
        try:
            # Calculate the cutoff time
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            cutoff_iso = cutoff_time.isoformat()
            
            # Get deals created since cutoff time
            response = self.storage.supabase.table('car_deals').select('*').gte('created_at', cutoff_iso).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"⚠️  Error fetching new deals: {response.error}")
                return []
                
            deals = response.data or []
            print(f"🆕 Found {len(deals)} new deals in last {hours_back} hours")
            return deals
            
        except Exception as e:
            print(f"❌ Error getting new deals: {e}")
            return []
    
    def send_notifications_for_new_deals(self, new_deals: List[Dict]) -> Dict:
        """
        Send email notifications to eligible users for new deals
        Returns summary of notification results
        """
        if not new_deals:
            print("📭 No new deals to notify about")
            return {'success': True, 'sent': 0, 'skipped': 0, 'errors': 0}
        
        print(f"📧 Processing notifications for {len(new_deals)} new deals...")
        
        # Get users eligible for notifications
        eligible_users = self.get_notification_eligible_users()
        
        if not eligible_users:
            print("📭 No users eligible for notifications")
            return {'success': True, 'sent': 0, 'skipped': 0, 'errors': 0}
        
        results = {
            'sent': 0,
            'skipped': 0,
            'errors': 0,
            'details': []
        }
        
        for user in eligible_users:
            user_id = user['user_id']  # Use user_id instead of id
            user_email = user['email']  # Email is already in the user object
            
            try:
                # Get user's preferences for filtering (still from user_preferences table)
                user_preferences = self.get_user_preferences(user_id)
                
                # Filter deals based on user's preferences
                user_deals = self.filter_deals_for_user(new_deals, user, user_preferences)
                
                if not user_deals:
                    print(f"📭 No relevant deals for user {user_email}")
                    results['skipped'] += 1
                    continue
                
                # Check notification frequency from notification_preferences
                notification_prefs = user.get('notification_preferences', {})
                frequency = notification_prefs.get('frequency', 'daily')
                
                if frequency == 'instant':
                    # Send immediately
                    pass
                elif frequency == 'daily':
                    # For daily notifications, we'll send them during the main scraping session
                    # which typically runs once per day
                    pass
                elif frequency == 'weekly':
                    # For weekly notifications, check if it's the user's preferred day
                    # For now, we'll include weekly users in daily processing
                    # and add day-of-week logic later
                    pass
                else:
                    print(f"⚠️  Unknown notification frequency: {frequency}")
                    results['skipped'] += 1
                    continue
                
                # Send notification
                notification_result = self.notification_service.send_deal_notification(
                    user_email=user_email,
                    deals=user_deals,
                    user_preferences=user_preferences
                )
                
                if notification_result['success']:
                    results['sent'] += 1
                    print(f"✅ Sent {len(user_deals)} deals to {user_email}")
                else:
                    results['errors'] += 1
                    print(f"❌ Failed to send to {user_email}: {notification_result.get('error')}")
                
                results['details'].append({
                    'user_email': user_email,
                    'deals_count': len(user_deals),
                    'success': notification_result['success'],
                    'error': notification_result.get('error')
                })
                
            except Exception as e:
                results['errors'] += 1
                print(f"❌ Error processing notifications for user {user_id}: {e}")
                results['details'].append({
                    'user_id': user_id,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"📧 Notification summary: {results['sent']} sent, {results['skipped']} skipped, {results['errors']} errors")
        results['success'] = results['errors'] == 0
        return results
    
    def process_daily_notifications(self) -> Dict:
        """
        Main function to process daily deal notifications
        This should be called after the daily scraping is complete
        """
        print(f"\n{'='*60}")
        print("📧 PROCESSING DAILY DEAL NOTIFICATIONS")
        print(f"{'='*60}")
        
        try:
            # Get new deals from the last 24 hours
            new_deals = self.get_new_deals_for_notification(hours_back=24)
            
            if not new_deals:
                print("📭 No new deals found in the last 24 hours")
                return {'success': True, 'sent': 0, 'skipped': 0, 'errors': 0, 'message': 'No new deals'}
            
            # Send notifications
            results = self.send_notifications_for_new_deals(new_deals)
            
            print(f"\n✅ Daily notification processing complete:")
            print(f"📊 New deals processed: {len(new_deals)}")
            print(f"📧 Notifications sent: {results['sent']}")
            print(f"⏭️  Users skipped: {results['skipped']}")
            print(f"❌ Errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in daily notification processing: {e}")
            return {'success': False, 'error': str(e), 'sent': 0, 'skipped': 0, 'errors': 1}


def test_notification_pipeline():
    """Test the notification pipeline with current deals"""
    pipeline = DealNotificationPipeline()
    
    print("🧪 Testing notification pipeline...")
    
    # Test getting eligible users
    users = pipeline.get_notification_eligible_users()
    print(f"📧 Found {len(users)} eligible users")
    
    if users:
        # Test with a sample user
        user = users[0]
        print(f"👤 Testing with user: {user['user_id']} ({user['name']})")
        print(f"📧 User email: {user['email']}")
        
        preferences = pipeline.get_user_preferences(user['user_id'])
        print(f"⚙️  User preferences: {preferences is not None}")
    
    # Test getting new deals
    new_deals = pipeline.get_new_deals_for_notification(hours_back=168)  # 7 days for testing
    print(f"🆕 Found {len(new_deals)} recent deals")
    
    if new_deals:
        print("📋 Sample deal:")
        sample_deal = new_deals[0]
        print(f"   {sample_deal.get('make')} {sample_deal.get('model')} - £{sample_deal.get('price_numeric', 0):,}")
        print(f"   Profit: {sample_deal.get('profit_potential_pct', 0):.1f}%")
        print(f"   Rating: {sample_deal.get('deal_rating')}")
    
    # Test the full pipeline
    if len(users) > 0 and len(new_deals) > 0:
        print(f"\n🚀 Testing full notification pipeline...")
        results = pipeline.process_daily_notifications()
        print(f"📊 Pipeline results: {results}")

if __name__ == "__main__":
    test_notification_pipeline()
