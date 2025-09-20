"""
Email notification service and pipeline for Car Dealer Bot
Handles sending deal notifications to users via Resend and manages the notification workflow
"""
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
from pathlib import Path
from src.storage import SupabaseStorage

# Load environment variables from config folder
config_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'config' / '.env'
load_dotenv(dotenv_path=config_path)


class NotificationService:
    """Core email notification service using Resend API"""
    
    def __init__(self):
        # Use environment variable for production, fallback for development
        self.api_key = os.getenv('RESEND_API_KEY', 're_XgwebpPG_MCUazVt1Vs5QQwacuLWGQaA2')
        self.base_url = "https://api.resend.com/emails"
        self.from_email = "Car Dealer Bot <noreply@cardealerbot.co.uk>"
        
    def send_deal_notification(self, user_email: str, deals: List[Dict], user_preferences: Dict = None) -> Dict:
        """
        Send deal notification email to a user
        
        Args:
            user_email: User's email address
            deals: List of deal dictionaries
            user_preferences: User's notification preferences
            
        Returns:
            Dict with success status and email ID
        """
        try:
            # Generate email content
            subject = self._generate_subject(deals, user_preferences)
            html_content = self._generate_html_content(deals, user_preferences)
            
            # Send email
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "from": self.from_email,
                "to": [user_email],
                "subject": subject,
                "html": html_content,
                "reply_to": "support@cardealerbot.co.uk"  # Can be changed to your actual email
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                email_id = response.json().get('id')
                print(f"✅ Deal notification sent to {user_email} (ID: {email_id})")
                return {
                    "success": True,
                    "email_id": email_id,
                    "user_email": user_email,
                    "deals_count": len(deals)
                }
            else:
                print(f"❌ Failed to send email to {user_email}: {response.status_code}")
                print(f"Response: {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "user_email": user_email
                }
                
        except Exception as e:
            print(f"❌ Exception sending email to {user_email}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "user_email": user_email
            }
    
    def send_daily_digest(self, user_email: str, daily_deals: List[Dict], user_preferences: Dict = None) -> Dict:
        """Send daily digest of deals to a user"""
        if not daily_deals:
            print(f"ℹ️  No deals to send to {user_email}")
            return {"success": True, "skipped": True, "reason": "no_deals"}
            
        return self.send_deal_notification(user_email, daily_deals, user_preferences)
    
    def _generate_subject(self, deals: List[Dict], user_preferences: Dict = None) -> str:
        """Generate email subject line"""
        deal_count = len(deals)
        
        if deal_count == 1:
            deal = deals[0]
            return f"🚗 New Deal: {deal.get('make', '')} {deal.get('model', '')} - £{deal.get('price_numeric', 0):,}"
        else:
            return f"🚗 {deal_count} New Car Deals Found - Car Dealer Bot"
    
    def _generate_html_content(self, deals: List[Dict], user_preferences: Dict = None) -> str:
        """Generate HTML email content"""
        
        # Calculate total potential profit
        total_profit = sum(deal.get('absolute_profit', 0) for deal in deals)
        
        # Generate deal cards HTML
        deal_cards = ""
        for deal in deals:
            deal_cards += self._generate_deal_card(deal)
        
        # Main email template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New Car Deals</title>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            
            <!-- Header -->
            <div style="text-align: center; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px;">
                <h1 style="margin: 0; font-size: 28px;">🚗 Car Dealer Bot</h1>
                <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;">New profitable deals found!</p>
            </div>
            
            <!-- Summary -->
            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #059669;">
                <h2 style="margin: 0 0 15px 0; color: #059669;">📊 Summary</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #2563eb;">{len(deals)}</div>
                        <div style="color: #6b7280; font-size: 14px;">New Deals</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #059669;">£{total_profit:,.0f}</div>
                        <div style="color: #6b7280; font-size: 14px;">Total Profit Potential</div>
                    </div>
                </div>
            </div>
            
            <!-- Deals -->
            <h2 style="color: #1f2937; margin-bottom: 20px;">🔥 Your Deals</h2>
            {deal_cards}
            
            <!-- Footer -->
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 14px;">
                <p>Happy trading! 🎯</p>
                <p>
                    <a href="https://cardealerbot.co.uk" style="color: #2563eb; text-decoration: none;">Visit Dashboard</a> | 
                    <a href="https://cardealerbot.co.uk/preferences" style="color: #2563eb; text-decoration: none;">Update Preferences</a>
                </p>
                <p style="margin-top: 20px;">
                    <small>This email was sent by Car Dealer Bot because you have notifications enabled. 
                    <a href="https://cardealerbot.co.uk/unsubscribe" style="color: #6b7280;">Unsubscribe</a></small>
                </p>
            </div>
            
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_deal_card(self, deal: Dict) -> str:
        """Generate HTML for a single deal card"""
        
        # Format values safely
        make = deal.get('make', 'Unknown')
        model = deal.get('model', 'Unknown')
        price = deal.get('price_numeric', 0)
        profit = deal.get('absolute_profit', 0)
        profit_pct = deal.get('profit_potential_pct', 0)
        year = deal.get('year', 'Unknown')
        mileage = deal.get('mileage', 0)
        location = deal.get('location', 'Unknown')
        rating = deal.get('deal_rating', 'Good Deal')
        url = deal.get('url', '#')
        image_url = deal.get('image_url', '')
        
        # Rating colors
        rating_colors = {
            'Excellent Deal': '#059669',
            'Good Deal': '#2563eb', 
            'Negotiation Target': '#d97706'
        }
        rating_color = rating_colors.get(rating, '#6b7280')
        
        # Deal card HTML
        card_html = f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 12px; margin-bottom: 20px; overflow: hidden; background: white;">
            
            <!-- Image -->
            {"<img src='" + image_url + "' style='width: 100%; height: 200px; object-fit: cover;' alt='Car Image'>" if image_url else ""}
            
            <!-- Content -->
            <div style="padding: 20px;">
                
                <!-- Title and Rating -->
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: #1f2937; font-size: 20px;">{make} {model}</h3>
                    <span style="background: {rating_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{rating}</span>
                </div>
                
                <!-- Key Details -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; font-size: 14px; color: #6b7280;">
                    <div><strong>Year:</strong> {year}</div>
                    <div><strong>Mileage:</strong> {mileage:,} miles</div>
                    <div><strong>Location:</strong> {location}</div>
                    <div><strong>Seller:</strong> {deal.get('seller_type', 'Unknown')}</div>
                </div>
                
                <!-- Price and Profit -->
                <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <div style="font-size: 18px; font-weight: bold; color: #1f2937;">£{price:,}</div>
                            <div style="color: #6b7280; font-size: 12px;">Asking Price</div>
                        </div>
                        <div>
                            <div style="font-size: 18px; font-weight: bold; color: #059669;">£{profit:,.0f}</div>
                            <div style="color: #6b7280; font-size: 12px;">Profit Potential ({profit_pct * 100:.1f}%)</div>
                        </div>
                    </div>
                </div>
                
                <!-- Action Button -->
                <a href="{url}" style="display: block; background: #2563eb; color: white; text-decoration: none; padding: 12px 20px; border-radius: 6px; text-align: center; font-weight: bold; transition: background-color 0.2s;">
                    View Deal on AutoTrader →
                </a>
                
            </div>
        </div>
        """
        
        return card_html


class DealNotificationPipeline:
    """Pipeline for managing notification workflow and user preferences"""
    
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
            
            # Filter users who have email notifications enabled
            eligible_users = []
            for user in users:
                prefs = user.get('notification_preferences', {})
                
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
                
                if email_enabled and user.get('email'):
                    eligible_users.append(user)
            
            # Users found for notifications - no detailed log
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
        
        # Filtering deals for user - silent operation
        
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
                # Deal type filtering done silently
        
        # Filter by price range if specified
        price_range = notification_prefs.get('price_range', {})
        if price_range.get('min'):
            before_count = len(filtered_deals)
            filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) >= price_range['min']]
            # Min price filtering done silently
        if price_range.get('max'):
            before_count = len(filtered_deals)
            filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) <= price_range['max']]
            # Max price filtering done silently
        
        # Filter by maximum distance if specified
        max_distance = notification_prefs.get('max_distance')
        if max_distance and user.get('postcode'):
            # For now, we'll skip distance filtering as it requires geolocation logic
            # This can be implemented later with postcode distance calculation
            pass  # Distance filtering not implemented yet
        
        # Apply user's general deal preferences if available (but don't be too restrictive)
        if user_preferences:
            # Apply user preferences silently
            # Min/max price (override notification prefs if more restrictive)
            if user_preferences.get('min_price_numeric'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) >= user_preferences['min_price_numeric']]
                # User min price filtering done silently
            if user_preferences.get('max_price_numeric'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('price_numeric', 0) <= user_preferences['max_price_numeric']]
                # User max price filtering done silently
            
            # Max mileage
            if user_preferences.get('max_mileage'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('mileage', 0) <= user_preferences['max_mileage']]
                # Mileage filtering done silently
            
            # Min year
            if user_preferences.get('min_year'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('year', 0) >= user_preferences['min_year']]
                # Min year filtering done silently
            
            # Max year  
            if user_preferences.get('max_year'):
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('year', 0) <= user_preferences['max_year']]
                # Max year filtering done silently
            
            # Preferred makes
            preferred_makes = user_preferences.get('preferred_makes', [])
            if preferred_makes:
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('make') in preferred_makes]
                # Preferred makes filtering done silently
            
            # Preferred models
            preferred_models = user_preferences.get('preferred_models', [])
            if preferred_models:
                before_count = len(filtered_deals)
                filtered_deals = [d for d in filtered_deals if d.get('model') in preferred_models]
                # Preferred models filtering done silently
        
        # Final filter count - no log needed
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
            # Deals found for notifications - no detailed log
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
            from src.output_manager import get_output_manager
            output_manager = get_output_manager()
            output_manager.info("No new deals to notify about")
            return {'success': True, 'sent': 0, 'skipped': 0, 'errors': 0}
        
        from src.output_manager import get_output_manager
        output_manager = get_output_manager()
        output_manager.info(f"Processing notifications for {len(new_deals)} new deals...")
        
        # Get users eligible for notifications
        eligible_users = self.get_notification_eligible_users()
        
        if not eligible_users:
            output_manager.info("No users eligible for notifications")
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
                    # No relevant deals for this user
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
                    # Deal sent successfully - log later in summary
                else:
                    results['errors'] += 1
                    # Error logged in details, no need to print here
                
                results['details'].append({
                    'user_email': user_email,
                    'deals_count': len(user_deals),
                    'success': notification_result['success'],
                    'error': notification_result.get('error')
                })
                
            except Exception as e:
                results['errors'] += 1
                # Error captured in results, not printing individually
                results['details'].append({
                    'user_id': user_id,
                    'success': False,
                    'error': str(e)
                })
        
        output_manager.info(f"Notification summary: {results['sent']} sent, {results['skipped']} skipped, {results['errors']} errors")
        results['success'] = results['errors'] == 0
        return results
    
    def process_daily_notifications(self) -> Dict:
        """
        Main function to process daily deal notifications
        This should be called after the daily scraping is complete
        """
        # Process daily notifications quietly without printing status messages
        try:
            # Get new deals from the last 24 hours
            new_deals = self.get_new_deals_for_notification(hours_back=24)
            
            if not new_deals:
                # No new deals found
                return {'success': True, 'sent': 0, 'skipped': 0, 'errors': 0, 'message': 'No new deals'}
            
            # Send notifications
            results = self.send_notifications_for_new_deals(new_deals)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in daily notification processing: {e}")
            return {'success': False, 'error': str(e), 'sent': 0, 'skipped': 0, 'errors': 1}


# Test functions
def test_notification_service():
    """Test the notification service with sample data"""
    service = NotificationService()
    
    # Sample deal data
    sample_deals = [
        {
            'make': 'BMW',
            'model': '3 Series',
            'price_numeric': 15500,
            'absolute_profit': 2800,
            'profit_potential_pct': 0.181,
            'year': 2018,
            'mileage': 45000,
            'location': 'London',
            'deal_rating': 'Excellent Deal',
            'seller_type': 'Dealer',
            'url': 'https://www.autotrader.co.uk/car-details/sample',
            'image_url': 'https://via.placeholder.com/400x200?text=BMW+3+Series'
        }
    ]
    
    # Test email
    test_email = input("Enter your email to test deal notification: ")
    result = service.send_deal_notification(test_email, sample_deals)
    
    if result['success']:
        print(f"✅ Test notification sent successfully!")
        print(f"📧 Email ID: {result['email_id']}")
    else:
        print(f"❌ Test failed: {result['error']}")


def test_notification_pipeline():
    """Test the notification pipeline with current deals"""
    pipeline = DealNotificationPipeline()
    
    print("🧪 Testing notification pipeline...")
    
    # Test getting eligible users
    users = pipeline.get_notification_eligible_users()
    print(f"📋 Found {len(users)} eligible users")
    
    # Test getting new deals (last 7 days for testing)
    new_deals = pipeline.get_new_deals_for_notification(hours_back=168)
    print(f"🚗 Found {len(new_deals)} deals from last 7 days")
    
    # Test the full pipeline if we have users and deals
    if len(users) > 0 and len(new_deals) > 0:
        print(f"🚀 Running notification test with {len(users)} users and {len(new_deals)} deals...")
        results = pipeline.process_daily_notifications()
        print(f"✅ Test complete: {results['sent']} emails sent")
    else:
        print("⚠️  Need at least 1 user and 1 deal to test full pipeline")


if __name__ == "__main__":
    print("Choose test option:")
    print("1. Test email service only (with sample data)")
    print("2. Test full notification pipeline (with real data)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_notification_service()
    elif choice == "2":
        test_notification_pipeline()
    else:
        print("Invalid choice. Running full pipeline test by default.")
        test_notification_pipeline()