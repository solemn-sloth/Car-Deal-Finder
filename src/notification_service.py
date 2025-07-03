"""
Email notification service for Car Dealer Bot
Handles sending deal notifications to users via Resend
"""
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NotificationService:
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
                <p>Happy dealing! 🎯</p>
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

# Test function
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
            'profit_potential_pct': 18.1,
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

if __name__ == "__main__":
    test_notification_service()
