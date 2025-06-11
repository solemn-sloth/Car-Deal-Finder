import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

class SupabaseStorage:
    """Class to handle Supabase storage operations"""
    
    def __init__(self, table_name='car_deals'):
        """Initialize Supabase connection"""
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables. Please set SUPABASE_URL and SUPABASE_KEY.")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name
        
        print(f"Supabase client initialized for table: {table_name}")
    
    def get_table_stats(self):
        """Get table statistics"""
        try:
            # Count total rows
            response = self.supabase.table(self.table_name).select('count', count='exact').execute()
            count = response.count if hasattr(response, 'count') else 0
            
            return {
                'status': 'available',
                'item_count': count,
                'table_name': self.table_name
            }
        except Exception as e:
            print(f"Error getting table stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def save_deal(self, deal):
        """Save a single deal to Supabase"""
        try:
            # Generate a unique ID if not present
            if 'deal_id' not in deal:
                deal['deal_id'] = str(uuid.uuid4())
            
            # Add timestamps
            current_time = datetime.now().isoformat()
            deal['created_at'] = deal.get('created_at', current_time)
            deal['updated_at'] = current_time
            
            # Save to Supabase
            response = self.supabase.table(self.table_name).insert(deal).execute()
            
            # Check for errors
            if hasattr(response, 'error') and response.error:
                return {"success": False, "error": response.error}
            
            return {"success": True, "deal_id": deal['deal_id']}
            
        except Exception as e:
            print(f"Error saving deal to Supabase: {e}")
            return {"success": False, "error": str(e)}
    
    def save_deals(self, deals):
        """Save multiple deals to Supabase"""
        results = {"total": len(deals), "saved": 0, "failed": 0, "errors": []}
        
        for deal in deals:
            result = self.save_deal(deal)
            if result["success"]:
                results["saved"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(result["error"])
        
        return results
    
    def get_all_deals(self):
        """Get all deals from Supabase"""
        try:
            # Query data
            response = self.supabase.table(self.table_name).select('*').execute()
            
            # Handle the response
            if hasattr(response, 'data'):
                return response.data
            return []
            
        except Exception as e:
            print(f"Error getting deals from Supabase: {e}")
            return []
    
    def get_deals_by_profit_potential(self, min_profit=0, limit=50):
        """Get deals filtered by minimum profit potential"""
        try:
            # Query with filter for profit potential
            response = self.supabase.table(self.table_name)\
                .select('*')\
                .gte('profit_potential_pct', min_profit)\
                .order('profit_potential_pct', desc=True)\
                .limit(limit)\
                .execute()
            
            # Handle the response
            if hasattr(response, 'data'):
                return response.data
            return []
            
        except Exception as e:
            print(f"Error getting deals by profit potential from Supabase: {e}")
            return []
    
    def delete_deal(self, deal_id):
        """Delete a deal from Supabase"""
        try:
            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq('deal_id', deal_id)\
                .execute()
                
            # Check for errors
            if hasattr(response, 'error') and response.error:
                return {"success": False, "error": response.error}
                
            return {"success": True}
            
        except Exception as e:
            print(f"Error deleting deal from Supabase: {e}")
            return {"success": False, "error": str(e)}
    
    def update_deal(self, deal_id, updated_data):
        """Update a deal in Supabase"""
        try:
            # Add updated timestamp
            updated_data['updated_at'] = datetime.now().isoformat()
            
            # Update in Supabase
            response = self.supabase.table(self.table_name)\
                .update(updated_data)\
                .eq('deal_id', deal_id)\
                .execute()
                
            # Check for errors
            if hasattr(response, 'error') and response.error:
                return {"success": False, "error": response.error}
                
            return {"success": True}
            
        except Exception as e:
            print(f"Error updating deal in Supabase: {e}")
            return {"success": False, "error": str(e)}
