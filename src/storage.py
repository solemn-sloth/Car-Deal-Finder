import os
from datetime import datetime, timedelta
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

# Load environment variables from config folder
config_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'config' / '.env'
load_dotenv(dotenv_path=config_path)

def extract_deal_ids_from_url(url: str) -> list:
    """
    Extract any deal IDs from a URL.
    This helps identify if a deal is already in the database but with a slightly different URL.
    
    Args:
        url: URL to extract deal IDs from
        
    Returns:
        List of extracted deal IDs
    """
    deal_ids = []
    try:
        import re
        # Look for car-details/NUMBER pattern
        matches = re.findall(r'car-details/(\d+)', url)
        if matches:
            deal_ids.extend(matches)
            
        # Look for classified/advert/NUMBER pattern
        matches = re.findall(r'classified/advert/(\d+)', url)
        if matches:
            deal_ids.extend(matches)
            
        # Look for any other numeric IDs that might be deal identifiers
        # Only include IDs with 10+ digits which are likely actual deal IDs
        matches = re.findall(r'/([0-9]{10,})(?:/|\?|$)', url)
        if matches:
            deal_ids.extend(matches)
            
        return deal_ids
    except Exception as e:
        # If parsing fails, log the error and return empty list
        print(f"Error extracting deal IDs: {e}")
        return []

def normalize_autotrader_url(url: str) -> str:
    """
    Normalize AutoTrader URL by ensuring the search parameters are preserved.
    This maintains the 'return to search results' functionality for users.
    """
    try:
        # For AutoTrader URLs, ensure we're keeping all search parameters
        if "autotrader.co.uk/car-details" in url:
            # Clean up any unnecessary tracking parameters if needed
            # but preserve search-related parameters
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            
            # Parse the URL
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # Keep only search-related parameters (this list can be adjusted)
            search_params = ['advertising-location', 'page', 'sort', 'postcode', 
                             'radius', 'year-from', 'year-to', 'price-from', 
                             'price-to', 'include-delivery-option', 'body-type',
                             'fuel-type', 'transmission', 'exclude-writeoff-categories']
            
            # Filter to keep only search-related parameters
            # Note: We're keeping ALL parameters to ensure search functionality works
            filtered_params = query_params
            
            # Reconstruct the URL with preserved parameters
            new_query = urlencode(filtered_params, doseq=True)
            new_url_parts = list(parsed_url)
            new_url_parts[4] = new_query
            
            return urlunparse(new_url_parts)
        
        # For other URLs, return as-is
        return url
    except Exception as e:
        # If parsing fails, log the error and return original URL
        print(f"Error normalizing URL: {e}")
        return url

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

        self._valid_columns = None

        
        # Supabase client initialized silently

    def get_valid_columns(self):
        """
        Get a list of valid column names from the database schema
        Uses caching to reduce database queries
        """
        if self._valid_columns is not None:
            return self._valid_columns
        
        try:
            # Query information_schema for column names
            response = self.supabase.table('information_schema.columns')\
                .select('column_name')\
                .eq('table_name', self.table_name)\
                .execute()
            
            if hasattr(response, 'data') and response.data:
                self._valid_columns = [col['column_name'] for col in response.data]
                # Retrieved valid columns
                return self._valid_columns
            
            # No columns found for table
            return []
        except Exception as e:
            print(f"❌ Error getting valid columns: {e}")
            # Fallback to an empty list if we can't get columns
            return []

    def filter_deal_fields(self, deal):
        """
        Filter deal object in-place to only include keys that exist as columns in the database
        """
        valid_columns = self.get_valid_columns()
        if not valid_columns:
            # No valid columns found. Proceeding with unfiltered data.
            return
        
        # Find keys to remove
        keys_to_remove = [k for k in list(deal.keys()) if k not in valid_columns]
        
        # Remove invalid keys
        for key in keys_to_remove:
            deal.pop(key, None)
        
        # Filtered out non-existent columns
        return deal

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
        """Save a single deal to Supabase with duplicate checking"""
        try:
            # Add this line to filter the deal fields in-place
            self.filter_deal_fields(deal)
        
            # Check if deal with this URL already exists
            url = deal.get('url')
            if url:
                existing = self.supabase.table(self.table_name)\
                    .select('id')\
                    .eq('url', url)\
                    .limit(1)\
                    .execute()
                
                if existing.data:
                    # Deal exists, update it
                    deal_id = existing.data[0]['id']
                    deal['updated_at'] = datetime.now().isoformat()
                    
                    response = self.supabase.table(self.table_name)\
                        .update(deal)\
                        .eq('id', deal_id)\
                        .execute()
                    
                    if hasattr(response, 'error') and response.error:
                        return {"success": False, "error": response.error}
                    
                    return {"success": True, "deal_id": deal_id, "action": "updated"}
                
                # Deal doesn't exist, insert new one
                if 'deal_id' not in deal:
                    deal['deal_id'] = str(uuid.uuid4())
                
                current_time = datetime.now().isoformat()
                deal['created_at'] = deal.get('created_at', current_time)
                deal['updated_at'] = current_time
                
                response = self.supabase.table(self.table_name).insert(deal).execute()
                
                if hasattr(response, 'error') and response.error:
                    return {"success": False, "error": response.error}
                
                return {"success": True, "deal_id": deal['deal_id'], "action": "inserted"}
            
            return {"success": False, "error": "No URL provided"}
            
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
    
    def normalize_url(self, url):
        """
        Normalize URL by removing query parameters to identify true duplicates.
        Example: '/classified/advert/202505052069381?sort=price-asc&searchId=abc123'
        Returns: '/classified/advert/202505052069381'
        """
        if not url:
            return url
        
        # Remove everything after the first '?'
        base_url = url.split('?')[0]
        return base_url
    
    def clean_all_duplicates(self, model, make=None):
        """
        Find and remove ALL duplicate deals based on normalized URLs.
        Keeps only the most recent version of each unique car.
        
        Returns:
            dict: Results with duplicate cleanup statistics
        """
        try:
            filter_desc = f"{make} {model}" if make else model
            # Starting duplicate cleanup
            
            # Get all existing deals for this model
            if make:
                existing_response = self.supabase.table(self.table_name)\
                    .select('*')\
                    .eq('make', make)\
                    .eq('model', model)\
                    .execute()
            else:
                existing_response = self.supabase.table(self.table_name)\
                    .select('*')\
                    .eq('model', model)\
                    .execute()
            
            if not existing_response.data:
                # No existing deals found
                return {"success": True, "duplicates_removed": 0}
            
            # Group deals by normalized URL
            url_groups = {}
            for deal in existing_response.data:
                normalized_url = self.normalize_url(deal.get('url', ''))
                if normalized_url:
                    if normalized_url not in url_groups:
                        url_groups[normalized_url] = []
                    url_groups[normalized_url].append(deal)
            
            # Find and remove duplicates (keep most recent)
            duplicates_to_remove = []
            total_duplicates = 0
            
            for normalized_url, deals_group in url_groups.items():
                if len(deals_group) > 1:
                    # Sort by updated_at or created_at, keep the most recent
                    deals_group.sort(key=lambda x: x.get('updated_at', x.get('created_at', '')), reverse=True)
                    
                    # Keep the first (most recent), remove the rest
                    deals_to_remove = deals_group[1:]  # All except the first (most recent)
                    duplicates_to_remove.extend(deals_to_remove)
                    total_duplicates += len(deals_to_remove)
                    
                    # Duplicates found - keeping most recent and removing others
            
            # Remove duplicates in batches
            removed_count = 0
            if duplicates_to_remove:
                # Removing duplicate deals silently
                
                # Remove duplicates one by one (safer for RLS)
                for deal in duplicates_to_remove:
                    try:
                        delete_response = self.supabase.table(self.table_name)\
                            .delete()\
                            .eq('deal_id', deal['deal_id'])\
                            .execute()
                        
                        if not (hasattr(delete_response, 'error') and delete_response.error):
                            removed_count += 1
                        
                    except Exception as e:
                        # Failed to remove duplicate - handled silently
                        pass
            
            # Duplicate cleanup complete
            
            return {
                "success": True,
                "duplicates_removed": removed_count,
                "total_duplicates_found": total_duplicates
            }
            
        except Exception as e:
            error_msg = f"Error during duplicate cleanup: {e}"
            print(f"   ❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "duplicates_removed": 0
            }

    def save_deals_with_upsert(self, deals, model, make=None):
        """
        Save deals using comprehensive URL-based deduplication.
        
        Process:
        1. Clean ALL existing duplicates first
        2. Get remaining unique URLs
        3. Insert only truly new deals
        
        Args:
            deals: List of deal dictionaries to save
            model: Model name (e.g., "Sportage") 
            make: Optional make name (e.g., "Kia")
            
        Returns:
            dict: Results with counts of inserted/skipped/cleaned deals
        """
        try:
            filter_desc = f"{make} {model}" if make else model
            
            if not deals:
                # No deals to process
                return {
                    "success": True,
                    "model": model,
                    "make": make,
                    "inserted_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "total_processed": 0,
                    "duplicates_cleaned": 0
                }

            # STEP 1: Clean ALL existing duplicates first
            cleanup_result = self.clean_all_duplicates(model, make)
            duplicates_cleaned = cleanup_result.get('duplicates_removed', 0)
            
            if not cleanup_result['success']:
                # Duplicate cleanup failed
                pass

            # STEP 2: Get remaining unique URLs after cleanup
            if make:
                existing_response = self.supabase.table(self.table_name)\
                    .select('url')\
                    .eq('make', make)\
                    .eq('model', model)\
                    .execute()
            else:
                existing_response = self.supabase.table(self.table_name)\
                    .select('url')\
                    .eq('model', model)\
                    .execute()
            
            # Normalize existing URLs for comparison
            existing_normalized_urls = set()
            if existing_response.data:
                for row in existing_response.data:
                    url = row.get('url', '')
                    if url:
                        normalized_url = self.normalize_url(url)
                        existing_normalized_urls.add(normalized_url)
            
            # Found existing URLs to check against
            
            # STEP 3: Process each scraped deal with normalized URL comparison
            current_time = datetime.now()
            inserted_count = 0
            skipped_count = 0
            
            deals_to_insert = []
            
            for deal in deals:
                deal_url = deal.get('url', '')
                
                if not deal_url:
                    # Skipping deal without URL
                    skipped_count += 1
                    continue
                
                # Normalize the deal URL for comparison
                normalized_deal_url = self.normalize_url(deal_url)
                
                if normalized_deal_url in existing_normalized_urls:
                    # Normalized URL already exists - skip to prevent duplicate
                    skipped_count += 1
                    # URL already exists, skipping
                    continue
                
                # New normalized URL - prepare for insertion
                deal['updated_at'] = current_time.isoformat()
                deal['date_added'] = current_time.strftime('%Y-%m-%d')
                deal['created_at'] = current_time.isoformat()
                
                if 'deal_id' not in deal:
                    deal['deal_id'] = str(uuid.uuid4())
                
                deals_to_insert.append(deal)
                inserted_count += 1
                # Will insert this deal
                
                
                # Add to existing set to prevent duplicates within this batch
                existing_normalized_urls.add(normalized_deal_url)
            
            # STEP 4: Perform batch insert if we have new deals
            if deals_to_insert:
                try:
                    insert_response = self.supabase.table(self.table_name)\
                        .insert(deals_to_insert)\
                        .execute()
                    
                    if hasattr(insert_response, 'error') and insert_response.error:
                        # Batch insert failed
                        return {
                            "success": False,
                            "error": str(insert_response.error),
                            "model": model,
                            "make": make,
                            "duplicates_cleaned": duplicates_cleaned
                        }
                    
                    # Batch insert completed successfully
                    
                except Exception as e:
                    # Insert error - returning with error message
                    return {
                        "success": False,
                        "error": str(e),
                        "model": model,
                        "make": make,
                        "duplicates_cleaned": duplicates_cleaned
                    }
            
            total_processed = inserted_count + skipped_count
            
            # Processing complete - returning results silently
            
            return {
                "success": True,
                "model": model,
                "make": make,
                "inserted_count": inserted_count,
                "updated_count": 0,  # No updates due to RLS policies
                "skipped_count": skipped_count,
                "total_processed": total_processed,
                "duplicates_cleaned": duplicates_cleaned,
                "timestamp": current_time.isoformat(),
                "approach": "comprehensive_url_deduplication"
            }
                
        except Exception as e:
            error_msg = f"Error inserting {model} deals: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "model": model,
                "make": make,
                "duplicates_cleaned": 0
            }
    
    def get_fresh_deals(self, hours_fresh=24, limit=None):
        """
        Get deals that are fresh (updated within specified hours).
        This is the recommended way to get current deals.
        
        Args:
            hours_fresh: Number of hours to consider as "fresh"
            limit: Optional limit on number of results
            
        Returns:
            list: Fresh deals from database
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours_fresh)).isoformat()
            
            query = self.supabase.table(self.table_name)\
                .select('*')\
                .gt('updated_at', cutoff_time)\
                .order('updated_at', desc=True)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if hasattr(response, 'data'):
                return response.data
            return []
            
        except Exception as e:
            print(f"Error getting fresh deals: {e}")
            return []
    
    def get_fresh_deals_by_model(self, model, make=None, hours_fresh=24):
        """
        Get fresh deals for a specific model.
        
        Args:
            model: Model name
            make: Optional make name
            hours_fresh: Number of hours to consider as "fresh"
            
        Returns:
            list: Fresh deals for the specified model
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours_fresh)).isoformat()
            
            query = self.supabase.table(self.table_name)\
                .select('*')\
                .eq('model', model)\
                .gt('updated_at', cutoff_time)\
                .order('updated_at', desc=True)
            
            if make:
                query = query.eq('make', make)
            
            response = query.execute()
            
            if hasattr(response, 'data'):
                return response.data
            return []
            
        except Exception as e:
            print(f"Error getting fresh deals for {model}: {e}")
            return []
    
    def cleanup_old_deals(self, days_old=7):
        """
        Remove deals older than X days as backup cleanup.
        This serves as a safety net for any stale data that might accumulate.
        Uses the updated_at field from existing schema.
        
        Args:
            days_old: Number of days after which deals are considered stale
            
        Returns:
            dict: Results with count of cleaned deals
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            # Delete deals older than cutoff date using updated_at field
            delete_response = self.supabase.table(self.table_name)\
                .delete()\
                .lt('updated_at', cutoff_date)\
                .execute()
            
            deleted_count = len(delete_response.data) if delete_response.data else 0
            print(f"🧹 Cleaned up {deleted_count} deals older than {days_old} days")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date
            }
            
        except Exception as e:
            error_msg = f"Error during cleanup: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_batch_stats(self):
        """
        Get statistics about recent batches and data freshness.
        Uses existing schema fields for tracking.
        
        Returns:
            dict: Statistics about batches and data freshness
        """
        try:
            # Get total count
            total_response = self.supabase.table(self.table_name)\
                .select('count', count='exact')\
                .execute()
            total_count = total_response.count if hasattr(total_response, 'count') else 0
            
            # Get fresh deals (last 24 hours) using updated_at field
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            
            fresh_response = self.supabase.table(self.table_name)\
                .select('count', count='exact')\
                .gt('updated_at', yesterday)\
                .execute()
            fresh_count = fresh_response.count if hasattr(fresh_response, 'count') else 0
            
            # Get unique models with recent data
            models_response = self.supabase.table(self.table_name)\
                .select('model')\
                .gt('updated_at', yesterday)\
                .execute()
            
            unique_models = set()
            if models_response.data:
                unique_models = {deal['model'] for deal in models_response.data}
            
            return {
                "total_deals": total_count,
                "fresh_deals_24h": fresh_count,
                "unique_models_24h": len(unique_models),
                "models_updated_24h": list(unique_models),
                "data_freshness_pct": (fresh_count / total_count * 100) if total_count > 0 else 0
            }
            
        except Exception as e:
            print(f"Error getting batch stats: {e}")
            return {"error": str(e)}
    
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
    
    def sync_deals_intelligently(self, new_deals):
        """
        Intelligent deal synchronization that:
        1. Adds new deals found in scrape
        2. Removes deals that are no longer available (sold/removed)
        3. Never uses time-based deletion
        
        This prevents the aggressive cleanup that deleted all deals after 24h.
        
        Args:
            new_deals: List of deal dictionaries from current scrape
            
        Returns:
            dict: Results with counts of added/removed/updated deals
        """
        try:
            # Starting intelligent deal synchronization
            
            # Create service role client to bypass RLS
            supabase_url = os.getenv('SUPABASE_URL')
            service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            if not service_role_key:
                raise ValueError("SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
                
            service_client = create_client(supabase_url, service_role_key)
            
            # Get URLs from new deals (normalize them)
            new_deal_urls = set()
            deals_by_url = {}
            
            for deal in new_deals:
                url = deal.get('url')
                if url:
                    normalized_url = normalize_autotrader_url(url)
                    new_deal_urls.add(normalized_url)
                    deals_by_url[normalized_url] = deal
            
            # Found unique deals in current scrape
            
            # Get all existing URLs from database
            existing_response = service_client.table(self.table_name)\
                .select('url, id')\
                .execute()
            
            existing_urls = set()
            existing_url_to_id = {}
            
            if existing_response.data:
                for row in existing_response.data:
                    url = row.get('url')
                    if url:
                        normalized_url = normalize_autotrader_url(url)
                        existing_urls.add(normalized_url)
                        existing_url_to_id[normalized_url] = row['id']
            
            # Found existing deals in database
            
            # Keep track of deal IDs from the current scrape
            current_deal_ids = set()
            for deal in new_deals:
                # First check if the deal already has a deal_id field
                if deal.get('deal_id'):
                    current_deal_ids.add(deal['deal_id'])
                    continue
                    
                # If not, extract deal ID from URL
                deal_url = deal.get('url', '')
                if deal_url:
                    # Use our extraction function to get all possible deal IDs
                    deal_ids = extract_deal_ids_from_url(deal_url)
                    if deal_ids:
                        # Use the first extracted ID as the deal_id
                        deal['deal_id'] = deal_ids[0]
                        current_deal_ids.add(deal_ids[0])
                        
                        # Add all extracted IDs to the tracking set
                        for deal_id in deal_ids:
                            current_deal_ids.add(deal_id)
                    
            # Get all deal IDs from database to cross-check
            existing_deal_ids = set()
            deal_id_response = service_client.table(self.table_name)\
                .select('deal_id')\
                .execute()
                
            if deal_id_response.data:
                existing_deal_ids = {row.get('deal_id') for row in deal_id_response.data if row.get('deal_id')}
            
            # Calculate what to add, update, and remove
            urls_to_add = new_deal_urls - existing_urls
            # Only remove URLs that aren't in current scrape AND don't have a deal_id that exists in current scrape
            urls_to_remove = {url for url in (existing_urls - new_deal_urls) 
                             if not any(url_id in current_deal_ids for url_id in extract_deal_ids_from_url(url))}
            urls_to_update = new_deal_urls & existing_urls  # These are already in the database, update them
            
            # Calculated deals to add, update, and remove
            
            added_count = 0
            updated_count = 0
            removed_count = 0
            
            # Add new deals - create a unique deal ID based on the URL to prevent duplicates
            for url in urls_to_add:
                if url in deals_by_url:
                    deal = deals_by_url[url].copy()
                    # Extract deal ID from URL or generate a unique one
                    try:
                        # Extract deal ID from URL (e.g., 202508245716619 from /car-details/202508245716619)
                        import re
                        match = re.search(r'car-details/(\d+)', url)
                        if match:
                            deal_id = match.group(1)
                            deal['deal_id'] = deal_id
                        else:
                            # If can't extract, generate UUID
                            deal['deal_id'] = str(uuid.uuid4())
                    except:
                        # Fallback to UUID
                        deal['deal_id'] = str(uuid.uuid4())
                    
                    deal['created_at'] = datetime.now().isoformat()
                    deal['updated_at'] = datetime.now().isoformat()
                    
                    try:
                        # Check if a deal with this deal_id already exists
                        existing_deal = service_client.table(self.table_name)\
                            .select('id')\
                            .eq('deal_id', deal.get('deal_id'))\
                            .execute()
                        
                        if existing_deal.data and len(existing_deal.data) > 0:
                            # Deal already exists with this deal_id - update it instead of inserting
                            deal_id = existing_deal.data[0]['id']
                            response = service_client.table(self.table_name)\
                                .update(deal)\
                                .eq('id', deal_id)\
                                .execute()
                                
                            if not (hasattr(response, 'error') and response.error):
                                updated_count += 1  # Count as update instead of add
                        else:
                            # Insert new deal
                            response = service_client.table(self.table_name)\
                                .insert(deal)\
                                .execute()
                            
                            if not (hasattr(response, 'error') and response.error):
                                added_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Failed to add deal {url}: {e}")
            
            # Update existing deals with fresh data
            for url in urls_to_update:
                if url in deals_by_url and url in existing_url_to_id:
                    deal = deals_by_url[url].copy()
                    deal['updated_at'] = datetime.now().isoformat()
                    deal_id = existing_url_to_id[url]
                    
                    try:
                        response = service_client.table(self.table_name)\
                            .update(deal)\
                            .eq('id', deal_id)\
                            .execute()
                        
                        if not (hasattr(response, 'error') and response.error):
                            updated_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Failed to update deal {url}: {e}")
            
            # Remove deals that are no longer available (sold/removed)
            if urls_to_remove:
                for url in urls_to_remove:
                    if url in existing_url_to_id:
                        deal_id = existing_url_to_id[url]
                        
                        # Extract database deal ID and verify it's not in current scrape
                        db_deal_response = service_client.table(self.table_name)\
                            .select('deal_id')\
                            .eq('id', deal_id)\
                            .execute()
                            
                        should_delete = True
                        if db_deal_response.data and len(db_deal_response.data) > 0:
                            db_deal_id = db_deal_response.data[0].get('deal_id')
                            if db_deal_id and db_deal_id in current_deal_ids:
                                # This deal was found in the current scrape with a different URL
                                # We should NOT delete it
                                should_delete = False
                                print(f"   ℹ️  Keeping deal {db_deal_id} - found in current scrape with different URL")
                        
                        if should_delete:
                            try:
                                response = service_client.table(self.table_name)\
                                    .delete()\
                                    .eq('id', deal_id)\
                                    .execute()
                                
                                if not (hasattr(response, 'error') and response.error):
                                    removed_count += 1
                            except Exception as e:
                                print(f"   ⚠️  Failed to remove deal {url}: {e}")
                        else:
                            # Count as updated instead of removed
                            updated_count += 1
            
            total_deals_after = len(existing_urls) + added_count - removed_count
            
            # Deal synchronization complete
            # Add some debug info to help understand what's happening
            print(f"   🔄 Sync debug: {len(new_deal_urls)} new URLs, {len(current_deal_ids)} deal IDs in scrape")
            print(f"   🔄 Sync debug: {len(existing_urls)} existing URLs, {len(existing_deal_ids)} deal IDs in DB")
            print(f"   🔄 Sync debug: {len(urls_to_add)} URLs to add, {len(urls_to_update)} to update, {len(urls_to_remove)} to remove")
            
            return {
                "success": True,
                "added_count": added_count,
                "updated_count": updated_count,
                "removed_count": removed_count,
                "total_deals": total_deals_after,
                "new_deals_found": len(new_deal_urls),
                "existing_deals_before": len(existing_urls),
                "preserved_deal_ids": len(current_deal_ids & existing_deal_ids)
            }
            
        except Exception as e:
            error_msg = f"Error during intelligent sync: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
