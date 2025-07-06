#!/usr/bin/env python3
"""
Enhanced Database Column Detection Test Script

This script provides comprehensive testing and debugging for database column detection
issues when saving car deals to the database.
"""

import sys
import os
from typing import List, Dict, Set

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Check if we have environment variables
HAVE_SUPABASE_CREDENTIALS = (
    os.getenv('SUPABASE_URL') is not None and 
    os.getenv('SUPABASE_KEY') is not None
)

if HAVE_SUPABASE_CREDENTIALS:
    from supabase_storage import SupabaseStorage
else:
    print("⚠️  Supabase credentials not found in environment variables.")
    print("   This test will run in mock mode for debugging purposes.")
    print("   To run full tests, set SUPABASE_URL and SUPABASE_KEY environment variables.")
    SupabaseStorage = None


class DatabaseColumnAnalyzer:
    """Comprehensive analyzer for database column detection issues"""
    
    def __init__(self, table_name='car_deals'):
        self.table_name = table_name
        self.storage = None
        self.expected_columns = [
            "id", "deal_id", "created_at", "updated_at", "make", "model", 
            "year", "spec", "registration", "engine_size", "fuel_type", 
            "transmission", "body_type", "doors", "mileage", "price_numeric", 
            "enhanced_retail_estimate", "enhanced_net_sale_price", 
            "enhanced_gross_cash_profit", "enhanced_gross_margin_pct", 
            "profit_potential_pct", "absolute_profit", "deal_rating", 
            "enhanced_rating", "spec_analysis", "location", "distance", 
            "seller_type", "title", "subtitle", "full_title", "url", 
            "comparison_url", "image_url", "image_url_2", "date_added", 
            "test_record", "price", "analysis_method"
        ]
        
    def test_database_connection(self) -> bool:
        """Enhanced database connection test with verbose column output"""
        print("🔍 Testing Database Connection & Column Detection")
        print("=" * 60)
        
        # Check if we have credentials
        if not HAVE_SUPABASE_CREDENTIALS:
            print("❌ Cannot test database connection - no Supabase credentials")
            print("   Set SUPABASE_URL and SUPABASE_KEY environment variables")
            return False
        
        try:
            # Initialize storage
            print("📡 Initializing Supabase connection...")
            self.storage = SupabaseStorage(self.table_name)
            print("✅ Connection established successfully")
            
            # Test basic table access
            print("\n🗂️  Testing table access...")
            table_stats = self.storage.get_table_stats()
            
            if table_stats.get('status') == 'available':
                print(f"✅ Table '{self.table_name}' is accessible")
                print(f"📊 Current record count: {table_stats.get('item_count', 'Unknown')}")
            else:
                print(f"❌ Table access failed: {table_stats.get('error', 'Unknown error')}")
                return False
            
            # Test column detection
            print("\n🔍 Testing column detection...")
            detected_columns = self.storage.get_valid_columns()
            
            if detected_columns:
                print(f"✅ Successfully detected {len(detected_columns)} columns")
                print("📋 Detected columns:")
                for i, col in enumerate(sorted(detected_columns), 1):
                    print(f"   {i:2d}. {col}")
            else:
                print("❌ No columns detected - this indicates a problem!")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Database connection test failed: {e}")
            return False
    
    def analyze_database_columns(self) -> Dict:
        """Compare detected columns with expected columns"""
        print("\n🔬 Analyzing Database Column Detection")
        print("=" * 60)
        
        if not HAVE_SUPABASE_CREDENTIALS:
            print("❌ Cannot analyze database columns - no Supabase credentials")
            print("   Running in mock mode - showing expected vs theoretical detected columns")
            
            # Mock analysis for demonstration
            return self._mock_column_analysis()
        
        if not self.storage:
            print("❌ No database connection available")
            return {"success": False, "error": "No database connection"}
        
        # Get detected columns
        detected_columns = self.storage.get_valid_columns()
        detected_set = set(detected_columns)
        expected_set = set(self.expected_columns)
        
        # Calculate differences
        missing_columns = expected_set - detected_set
        extra_columns = detected_set - expected_set
        matching_columns = expected_set & detected_set
        
        print(f"📊 Column Analysis Results:")
        print(f"   Expected columns: {len(self.expected_columns)}")
        print(f"   Detected columns: {len(detected_columns)}")
        print(f"   Matching columns: {len(matching_columns)}")
        print(f"   Missing columns:  {len(missing_columns)}")
        print(f"   Extra columns:    {len(extra_columns)}")
        
        # Show detailed breakdown
        if missing_columns:
            print(f"\n❌ Missing Columns ({len(missing_columns)}):")
            for col in sorted(missing_columns):
                print(f"   • {col}")
        
        if extra_columns:
            print(f"\n➕ Extra Columns ({len(extra_columns)}):")
            for col in sorted(extra_columns):
                print(f"   • {col}")
        
        if matching_columns:
            print(f"\n✅ Matching Columns ({len(matching_columns)}):")
            for col in sorted(matching_columns):
                print(f"   • {col}")
        
        # Calculate success rate
        success_rate = (len(matching_columns) / len(self.expected_columns)) * 100
        print(f"\n📈 Column Detection Success Rate: {success_rate:.1f}%")
        
        return {
            "success": True,
            "expected_count": len(self.expected_columns),
            "detected_count": len(detected_columns),
            "matching_count": len(matching_columns),
            "missing_count": len(missing_columns),
            "extra_count": len(extra_columns),
            "success_rate": success_rate,
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "matching_columns": list(matching_columns)
        }
    
    def _mock_column_analysis(self) -> Dict:
        """Mock column analysis for demonstration when no database connection"""
        # Simulated detected columns (what might be found in a real scenario)
        mock_detected = [
            "id", "deal_id", "created_at", "updated_at", "make", "model", 
            "year", "engine_size", "fuel_type", "transmission", "body_type", 
            "doors", "mileage", "price_numeric", "location", "distance", 
            "seller_type", "title", "subtitle", "url", "image_url", 
            "date_added", "price"
        ]
        
        detected_set = set(mock_detected)
        expected_set = set(self.expected_columns)
        
        missing_columns = expected_set - detected_set
        extra_columns = detected_set - expected_set
        matching_columns = expected_set & detected_set
        
        print(f"📊 Mock Column Analysis Results:")
        print(f"   Expected columns: {len(self.expected_columns)}")
        print(f"   Mock detected columns: {len(mock_detected)}")
        print(f"   Matching columns: {len(matching_columns)}")
        print(f"   Missing columns:  {len(missing_columns)}")
        print(f"   Extra columns:    {len(extra_columns)}")
        
        if missing_columns:
            print(f"\n❌ Missing Columns ({len(missing_columns)}):")
            for col in sorted(missing_columns):
                print(f"   • {col}")
        
        if extra_columns:
            print(f"\n➕ Extra Columns ({len(extra_columns)}):")
            for col in sorted(extra_columns):
                print(f"   • {col}")
        
        success_rate = (len(matching_columns) / len(self.expected_columns)) * 100
        print(f"\n📈 Mock Column Detection Success Rate: {success_rate:.1f}%")
        
        return {
            "success": True,
            "expected_count": len(self.expected_columns),
            "detected_count": len(mock_detected),
            "matching_count": len(matching_columns),
            "missing_count": len(missing_columns),
            "extra_count": len(extra_columns),
            "success_rate": success_rate,
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "matching_columns": list(matching_columns),
            "mock_mode": True
        }
    
    def check_case_sensitivity_issues(self) -> Dict:
        """Check for case sensitivity and whitespace issues in column names"""
        print("\n🔍 Checking Case Sensitivity & Whitespace Issues")
        print("=" * 60)
        
        if not HAVE_SUPABASE_CREDENTIALS:
            print("❌ Cannot check case sensitivity - no Supabase credentials")
            print("   Running in mock mode for demonstration")
            return self._mock_case_sensitivity_check()
        
        if not self.storage:
            print("❌ No database connection available")
            return {"success": False, "error": "No database connection"}
        
        detected_columns = self.storage.get_valid_columns()
        issues_found = []
        
        # Check for case sensitivity issues
        detected_lower = [col.lower() for col in detected_columns]
        expected_lower = [col.lower() for col in self.expected_columns]
        
        case_issues = []
        for expected_col in self.expected_columns:
            if expected_col not in detected_columns:
                # Check if there's a case-insensitive match
                for detected_col in detected_columns:
                    if expected_col.lower() == detected_col.lower():
                        case_issues.append({
                            "expected": expected_col,
                            "detected": detected_col,
                            "issue": "case_mismatch"
                        })
        
        # Check for whitespace issues
        whitespace_issues = []
        for detected_col in detected_columns:
            if detected_col != detected_col.strip():
                whitespace_issues.append({
                    "column": detected_col,
                    "issue": "whitespace",
                    "cleaned": detected_col.strip()
                })
        
        # Report findings
        if case_issues:
            print(f"⚠️  Case Sensitivity Issues Found ({len(case_issues)}):")
            for issue in case_issues:
                print(f"   • Expected: '{issue['expected']}' vs Detected: '{issue['detected']}'")
        
        if whitespace_issues:
            print(f"⚠️  Whitespace Issues Found ({len(whitespace_issues)}):")
            for issue in whitespace_issues:
                print(f"   • Column: '{issue['column']}' (has leading/trailing whitespace)")
        
        if not case_issues and not whitespace_issues:
            print("✅ No case sensitivity or whitespace issues found")
        
        return {
            "success": True,
            "case_issues": case_issues,
            "whitespace_issues": whitespace_issues,
            "total_issues": len(case_issues) + len(whitespace_issues)
        }
    
    def _mock_case_sensitivity_check(self) -> Dict:
        """Mock case sensitivity check for demonstration"""
        # Mock some common issues
        mock_case_issues = [
            {"expected": "deal_id", "detected": "Deal_ID", "issue": "case_mismatch"},
            {"expected": "fuel_type", "detected": "Fuel_Type", "issue": "case_mismatch"}
        ]
        
        mock_whitespace_issues = [
            {"column": " price ", "issue": "whitespace", "cleaned": "price"}
        ]
        
        print(f"📊 Mock Case Sensitivity Check Results:")
        
        if mock_case_issues:
            print(f"⚠️  Mock Case Issues ({len(mock_case_issues)}):")
            for issue in mock_case_issues:
                print(f"   • Expected: '{issue['expected']}' vs Detected: '{issue['detected']}'")
        
        if mock_whitespace_issues:
            print(f"⚠️  Mock Whitespace Issues ({len(mock_whitespace_issues)}):")
            for issue in mock_whitespace_issues:
                print(f"   • Column: '{issue['column']}' (has leading/trailing whitespace)")
        
        return {
            "success": True,
            "case_issues": mock_case_issues,
            "whitespace_issues": mock_whitespace_issues,
            "total_issues": len(mock_case_issues) + len(mock_whitespace_issues),
            "mock_mode": True
        }
    
    def debug_get_valid_columns_method(self) -> Dict:
        """Provide detailed debugging of the get_valid_columns method"""
        print("\n🛠️  Debugging get_valid_columns() Method")
        print("=" * 60)
        
        if not HAVE_SUPABASE_CREDENTIALS:
            print("❌ Cannot debug get_valid_columns - no Supabase credentials")
            print("   Running in mock mode for demonstration")
            return self._mock_debug_method()
        
        if not self.storage:
            print("❌ No database connection available")
            return {"success": False, "error": "No database connection"}
        
        # Clear cache to force fresh query
        self.storage._valid_columns = None
        
        print("🔄 Clearing column cache and performing fresh query...")
        
        try:
            # Debug the query step by step
            print(f"📋 Querying information_schema.columns for table: '{self.table_name}'")
            
            # Make the query directly to see what we get
            response = self.storage.supabase.table('information_schema.columns')\
                .select('column_name, data_type, is_nullable')\
                .eq('table_name', self.table_name)\
                .execute()
            
            print(f"📡 Query executed successfully")
            print(f"📊 Response type: {type(response)}")
            print(f"📊 Has data attribute: {hasattr(response, 'data')}")
            
            if hasattr(response, 'data'):
                print(f"📊 Data length: {len(response.data) if response.data else 0}")
                
                if response.data:
                    print(f"📋 Raw column data (first 3 entries):")
                    for i, col_data in enumerate(response.data[:3]):
                        print(f"   {i+1}. {col_data}")
                    
                    # Extract just column names
                    column_names = [col['column_name'] for col in response.data]
                    print(f"\n📝 Extracted column names ({len(column_names)}):")
                    for col in sorted(column_names):
                        print(f"   • {col}")
                    
                    return {
                        "success": True,
                        "raw_data": response.data,
                        "column_names": column_names,
                        "query_successful": True
                    }
                else:
                    print("❌ Query returned empty data")
                    return {
                        "success": False,
                        "error": "Query returned empty data",
                        "query_successful": True
                    }
            else:
                print("❌ Response has no data attribute")
                return {
                    "success": False,
                    "error": "Response has no data attribute",
                    "query_successful": False
                }
                
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_successful": False
            }
    
    def _mock_debug_method(self) -> Dict:
        """Mock debug method for demonstration"""
        print("🔄 Mock: Clearing column cache and performing fresh query...")
        print(f"📋 Mock: Querying information_schema.columns for table: '{self.table_name}'")
        print(f"📡 Mock: Query executed successfully")
        print(f"📊 Mock: Response type: <class 'postgrest.APIResponse'>")
        print(f"📊 Mock: Has data attribute: True")
        print(f"📊 Mock: Data length: 23")
        
        # Mock raw data
        mock_raw_data = [
            {"column_name": "id", "data_type": "bigint", "is_nullable": "NO"},
            {"column_name": "deal_id", "data_type": "text", "is_nullable": "YES"},
            {"column_name": "created_at", "data_type": "timestamp with time zone", "is_nullable": "YES"}
        ]
        
        print(f"📋 Mock: Raw column data (first 3 entries):")
        for i, col_data in enumerate(mock_raw_data):
            print(f"   {i+1}. {col_data}")
        
        mock_column_names = [
            "id", "deal_id", "created_at", "updated_at", "make", "model", 
            "year", "engine_size", "fuel_type", "transmission", "body_type", 
            "doors", "mileage", "price_numeric", "location", "distance", 
            "seller_type", "title", "subtitle", "url", "image_url", 
            "date_added", "price"
        ]
        
        print(f"\n📝 Mock: Extracted column names ({len(mock_column_names)}):")
        for col in sorted(mock_column_names):
            print(f"   • {col}")
        
        return {
            "success": True,
            "raw_data": mock_raw_data,
            "column_names": mock_column_names,
            "query_successful": True,
            "mock_mode": True
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive debug report"""
        print("\n📊 Generating Comprehensive Debug Report")
        print("=" * 60)
        
        report = {
            "timestamp": None,
            "table_name": self.table_name,
            "connection_test": None,
            "column_analysis": None,
            "case_sensitivity_check": None,
            "debug_method": None,
            "sample_data_filtering": None,
            "recommendations": [],
            "mock_mode": not HAVE_SUPABASE_CREDENTIALS
        }
        
        # Import datetime for timestamp
        from datetime import datetime
        report["timestamp"] = datetime.now().isoformat()
        
        # Run all tests
        print("🔍 Running comprehensive analysis...")
        
        # Test 1: Database connection
        connection_success = self.test_database_connection()
        report["connection_test"] = {"success": connection_success}
        
        # Always run analysis - either real or mock
        if connection_success or not HAVE_SUPABASE_CREDENTIALS:
            # Test 2: Column analysis
            report["column_analysis"] = self.analyze_database_columns()
            
            # Test 3: Case sensitivity check
            report["case_sensitivity_check"] = self.check_case_sensitivity_issues()
            
            # Test 4: Debug method
            report["debug_method"] = self.debug_get_valid_columns_method()
            
            # Test 5: Sample data filtering
            report["sample_data_filtering"] = self.test_sample_data_filtering()
            
            # Generate recommendations
            report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def test_sample_data_filtering(self) -> Dict:
        """Test how the filter_deal_fields method works with sample data"""
        print("\n🧪 Testing Sample Data Filtering")
        print("=" * 60)
        
        # Sample deal data that might be passed to the storage
        sample_deal = {
            "id": 12345,
            "deal_id": "202505222699281",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "make": "ABARTH",
            "model": "595",
            "year": 2018,
            "spec": "1.4 T-Jet",
            "registration": "AB18 XYZ",
            "engine_size": 1.4,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "body_type": "Hatchback",
            "doors": 3,
            "mileage": 88803,
            "price_numeric": 6899,
            "enhanced_retail_estimate": 7500,
            "enhanced_net_sale_price": 6800,
            "enhanced_gross_cash_profit": 1200,
            "enhanced_gross_margin_pct": 15.5,
            "profit_potential_pct": 12.3,
            "absolute_profit": 901,
            "deal_rating": "Good Deal",
            "enhanced_rating": "Excellent Deal",
            "spec_analysis": {"comprehensive_factors_applied": True},
            "location": "York",
            "distance": "58 miles",
            "seller_type": "TRADE",
            "title": "Abarth 595",
            "subtitle": "1.4 T-Jet 70th Euro 6 3dr",
            "full_title": "Abarth 595 1.4 T-Jet 70th Euro 6 3dr",
            "url": "https://www.autotrader.co.uk/car-details/202505222699281",
            "comparison_url": "https://www.autotrader.co.uk/compare/202505222699281",
            "image_url": "https://example.com/image1.jpg",
            "image_url_2": "https://example.com/image2.jpg",
            "date_added": "2025-01-01",
            "test_record": False,
            "price": "£6,899",
            "analysis_method": "enhanced",
            # Some fields that might not exist in database
            "invalid_field_1": "should_be_removed",
            "invalid_field_2": "should_also_be_removed"
        }
        
        print(f"📊 Sample deal data contains {len(sample_deal)} fields")
        print(f"📋 Sample fields: {', '.join(list(sample_deal.keys())[:10])}...")
        
        if not HAVE_SUPABASE_CREDENTIALS:
            print("❌ Cannot test real filtering - no Supabase credentials")
            print("   Running mock filtering demonstration")
            return self._mock_data_filtering(sample_deal)
        
        if not self.storage:
            print("❌ No database connection available")
            return {"success": False, "error": "No database connection"}
        
        try:
            # Make a copy to avoid modifying the original
            test_deal = sample_deal.copy()
            
            print(f"🔧 Applying filter_deal_fields to sample data...")
            filtered_deal = self.storage.filter_deal_fields(test_deal)
            
            print(f"📊 After filtering: {len(filtered_deal)} fields remain")
            
            # Compare before and after
            original_fields = set(sample_deal.keys())
            filtered_fields = set(filtered_deal.keys())
            removed_fields = original_fields - filtered_fields
            
            if removed_fields:
                print(f"🗑️  Removed {len(removed_fields)} invalid fields:")
                for field in sorted(removed_fields):
                    print(f"   • {field}")
            else:
                print("✅ No fields were removed")
            
            print(f"📋 Remaining fields: {', '.join(sorted(filtered_fields))}")
            
            return {
                "success": True,
                "original_field_count": len(sample_deal),
                "filtered_field_count": len(filtered_deal),
                "removed_field_count": len(removed_fields),
                "removed_fields": list(removed_fields),
                "remaining_fields": list(filtered_fields)
            }
            
        except Exception as e:
            print(f"❌ Error during filtering test: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _mock_data_filtering(self, sample_deal: Dict) -> Dict:
        """Mock data filtering for demonstration"""
        # Mock detected columns (subset of expected)
        mock_detected = [
            "id", "deal_id", "created_at", "updated_at", "make", "model", 
            "year", "engine_size", "fuel_type", "transmission", "body_type", 
            "doors", "mileage", "price_numeric", "location", "distance", 
            "seller_type", "title", "subtitle", "url", "image_url", 
            "date_added", "price"
        ]
        
        original_fields = set(sample_deal.keys())
        mock_valid_fields = set(mock_detected)
        
        # Fields that would be removed
        removed_fields = original_fields - mock_valid_fields
        remaining_fields = original_fields & mock_valid_fields
        
        print(f"🔧 Mock: Applying filter_deal_fields to sample data...")
        print(f"📊 Mock: After filtering: {len(remaining_fields)} fields remain")
        
        if removed_fields:
            print(f"🗑️  Mock: Would remove {len(removed_fields)} invalid fields:")
            for field in sorted(removed_fields):
                print(f"   • {field}")
        else:
            print("✅ Mock: No fields would be removed")
        
        return {
            "success": True,
            "original_field_count": len(sample_deal),
            "filtered_field_count": len(remaining_fields),
            "removed_field_count": len(removed_fields),
            "removed_fields": list(removed_fields),
            "remaining_fields": list(remaining_fields),
            "mock_mode": True
        }
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        # Handle mock mode
        if report.get("mock_mode", False):
            recommendations.append("🔧 Running in mock mode due to missing Supabase credentials.")
            recommendations.append("⚠️ Set SUPABASE_URL and SUPABASE_KEY environment variables for real testing.")
        
        if report.get("column_analysis", {}).get("success"):
            analysis = report["column_analysis"]
            
            if analysis.get("missing_count", 0) > 0:
                recommendations.append(
                    f"🔧 {analysis['missing_count']} expected columns are missing. "
                    f"Check if the table schema matches expectations."
                )
            
            if analysis.get("success_rate", 0) < 50:
                recommendations.append(
                    "⚠️ Low column detection success rate. Verify table name and schema."
                )
        
        if report.get("case_sensitivity_check", {}).get("success"):
            case_check = report["case_sensitivity_check"]
            
            if case_check.get("total_issues", 0) > 0:
                recommendations.append(
                    f"🔤 {case_check['total_issues']} case sensitivity/whitespace issues found. "
                    f"Review column naming conventions."
                )
        
        if report.get("debug_method", {}).get("success"):
            debug_info = report["debug_method"]
            
            if not debug_info.get("query_successful", False):
                recommendations.append(
                    "🔧 Column query failed. Check database permissions and table existence."
                )
        
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        # Handle mock mode
        if report.get("mock_mode", False):
            recommendations.append("🔧 Running in mock mode due to missing Supabase credentials.")
            recommendations.append("⚠️ Set SUPABASE_URL and SUPABASE_KEY environment variables for real testing.")
        
        if report.get("column_analysis", {}).get("success"):
            analysis = report["column_analysis"]
            
            if analysis.get("missing_count", 0) > 0:
                recommendations.append(
                    f"🔧 {analysis['missing_count']} expected columns are missing. "
                    f"Check if the table schema matches expectations."
                )
            
            if analysis.get("success_rate", 0) < 50:
                recommendations.append(
                    "⚠️ Low column detection success rate. Verify table name and schema."
                )
        
        if report.get("case_sensitivity_check", {}).get("success"):
            case_check = report["case_sensitivity_check"]
            
            if case_check.get("total_issues", 0) > 0:
                recommendations.append(
                    f"🔤 {case_check['total_issues']} case sensitivity/whitespace issues found. "
                    f"Review column naming conventions."
                )
        
        if report.get("debug_method", {}).get("success"):
            debug_info = report["debug_method"]
            
            if not debug_info.get("query_successful", False):
                recommendations.append(
                    "🔧 Column query failed. Check database permissions and table existence."
                )
        
        if report.get("sample_data_filtering", {}).get("success"):
            filtering_info = report["sample_data_filtering"]
            
            if filtering_info.get("removed_field_count", 0) > 0:
                recommendations.append(
                    f"🗑️  {filtering_info['removed_field_count']} fields would be removed during filtering. "
                    f"Check if these are expected to be filtered out."
                )
        
        if not recommendations:
            recommendations.append("✅ No major issues detected. Column detection appears to be working correctly.")
        
        return recommendations


def main():
    """Main function to run the comprehensive database column analysis"""
    print("🚀 Enhanced Database Column Detection Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DatabaseColumnAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Print final summary
    print("\n📋 Final Summary & Recommendations")
    print("=" * 60)
    
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"{i}. {recommendation}")
    
    print(f"\n📊 Analysis completed at: {report['timestamp']}")
    print("🔍 For detailed debugging, check the output above.")
    
    return report


if __name__ == "__main__":
    try:
        report = main()
        
        # Exit with appropriate code
        if report["connection_test"]["success"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)