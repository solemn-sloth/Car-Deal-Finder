"""
FastAPI server to expose AutoTrader scraper data.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import os
from datetime import datetime
import json

app = FastAPI(title="AutoTrader Deal Finder API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "http://localhost:3000",  # Local Next.js development
        "https://car-deal-bot.vercel.app",  # Main Vercel deployment
        "https://car-deal-bot-*.vercel.app",  # Preview deployments
        "https://frontend-2abdtmplu-hishams-projects-39cc6fd4.vercel.app",  # Current deployment
        "https://car-deal-bot.netlify.app",  # Netlify main site
        "https://*.netlify.app",  # Any Netlify preview deployments
        "*"  # Allow all origins for debugging
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/deals")
@app.get("/api/deals")
async def get_deals():
    """Get all deals from Supabase only."""
    try:
        from supabase_storage import SupabaseStorage
        supabase_storage = SupabaseStorage()
        vehicles_data = supabase_storage.get_all_deals()
        
        if not vehicles_data:
            return {"deals": [], "data_source": "Supabase", "count": 0}
        
        print("✅ Successfully retrieved deals from Supabase")
        data_source = "Supabase"
        
        if not vehicles_data:
            return {"deals": [], "data_source": data_source, "count": 0}
            
        # Transform data for frontend (handle both DynamoDB and Google Sheets formats)
        deals = []
        for vehicle in vehicles_data:
            deal = {
                "url": vehicle.get("url", ""),
                "make": vehicle.get("make", ""),
                "model": vehicle.get("model", ""),
                "spec": vehicle.get("spec", ""),
                "year": vehicle.get("year", 0),
                "mileage": vehicle.get("mileage", 0),
                "registration": vehicle.get("registration", ""),
                "price_numeric": vehicle.get("price_numeric", vehicle.get("price", 0)),
                "profit_potential_pct": vehicle.get("profit_potential_pct", 0),
                "absolute_profit": vehicle.get("absolute_profit", 0),
                "location": vehicle.get("location", ""),
                "deal_rating": vehicle.get("deal_rating", "Unknown"),
                "image_url": vehicle.get("image_url", ""),
                "image_url_2": vehicle.get("image_url_2", ""),
                "comparisonUrl": vehicle.get("comparisonUrl", ""),
                "dateAdded": vehicle.get("date_added", vehicle.get("dateAdded", datetime.now().strftime("%Y-%m-%d"))),
                "deal_id": vehicle.get("deal_id", ""),  # DynamoDB specific field
                "created_at": vehicle.get("created_at", ""),  # DynamoDB specific field
            }
            deals.append(deal)
        
        return {
            "deals": deals,
            "data_source": data_source,
            "count": len(deals),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving deals: {str(e)}")

@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Check health of both storage systems."""
    health = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "storage_systems": {}
    }
    
    # Check Supabase
    try:
        from supabase_storage import SupabaseStorage
        supabase_storage = SupabaseStorage()
        stats = supabase_storage.get_table_stats()
        health["storage_systems"]["supabase"] = {
            "status": "connected",
            "table_name": stats.get("table_name"),
            "item_count": stats.get("item_count", 0)
        }
    except Exception as e:
        health["storage_systems"]["supabase"] = {
            "status": "error",
            "error": str(e)
        }
        health["status"] = "unhealthy"
    
    return health

@app.get("/deals/top")
@app.get("/api/deals/top")
async def get_top_deals(limit: int = 50, min_profit: float = 0.0):
    """Get top deals by profit potential from Supabase."""
    try:
        from supabase_storage import SupabaseStorage
        supabase_storage = SupabaseStorage()
        
        # Get deals filtered by profit potential
        vehicles_data = supabase_storage.get_deals_by_profit_potential(
            min_profit=min_profit,
            limit=limit
        )
        
        if not vehicles_data:
            return {"deals": [], "data_source": "Supabase", "count": 0}
        
        # Transform data for frontend
        deals = []
        for vehicle in vehicles_data:
            deal = {
                "url": vehicle.get("url", ""),
                "make": vehicle.get("make", ""),
                "model": vehicle.get("model", ""),
                "spec": vehicle.get("spec", ""),
                "year": vehicle.get("year", 0),
                "mileage": vehicle.get("mileage", 0),
                "registration": vehicle.get("registration", ""),
                "price_numeric": vehicle.get("price_numeric", vehicle.get("price", 0)),
                "profit_potential_pct": vehicle.get("profit_potential_pct", 0),
                "absolute_profit": vehicle.get("absolute_profit", 0),
                "location": vehicle.get("location", ""),
                "deal_rating": vehicle.get("deal_rating", "Unknown"),
                "image_url": vehicle.get("image_url", ""),
                "image_url_2": vehicle.get("image_url_2", ""),
                "comparisonUrl": vehicle.get("comparisonUrl", ""),
                "dateAdded": vehicle.get("date_added", datetime.now().strftime("%Y-%m-%d")),
                "deal_id": vehicle.get("deal_id", ""),
                "created_at": vehicle.get("created_at", ""),
            }
            deals.append(deal)
        
        return {
            "deals": deals,
            "data_source": "Supabase",
            "count": len(deals),
            "filters": {
                "min_profit": min_profit,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving top deals: {str(e)}")

@app.get("/stats")
@app.get("/api/stats")
async def get_storage_stats():
    """Get statistics from Supabase storage system."""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "storage_systems": {}
    }
    
    # Supabase stats only
    try:
        from supabase_storage import SupabaseStorage
        supabase_storage = SupabaseStorage()
        supabase_stats = supabase_storage.get_table_stats()
        stats["storage_systems"]["supabase"] = supabase_stats
    except Exception as e:
        stats["storage_systems"]["supabase"] = {"error": str(e)}
    
    return stats
