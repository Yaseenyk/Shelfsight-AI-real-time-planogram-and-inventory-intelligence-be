"""v1 router aggregation — one module per retail-vision capability."""

from fastapi import APIRouter

from app.api.v1 import expiry, freshness, insights, inventory, planogram

api_router = APIRouter()
api_router.include_router(inventory.router, prefix="/inventory", tags=["inventory"])
api_router.include_router(planogram.router, prefix="/planogram", tags=["planogram"])
api_router.include_router(freshness.router, prefix="/freshness", tags=["freshness"])
api_router.include_router(expiry.router, prefix="/expiry", tags=["expiry"])
api_router.include_router(insights.router, prefix="/insights", tags=["insights"])
