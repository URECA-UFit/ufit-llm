from fastapi import APIRouter, Query
from ufit.services.recommend_service import search_similar_plans

recommend_router = APIRouter()

@recommend_router.get("/recommend")
def recommend_plan(query: str = Query(..., description="query about plans")):
    results = search_similar_plans(query)
    return {"query": query, "recommendations": results}