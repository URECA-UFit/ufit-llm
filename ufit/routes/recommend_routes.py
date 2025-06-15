from fastapi import APIRouter, Query, Depends
from ufit.services.recommend_service import search_similar_plans, make_recommend
from sqlalchemy.orm import Session
from pymongo.database import Database
from ufit.database.database import get_db, get_mongo_db
from pydantic import BaseModel
from typing import List

recommend_router = APIRouter()

@recommend_router.get("/recommend")
def recommend_plan(query: str = Query(..., description="query about plans")):
    results = search_similar_plans(query)
    return {
        "query": query, 
        "recommendations": [
            {
                "content": result["content"],
                "scoer": result["score"]
            } for result in results
        ]
    }   

class RecommendRequest(BaseModel):
    user_id: int
    context: str
    chat_room_id: int

@recommend_router.post("/api/chats/message/ai")
def make_recommend_endpoint(
    req: RecommendRequest,
    db: Session = Depends(get_db),
    mongo_db: Database = Depends(get_mongo_db),
):
    recs = make_recommend(
        user_id=req.user_id,
        base_prompt=req.context,
        chat_room_id=req.chat_room_id,
        postgre_db=db,
        mongo_db=mongo_db,
    )
    return recs