from fastapi import APIRouter, Query, Depends
from ufit.services.recommend_service import make_recommend
from sqlalchemy.orm import Session
from pymongo.database import Database
from ufit.database.database import get_db, get_mongo_db
from ufit.dto.recommend import RecommendRequest, RecommendResponse


recommend_router = APIRouter()

@recommend_router.post(
    "/api/chats/message/ai",
    response_model=RecommendResponse,
)
def make_recommend_endpoint(
    req: RecommendRequest,
    db: Session = Depends(get_db),
    mongo_db: Database = Depends(get_mongo_db),
) -> RecommendResponse:
    recs = make_recommend(
        user_id=req.userId,
        base_prompt=req.content,
        chat_room_id=req.chatRoomId,
        postgre_db=db,
        mongo_db=mongo_db,
    )
    return recs