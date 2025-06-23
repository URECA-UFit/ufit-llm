# LLM 응답 처리
from fastapi import APIRouter, Depends, Body
from pymongo.database import Database
from ufit.database.database import get_mongo_db
from ufit.services.chat_bot_review_user_query import summarize_user_queries_with_llm, get_user_queries_after_recommendation

chat_router = APIRouter()

@chat_router.post("/api/chats/review/{chatroomId}")
def review_user_query_summary_by_id(
    chatroomId: int,
    recommendation_message_id: str = Body(..., embed=True),
    mongo_db: Database = Depends(get_mongo_db)
):
    user_msgs = get_user_queries_after_recommendation(mongo_db, chatroomId, recommendation_message_id)
    summary = summarize_user_queries_with_llm(user_msgs)
    return {"summary": summary}

