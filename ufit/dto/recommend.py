from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    userId: int
    content: str
    chatRoomId: int

class PlanDTO(BaseModel):
    planId: str
    name: str

class RecommendResponse(BaseModel):
    messagesId: str
    answer: str
    recommendPlan: List[PlanDTO]